#
# Paper B - B7-Closeout: corrected per-family dynamic features (CPU-only).
#
# Two analysis fixes over B7-Fix (task §2):
#  1. gradient temporal statistics over VALID samples only
#     (random-camera radii > 0 AND gradient_norm > 0); slope / direction
#     consistency -> NA when fewer than 2 valid samples.
#  2. slopes use the TRUE sample index of non-NaN entries (no re-indexing
#     after NaN removal) for demand / radius / gradient series.
#
# Recomputes the Gaussian-following demand series and fixed-view radius
# series from the cached dyn bundles (no new oracles / training / replay),
# merges with the static columns of b7fix features, and stores per-sample
# series for auditability.
#

import os
import sys
import json
import csv

import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams  # noqa: E402
from diagnostics.common import project_to_pixel, roi_box, projected_gaussian_geometry  # noqa: E402
from diagnostics.diagnostic_b7fix import build_cpu_cams, _cam_id  # noqa: E402

B7 = "paper_b/b7_dynamic_capacity_signal"
B7F = "paper_b/b7_fix_dynamic_signal"
OUT = "paper_b/b7_closeout_dynamic_ablation"
TILE_MARGIN_PX = 16


def trueidx_slope(v):
    v = np.asarray(v, dtype=np.float64)
    idx = np.where(~np.isnan(v))[0]
    if len(idx) < 2:
        return None
    return float(np.polyfit(idx, v[idx], 1)[0])


def stats_trueidx(v, require_two=True):
    v = np.asarray(v, dtype=np.float64)
    vv = v[~np.isnan(v)]
    return {"mean": float(np.mean(vv)) if len(vv) else None,
            "std": float(np.std(vv)) if len(vv) else None,
            "slope": trueidx_slope(v),
            "lf": float(vv[-1] - vv[0]) if (len(vv) >= 2 and not require_two) or len(vv) >= 2 else None,
            "cv": float(np.std(vv) / (abs(np.mean(vv)) + 1e-8)) if len(vv) else None}


def main():
    parser = ArgumentParser("Paper B B7-Closeout feature recompute")
    lp = ModelParams(parser)
    args = parser.parse_args()
    dataset = lp.extract(args)

    rows_fix = json.load(open(f"{B7F}/data/b7fix_dynamic_features.json"))
    iters = sorted({r["iteration"] for r in rows_fix})
    ident = json.load(open(f"{B7}/cache/camera_identity.json"))
    pool_cams = build_cpu_cams(dataset.source_path, ident["train"], 30)
    assert [_cam_id(c) for c in pool_cams] == ident["pool"]
    print("[b7close] camera identity assertion PASS (CPU shim)")

    out_rows = []
    for it in iters:
        snap = torch.load(f"{B7}/cache/snap7_{it}.pt", map_location="cpu")
        dyn = torch.load(f"{B7}/cache/dyn_{it}.pt", map_location="cpu")
        rot_snap = snap["snapshot"][5]
        members = [r for r in rows_fix if r["iteration"] == it]
        idxs = [r["parent_index"] for r in members]
        dxyz = dyn["xyz"].float()
        dsc = dyn["scale"].float()
        dop = dyn["opacity"].float()
        gn = dyn["gradn"].float()
        gdx = dyn["gdx"].float()
        gdy = dyn["gdy"].float()
        rad_rand = dyn["radii"].float()
        l1maps = dyn["l1"].float()
        S = dxyz.shape[0]
        n_views = l1maps.shape[1]
        cost_cams = pool_cams[:n_views]

        # per-sample series (Gaussian-following demand + fixed-view radius)
        demand = {i: np.full(S, np.nan) for i in idxs}
        fvradius = {i: np.full(S, np.nan) for i in idxs}
        fvvis = {i: np.zeros(S) for i in idxs}
        for si in range(S):
            for idx in idxs:
                p = dxyz[si, idx]
                sc = dsc[si, idx]
                ro = rot_snap[idx].float()
                energies, rads, vis = [], [], 0
                for vi, cam in enumerate(cost_cams):
                    pg = projected_gaussian_geometry(cam, p, sc, ro)
                    if pg is None:
                        continue
                    r_px = pg["radius_3sigma_px"]
                    u, v, _, ok = project_to_pixel(cam, p)
                    if not ok or r_px <= 0:
                        continue
                    x0, y0, x1, y1 = roi_box(u, v, float(r_px), cam.image_width,
                                             cam.image_height, TILE_MARGIN_PX)
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        continue
                    energies.append(float(l1maps[si, vi, y0:y1, x0:x1].mean()))
                    rads.append(r_px)
                    vis += 1
                if energies:
                    demand[idx][si] = float(np.mean(energies))
                if rads:
                    fvradius[idx][si] = float(np.mean(rads))
                fvvis[idx][si] = vis / n_views
            print(f"[b7close] it={it} sample {si + 1}/{S}")

        allE = [e for i in idxs for e in demand[i] if not np.isnan(e)]
        thr = float(np.median(allE))

        for r, idx in zip(members, idxs):
            gn_i = gn[:, idx].numpy()
            valid = (rad_rand[:, idx].numpy() > 0) & (gn_i > 0)
            gv = np.where(valid, gn_i, np.nan)
            gs = stats_trueidx(gv)
            if valid.sum() >= 2:
                th2 = 2.0 * np.arctan2(gdy[valid, idx].numpy(), gdx[valid, idx].numpy() + 1e-12)
                gdc = float(np.hypot(np.mean(np.cos(th2)), np.mean(np.sin(th2))))
            else:
                gdc = None
            dem = demand[idx]
            ds = stats_trueidx(dem)
            high_frac = float(np.mean(dem[~np.isnan(dem)] > thr)) if (~np.isnan(dem)).any() else None
            row = dict(r)  # static + labels + old/fixed features preserved
            row.update({
                "cl_gradn_mean_valid": gs["mean"],
                "cl_gradn_std_valid": gs["std"],
                "cl_gradn_slope_valid": gs["slope"],
                "cl_gradn_cv_valid": gs["cv"],
                "cl_grad_dir_consistency": gdc,
                "cl_Exposure_valid": float(np.nansum(np.where(valid, gn_i, 0.0))),
                "cl_demand_mean": ds["mean"], "cl_demand_std": ds["std"],
                "cl_demand_slope_trueidx": ds["slope"],
                "cl_demand_last_first": ds["lf"], "cl_demand_cv": ds["cv"],
                "cl_demand_high_fraction": high_frac,
                "cl_fixedview_radius_mean": float(np.nanmean(fvradius[idx])) if np.any(~np.isnan(fvradius[idx])) else None,
                "cl_fixedview_radius_slope_trueidx": trueidx_slope(fvradius[idx]),
                "cl_fixedview_visibility_fraction": float(np.mean(fvvis[idx])),
                "cl_n_valid_grad_samples": int(valid.sum()),
                "cl_demand_series": [None if np.isnan(x) else float(x) for x in dem],
            })
            out_rows.append(row)
        del snap, dyn, l1maps
        print(f"[b7close] it={it}: {len(members)} candidates done")

    with open(f"{OUT}/data/b7close_features.csv", "w", newline="") as f:
        cols = [c for c in out_rows[0].keys() if c != "cl_demand_series"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow({k: ("NA" if r.get(k) is None else (f"{r[k]:.8g}" if isinstance(r[k], float) else r[k]))
                        for k in cols})
    json.dump(out_rows, open(f"{OUT}/data/b7close_features.json", "w"))
    print(f"[b7close] saved {len(out_rows)} rows")


if __name__ == "__main__":
    main()
