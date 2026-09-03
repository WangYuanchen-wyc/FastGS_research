#
# Paper B - B7-Fix: corrected dynamic features from cached data (CPU-only).
#
# Fixes vs B7:
#  1. gradient direction consistency: arctan2 already returns radians
#     (old code double-converted with deg2rad); only valid/visible gradient
#     samples enter the statistic.
#  2. DemandPersistence: Gaussian-following demand history — at each历史
#     sample the candidate's HISTORICAL xyz/scale are re-projected onto the
#     same fixed cost views to build ROI_i(t); residual energy is read from
#     that sample's stored residual map. (Old code used the snapshot-T ROI
#     for all samples.)
#  3. radius/visibility: the per-iteration radii came from the RANDOM
#     training camera — removed as features (kept only as a sample-validity
#     proxy). New fixed-view projected radius/visibility computed from
#     historical xyz/scale (snapshot rotation) on the fixed cost views.
#  4. separate Ineff_xyz / Ineff_scale / Ineff_opacity (no mixed-unit
#     denominator); OptimizationExposure accumulates gradient magnitude over
#     visible samples only.
#
# Reuses cache (snap7_*.pt / dyn_*.pt / oracle labels via b7_dynamic_features
# .json); NO new 100-step oracles, NO replay, no FastGS change.
#

import os
import sys
import json
import csv

import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from diagnostics.common import project_to_pixel, roi_box, \
    projected_gaussian_geometry  # noqa: E402


class CamShim:
    """CPU camera replica (no .cuda()) — only the attributes used by the
    projection helpers: name/uid, FoV, size, world_view / full_proj matrices.
    Math replicated exactly from utils/camera_utils.loadCam + scene/cameras.py
    (trans=0, scale=1); order taken from the SAVED camera identity list."""


def build_cpu_cams(source_path, ident_train, view_pool):
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
    from scene.dataset_readers import focal2fov
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    extr = read_extrinsics_binary(os.path.join(source_path, "sparse/0", "images.bin"))
    intr = read_intrinsics_binary(os.path.join(source_path, "sparse/0", "cameras.bin"))
    info = {}
    for k, im in extr.items():
        c = intr[im.camera_id]
        if c.model == "SIMPLE_PINHOLE":
            fx = fy = c.params[0]
        elif c.model == "PINHOLE":
            fx, fy = c.params[0], c.params[1]
        else:
            raise RuntimeError(f"camera model {c.model} unsupported")
        W0, H0 = c.width, c.height
        down = (W0 / 1600.0) if W0 > 1600 else 1.0
        W, H = int(W0 / down), int(H0 / down)
        name = os.path.basename(im.name).split(".")[0]
        info[name] = (np.transpose(qvec2rotmat(im.qvec)), np.array(im.tvec),
                      focal2fov(fx, W), focal2fov(fy, H), W, H)
    cams = []
    for name, uid in ident_train:
        R, T, fovx, fovy, W, H = info[name]
        c = CamShim()
        c.image_name, c.uid = name, uid
        c.FoVx, c.FoVy, c.image_width, c.image_height = fovx, fovy, W, H
        wvt = torch.tensor(getWorld2View2(R, T, np.zeros(3), 1.0)).transpose(0, 1)
        pm = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1)
        c.world_view_transform = wvt
        c.full_proj_transform = wvt.unsqueeze(0).bmm(pm.unsqueeze(0)).squeeze(0)
        cams.append(c)
    return cams[:view_pool]

B7 = "paper_b/b7_dynamic_capacity_signal"
OUT = "paper_b/b7_fix_dynamic_signal"
TILE_MARGIN_PX = 16
W, SAMP = 100, 10


def _cam_id(c):
    return [str(c.image_name), int(c.uid)]


def main():
    parser = ArgumentParser("Paper B B7-Fix: corrected dynamic features")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args()
    dataset = lp.extract(args)

    rows_old = json.load(open(f"{B7}/data/b7_dynamic_features.json"))
    iters = sorted({r["iteration"] for r in rows_old})

    # CPU camera shim rebuilt from COLMAP in the SAVED identity order
    import random as _rnd
    _rnd.seed(0); np.random.seed(0); torch.manual_seed(0)
    ident = json.load(open(f"{B7}/cache/camera_identity.json"))
    pool_cams = build_cpu_cams(dataset.source_path, ident["train"], 30)
    assert [_cam_id(c) for c in pool_cams] == ident["pool"], "pool identity FAIL"
    print("[b7fix] camera identity assertion PASS (CPU shim, order == collect)")

    out_rows = []
    for it in iters:
        snap = torch.load(f"{B7}/cache/snap7_{it}.pt", map_location="cpu")
        dyn = torch.load(f"{B7}/cache/dyn_{it}.pt", map_location="cpu")
        xyz_snap = snap["snapshot"][1]
        rot_snap = snap["snapshot"][5]                       # (N,4) raw rotation
        members = [r for r in rows_old if r["iteration"] == it]
        idxs = [r["parent_index"] for r in members]
        dxyz = dyn["xyz"].float()                            # (S,N,3)
        dsc = dyn["scale"].float()                           # (S,N,3) activated
        dop = dyn["opacity"].float()
        gn = dyn["gradn"].float()
        gdx = dyn["gdx"].float()
        gdy = dyn["gdy"].float()
        rad_rand = dyn["radii"].float()                      # per-iteration random-camera radii
        l1maps = dyn["l1"].float()                           # (S,8,H,W)
        S = dxyz.shape[0]
        xs = np.arange(S, dtype=np.float64)
        n_views = l1maps.shape[1]
        cost_cams = pool_cams[:n_views]
        H, Wp = int(cost_cams[0].image_height), int(cost_cams[0].image_width)

        def stats(v):
            v = np.asarray(v, dtype=np.float64)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                return {"mean": None, "std": None, "slope": None, "lf": None, "cv": None}
            return {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "slope": float(np.polyfit(xs[:len(v)], v, 1)[0]) if len(v) >= 2 else 0.0,
                    "lf": float(v[-1] - v[0]) if len(v) >= 2 else 0.0,
                    "cv": float(np.std(v) / (abs(np.mean(v)) + 1e-8))}

        # pass 1: Gaussian-following demand energy per candidate per sample
        demand = {i: [] for i in idxs}
        fv_radius = {i: [] for i in idxs}                    # mean fixed-view radius per sample
        fv_vis = {i: [] for i in idxs}                       # fraction of views visible per sample
        for si in range(S):
            for ci, idx in enumerate(idxs):
                p = dxyz[si, idx].numpy()
                sc = dsc[si, idx].numpy()
                ro = rot_snap[idx].float()
                energies, rads, vis = [], [], 0
                for vi, cam in enumerate(cost_cams):
                    pg = projected_gaussian_geometry(cam, torch.tensor(p), torch.tensor(sc), ro)
                    if pg is None:
                        continue
                    r_px = pg["radius_3sigma_px"]
                    u, v, _, ok = project_to_pixel(cam, torch.tensor(p))
                    if not ok or r_px <= 0:
                        continue
                    x0, y0, x1, y1 = roi_box(u, v, float(r_px), cam.image_width,
                                              cam.image_height, TILE_MARGIN_PX)
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        continue
                    energies.append(float(l1maps[si, vi, y0:y1, x0:x1].mean()))
                    rads.append(r_px)
                    vis += 1
                demand[idx].append(float(np.mean(energies)) if energies else np.nan)
                fv_radius[idx].append(float(np.mean(rads)) if rads else np.nan)
                fv_vis[idx].append(vis / n_views)
            if (si + 1) % 5 == 0:
                print(f"[b7fix] it={it} sample {si + 1}/{S}")

        # snapshot-level demand median for high_fraction (all candidate×sample energies)
        allE = [e for i in idxs for e in demand[i] if not np.isnan(e)]
        thr = float(np.median(allE))

        for r, idx in zip(members, idxs):
            xyz_i = dxyz[:, idx].numpy()
            sc_i = np.max(dsc[:, idx].numpy(), axis=1)
            op_i = dop[:, idx].numpy()
            gn_i = gn[:, idx].numpy()
            gdx_i, gdy_i = gdx[:, idx].numpy(), gdy[:, idx].numpy()
            rr_i = rad_rand[:, idx].numpy() > 0             # sample validity (rendered that iter)
            step_xyz = np.linalg.norm(np.diff(xyz_i, axis=0), axis=1)
            step_sc = np.abs(np.diff(sc_i))
            step_op = np.abs(np.diff(op_i))
            path_xyz, net_xyz = float(np.sum(step_xyz)), float(np.linalg.norm(xyz_i[-1] - xyz_i[0]))
            path_sc, path_op = float(np.sum(step_sc)), float(np.sum(step_op))
            gs = stats(gn_i)
            # corrected direction consistency: radians only, valid samples only
            valid = rr_i & (gn_i > 0)
            if valid.sum() >= 2:
                th2 = 2.0 * np.arctan2(gdy_i[valid], gdx_i[valid] + 1e-12)
                gdc = float(np.hypot(np.mean(np.cos(th2)), np.mean(np.sin(th2))))
            else:
                gdc = None
            dem = np.array(demand[idx], dtype=np.float64)
            ds = stats(dem)
            high_frac = float(np.mean(dem[~np.isnan(dem)] > thr)) if (~np.isnan(dem)).any() else None
            rads_t = np.array([x for x in fv_radius[idx] if not np.isnan(x)])
            fvr_mean = float(np.mean(rads_t)) if len(rads_t) else None
            fvr_slope = float(np.polyfit(np.arange(len(rads_t)), rads_t, 1)[0]) if len(rads_t) >= 2 else None
            fv_vis_frac = float(np.mean(fv_vis[idx]))
            row = dict(r)  # keep iteration/parent/labels/static/old-dynamic cols
            row.update({
                "fx_gradn_mean": gs["mean"], "fx_gradn_std": gs["std"],
                "fx_gradn_slope": gs["slope"], "fx_gradn_cv": gs["cv"],
                "fx_grad_dir_consistency": gdc,
                "fx_xyz_path": path_xyz, "fx_xyz_net": net_xyz,
                "fx_xyz_net_over_path": net_xyz / (path_xyz + 1e-8),
                "fx_scale_path": path_sc, "fx_scale_slope": stats(sc_i)["slope"],
                "fx_opacity_path": path_op, "fx_opacity_net": float(op_i[-1] - op_i[0]),
                "fx_demand_mean": ds["mean"], "fx_demand_std": ds["std"],
                "fx_demand_slope": ds["slope"], "fx_demand_last_first": ds["lf"],
                "fx_demand_cv": ds["cv"], "fx_demand_high_fraction": high_frac,
                "fx_OptimizationExposure": float(np.sum(gn_i[rr_i])),
                "fx_Ineff_xyz": (ds["mean"] / (path_xyz + 1e-8)) if ds["mean"] is not None else None,
                "fx_Ineff_scale": (ds["mean"] / (path_sc + 1e-8)) if ds["mean"] is not None else None,
                "fx_Ineff_opacity": (ds["mean"] / (path_op + 1e-8)) if ds["mean"] is not None else None,
                "fx_fixedview_radius_mean": fvr_mean,
                "fx_fixedview_radius_slope": fvr_slope,
                "fx_fixedview_visibility_fraction": fv_vis_frac,
            })
            out_rows.append(row)
        del snap, dyn, l1maps
        print(f"[b7fix] it={it}: {len(members)} candidates done")

    with open(f"{OUT}/data/b7fix_dynamic_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow({k: ("NA" if v is None else (f"{v:.8g}" if isinstance(v, float) else v))
                        for k, v in r.items()})
    json.dump(out_rows, open(f"{OUT}/data/b7fix_dynamic_features.json", "w"))
    print(f"[b7fix] saved {len(out_rows)} rows")


if __name__ == "__main__":
    main()
