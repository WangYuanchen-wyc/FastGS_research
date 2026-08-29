#
# Paper B - B3: Pre-Action Oracle Predictability — dataset builder
#
# Loads the B2-C PERSISTED master snapshot (cache/master_snapshot.pt) — never a
# fresh warmup — and computes pre-action descriptors for the exact 881 oracle
# candidates (identity = parent_index within that snapshot, verified via the
# snapshot fingerprint). Reuses v2.build_candidate (B2-A formulas) for the
# FastGS/parent-state, residual and residual-Gaussian-geometry fields, then
# adds the B3 multi-view consistency descriptors (CV, direction consistency).
#
# Features are strictly pre-action: no child geometry, no split result, no
# future quality / tile values. Oracle labels are joined from the B2-C cache
# (supervision only).
#
# Output -> paper_b/b3_action_predictability/data/b3_candidate_dataset.{csv,json}
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
from gaussian_renderer import render_fastgs  # noqa: E402
from utils.fast_utils import get_loss  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from diagnostics.common import clone_tree, install_c_proxy, seed_all  # noqa: E402
import diagnostics.diagnostic_v2 as v2  # noqa: E402

B2C = "paper_b/b2_c_oracle_scalability"
OUT = "paper_b/b3_action_predictability"
TILE_MARGIN_PX = 16


def main():
    parser = ArgumentParser("Paper B B3: pre-action descriptor collection")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--n_probe", type=int, default=8)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    assert opt.optimizer_type == "default"

    install_c_proxy()
    seed_all(0)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    v2.G.update({"pipe": pipe, "bg": bg, "mult": opt.mult,
                 "loss_thresh": opt.loss_thresh, "sh_degree": dataset.sh_degree})

    # ---------------- load B2-C artifacts ----------------------------------
    blob = torch.load(f"{B2C}/cache/master_snapshot.pt")
    protocol = json.load(open(f"{B2C}/cache/protocol.json"))
    snapshot = blob["snapshot"]
    xyz_snap = snapshot[1]
    n_before = int(xyz_snap.shape[0])
    snap_fp = "%d_%.4f_%.4f" % (n_before, float(xyz_snap.sum()), float(snapshot[6].sum()))
    assert snap_fp == protocol["snapshot_fp"], \
        f"snapshot fingerprint mismatch: {snap_fp} vs {protocol['snapshot_fp']} — refusing to join"
    print(f"[b3] master snapshot loaded & verified (N={n_before}, it={protocol['diag_iter']})")

    with open(f"{B2C}/data/b2c_candidate_oracles.json") as f:
        oracle_list = json.load(f)["oracles"]
    oracle = {o["parent_index"]: o for o in oracle_list}
    print(f"[b3] oracle labels: {len(oracle)} candidates")

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.restore(clone_tree(snapshot), opt)
    train_cams = scene.getTrainCameras()
    pool_cams = train_cams[:len(blob["pool_radii"])]

    # per-view residual maps from the snapshot state (action 前唯一信息)
    pool_data = []
    with torch.no_grad():
        for i, cam in enumerate(pool_cams):
            out = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
            l1_map = torch.mean(torch.abs(out["render"] - cam.original_image.cuda()), dim=0)
            pool_data.append({"cam": cam, "radii": blob["pool_radii"][i],
                              "l1_map": l1_map, "mask": blob["pool_masks"][i]})

    # gradient norms from the snapshot accumulators (native definition)
    gv = gaussians.xyz_gradient_accum / gaussians.denom
    gv[gv.isnan()] = 0.0
    ga = gaussians.xyz_gradient_accum_abs / gaussians.denom
    ga[ga.isnan()] = 0.0
    grad_norm = torch.norm(gv, dim=-1)
    grad_abs_norm = torch.norm(ga, dim=-1)
    imp = blob["imp"]
    dummy_pruning = torch.zeros_like(imp, dtype=torch.float32)

    clone_set = blob["clone_set"]
    split_set = blob["split_set"]
    native_action_of = {int(i): "clone" for i in np.where(clone_set.cpu().numpy())[0]}
    native_action_of.update({int(i): "split" for i in np.where(split_set.cpu().numpy())[0]})

    # ---------------- per-candidate descriptors ---------------------------
    rows = []
    eps = 1e-8
    for n_done, idx in enumerate(sorted(oracle.keys())):
        cand = v2.build_candidate(gaussians, int(idx), native_action_of[int(idx)],
                                  protocol["diag_iter"], imp, dummy_pruning,
                                  grad_norm, grad_abs_norm, pool_data, args.n_probe)
        if cand is None:
            rows.append({"parent_index": int(idx), "valid_views": 0,
                         "oracle_action": oracle[idx]["oracle_action"]})
            continue
        pv = cand["per_view"]
        rdirs = [v["residual"]["residual_direction_deg"] for v in pv
                 if v.get("residual") is not None
                 and not np.isnan(v["residual"]["residual_direction_deg"])]
        # multi-view direction consistency via doubled-angle resultant (180-deg symmetric)
        if rdirs:
            a2 = np.deg2rad(np.array(rdirs)) * 2.0
            c2, s2 = float(np.mean(np.cos(a2))), float(np.mean(np.sin(a2)))
            dir_consistency = float(np.hypot(c2, s2))
        else:
            dir_consistency = None

        def cv(key):
            vals = [v["residual"][key] for v in pv if v.get("residual") is not None]
            vals = [x for x in vals if not np.isnan(x)]
            if len(vals) < 2:
                return None
            m_, s_ = float(np.mean(vals)), float(np.std(vals))
            return s_ / (abs(m_) + eps)

        # demand pixel ratio: demand pixels / ROI area, per fixed view
        dpr = []
        for vd in cand["_views"]:
            x0, y0, x1, y1 = vd["box"]
            area = max((x1 - x0) * (y1 - y0), 1)
            dpr.append(float(int(vd["mask"][y0:y1, x0:x1].sum())) / area)

        o = oracle[idx]
        pxyz = gaussians.get_xyz[int(idx)].detach().cpu().numpy()
        row = {
            "parent_index": int(idx),
            "native_action": cand["native_action"],
            "x": float(pxyz[0]), "y": float(pxyz[1]), "z": float(pxyz[2]),
            # --- FastGS / parent state baseline ---
            "importance_score": cand["importance_score"],
            "grad": cand["grad"], "grad_abs": cand["grad_abs"],
            "opacity": cand["opacity"],
            "scale_x": cand["scale_x"], "scale_y": cand["scale_y"], "scale_z": cand["scale_z"],
            "scale_max": cand["scale_max"], "scale_min": cand["scale_min"],
            "scale_anisotropy": cand["scale_anisotropy"],
            "projected_radius_mean": cand["projected_radius_mean"],
            "projected_radius_max": cand["projected_radius_max"],
            "footprint_mean": cand["footprint_mean"], "footprint_max": cand["footprint_max"],
            "footprint_area_mean": cand["footprint_area_mean"],
            "visibility_count": cand["num_visible_views"],
            "valid_views": cand["num_valid_local_views"],
            # --- residual descriptors (B2-A formulas) ---
            "residual_energy_mean": cand["residual_energy_mean"],
            "residual_energy_std": cand["residual_energy_std"],
            "residual_anisotropy_mean": cand["residual_anisotropy_mean"],
            "residual_anisotropy_std": cand["residual_anisotropy_std"],
            "residual_extent_mean": cand["residual_extent_mean"],
            "residual_centroid_offset_mean": cand["residual_centroid_offset_mean"],
            "demand_pixel_ratio_mean": float(np.mean(dpr)) if dpr else None,
            # --- residual-Gaussian relative geometry (B2-A) ---
            "residual_parent_alignment_mean": cand["residual_parent_alignment_mean"],
            "residual_extent_ratio_mean": cand["residual_extent_ratio_mean"],
            "proj_anisotropy_mean": cand["proj_anisotropy_mean"],
            # --- multi-view consistency (B3 new) ---
            "residual_energy_cv": cv("residual_energy_mean"),
            "residual_anisotropy_cv": cv("residual_anisotropy"),
            "residual_extent_cv": cv("residual_extent"),
            "residual_direction_consistency": dir_consistency,
            "residual_extent_std": None,
            "residual_centroid_offset_std": None,
            # --- supervision labels only (NEVER model input) ---
            "oracle_action": o["oracle_action"],
            "oracle_gap": o["oracle_gap"],
            "split_sem": o["split_sem"],
            "split_dQ_std_100": o["split_dQ_std_100"],
            "oracle_high_confident": o["oracle_high_confident"],
        }
        # std of extent & centroid offset across views need per-view alignment rows
        offs = [v["alignment"]["residual_centroid_offset"] for v in pv
                if v.get("alignment") and v["alignment"].get("residual_centroid_offset") is not None
                and not np.isnan(v["alignment"]["residual_centroid_offset"])]
        row["residual_centroid_offset_std"] = float(np.std(offs)) if len(offs) >= 2 else None
        exts = [v["residual"]["residual_extent"] for v in pv
                if v.get("residual") is not None and not np.isnan(v["residual"]["residual_extent"])]
        row["residual_extent_std"] = float(np.std(exts)) if len(exts) >= 2 else None
        rows.append(row)
        if (n_done + 1) % 100 == 0:
            print(f"[b3] descriptors {n_done + 1}/{len(oracle)}")

    # ---------------- save --------------------------------------------------
    os.makedirs(f"{OUT}/data", exist_ok=True)
    meta = {"snapshot_fp": snap_fp, "diag_iter": protocol["diag_iter"],
            "n_snapshot": n_before, "n_candidates": len(rows),
            "n_valid_oracle": sum(1 for r in rows if r.get("oracle_action") in ("clone", "split")),
            "n_high_conf": sum(1 for r in rows if r.get("oracle_high_confident")),
            "note": "oracle_* fields are supervision labels only, never model inputs"}
    with open(f"{OUT}/data/b3_candidate_dataset.json", "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    cols = [c for c in rows[0].keys() if c != "_x"]
    with open(f"{OUT}/data/b3_candidate_dataset.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["NA" if r.get(c) is None else (f"{r[c]:.8g}" if isinstance(r[c], float)
                        else r[c]) for c in cols])
    print(f"[b3] saved {OUT}/data/b3_candidate_dataset.csv / .json ({len(rows)} candidates)")
    print(f"[b3] valid oracle: {meta['n_valid_oracle']}, high-conf: {meta['n_high_conf']}")


if __name__ == "__main__":
    main()
