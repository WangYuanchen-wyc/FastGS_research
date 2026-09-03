#
# Paper B - B6-Fix: Action-Prior Audit
#
# Question: does Pred-How's gain come from candidate-specific signal or from a
# global Clone prior? On B5's RW repeat-0 subset (identical 50 candidates per
# iteration) compare, under one replay protocol:
#     RW-NativeHow / RW-AllClone / RW-AllSplit / RW-PredHow / RW-OracleHow
#
# Fixed predictors (NO per-stage model selection, no test leakage):
#     Who = Ridge, How = DecisionTreeRegressor(depth 3)
# LOSO predictions are regenerated from the existing b6_features (no feature
# re-extraction). New GPU branches: AllClone / AllSplit / PredHow on the RW
# subset; NativeHow / OracleHow reuse B5's repeat-0 records.
#
# No allocator, no FastGS modification, no B7.
#

import os
import sys
import json
import random
import csv

import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from diagnostics.common import (install_c_proxy, seed_all,  # noqa: E402
                                project_to_pixel, roi_box, native_train_one_iter)
import diagnostics.diagnostic_v2 as v2  # noqa: E402
from diagnostics.diagnostic_b2c import restore_from, apply_assignment, group_eval  # noqa: E402
from diagnostics.diagnostic_b6 import build_scene  # noqa: E402

B5 = "paper_b/b5_cross_stage_capacity_oracle"
B6 = "paper_b/b6_practical_oracle_approximation"
OUT = "paper_b/b6_action_prior_audit"
TILE_MARGIN_PX = 16


@torch.no_grad()
def latency_light(g, views, n_views=3, reps=3):
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    vv = views[:n_views]
    for cam in vv:
        render_fastgs(cam, g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])
    torch.cuda.synchronize()
    lats = []
    for _ in range(reps):
        for cam in vv:
            ev0.record()
            render_fastgs(cam, g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])
            ev1.record()
            torch.cuda.synchronize()
            lats.append(ev0.elapsed_time(ev1))
    return float(np.mean(lats))


def main():
    parser = ArgumentParser("Paper B B6-Fix: action-prior audit replay")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iters", type=str, default="1000,2000,5000,12000")
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--diag_steps", type=int, default=100)
    parser.add_argument("--view_pool", type=int, default=30)
    parser.add_argument("--n_cost_views", type=int, default=8)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    assert opt.optimizer_type == "default"
    iters = sorted(int(x) for x in args.iters.split(","))
    install_c_proxy()
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    v2.G.update({"pipe": pipe, "bg": bg, "mult": opt.mult,
                 "loss_thresh": opt.loss_thresh, "sh_degree": dataset.sh_degree})
    scene, train_cams, test_cams, pool_cams = build_scene(dataset, opt, args.view_pool)
    global_probe = (test_cams[:5] if test_cams and len(test_cams) > 0 else train_cams[:5])
    cost_views = pool_cams[:args.n_cost_views]
    cam_seq_rng = random.Random(2024)
    cam_seq = [cam_seq_rng.randint(0, len(train_cams) - 1) for _ in range(args.diag_steps)]

    pred = json.load(open(f"{OUT}/cache/b6fix_predictions.json"))["preds"]
    M = 50
    rows = []
    for it in iters:
        snap = torch.load(f"{B5}/cache/snap_{it}.pt")
        snapshot, radii_snap = snap["snapshot"], snap["radii"].clone()
        xyz_snap = snapshot[1]
        with open(f"{B5}/data/groups/group_it{it}_seed0.json") as f:
            members = [m["parent_index"] for m in json.load(f)["members"]]
        native_action_of = {}
        for i in np.where(snap["clone_set"].cpu().numpy())[0]:
            native_action_of[int(i)] = "clone"
        for i in np.where(snap["split_set"].cpu().numpy())[0]:
            native_action_of[int(i)] = "split"

        # EXACT B5 RW repeat-0 subset (rw_base + seed*100000 + it*100 + round(rho*100)*10 + rep)
        rr = np.random.RandomState(80000 + it * 100 + int(round(0.5 * 100)) * 10 + 0)
        subset = [int(i) for i in rr.choice(np.array(members), size=M, replace=False)]

        p_gap = {int(k): v for k, v in pred[str(it)]["pred_q_gap"].items()}
        pred_act = {i: ("clone" if p_gap[i] > 0 else "split") for i in subset}
        n_pred_clone = sum(1 for a in pred_act.values() if a == "clone")
        print(f"\n[b6fix] it={it}: subset={M}, pred clone ratio = {n_pred_clone}/{M}")

        # fixed full-group ROI (same rule as B5/B6)
        support_masks, demand_masks = [], []
        for vi, cam in enumerate(cost_views):
            sm = torch.zeros(int(cam.image_height), int(cam.image_width),
                             dtype=torch.bool, device="cuda")
            for idx in members:
                vr = snap["pool_radii"][vi]
                if int(vr[idx]) <= 0:
                    continue
                u, v, _, ok = project_to_pixel(cam, xyz_snap[idx])
                if not ok:
                    continue
                x0, y0, x1, y1 = roi_box(u, v, float(vr[idx]), cam.image_width,
                                         cam.image_height, TILE_MARGIN_PX)
                sm[y0:y1, x0:x1] = True
            support_masks.append(sm)
            demand_masks.append(sm & snap["pool_masks"][vi])

        policies = {
            "rw_allclone": [(i, "clone") for i in subset],
            "rw_allsplit": [(i, "split") for i in subset],
            "rw_predhow": [(i, pred_act[i]) for i in subset],
        }
        for pname, assign in policies.items():
            g = restore_from(snapshot, opt)
            apply_assignment(g, assign, radii_snap)
            n_after = int(g.get_xyz.shape[0])
            seed_all(1234)
            h = args.diag_steps // 2
            ev = {0: group_eval(g, cost_views, support_masks, demand_masks, global_probe)}
            for i in range(1, args.diag_steps + 1):
                native_train_one_iter(it + i, train_cams[cam_seq[i - 1]], g, pipe, bg, opt)
                with torch.no_grad():
                    if opt.optimizer_type == "default":
                        g.optimizer_step(it + i)
                    if i in (h, args.diag_steps):
                        ev[i] = group_eval(g, cost_views, support_masks, demand_masks,
                                           global_probe, with_lpips=(i == args.diag_steps))
            lat = latency_light(g, cost_views)
            del g
            torch.cuda.empty_cache()
            rows.append({"iteration": it, "policy": pname, "M": M,
                         "delta_num_gaussians": n_after - int(xyz_snap.shape[0]),
                         "pred_clone_ratio": n_pred_clone / M,
                         "demand_l1_100": ev[args.diag_steps]["demand"]["l1"],
                         "demand_psnr_100": ev[args.diag_steps]["demand"]["psnr"],
                         "global_psnr_100": ev[args.diag_steps]["global"]["psnr"],
                         "global_ssim_100": ev[args.diag_steps]["global"].get("ssim"),
                         "tile_0": ev[0]["tile_pairs_mean"],
                         "latency_ms": lat})
            print(f"[b6fix] it={it} {pname}: dN={rows[-1]['delta_num_gaussians']:+d} "
                  f"dL1={rows[-1]['demand_l1_100']:.6f} "
                  f"gPSNR={rows[-1]['global_psnr_100']:.4f}")
        del snap
        torch.cuda.empty_cache()

    # join B5 repeat-0 records for the same subset (rw_nh/rw_oh rep 0)
    b5 = json.load(open(f"{B5}/data/b5_group_results.json"))["records"]
    for it in iters:
        for p in ("rw_nh", "rw_oh"):
            r = next(x for x in b5 if x["iteration"] == it and x["group_seed"] == 0
                     and x["rho"] == 0.5 and x["policy"] == p and x["rw_repeat"] == 0)
            rows.append({"iteration": it, "policy": "rw_nativehow" if p == "rw_nh" else "rw_oraclehow",
                         "M": r["M"], "delta_num_gaussians": r["delta_num_gaussians"],
                         "pred_clone_ratio": None,
                         "demand_l1_100": r["demand_l1_100"],
                         "demand_psnr_100": r["demand_psnr_100"],
                         "global_psnr_100": r["global_psnr_100"],
                         "global_ssim_100": r["global_ssim_100"],
                         "tile_0": r["tile_0"], "latency_ms": None})

    with open(f"{OUT}/data/b6fix_action_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("NA" if v is None else (f"{v:.8g}" if isinstance(v, float) else v))
                        for k, v in r.items()})
    json.dump(rows, open(f"{OUT}/data/b6fix_action_results.json", "w"), indent=1)
    print(f"\n[b6fix] saved {len(rows)} rows")


if __name__ == "__main__":
    main()
