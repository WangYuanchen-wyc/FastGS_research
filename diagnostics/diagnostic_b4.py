#
# Paper B - B4: Joint Structural Capacity Allocation Oracle (diagnostic)
#
# Questions:
#   WHO : at equal added-GS budget M, does choosing the right candidates
#         (top-M by oracle value) beat random selection?
#   HOW : on the SAME candidate subset, does the oracle Clone/Split action
#         beat the FastGS native action?
#   JOINT frontier: OW-OH at rho = 0.5 / 0.75 vs Native-Full at rho = 1.0.
#
# Everything replays from the B2-C PERSISTED master snapshot (fingerprint-
# verified); splits use the identical B2-C repeat-0 seeds (300000 + idx*100).
# Fixed full-group support/demand ROI shared by every policy/rho within a
# (K, seed). M = round(rho*K) is a diagnostic knob only, not a method quota.
#
# No predictor / allocator / budget / new densification rule is implemented.
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
from diagnostics.common import (clone_tree, install_c_proxy, seed_all,  # noqa: E402
                                project_to_pixel, roi_box, native_train_one_iter)
import diagnostics.diagnostic_v2 as v2  # noqa: E402
from diagnostics.diagnostic_b2c import restore_from, apply_assignment, group_eval  # noqa: E402

B2C = "paper_b/b2_c_oracle_scalability"
OUT = "paper_b/b4_structural_capacity_oracle"
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
    parser = ArgumentParser("Paper B B4: joint structural capacity oracle")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--Ks", type=str, default="100")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--rhos", type=str, default="0.25,0.50,0.75,1.00")
    parser.add_argument("--rw_repeats", type=int, default=5)
    parser.add_argument("--diag_steps", type=int, default=100)
    parser.add_argument("--view_pool", type=int, default=30)
    parser.add_argument("--n_cost_views", type=int, default=8)
    parser.add_argument("--train_seed", type=int, default=1234)
    parser.add_argument("--camseq_seed", type=int, default=2024)
    parser.add_argument("--rw_base_seed", type=int, default=70000)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    assert opt.optimizer_type == "default"
    Ks = [int(k) for k in args.Ks.split(",")]
    rhos = [float(r) for r in args.rhos.split(",")]

    install_c_proxy()
    seed_all(0)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    v2.G.update({"pipe": pipe, "bg": bg, "mult": opt.mult,
                 "loss_thresh": opt.loss_thresh, "sh_degree": dataset.sh_degree})

    # ---------------- B2-C artifacts ---------------------------------------
    blob = torch.load(f"{B2C}/cache/master_snapshot.pt")
    protocol = json.load(open(f"{B2C}/cache/protocol.json"))
    snapshot = blob["snapshot"]
    xyz_snap = snapshot[1]
    n_before = int(xyz_snap.shape[0])
    snap_fp = "%d_%.4f_%.4f" % (n_before, float(xyz_snap.sum()), float(snapshot[6].sum()))
    assert snap_fp == protocol["snapshot_fp"], "snapshot fingerprint mismatch"
    print(f"[b4] master snapshot verified (N={n_before}, it={protocol['diag_iter']})")

    with open(f"{B2C}/data/b2c_candidate_oracles.json") as f:
        oracle = {o["parent_index"]: o for o in json.load(f)["oracles"]}
    n_fallback_total = sum(1 for o in oracle.values() if o["oracle_action"] not in ("clone", "split"))
    print(f"[b4] oracles: {len(oracle)} ({n_fallback_total} fallback->native)")

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    del gaussians
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    global_probe = (test_cams[:5] if test_cams and len(test_cams) > 0 else train_cams[:5])
    pool_cams = train_cams[:args.view_pool]
    cost_views = pool_cams[:args.n_cost_views]
    cam_seq_rng = random.Random(args.camseq_seed)
    cam_seq = [cam_seq_rng.randint(0, len(train_cams) - 1) for _ in range(args.diag_steps)]
    pool_radii, pool_masks = blob["pool_radii"], blob["pool_masks"]
    radii_snap = blob["radii"].clone()
    diag_iter0 = protocol["diag_iter"]
    del blob
    torch.cuda.empty_cache()

    records = []
    h = args.diag_steps // 2
    for K in Ks:
        for seed_id in range(args.n_seeds):
            with open(f"{B2C}/data/groups/group_K{K}_seed{seed_id}.json") as f:
                members = [m["parent_index"] for m in json.load(f)["members"]]
            assert len(members) == K

            # candidate values (fallback -> native action, q = 0)
            info = {}
            for i in members:
                o = oracle[i]
                q_c, q_s = o.get("clone_dQ_100"), o.get("split_dQ_mean_100")
                if q_c is None or q_s is None:
                    info[i] = {"fallback": True, "q_best": 0.0, "q_native": 0.0,
                               "oracle_action": o["native_action"], "native_action": o["native_action"]}
                else:
                    oa = "clone" if q_c > q_s else "split"
                    info[i] = {"fallback": False,
                               "q_clone": q_c, "q_split": q_s,
                               "q_best": max(q_c, q_s), "q_native": q_c if o["native_action"] == "clone" else q_s,
                               "oracle_action": oa, "native_action": o["native_action"]}
            n_fallback_group = sum(1 for i in members if info[i]["fallback"])
            print(f"\n[b4] K={K} seed={seed_id}: {K} members, {n_fallback_group} fallback")

            # fixed full-group support/demand ROI (union over ALL K, pre-action)
            support_masks, demand_masks = [], []
            for vi, cam in enumerate(cost_views):
                sm = torch.zeros(int(cam.image_height), int(cam.image_width),
                                 dtype=torch.bool, device="cuda")
                for idx in members:
                    vr = pool_radii[vi]
                    if int(vr[idx]) <= 0:
                        continue
                    u, v, _, ok = project_to_pixel(cam, xyz_snap[idx])
                    if not ok:
                        continue
                    x0, y0, x1, y1 = roi_box(u, v, float(vr[idx]), cam.image_width,
                                             cam.image_height, TILE_MARGIN_PX)
                    sm[y0:y1, x0:x1] = True
                support_masks.append(sm)
                demand_masks.append(sm & pool_masks[vi])

            # branch plan: (policy, rho, rw_repeat, assignment)
            branches = [("keep_all", 0.0, None, None),
                        ("native_full", 1.0, None,
                         [(i, info[i]["native_action"]) for i in members])]
            for rho in rhos:
                M = int(round(rho * K))
                by_qn = sorted(members, key=lambda i: -info[i]["q_native"])[:M]
                by_qb = sorted(members, key=lambda i: -info[i]["q_best"])[:M]
                branches.append(("ow_nh", rho, None, [(i, info[i]["native_action"]) for i in by_qn]))
                branches.append(("ow_oh", rho, None, [(i, info[i]["oracle_action"]) for i in by_qb]))
                for r in range(args.rw_repeats):
                    rng = np.random.RandomState(args.rw_base_seed + seed_id * 100000
                                                + K * 1000 + int(round(rho * 100)) * 10 + r)
                    subset = [int(i) for i in rng.choice(np.array(members), size=M, replace=False)]
                    branches.append(("rw_nh", rho, r, [(i, info[i]["native_action"]) for i in subset]))
                    branches.append(("rw_oh", rho, r, [(i, info[i]["oracle_action"]) for i in subset]))

            for policy, rho, rep, assign in branches:
                g = restore_from(snapshot, opt)
                if assign is not None:
                    apply_assignment(g, assign, radii_snap)
                n_after = int(g.get_xyz.shape[0])

                seed_all(args.train_seed)
                ev = {0: group_eval(g, cost_views, support_masks, demand_masks, global_probe)}
                for i in range(1, args.diag_steps + 1):
                    cam = train_cams[cam_seq[i - 1]]
                    native_train_one_iter(diag_iter0 + i, cam, g, pipe, bg, opt)
                    with torch.no_grad():
                        if opt.optimizer_type == "default":
                            g.optimizer_step(diag_iter0 + i)
                        if i in (h, args.diag_steps):
                            ev[i] = group_eval(g, cost_views, support_masks, demand_masks,
                                               global_probe, with_lpips=(i == args.diag_steps))
                lat = latency_light(g, cost_views)
                del g
                torch.cuda.empty_cache()

                rec = {"K": K, "group_seed": seed_id, "rho": rho,
                       "M": 0 if assign is None else len(assign),
                       "policy": policy, "rw_repeat": rep,
                       "n_selected": 0 if assign is None else len(assign),
                       "n_fallback_selected": 0 if assign is None else sum(1 for i, _ in assign if info[i]["fallback"]),
                       "num_gaussians_before": n_before, "num_gaussians_after": n_after,
                       "delta_num_gaussians": n_after - n_before,
                       "demand_l1_0": ev[0]["demand"]["l1"], "demand_l1_50": ev[h]["demand"]["l1"],
                       "demand_l1_100": ev[args.diag_steps]["demand"]["l1"],
                       "demand_psnr_100": ev[args.diag_steps]["demand"]["psnr"],
                       "support_l1_100": ev[args.diag_steps]["support"]["l1"],
                       "support_psnr_100": ev[args.diag_steps]["support"]["psnr"],
                       "global_psnr_0": ev[0]["global"]["psnr"],
                       "global_psnr_50": ev[h]["global"]["psnr"],
                       "global_psnr_100": ev[args.diag_steps]["global"]["psnr"],
                       "global_ssim_100": ev[args.diag_steps]["global"].get("ssim"),
                       "global_lpips_100": ev[args.diag_steps]["global"].get("lpips"),
                       "tile_0": ev[0]["tile_pairs_mean"],
                       "tile_100": ev[args.diag_steps]["tile_pairs_mean"],
                       "latency_ms": lat, "fps": 1000.0 / lat}
                records.append(rec)
                dl1 = rec["demand_l1_100"]
                print(f"    {policy:11s} rep={rep} M={rec['M']:3d} dN={rec['delta_num_gaussians']:+4d} "
                      f"dL1@100={'NA' if dl1 is None else '%.6f' % dl1} "
                      f"gPSNR={rec['global_psnr_100']:.4f}")

    # rho tagging for constrained policies + keep reference deltas
    for K in Ks:
        for s in range(args.n_seeds):
            sub = [r for r in records if r["K"] == K and r["group_seed"] == s]
            keep = next(r for r in sub if r["policy"] == "keep_all")
            for r in sub:
                r["dTile_0"] = r["tile_0"] - keep["tile_0"]
                r["dTile_100"] = r["tile_100"] - keep["tile_100"]
    os.makedirs(f"{OUT}/data", exist_ok=True)
    with open(f"{OUT}/data/b4_group_results.json", "w") as f:
        json.dump({"n_before": n_before, "diag_iter": protocol["diag_iter"],
                   "config": vars(args), "records": records}, f, indent=1)
    cols = list(records[0].keys())
    with open(f"{OUT}/data/b4_group_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow(["NA" if r[c] is None else (f"{r[c]:.8g}" if isinstance(r[c], float) else r[c])
                        for c in cols])
    print(f"\n[b4] saved {OUT}/data/b4_group_results.json / .csv ({len(records)} records)")


if __name__ == "__main__":
    main()
