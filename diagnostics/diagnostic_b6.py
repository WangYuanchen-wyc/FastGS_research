#
# Paper B - B6: Practical Oracle Approximation (offline reachability)
#
# Can pre-action information approximate the B5-Fix candidate oracles?
#   WHO : rank by q_best = max(q_clone, q_split)
#   HOW : q_gap = q_clone - q_split   (sign -> Clone/Split)
#
# features mode: extract pre-action descriptors for every valid B5 oracle
#   candidate from the SAME persisted snapshots (no future/action info).
#   B5-Fix lesson applied: seed_all(0) before Scene + camera identity
#   assertion against B5's collect-time record.
# replay mode: run the practical policies (Pred-Who / Pred-How) under the B5
#   replay protocol on the B5 groups (seed 0, rho=0.5), reusing B5's stored
#   native_full / rw_nh / ow_oh records as references.
#
# No full-training method, no FastGS modification.
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
from diagnostics.diagnostic_b2c import (restore_from, apply_assignment, group_eval,  # noqa: E402
                                        split_seed_for)

B5 = "paper_b/b5_cross_stage_capacity_oracle"
OUT = "paper_b/b6_practical_oracle_approximation"
TILE_MARGIN_PX = 16


def _cam_id(c):
    return [str(c.image_name), int(c.uid)]


def build_scene(dataset, opt, view_pool):
    """Scene creation with the B5-Fix protocol: seeded RNG + identity assert."""
    seed_all(0)
    g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, g)
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    pool_cams = train_cams[:view_pool]
    ident = json.load(open(f"{B5}/cache/camera_identity.json"))
    for name, current, saved in (("train", [_cam_id(c) for c in train_cams], ident["train"]),
                                 ("test", [_cam_id(c) for c in test_cams], ident["test"]),
                                 ("pool", [_cam_id(c) for c in pool_cams], ident["pool"])):
        if current != saved:
            raise RuntimeError(f"[b6] camera identity assertion FAIL on '{name}' — aborting")
    print("[b6] camera identity assertion PASS")
    return scene, train_cams, test_cams, pool_cams


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


def extract_features(dataset, opt, pipe, args, iters):
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    v2.G.update({"pipe": pipe, "bg": bg, "mult": opt.mult,
                 "loss_thresh": opt.loss_thresh, "sh_degree": dataset.sh_degree})
    scene, train_cams, test_cams, pool_cams = build_scene(dataset, opt, args.view_pool)

    # labels from the B5-Fix oracle table
    labels = {}
    with open(f"{B5}/data/b5_candidate_oracles.csv") as f:
        for r in csv.DictReader(f):
            labels.setdefault(int(r["iteration"]), []).append(
                {"parent_index": int(r["parent_index"]),
                 "q_clone": float(r["q_clone"]), "q_split": float(r["q_split_mean"]),
                 "q_best": float(r["q_best"]),
                 "oracle_action": r["oracle_action"], "native_action": r["native_action"]})

    rows = []
    for it in iters:
        snap = torch.load(f"{B5}/cache/snap_{it}.pt")
        snapshot, radii_snap = snap["snapshot"], snap["radii"].clone()
        xyz_snap = snapshot[1]
        clone_idx = set(int(i) for i in np.where(snap["clone_set"].cpu().numpy())[0])
        gaussians = GaussianModel(v2.G["sh_degree"], opt.optimizer_type)
        gaussians.restore(clone_tree(snapshot), opt)

        pool_data = []
        with torch.no_grad():
            for i, cam in enumerate(pool_cams):
                out = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
                l1_map = torch.mean(torch.abs(out["render"] - cam.original_image.cuda()), dim=0)
                pool_data.append({"cam": cam, "radii": snap["pool_radii"][i],
                                  "l1_map": l1_map, "mask": snap["pool_masks"][i]})

        gv = gaussians.xyz_gradient_accum / gaussians.denom
        gv[gv.isnan()] = 0.0
        ga = gaussians.xyz_gradient_accum_abs / gaussians.denom
        ga[ga.isnan()] = 0.0
        grad_norm, grad_abs_norm = torch.norm(gv, dim=-1), torch.norm(ga, dim=-1)
        imp = snap["imp"]
        dummy_pruning = torch.zeros_like(imp, dtype=torch.float32)

        for n_done, lab in enumerate(labels.get(it, [])):
            idx = lab["parent_index"]
            cand = v2.build_candidate(gaussians, idx, "clone" if idx in clone_idx else "split",
                                      it, imp, dummy_pruning, grad_norm, grad_abs_norm,
                                      pool_data, args.n_probe)
            if cand is None:
                rows.append({"iteration": it, "parent_index": idx, "valid_views": 0,
                             **{k: lab[k] for k in ("q_clone", "q_split", "q_best",
                                                    "oracle_action", "native_action")}})
                continue
            pv = cand["per_view"]
            rdirs = [v["residual"]["residual_direction_deg"] for v in pv
                     if v.get("residual") is not None
                     and not np.isnan(v["residual"]["residual_direction_deg"])]
            if rdirs:
                a2 = np.deg2rad(np.array(rdirs)) * 2.0
                dir_cons = float(np.hypot(np.mean(np.cos(a2)), np.mean(np.sin(a2))))
            else:
                dir_cons = None
            eps = 1e-8

            def cv(key):
                vals = [v["residual"][key] for v in pv if v.get("residual") is not None]
                vals = [x for x in vals if not np.isnan(x)]
                if len(vals) < 2:
                    return None
                return float(np.std(vals)) / (abs(float(np.mean(vals))) + eps)

            dpr = []
            for vd in cand["_views"]:
                x0, y0, x1, y1 = vd["box"]
                dpr.append(float(int(vd["mask"][y0:y1, x0:x1].sum()))
                           / max((x1 - x0) * (y1 - y0), 1))
            row = {"iteration": it, "parent_index": idx,
                   "x": float(gaussians.get_xyz[idx][0]),
                   "y": float(gaussians.get_xyz[idx][1]),
                   "z": float(gaussians.get_xyz[idx][2]),
                   "native_action": cand["native_action"],
                   "importance_score": cand["importance_score"],
                   "grad": cand["grad"], "grad_abs": cand["grad_abs"],
                   "opacity": cand["opacity"],
                   "scale_max": cand["scale_max"], "scale_min": cand["scale_min"],
                   "scale_anisotropy": cand["scale_anisotropy"],
                   "projected_radius_mean": cand["projected_radius_mean"],
                   "footprint_mean": cand["footprint_mean"],
                   "footprint_area_mean": cand["footprint_area_mean"],
                   "visibility_count": cand["num_visible_views"],
                   "residual_energy_mean": cand["residual_energy_mean"],
                   "residual_energy_std": cand["residual_energy_std"],
                   "residual_anisotropy_mean": cand["residual_anisotropy_mean"],
                   "residual_anisotropy_std": cand["residual_anisotropy_std"],
                   "residual_extent_mean": cand["residual_extent_mean"],
                   "residual_centroid_offset_mean": cand["residual_centroid_offset_mean"],
                   "demand_pixel_ratio_mean": float(np.mean(dpr)) if dpr else None,
                   "residual_parent_alignment_mean": cand["residual_parent_alignment_mean"],
                   "residual_extent_ratio_mean": cand["residual_extent_ratio_mean"],
                   "proj_anisotropy_mean": cand["proj_anisotropy_mean"],
                   "residual_energy_cv": cv("residual_energy_mean"),
                   "residual_anisotropy_cv": cv("residual_anisotropy"),
                   "residual_extent_cv": cv("residual_extent"),
                   "residual_direction_consistency": dir_cons,
                   # labels (NEVER model inputs)
                   "q_clone": lab["q_clone"], "q_split": lab["q_split"],
                   "q_best": lab["q_best"], "q_gap": lab["q_clone"] - lab["q_split"],
                   "oracle_action": lab["oracle_action"]}
            rows.append(row)
            if (n_done + 1) % 100 == 0:
                print(f"[b6-feat] it={it}: {n_done + 1}/{len(labels[it])}")
        del gaussians, snap
        torch.cuda.empty_cache()

    with open(f"{OUT}/data/b6_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("NA" if v is None else
                            (f"{v:.8g}" if isinstance(v, float) else v)) for k, v in r.items()})
    json.dump(rows, open(f"{OUT}/data/b6_features.json", "w"))
    print(f"[b6-feat] saved {len(rows)} rows -> {OUT}/data/b6_features.csv")


def run_replay(dataset, opt, pipe, args, iters):
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    v2.G.update({"pipe": pipe, "bg": bg, "mult": opt.mult,
                 "loss_thresh": opt.loss_thresh, "sh_degree": dataset.sh_degree})
    scene, train_cams, test_cams, pool_cams = build_scene(dataset, opt, args.view_pool)
    global_probe = (test_cams[:5] if test_cams and len(test_cams) > 0 else train_cams[:5])
    cost_views = pool_cams[:args.n_cost_views]
    cam_seq_rng = random.Random(2024)
    cam_seq = [cam_seq_rng.randint(0, len(train_cams) - 1) for _ in range(args.diag_steps)]

    # LOSO predictions produced by analyze_b6.py
    pred = json.load(open(f"{OUT}/cache/b6_predictions.json"))["preds"]
    K, M = args.K, int(round(0.5 * args.K))
    replay_rows = []
    for it in iters:
        snap = torch.load(f"{B5}/cache/snap_{it}.pt")
        snapshot, radii_snap = snap["snapshot"], snap["radii"].clone()
        xyz_snap = snapshot[1]
        with open(f"{B5}/data/groups/group_it{it}_seed0.json") as f:
            members = [m["parent_index"] for m in json.load(f)["members"]]
        assert len(members) == K
        native_action_of = {}
        for i in np.where(snap["clone_set"].cpu().numpy())[0]:
            native_action_of[int(i)] = "clone"
        for i in np.where(snap["split_set"].cpu().numpy())[0]:
            native_action_of[int(i)] = "split"

        p_best = {int(k): v for k, v in pred[str(it)]["pred_q_best"].items()}
        p_gap = {int(k): v for k, v in pred[str(it)]["pred_q_gap"].items()}
        pred_act = {i: ("clone" if p_gap[i] > 0 else "split") for i in members}
        pred_who = sorted(members, key=lambda i: -p_best[i])[:M]

        # B5 rw rep0 subset (paired with B5's rw_nh rep0): same rng formula
        # rw_base + seed*100000 + it*100 + round(rho*100)*10 + repeat
        rr = np.random.RandomState(80000 + 0 * 100000 + it * 100 + int(round(0.5 * 100)) * 10 + 0)
        rw_subset = [int(i) for i in rr.choice(np.array(members), size=M, replace=False)]

        # fixed full-group ROI (same rule as B5)
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
            "predwho_nativehow": [(i, native_action_of[i]) for i in pred_who],
            "randomwho_predhow": [(i, pred_act[i]) for i in rw_subset],
            "predwho_predhow": [(i, pred_act[i]) for i in pred_who],
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
            rec = {"iteration": it, "policy": pname, "M": M, "K": K,
                   "delta_num_gaussians": n_after - int(xyz_snap.shape[0]),
                   "demand_l1_100": ev[args.diag_steps]["demand"]["l1"],
                   "demand_psnr_100": ev[args.diag_steps]["demand"]["psnr"],
                   "global_psnr_100": ev[args.diag_steps]["global"]["psnr"],
                   "tile_0": ev[0]["tile_pairs_mean"],
                   "latency_ms": lat}
            replay_rows.append(rec)
            print(f"[b6-replay] it={it} {pname}: dN={rec['delta_num_gaussians']:+d} "
                  f"dL1={rec['demand_l1_100']:.6f} gPSNR={rec['global_psnr_100']:.4f}")
        del snap
        torch.cuda.empty_cache()

    # join B5 reference records (seed 0, rho 0.5; keep_all/native_full stored
    # with rho 0.0/1.0 in B5 -> matched by policy across rhos)
    b5 = json.load(open(f"{B5}/data/b5_group_results.json"))["records"]
    for it in iters:
        sub05 = [r for r in b5 if r["iteration"] == it and r["group_seed"] == 0 and r["rho"] == 0.5]
        subany = [r for r in b5 if r["iteration"] == it and r["group_seed"] == 0]
        keep = next(r for r in subany if r["policy"] == "keep_all")
        for p in ("native_full", "ow_oh", "rw_nh", "rw_oh"):
            pool = subany if p == "native_full" else sub05
            for r in [x for x in pool if x["policy"] == p]:
                replay_rows.append({"iteration": it, "policy": p, "M": r["M"], "K": r["K"],
                                    "delta_num_gaussians": r["delta_num_gaussians"],
                                    "demand_l1_100": r["demand_l1_100"],
                                    "demand_psnr_100": r["demand_psnr_100"],
                                    "global_psnr_100": r["global_psnr_100"],
                                    "tile_0": r["tile_0"], "latency_ms": None})
        for r in [x for x in replay_rows if x["iteration"] == it]:
            r["dTile_0"] = r["tile_0"] - keep["tile_0"]

    with open(f"{OUT}/data/b6_replay_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(replay_rows[0].keys()))
        w.writeheader()
        for r in replay_rows:
            w.writerow({k: ("NA" if v is None else (f"{v:.8g}" if isinstance(v, float) else v))
                        for k, v in r.items()})
    json.dump(replay_rows, open(f"{OUT}/data/b6_replay_results.json", "w"), indent=1)
    print(f"[b6-replay] saved {len(replay_rows)} rows")


def main():
    parser = ArgumentParser("Paper B B6: practical oracle approximation")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--mode", type=str, default="features", choices=["features", "replay"])
    parser.add_argument("--iters", type=str, default="1000,1500,2000,5000,12000")
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--diag_steps", type=int, default=100)
    parser.add_argument("--view_pool", type=int, default=30)
    parser.add_argument("--n_cost_views", type=int, default=8)
    parser.add_argument("--n_probe", type=int, default=8)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    assert opt.optimizer_type == "default"
    iters = sorted(int(x) for x in args.iters.split(","))
    install_c_proxy()
    if args.mode == "features":
        extract_features(dataset, opt, pipe, args, iters)
    else:
        run_replay(dataset, opt, pipe, args, iters)


if __name__ == "__main__":
    main()
