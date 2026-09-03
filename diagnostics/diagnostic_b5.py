#
# Paper B - B5: Cross-Stage Structural Capacity Allocation Oracle
#
# Does the B4 Who / How / Joint allocation headroom persist across
# densification snapshots of ONE native FastGS trajectory?
#
# Snapshots at iterations {1000, 1500, 2000, 5000, 12000}, each captured BEFORE
# that event's native densification. ALL snapshots come from a single warmup
# pass and are persisted (cache/snap_{iter}.pt), so the phased processing
# (1000-2000 first, then 5000/12000) still sits on the SAME trajectory.
#
# Per snapshot: deterministic oversampling yields K=100 VALID candidates per
# group seed (invalid = no demand-valid view; they never enter the group and
# never join Oracle-Who ranking). Oracles use the B2-C protocol (Clone x1 +
# Split x5, 100-step replay, split seed 300000+idx*100+r; group realization =
# repeat 0). Policies per rho in {0.50, 0.75}: RW-NH x5 / RW-OH x5 (paired
# subsets) / OW-NH / OW-OH (identical top-M(q_best) subset) + Keep-All +
# Native-Full references. Fixed full-group ROI shared by everything within a
# snapshot/seed.
#
# No predictor / allocator / budget / new densification rule.
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
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras, get_loss  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from diagnostics.common import (clone_tree, install_c_proxy, seed_all,  # noqa: E402
                                project_to_pixel, roi_box, native_train_one_iter)
import diagnostics.diagnostic_v2 as v2  # noqa: E402
from diagnostics.diagnostic_b2c import (restore_from, apply_assignment, group_eval,  # noqa: E402
                                        split_seed_for, build_cand_views, demand_l1_on_views)

OUT = "paper_b/b5_cross_stage_capacity_oracle"
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


# ------------------------------------------------------------- collect -----

def _cam_id(c):
    return [str(c.image_name), int(c.uid)]


def collect(dataset, opt, pipe, args, diag_iters):
    seed_all(0)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    train_cams = scene.getTrainCameras()
    pool_cams = train_cams[:args.view_pool]
    # persist camera identity for the process-mode assertion (B5-Fix #1)
    ident = {"train": [_cam_id(c) for c in train_cams],
             "test": [_cam_id(c) for c in scene.getTestCameras()],
             "pool": [_cam_id(c) for c in pool_cams]}
    with open(f"{OUT}/cache/camera_identity.json", "w") as f:
        json.dump(ident, f)
    print(f"[b5-collect] camera identity saved: train={len(ident['train'])} "
          f"test={len(ident['test'])} pool={len(ident['pool'])}")

    stats = []
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    for iteration in range(1, diag_iters[-1] + 1):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = random.randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)
        _, vpt, vis_filter, radii = native_train_one_iter(
            iteration, viewpoint_cam, gaussians, pipe, bg, opt)
        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[vis_filter] = torch.max(
                    gaussians.max_radii2D[vis_filter], radii[vis_filter])
                gaussians.add_densification_stats(vpt, vis_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    my_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_stack)
                    imp, pru = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                    snap = None
                    if iteration in diag_iters:
                        clone_set, split_set, _, _, _ = v2.native_selection_sets(
                            gaussians, imp, opt, scene.cameras_extent)
                        pool_radii, pool_masks = [], []
                        with torch.no_grad():
                            for cam in pool_cams:
                                out = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
                                pool_radii.append(out["radii"])
                                pool_masks.append((get_loss(out["render"], cam.original_image.cuda())
                                                   > opt.loss_thresh).detach())
                        snap = {"snapshot": clone_tree(gaussians.capture(opt.optimizer_type)),
                                "radii": radii.clone(), "imp": imp,
                                "clone_set": clone_set, "split_set": split_set,
                                "pool_radii": pool_radii, "pool_masks": pool_masks}
                        torch.save(snap, f"{OUT}/cache/snap_{iteration}.pt")
                        n_c, n_s = int(clone_set.sum()), int(split_set.sum())
                        stats.append({"iteration": iteration,
                                      "num_gaussians_before": int(gaussians.get_xyz.shape[0]),
                                      "candidate_count": n_c + n_s,
                                      "native_clone_count": n_c, "native_split_count": n_s,
                                      "prune_count": None})
                        print(f"[b5-collect] snapshot it={iteration}: N={stats[-1]['num_gaussians_before']}, "
                              f"candidates={n_c + n_s} (clone {n_c} / split {n_s})")
                    n_before_ev = int(gaussians.get_xyz.shape[0])
                    gaussians.densify_and_prune_fastgs(
                        max_screen_size=(20 if iteration > opt.opacity_reset_interval else None),
                        min_opacity=0.005, extent=scene.cameras_extent, radii=radii, args=opt,
                        importance_score=imp, pruning_score=pru)
                    if snap is not None:
                        n_sel = n_c + n_s if snap else 0
                        n_after = int(gaussians.get_xyz.shape[0])
                        stats[-1]["prune_count"] = n_before_ev + n_sel - n_after
                        stats[-1]["num_gaussians_after_event"] = n_after
                if iteration % opt.opacity_reset_interval == 0 or \
                        (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            if iteration % 3000 == 0 and 15_000 < iteration < 30_000:
                pass  # not reached before 12000 in B5
            if opt.optimizer_type == "default":
                gaussians.optimizer_step(iteration)
    with open(f"{OUT}/data/b5_snapshot_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        for s in stats:
            w.writerow(s)
    print(f"[b5-collect] done; stats -> {OUT}/data/b5_snapshot_stats.csv")


# ------------------------------------------------------------- process -----

def process(dataset, opt, pipe, args, iters, train_cams, global_probe, cam_seq):
    bg = v2.G["bg"]
    h = args.diag_steps // 2
    path = f"{OUT}/data/b5_group_results.json"
    if os.path.exists(path):
        records = [r for r in json.load(open(path))["records"]
                   if r["iteration"] not in iters]  # drop re-processed snapshots
    else:
        records = []

    for it in iters:
        snap = torch.load(f"{OUT}/cache/snap_{it}.pt")
        snapshot, radii_snap = snap["snapshot"], snap["radii"].clone()
        xyz_snap = snapshot[1]
        n0 = int(xyz_snap.shape[0])
        clone_idx = np.where(snap["clone_set"].cpu().numpy())[0]
        split_idx = np.where(snap["split_set"].cpu().numpy())[0]
        population = np.concatenate([clone_idx, split_idx])
        native_action_of = {int(i): "clone" for i in clone_idx}
        native_action_of.update({int(i): "split" for i in split_idx})
        pool_data = [{"cam": cam, "radii": snap["pool_radii"][i], "mask": snap["pool_masks"][i]}
                     for i, cam in enumerate(train_cams[:args.view_pool])]
        cost_views = [pd["cam"] for pd in pool_data[:args.n_cost_views]]
        n_cand = len(population)
        print(f"\n[b5] === snapshot it={it}: N={n0}, population={n_cand} "
              f"(clone {len(clone_idx)} / split {len(split_idx)}) ===")

        # shared Keep reference (trained once per snapshot)
        keep_err_path = f"{OUT}/cache/keep_{it}.json"
        gk = restore_from(snapshot, opt)
        seed_all(args.train_seed)
        for i in range(1, args.diag_steps + 1):
            native_train_one_iter(it + i, train_cams[cam_seq[i - 1]], gk, pipe, bg, opt)
            with torch.no_grad():
                if opt.optimizer_type == "default":
                    gk.optimizer_step(it + i)

        def oracle_of(idx):
            """B2-C protocol oracle for one candidate (cached per snapshot)."""
            cp = f"{OUT}/cache/oracle_{it}_{idx}.json"
            if os.path.exists(cp):
                return json.load(open(cp))
            views = build_cand_views(idx, pool_data, xyz_snap, args.n_probe)
            e_keep = demand_l1_on_views(gk, views)
            if e_keep is None:
                out = {"parent_index": int(idx), "valid": False}
                json.dump(out, open(cp, "w"))
                return out
            e_clone, e_splits = None, []
            g = restore_from(snapshot, opt)
            g.tmp_radii = radii_snap.clone()
            single = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            single[idx] = True
            g.densify_and_clone_fastgs(single, torch.ones_like(single))
            g.tmp_radii = None
            seed_all(args.train_seed)
            for i in range(1, args.diag_steps + 1):
                native_train_one_iter(it + i, train_cams[cam_seq[i - 1]], g, pipe, bg, opt)
                with torch.no_grad():
                    g.optimizer_step(it + i)
            e_clone = demand_l1_on_views(g, views)
            del g
            for r in range(5):
                g = restore_from(snapshot, opt)
                g.tmp_radii = radii_snap.clone()
                single = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
                single[idx] = True
                seed_all(split_seed_for(idx, r))
                g.densify_and_split_fastgs(single, torch.ones_like(single), N=2)
                g.tmp_radii = None
                seed_all(args.train_seed)
                for i in range(1, args.diag_steps + 1):
                    native_train_one_iter(it + i, train_cams[cam_seq[i - 1]], g, pipe, bg, opt)
                    with torch.no_grad():
                        g.optimizer_step(it + i)
                e_splits.append(demand_l1_on_views(g, views))
                del g
            torch.cuda.empty_cache()
            q_c = e_keep - e_clone
            q_s = float(np.mean([e_keep - e for e in e_splits]))
            out = {"parent_index": int(idx), "valid": True,
                   "native_action": native_action_of[int(idx)],
                   "q_clone": q_c, "q_split_mean": q_s,
                   "q_split_std": float(np.std([e_keep - e for e in e_splits])),
                   "q_best": max(q_c, q_s),
                   "oracle_action": "clone" if q_c > q_s else "split"}
            json.dump(out, open(cp, "w"))
            return out

        # Keep-All eval on the shared keep model (per seed below via ROI)
        for seed_id in range(args.n_seeds):
            rng = np.random.RandomState(args.cand_seed + it * 10 + seed_id)
            order = rng.permutation(population)
            members, tried, invalid = [], 0, 0
            for idx in order:
                if len(members) >= args.K:
                    break
                tried += 1
                o = oracle_of(int(idx))
                if o.get("valid"):
                    members.append(o)
                else:
                    invalid += 1
            print(f"[b5] it={it} seed={seed_id}: {len(members)} valid members "
                  f"({tried} tried, {invalid} invalid)")
            with open(f"{OUT}/data/groups/group_it{it}_seed{seed_id}.json", "w") as f:
                json.dump({"iteration": it, "seed": seed_id,
                           "members": [{"parent_index": m["parent_index"],
                                        "native_action": m["native_action"],
                                        "q_best": m["q_best"]} for m in members]}, f, indent=1)
            if len(members) < args.K:
                print(f"[b5] WARNING: only {len(members)} valid candidates (< {args.K})")

            idxs = [m["parent_index"] for m in members]
            qbest = {m["parent_index"]: m["q_best"] for m in members}

            # fixed full-group ROI
            support_masks, demand_masks = [], []
            for vi, cam in enumerate(cost_views):
                sm = torch.zeros(int(cam.image_height), int(cam.image_width),
                                 dtype=torch.bool, device="cuda")
                for idx in idxs:
                    vr = pool_data[vi]["radii"]
                    if int(vr[idx]) <= 0:
                        continue
                    u, v, _, ok = project_to_pixel(cam, xyz_snap[idx])
                    if not ok:
                        continue
                    x0, y0, x1, y1 = roi_box(u, v, float(vr[idx]), cam.image_width,
                                             cam.image_height, TILE_MARGIN_PX)
                    sm[y0:y1, x0:x1] = True
                support_masks.append(sm)
                demand_masks.append(sm & pool_data[vi]["mask"])

            def run_branch(policy, rho, rep, assign):
                g = restore_from(snapshot, opt)
                if assign is not None:
                    apply_assignment(g, assign, radii_snap)
                n_after = int(g.get_xyz.shape[0])
                seed_all(args.train_seed)
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
                rec = {"iteration": it, "group_seed": seed_id, "rho": rho, "policy": policy,
                       "rw_repeat": rep, "M": 0 if assign is None else len(assign),
                       "K": len(idxs), "num_gaussians_before": n0,
                       "num_gaussians_after": n_after, "delta_num_gaussians": n_after - n0,
                       "demand_l1_0": ev[0]["demand"]["l1"], "demand_l1_50": ev[h]["demand"]["l1"],
                       "demand_l1_100": ev[args.diag_steps]["demand"]["l1"],
                       "demand_psnr_100": ev[args.diag_steps]["demand"]["psnr"],
                       "support_l1_100": ev[args.diag_steps]["support"]["l1"],
                       "global_psnr_100": ev[args.diag_steps]["global"]["psnr"],
                       "global_ssim_100": ev[args.diag_steps]["global"].get("ssim"),
                       "global_lpips_100": ev[args.diag_steps]["global"].get("lpips"),
                       "tile_0": ev[0]["tile_pairs_mean"],
                       "tile_100": ev[args.diag_steps]["tile_pairs_mean"],
                       "latency_ms": lat, "fps": 1000.0 / lat}
                records.append(rec)
                dl1 = rec["demand_l1_100"]
                print(f"    {policy:9s} rep={rep} rho={rho} M={rec['M']:3d} "
                      f"dL1@100={'NA' if dl1 is None else '%.6f' % dl1}")

            # references
            run_branch("keep_all", 0.0, None, None)
            run_branch("native_full", 1.0, None, [(i, native_action_of[i]) for i in idxs])
            for rho in args.rhos:
                M = int(round(rho * args.K))
                ow = sorted(idxs, key=lambda i: -qbest[i])[:M]
                run_branch("ow_nh", rho, None, [(i, native_action_of[i]) for i in ow])
                run_branch("ow_oh", rho, None,
                           [(i, next(m["oracle_action"] for m in members if m["parent_index"] == i))
                            for i in ow])
                for r in range(args.rw_repeats):
                    rr = np.random.RandomState(args.rw_base + seed_id * 100000 + it * 100
                                               + int(round(rho * 100)) * 10 + r)
                    subset = [int(i) for i in rr.choice(np.array(idxs), size=M, replace=False)]
                    run_branch("rw_nh", rho, r, [(i, native_action_of[i]) for i in subset])
                    run_branch("rw_oh", rho, r,
                               [(i, next(m["oracle_action"] for m in members if m["parent_index"] == i))
                                for i in subset])
            # keep tile reference for deltas
            sub = [r for r in records if r["iteration"] == it and r["group_seed"] == seed_id]
            keep = next(r for r in sub if r["policy"] == "keep_all")
            for r in sub:
                r["dTile_0"] = r["tile_0"] - keep["tile_0"]
                r["dTile_100"] = r["tile_100"] - keep["tile_100"]

        del gk
        del snap
        torch.cuda.empty_cache()
        json.dump({"records": records}, open(path, "w"), indent=1)
        print(f"[b5] saved running results ({len(records)} records)")

    cols = list(records[0].keys())
    with open(f"{OUT}/data/b5_group_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow(["NA" if r[c] is None else (f"{r[c]:.8g}" if isinstance(r[c], float) else r[c])
                        for c in cols])
    print(f"[b5] final saved {OUT}/data/b5_group_results.csv ({len(records)} records)")


def main():
    parser = ArgumentParser("Paper B B5: cross-stage capacity oracle")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--mode", type=str, default="collect",
                        choices=["collect", "process", "camident"])
    parser.add_argument("--iters", type=str, default="1000,1500,2000")
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=2)
    parser.add_argument("--rhos", type=str, default="0.50,0.75")
    parser.add_argument("--rw_repeats", type=int, default=5)
    parser.add_argument("--diag_steps", type=int, default=100)
    parser.add_argument("--view_pool", type=int, default=30)
    parser.add_argument("--n_cost_views", type=int, default=8)
    parser.add_argument("--n_probe", type=int, default=8)
    parser.add_argument("--train_seed", type=int, default=1234)
    parser.add_argument("--camseq_seed", type=int, default=2024)
    parser.add_argument("--cand_seed", type=int, default=777)
    parser.add_argument("--rw_base", type=int, default=80000)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    assert opt.optimizer_type == "default"
    iters = sorted(int(x) for x in args.iters.split(","))
    for it in iters:
        assert it > opt.densify_from_iter and it % opt.densification_interval == 0 \
            and it < opt.densify_until_iter, f"{it} is not a native densification event"

    install_c_proxy()
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    v2.G.update({"pipe": pipe, "bg": bg, "mult": opt.mult,
                 "loss_thresh": opt.loss_thresh, "sh_degree": dataset.sh_degree})

    if args.mode == "camident":
        # backfill reference camera identity WITHOUT warmup: replicates the
        # collect-phase path (seed_all(0) -> GaussianModel -> Scene); Scene
        # construction is file-deterministic (no RNG), so this equals what
        # collect saw for the existing snapshots.
        seed_all(0)
        g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        scene = Scene(dataset, g)
        ident = {"train": [_cam_id(c) for c in scene.getTrainCameras()],
                 "test": [_cam_id(c) for c in scene.getTestCameras()],
                 "pool": [_cam_id(c) for c in scene.getTrainCameras()[:args.view_pool]]}
        with open(f"{OUT}/cache/camera_identity.json", "w") as f:
            json.dump(ident, f)
        print(f"[b5-camident] saved: train={len(ident['train'])} test={len(ident['test'])} "
              f"pool={len(ident['pool'])}")
        return

    if args.mode == "collect":
        collect(dataset, opt, pipe, args, iters)
        return

    # process mode: needs scene cameras + cam_seq only (no warmup)
    seed_all(0)  # B5-Fix #1: same RNG state at Scene creation as collect
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    del gaussians
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    global_probe = (test_cams[:5] if test_cams and len(test_cams) > 0 else train_cams[:5])
    pool_cams_now = train_cams[:args.view_pool]

    # B5-Fix #1: camera identity assertion against the collect-phase record
    ident_path = f"{OUT}/cache/camera_identity.json"
    if not os.path.exists(ident_path):
        raise RuntimeError(f"missing {ident_path}: rerun collect first")
    ident = json.load(open(ident_path))
    for name, current, saved in (("train", [_cam_id(c) for c in train_cams], ident["train"]),
                                 ("test", [_cam_id(c) for c in test_cams], ident["test"]),
                                 ("pool", [_cam_id(c) for c in pool_cams_now], ident["pool"])):
        if current != saved:
            bad = next((i for i, (a, b) in enumerate(zip(current, saved)) if a != b),
                       min(len(current), len(saved)))
            raise RuntimeError(
                f"[b5] camera identity assertion FAIL on '{name}' order "
                f"(len {len(current)} vs {len(saved)}, first mismatch at {bad}: "
                f"{current[bad] if bad < len(current) else None} vs "
                f"{saved[bad] if bad < len(saved) else None}) — aborting")
    print("[b5] camera identity assertion PASS (train/test/pool order identical to collect)")

    cam_seq_rng = random.Random(args.camseq_seed)
    cam_seq = [cam_seq_rng.randint(0, len(train_cams) - 1) for _ in range(args.diag_steps)]
    args.rhos = [float(r) for r in args.rhos.split(",")]
    process(dataset, opt, pipe, args, iters, train_cams, global_probe, cam_seq)


if __name__ == "__main__":
    main()
