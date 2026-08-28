#
# Paper B - B2-B: Multi-Candidate Action Scalability Diagnostic
#
# Question: do single-candidate action-specific quality/compute conclusions
# survive when many FastGS candidates are processed simultaneously?
#
# Groups: K in {30, 100, 300} sampled NATURALLY (uniform, real population
# distribution) from the FastGS candidate population at a fixed real
# densification iteration (default 2000, after two native events).
# Branches per group (same checkpoint / optimizer / camera seq / training RNG,
# 100-step continuation, densify/prune/reset OFF):
#   keep      : no action
#   native    : FastGS scale-heuristic action per candidate (native clone/split)
#   all_clone : force clone on every candidate
#   all_split : force split on every candidate (candidate-specific seeds)
#   oracle_mix: NA (no per-candidate quality oracle exists for this fresh
#               population; not faked, no predictor trained)
#
# Compute additivity (core): per candidate i measure dTile_i@0 via
# action + K=0 render on a solo-restored model; predicted group dTile = sum_i;
# compare with the actually-executed group's dTile@0.
#
# Split-seed protocol: seed_i = SPLIT_SEED_BASE + parent_index*100 + 1, keyed
# by parent identity so the solo measurement and the group execution produce
# IDENTICAL children for candidate i (group splits applied in DESCENDING
# original-index order so parent rows are never disturbed by earlier splits).
#
# Outputs -> paper_b/b2_b_scalability/ . No training-logic modification.
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
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402
from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from diagnostics.common import (seed_all, clone_tree, install_c_proxy,  # noqa: E402
                                native_train_one_iter, project_to_pixel, roi_box)
import diagnostics.diagnostic_v2 as v2  # noqa: E402

OUT_DIR = "paper_b/b2_b_scalability"
TILE_MARGIN_PX = 16
SPLIT_SEED_BASE = 200000


def split_seed_for(parent_index):
    return SPLIT_SEED_BASE + int(parent_index) * 100 + 1


def restore_from(snapshot, opt):
    g = GaussianModel(v2.G["sh_degree"], opt.optimizer_type)
    g.restore(clone_tree(snapshot), opt)
    return g


def apply_single_action(g, action, parent_idx, radii_snap):
    g.tmp_radii = radii_snap.clone()
    single = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    single[parent_idx] = True
    ones = torch.ones_like(single)
    if action == "clone":
        g.densify_and_clone_fastgs(single, ones)
    else:
        seed_all(split_seed_for(parent_idx))
        g.densify_and_split_fastgs(single, ones, N=2)
    g.tmp_radii = None


@torch.no_grad()
def tiles_on_views(g, views):
    proxy = install_c_proxy()
    ts = []
    for cam in views:
        render_fastgs(cam, g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])
        ts.append(proxy.last_num_rendered)
    return float(np.mean(ts))


def apply_group_policy(g, policy, members, radii_snap):
    """members: list of (parent_idx, native_action). Splits are applied in
    DESCENDING original-index order so every parent row still holds its
    snapshot values when split (matches solo measurement exactly).
    tmp_radii is set ONCE to the snapshot length; the native postfix/prune
    machinery keeps it row-aligned through every clone/split."""
    if policy == "keep":
        return
    g.tmp_radii = radii_snap.clone()
    if policy in ("native", "all_clone"):
        if policy == "all_clone":
            clone_idx = [m[0] for m in members]
        else:
            clone_idx = [m[0] for m in members if m[1] == "clone"]
        if clone_idx:
            mask = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            mask[torch.tensor(clone_idx, dtype=torch.long, device="cuda")] = True
            g.densify_and_clone_fastgs(mask, torch.ones_like(mask))
    if policy in ("native", "all_split"):
        if policy == "all_split":
            split_idx = [m[0] for m in members]
        else:
            split_idx = [m[0] for m in members if m[1] == "split"]
        for idx in sorted(split_idx, reverse=True):
            single = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            single[idx] = True
            seed_all(split_seed_for(idx))
            g.densify_and_split_fastgs(single, torch.ones_like(single), N=2)
    g.tmp_radii = None


@torch.no_grad()
def group_local_eval(g, views, union_masks):
    """Local metrics on the fixed Group ROI (union of pre-action candidate
    ROIs) — one boolean mask per view; never a sum of per-candidate errors.
    Views whose union mask is empty (no member visible) are skipped."""
    proxy = install_c_proxy()
    l1s, mses, psnrs, tiles = [], [], [], []
    for cam, umask in zip(views, union_masks):
        out = render_fastgs(cam, g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])
        tiles.append(proxy.last_num_rendered)
        if int(umask.sum()) == 0:
            continue
        ri = out["render"][:, umask]
        gi = cam.original_image.cuda()[:, umask]
        diff = (ri - gi).abs()
        l1s.append(float(diff.mean()))
        mse = float((diff ** 2).mean())
        mses.append(mse)
        psnrs.append(10.0 * np.log10(1.0 / max(mse, 1e-10)))
    if not l1s:
        return {"l1": None, "mse": None, "psnr": None,
                "tile_pairs_mean": float(np.mean(tiles)), "n_valid_views": 0}
    return {"l1": float(np.mean(l1s)), "mse": float(np.mean(mses)),
            "psnr": float(np.mean(psnrs)), "tile_pairs_mean": float(np.mean(tiles)),
            "n_valid_views": len(l1s)}


@torch.no_grad()
def global_eval(g, views, with_lpips=False):
    psnrs, ssims, losses, lpipss = [], [], [], []
    for cam in views:
        image = render_fastgs(cam, g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])["render"]
        gt = cam.original_image.cuda()
        Ll1 = float(l1_loss(image, gt))
        ss = float(fast_ssim(image.unsqueeze(0), gt.unsqueeze(0)))
        losses.append(0.8 * Ll1 + 0.2 * (1.0 - ss))
        image_c = torch.clamp(image, 0.0, 1.0)
        gt_c = torch.clamp(gt, 0.0, 1.0)
        psnrs.append(float(psnr(image_c, gt_c).mean()))
        ssims.append(ss)
        if with_lpips:
            try:
                from lpipsPyTorch import lpips as lpips_fn
                lpipss.append(float(lpips_fn(image_c, gt_c, net_type='vgg').mean()))
            except Exception:
                lpipss.append(float("nan"))
    out = {"loss": float(np.mean(losses)), "psnr": float(np.mean(psnrs)),
           "ssim": float(np.mean(ssims))}
    if with_lpips:
        out["lpips"] = float(np.nanmean(lpipss)) if lpipss else None
    return out


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
    parser = ArgumentParser("Paper B B2-B: group scalability diagnostic")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--diag_iter", type=int, default=2000)
    parser.add_argument("--Ks", type=str, default="30,100,300")
    parser.add_argument("--n_group_seeds", type=int, default=3)
    parser.add_argument("--pool_size", type=int, default=30)
    parser.add_argument("--n_cost_views", type=int, default=8)
    parser.add_argument("--diag_steps", type=int, default=100)
    parser.add_argument("--warmup_seed", type=int, default=0)
    parser.add_argument("--train_seed", type=int, default=1234)
    parser.add_argument("--camseq_seed", type=int, default=2024)
    parser.add_argument("--group_base_seed", type=int, default=31337)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    diag_iter = args.diag_iter
    assert diag_iter > opt.densify_from_iter and diag_iter % opt.densification_interval == 0 \
        and diag_iter < opt.densify_until_iter
    assert opt.optimizer_type == "default"
    Ks = [int(k) for k in args.Ks.split(",")]

    install_c_proxy()
    seed_all(args.warmup_seed)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    v2.G.update({"pipe": pipe, "bg": bg, "mult": opt.mult,
                 "loss_thresh": opt.loss_thresh, "sh_degree": dataset.sh_degree})

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    global_probe = (test_cams[:5] if test_cams and len(test_cams) > 0 else train_cams[:5])
    pool_cams = train_cams[:args.pool_size]
    cost_views = pool_cams[:args.n_cost_views]
    cam_seq_rng = random.Random(args.camseq_seed)
    cam_seq = [cam_seq_rng.randint(0, len(train_cams) - 1) for _ in range(args.diag_steps)]
    print(f"[b2b] iter={diag_iter} Ks={Ks} seeds={args.n_group_seeds} cost_views={len(cost_views)}")

    # ---------------- warmup to diag_iter (native, with hook) -------------
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    hook = None
    for iteration in range(1, diag_iter + 1):
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
                    if iteration == diag_iter:
                        clone_set, split_set, _, _, _ = v2.native_selection_sets(
                            gaussians, imp, opt, scene.cameras_extent)
                        # pool renders (pre-action): per-view radii for ROI building
                        pool_radii = []
                        with torch.no_grad():
                            for cam in pool_cams:
                                out = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
                                pool_radii.append(out["radii"])
                        hook = {"importance": imp, "clone_set": clone_set, "split_set": split_set,
                                "pool_radii": pool_radii, "radii": radii.clone(),
                                "snapshot": clone_tree(gaussians.capture(opt.optimizer_type))}
                    gaussians.densify_and_prune_fastgs(
                        max_screen_size=(20 if iteration > opt.opacity_reset_interval else None),
                        min_opacity=0.005, extent=scene.cameras_extent, radii=radii, args=opt,
                        importance_score=imp, pruning_score=pru)
                if iteration % opt.opacity_reset_interval == 0 or \
                        (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            if iteration % 3000 == 0 and 15_000 < iteration < 30_000:
                my_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_stack)
                _, pru = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
                gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pru)
            if opt.optimizer_type == "default":
                gaussians.optimizer_step(iteration)

    assert hook is not None
    snapshot = hook["snapshot"]  # captured BEFORE the event's native densification
    radii_snap = hook["radii"].clone()
    n_before = int(snapshot[1].shape[0])
    xyz_snap = snapshot[1]

    clone_idx_all = np.where(hook["clone_set"].cpu().numpy())[0]
    split_idx_all = np.where(hook["split_set"].cpu().numpy())[0]
    population = np.concatenate([clone_idx_all, split_idx_all])
    native_action_of = {int(i): "clone" for i in clone_idx_all}
    native_action_of.update({int(i): "split" for i in split_idx_all})
    imp = hook["importance"].to(torch.float32).cpu().numpy()
    print(f"[b2b] population at iter {diag_iter}: {len(population)} "
          f"(clone {len(clone_idx_all)} / split {len(split_idx_all)}), N={n_before}")
    del gaussians
    torch.cuda.empty_cache()

    # keep baseline tiles (solo state == snapshot)
    g0 = restore_from(snapshot, opt)
    tile_keep_0 = tiles_on_views(g0, cost_views)
    del g0

    # ---------------- groups ----------------------------------------------
    os.makedirs(f"{OUT_DIR}/data", exist_ok=True)
    records = []
    h = args.diag_steps // 2
    for K in Ks:
        for seed_id in range(args.n_group_seeds):
            rng = np.random.RandomState(args.group_base_seed + K * 10 + seed_id)
            gidx = rng.choice(population, size=min(K, len(population)), replace=False)
            members = [(int(i), native_action_of[int(i)]) for i in sorted(gidx)]
            n_clone = sum(1 for _, a in members if a == "clone")
            print(f"\n[b2b] K={K} seed={seed_id}: {len(members)} members "
                  f"(native clone {n_clone} / split {len(members)-n_clone})")

            # ---- per-candidate ROI / Group ROI union (pre-action, fixed) ----
            union_masks, vis_counts = [], []
            for vi, cam in enumerate(cost_views):
                umask = torch.zeros(int(cam.image_height), int(cam.image_width),
                                    dtype=torch.bool, device="cuda")
                cnt = 0
                for idx, _ in members:
                    vr = hook["pool_radii"][vi]
                    if int(vr[idx]) <= 0:
                        continue
                    u, v, _, ok = project_to_pixel(cam, xyz_snap[idx])
                    if not ok:
                        continue
                    x0, y0, x1, y1 = roi_box(u, v, float(vr[idx]),
                                             cam.image_width, cam.image_height, TILE_MARGIN_PX)
                    umask[y0:y1, x0:x1] = True
                    cnt += 1
                union_masks.append(umask)
                vis_counts.append(cnt)
            cov_px = [int(m.sum()) for m in union_masks]
            print(f"    group ROI px/view (mean={np.mean(cov_px):.0f}), visible/view={np.mean(vis_counts):.1f}")

            # ---- save group membership ----
            with open(f"{OUT_DIR}/data/group_candidates_K{K}_seed{seed_id}.json", "w") as f:
                json.dump({"iteration": diag_iter, "K": K, "seed": seed_id,
                           "members": [{"parent_index": i, "native_action": a,
                                        "importance_score": int(imp[i]),
                                        "split_seed": split_seed_for(i)} for i, a in members],
                           "group_roi_px_per_view": cov_px}, f, indent=1)

            # ---- singles: dTile_i@0 for clone and split (solo restore) ----
            solo = {}
            for idx, act in members:
                row = {}
                for action in ("clone", "split"):
                    g = restore_from(snapshot, opt)
                    apply_single_action(g, action, idx, radii_snap)
                    row[action] = tiles_on_views(g, cost_views) - tile_keep_0
                    del g
                torch.cuda.empty_cache()
                solo[idx] = row

            # ---- group branches ----
            for policy in ("keep", "native", "all_clone", "all_split"):
                g = restore_from(snapshot, opt)
                if policy != "keep":
                    apply_group_policy(g, policy, members, radii_snap)
                n_after = int(g.get_xyz.shape[0])

                seed_all(args.train_seed)
                ev = {0: (group_local_eval(g, cost_views, union_masks),
                          global_eval(g, global_probe))}
                for i in range(1, args.diag_steps + 1):
                    cam = train_cams[cam_seq[i - 1]]
                    native_train_one_iter(diag_iter + i, cam, g, pipe, bg, opt)
                    with torch.no_grad():
                        if opt.optimizer_type == "default":
                            g.optimizer_step(diag_iter + i)
                        if i in (h, args.diag_steps):
                            ev[i] = (group_local_eval(g, cost_views, union_masks),
                                     global_eval(g, global_probe, with_lpips=(i == args.diag_steps)))
                lat = latency_light(g, cost_views)

                tile_policy_0 = ev[0][0]["tile_pairs_mean"]
                actual_d = None if policy == "keep" else tile_policy_0 - tile_keep_0
                if policy == "keep":
                    pred_d = None
                elif policy == "native":
                    pred_d = sum(solo[i]["clone" if a == "clone" else "split"] for i, a in members)
                elif policy == "all_clone":
                    pred_d = sum(solo[i]["clone"] for i, _ in members)
                else:
                    pred_d = sum(solo[i]["split"] for i, _ in members)
                rec = {
                    "K": K, "group_seed": seed_id, "policy": policy,
                    "num_candidates": len(members),
                    "native_clone_count": n_clone, "native_split_count": len(members) - n_clone,
                    "num_gaussians_before": n_before, "num_gaussians_after": n_after,
                    "delta_num_gaussians": n_after - n_before,
                    "tile_keep_0": tile_keep_0, "tile_policy_0": tile_policy_0,
                    "actual_delta_tile": actual_d, "predicted_delta_tile": pred_d,
                    "tile_additivity_absolute_error": None if pred_d is None else actual_d - pred_d,
                    "tile_additivity_relative_error": None if pred_d is None else
                        abs(actual_d - pred_d) / max(abs(actual_d), 1e-6),
                    "global_psnr_0": ev[0][1]["psnr"], "global_psnr_50": ev[h][1]["psnr"],
                    "global_psnr_100": ev[args.diag_steps][1]["psnr"],
                    "global_loss_0": ev[0][1]["loss"], "global_loss_100": ev[args.diag_steps][1]["loss"],
                    "global_ssim_100": ev[args.diag_steps][1].get("ssim"),
                    "global_lpips_100": ev[args.diag_steps][1].get("lpips"),
                    "group_local_l1_0": ev[0][0]["l1"], "group_local_l1_50": ev[h][0]["l1"],
                    "group_local_l1_100": ev[args.diag_steps][0]["l1"],
                    "group_local_psnr_0": ev[0][0]["psnr"],
                    "group_local_psnr_100": ev[args.diag_steps][0]["psnr"],
                    "tile_policy_100": ev[args.diag_steps][0]["tile_pairs_mean"],
                    "render_latency_ms": lat, "fps": 1000.0 / lat,
                    "group_roi_px_mean": float(np.mean(cov_px)),
                }
                records.append(rec)
                ae = rec["tile_additivity_absolute_error"]
                gl1 = rec["group_local_l1_100"]
                print(f"    {policy:9s}: N{rec['delta_num_gaussians']:+d} "
                      f"dTile={actual_d if actual_d is None else round(actual_d,2)} "
                      f"(pred {pred_d if pred_d is None else round(pred_d,2)}, "
                      f"err {ae if ae is None else round(ae,4)}) "
                      f"gL1@100={'NA' if gl1 is None else f'{gl1:.6f}'} "
                      f"gPSNR@100={rec['global_psnr_100']:.4f}")
                del g
                torch.cuda.empty_cache()

    # ---------------- save -------------------------------------------------
    with open(f"{OUT_DIR}/data/b2b_results.json", "w") as f:
        json.dump({"scene": os.path.abspath(dataset.source_path),
                   "diag_iter": diag_iter, "n_before": n_before,
                   "population": {"clone": len(clone_idx_all), "split": len(split_idx_all)},
                   "config": {k: v for k, v in vars(args).items()},
                   "oracle_mix": "NA",
                   "single_candidate_quality_scaling": "NOT AVAILABLE",
                   "records": records}, f, indent=1)
    cols = list(records[0].keys())
    with open(f"{OUT_DIR}/data/b2b_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow(["NA" if r[c] is None else (f"{r[c]:.8g}" if isinstance(r[c], float) else r[c])
                        for c in cols])
    print(f"\n[b2b] saved {OUT_DIR}/data/b2b_results.json / .csv , {len(records)} records")


if __name__ == "__main__":
    main()
