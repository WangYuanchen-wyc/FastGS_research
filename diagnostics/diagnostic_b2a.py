#
# Paper B - B2-A: Split Stochasticity + Residual-Gaussian Geometric Alignment
#
# Builds on Diagnostic V2 (same candidates, same protocol) and adds:
#   * Split repeated 5x per candidate with candidate-specific seeds
#       seed = split_base_seed + candidate_id * 100 + repeat_id
#     (only the split child sampling varies; checkpoint / optimizer /
#      camera sequence / training RNG stay identical)
#   * pre-action parent projected Gaussian geometry (EWA 2D covariance,
#     self-calibrated against native radii)
#   * residual <-> parent alignment (|cos|), extent ratio, centroid offset
#   * per-split-repeat child separation direction and
#     residual_split_alignment = |cos(residual dir, child1->child2 dir)|
#
# Outputs -> paper_b/b2_a_split_alignment/data/b2a_results.{json,csv}
# Diagnostic only: no predictor / allocator / budget / lineage / rollback.
#

import os
import sys
import json
import random

import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel  # noqa: E402
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from diagnostics.common import seed_all, clone_tree, install_c_proxy, native_train_one_iter  # noqa: E402
import diagnostics.diagnostic_v2 as v2  # noqa: E402  (build_batch / local_eval / G)

OUT_DIR = "paper_b/b2_a_split_alignment"


def restore_from(snapshot, opt):
    g = GaussianModel(v2.G["sh_degree"], opt.optimizer_type)
    g.restore(clone_tree(snapshot), opt)
    return g


def apply_action(g, branch, parent_idx, radii_snap, seed):
    if branch == "keep":
        return
    seed_all(seed)
    g.tmp_radii = radii_snap.clone()
    single = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    single[parent_idx] = True
    if branch == "clone":
        g.densify_and_clone_fastgs(single, torch.ones_like(single))
    else:
        g.densify_and_split_fastgs(single, torch.ones_like(single), N=2)
    g.tmp_radii = None


def child_separation_alignment(g, cand_views):
    """K=0 (post-action, pre-training): project the two children of the last
    split into every fixed view; residual_split_alignment = |cos(residual
    principal direction, child1->child2 screen direction)| per view."""
    from diagnostics.common import project_to_pixel
    c1 = g.get_xyz[-2].detach()
    c2 = g.get_xyz[-1].detach()
    rows, valid = [], 0
    for vd in cand_views:
        rd = vd.get("residual")
        if rd is None or np.isnan(rd["residual_direction_deg"]):
            rows.append(None)
            continue
        u1, v1, _, ok1 = project_to_pixel(vd["cam"], c1)
        u2, v2, _, ok2 = project_to_pixel(vd["cam"], c2)
        if not (ok1 and ok2):
            rows.append(None)
            continue
        dx, dy = u2 - u1, v2 - v1
        nrm = float(np.hypot(dx, dy))
        if nrm < 1e-3:  # degenerate separation (same pixel)
            rows.append(None)
            continue
        theta = np.deg2rad(rd["residual_direction_deg"])
        cosv = abs(float((dx * np.cos(theta) + dy * np.sin(theta)) / nrm))
        rows.append({"sep_px": nrm, "sep_dir_deg": float(np.degrees(np.arctan2(dy, dx)) % 180.0),
                     "residual_split_alignment": cosv})
        valid += 1
    mean_al = float(np.mean([r["residual_split_alignment"] for r in rows if r])) if valid else None
    return {"per_view": rows, "mean": mean_al, "n_valid": valid}


def run_branch(branch, seed, cand, snapshot, radii_snap, opt, train_cams,
               cam_seq, diag_iter, diag_steps, train_seed):
    g = restore_from(snapshot, opt)
    apply_action(g, branch, cand["parent_index"], radii_snap, seed)
    n_after = int(g.get_xyz.shape[0])
    sep_info = child_separation_alignment(g, cand["_views"]) if branch == "split" else None

    seed_all(train_seed)
    views = cand["_views"]
    evals = {0: v2.local_eval(g, views)}
    for i in range(1, diag_steps + 1):
        cam = train_cams[cam_seq[i - 1]]
        native_train_one_iter(diag_iter + i, cam, g, v2.G["pipe"], v2.G["bg"], opt)
        with torch.no_grad():
            if opt.optimizer_type == "default":
                g.optimizer_step(diag_iter + i)
            if i in (diag_steps // 2, diag_steps):
                evals[i] = v2.local_eval(g, views)
    del g
    torch.cuda.empty_cache()
    return {"n_after": n_after, "evals": evals, "sep": sep_info}


def main():
    parser = ArgumentParser("Paper B B2-A: split stochasticity + alignment")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--diag_iters", type=str, default="1000,1500,2000")
    parser.add_argument("--n_cand", type=int, default=20)
    parser.add_argument("--pool_size", type=int, default=30)
    parser.add_argument("--n_probe", type=int, default=8)
    parser.add_argument("--diag_steps", type=int, default=100)
    parser.add_argument("--n_split_repeats", type=int, default=5)
    parser.add_argument("--force_one_each", action="store_true")
    parser.add_argument("--warmup_seed", type=int, default=0)
    parser.add_argument("--action_seed", type=int, default=1234)
    parser.add_argument("--train_seed", type=int, default=1234)
    parser.add_argument("--camseq_seed", type=int, default=2024)
    parser.add_argument("--cand_seed", type=int, default=777)
    parser.add_argument("--split_base_seed", type=int, default=90000)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    diag_iters = sorted(int(x) for x in args.diag_iters.split(",") if x.strip())
    for it in diag_iters:
        assert it > opt.densify_from_iter and it % opt.densification_interval == 0 \
            and it < opt.densify_until_iter, f"diag_iter {it} is not a native event"
    assert opt.optimizer_type == "default"

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
    pool_cams = train_cams[:args.pool_size]
    cam_seq_rng = random.Random(args.camseq_seed)
    cam_seq = [cam_seq_rng.randint(0, len(train_cams) - 1) for _ in range(args.diag_steps)]
    print(f"[b2a] pool={len(pool_cams)} cams, repeats={args.n_split_repeats}")

    # ---------------- native warmup with hooks (identical to V2) ----------
    batches = {}
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    cand_seed = args.cand_seed

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
                    importance_score, pruning_score = compute_gaussian_score_fastgs(
                        camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if iteration in diag_iters:
                        print(f"\n[b2a] === hook at iteration {iteration} ===")
                        batch = v2.build_batch(gaussians, scene, opt, pool_cams,
                                               importance_score, pruning_score, iteration,
                                               args.n_cand, args.n_probe, cand_seed,
                                               args.force_one_each)
                        cand_seed += 1
                        batch["snapshot"] = clone_tree(gaussians.capture(opt.optimizer_type))
                        batch["radii"] = radii.clone()
                        batches[iteration] = batch
                    gaussians.densify_and_prune_fastgs(
                        max_screen_size=size_threshold, min_opacity=0.005,
                        extent=scene.cameras_extent, radii=radii, args=opt,
                        importance_score=importance_score, pruning_score=pruning_score)

                if iteration % opt.opacity_reset_interval == 0 or \
                        (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 3000 == 0 and 15_000 < iteration < 30_000:
                my_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_stack)
                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
                gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)

            if opt.optimizer_type == "default":
                gaussians.optimizer_step(iteration)

    del gaussians
    torch.cuda.empty_cache()

    # ---------------- per-candidate branches ------------------------------
    records = []
    cid = 0  # global candidate counter used in split seeds
    h = args.diag_steps // 2
    for diag_iter in diag_iters:
        batch = batches[diag_iter]
        snapshot, radii_snap = batch["snapshot"], batch["radii"]
        print(f"\n[b2a] batch iter={diag_iter}: {len(batch['candidates'])} candidates "
              f"(clone set={batch['n_clone_set']}, split set={batch['n_split_set']})")
        for ci, cand in enumerate(batch["candidates"]):
            rec = {k: vv for k, vv in cand.items() if not k.startswith("_")}
            rec["candidate_id"] = f"it{diag_iter}_c{ci:02d}"

            keep = run_branch("keep", 0, cand, snapshot, radii_snap, opt,
                              train_cams, cam_seq, diag_iter, args.diag_steps, args.train_seed)
            clone = run_branch("clone", args.action_seed, cand, snapshot, radii_snap,
                               opt, train_cams, cam_seq, diag_iter, args.diag_steps, args.train_seed)
            splits, split_detail = [], []
            for r in range(args.n_split_repeats):
                seed = args.split_base_seed + cid * 100 + r
                sb = run_branch("split", seed, cand, snapshot, radii_snap, opt,
                                train_cams, cam_seq, diag_iter, args.diag_steps, args.train_seed)
                splits.append(sb)
                split_detail.append({
                    "repeat": r, "seed": seed,
                    "demand_l1": {str(s): sb["evals"][s]["demand"]["l1"] for s in (0, h, args.diag_steps)},
                    "support_l1": {str(s): sb["evals"][s]["support"]["l1"] for s in (0, h, args.diag_steps)},
                    "tile_pairs": {str(s): sb["evals"][s]["tile_pairs_mean"] for s in (0, args.diag_steps)},
                    "residual_split_alignment": sb["sep"]["mean"] if sb["sep"] else None,
                    "sep_per_view": sb["sep"]["per_view"] if sb["sep"] else None,
                })

            def ev(br, s, kind):
                return br["evals"][s][kind]["l1"]

            rec["num_gaussians_keep"] = keep["n_after"]
            rec["num_gaussians_clone"] = clone["n_after"]
            for step, tag in ((0, "0"), (h, str(h)), (args.diag_steps, "100")):
                rec[f"keep_local_error_{tag}"] = ev(keep, step, "support")
                rec[f"clone_local_error_{tag}"] = ev(clone, step, "support")
                rec[f"keep_demand_error_{tag}"] = ev(keep, step, "demand")
                rec[f"clone_demand_error_{tag}"] = ev(clone, step, "demand")
                rec[f"tile_keep_{tag}"] = keep["evals"][step]["tile_pairs_mean"]
                rec[f"tile_clone_{tag}"] = clone["evals"][step]["tile_pairs_mean"]
            rec["deltaQ_demand_clone_100"] = rec["keep_demand_error_100"] - rec["clone_demand_error_100"]

            dqs = [ev(sb, args.diag_steps, "demand") for sb in splits]
            dqs = [d for d in dqs if d is not None]
            sqs = [ev(sb, args.diag_steps, "support") for sb in splits]
            tiles0 = [sb["evals"][0]["tile_pairs_mean"] for sb in splits]
            tiles100 = [sb["evals"][args.diag_steps]["tile_pairs_mean"] for sb in splits]
            dqs0 = [ev(sb, 0, "demand") for sb in splits]

            rec["split_dQ_mean"] = float(np.mean([rec["keep_demand_error_100"] - d for d in dqs]))
            rec["split_dQ_std"] = float(np.std([rec["keep_demand_error_100"] - d for d in dqs]))
            rec["split_dQ_min"] = float(np.min([rec["keep_demand_error_100"] - d for d in dqs]))
            rec["split_dQ_max"] = float(np.max([rec["keep_demand_error_100"] - d for d in dqs]))
            rec["split_dQ0_mean"] = float(np.mean([rec["keep_demand_error_0"] - d for d in dqs0])) if all(
                d is not None for d in dqs0) else None
            rec["split_dQ_support_mean"] = float(np.mean(
                [rec["keep_local_error_100"] - s for s in sqs]))
            rec["split_stochastic_range"] = rec["split_dQ_max"] - rec["split_dQ_min"]
            rec["split_tile_mean_0"] = float(np.mean(tiles0))
            rec["split_tile_std_0"] = float(np.std(tiles0))
            rec["split_tile_mean_100"] = float(np.mean(tiles100))
            rec["split_tile_std_100"] = float(np.std(tiles100))
            rec["deltaTile_clone_0"] = rec["tile_clone_0"] - rec["tile_keep_0"]
            rec["deltaTile_split_0"] = rec["split_tile_mean_0"] - rec["tile_keep_0"]
            rec["deltaTile_clone_100"] = rec["tile_clone_100"] - rec["tile_keep_100"]
            rec["deltaTile_split_100"] = rec["split_tile_mean_100"] - rec["tile_keep_100"]
            rec["num_gaussians_split"] = int(np.mean([sb["n_after"] for sb in splits]))

            # stable action gap (split = mean over 5 repeats)
            rec["quality_action_gap"] = rec["split_dQ_mean"] - rec["deltaQ_demand_clone_100"]
            rec["clone_split_mean_gap_abs"] = abs(rec["split_dQ_mean"] - rec["deltaQ_demand_clone_100"])

            # residual-split alignment per repeat
            aligns = [d["residual_split_alignment"] for d in split_detail
                      if d["residual_split_alignment"] is not None]
            rec["residual_split_alignment_mean"] = float(np.mean(aligns)) if aligns else None
            rec["residual_split_alignment_std"] = float(np.std(aligns)) if aligns else None

            # winner (split uses mean quality over repeats)
            errs = {"keep": rec["keep_demand_error_100"],
                    "clone": rec["clone_demand_error_100"],
                    "split": float(np.mean(dqs))}
            rec["oracle_basis"] = "demand"
            rec["oracle_winner"] = min(errs, key=errs.get)

            rec["_split_detail"] = split_detail
            records.append(rec)
            print(f"  cid {cid} ({rec['candidate_id']}, {cand['native_action']}): "
                  f"dQ_clone={rec['deltaQ_demand_clone_100']:+.6f} "
                  f"dQ_split={rec['split_dQ_mean']:+.6f}±{rec['split_dQ_std']:.6f} "
                  f"range={rec['split_stochastic_range']:.6f} "
                  f"align={rec['residual_split_alignment_mean']} "
                  f"geom_mismatch_max={rec.get('geom_radius_mismatch_max')}")
            cid += 1

    # ---------------- save -------------------------------------------------
    os.makedirs(f"{OUT_DIR}/data", exist_ok=True)
    json_out = {"scene": os.path.abspath(dataset.source_path),
                "config": {k: v for k, v in vars(args).items()},
                "batches": {str(k): {"n_clone_set": b["n_clone_set"],
                                     "n_split_set": b["n_split_set"],
                                     "n_gaussians_at_event": b["n_gaussians_at_event"]}
                            for k, b in batches.items()},
                "records": records}
    with open(f"{OUT_DIR}/data/b2a_results.json", "w") as f:
        json.dump(json_out, f, indent=1)

    cols = ["candidate_id", "iteration", "parent_index", "native_action",
            "importance_score", "grad", "grad_abs", "opacity",
            "scale_max", "scale_anisotropy", "footprint_mean",
            "proj_anisotropy_mean",
            "residual_energy_mean", "residual_anisotropy_mean",
            "residual_parent_alignment_mean", "residual_extent_ratio_mean",
            "residual_centroid_offset_mean",
            "keep_demand_error_100", "clone_demand_error_100",
            "deltaQ_demand_clone_100",
            "split_dQ_mean", "split_dQ_std", "split_dQ_min", "split_dQ_max",
            "split_stochastic_range", "clone_split_mean_gap_abs",
            "split_dQ0_mean", "split_dQ_support_mean",
            "quality_action_gap", "oracle_winner", "oracle_basis",
            "deltaTile_clone_0", "deltaTile_split_0",
            "split_tile_std_0",
            "deltaTile_clone_100", "deltaTile_split_100",
            "tile_keep_0", "tile_clone_0", "split_tile_mean_0",
            "tile_keep_100", "tile_clone_100", "split_tile_mean_100",
            "residual_split_alignment_mean", "residual_split_alignment_std",
            "num_gaussians_keep", "num_gaussians_clone", "num_gaussians_split",
            "num_valid_local_views", "num_valid_demand_views",
            "geom_radius_mismatch_max"]
    import csv
    with open(f"{OUT_DIR}/data/b2a_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            row = []
            for c in cols:
                v = r.get(c, None)
                row.append("NA" if v is None else (f"{v:.8g}" if isinstance(v, float) else v))
            w.writerow(row)
    print(f"\n[b2a] saved {OUT_DIR}/data/b2a_results.json , .csv , {len(records)} candidates")


if __name__ == "__main__":
    main()
