#
# Paper B - B9-T: Temporal Split Demand (model-level complete training)
#
# Question: does the value of a stronger split (parent -> 3 children instead
# of FastGS's native 2) depend on the training stage?
#
# Strategies (identical FastGS settings except split cardinality in the
# designated stage window; densification active range derived from the real
# schedule: events at 1000..14500, step 500):
#     native    : Split-2 everywhere
#     early3    : Split-3 for events in [1000, 5500)
#     middle3   : Split-3 for events in [5500, 10000)
#     late3     : Split-3 for events in [10000, 14500)
#     always3   : Split-3 everywhere
#
# Implementation: a thin wrapper around the NATIVE training loop. At each
# native densification event we compute the native scores exactly as
# train.py does, then apply densify_and_clone_fastgs to the native clone
# set (unchanged) and densify_and_split_fastgs with N=2 or N=3 for the
# native split set depending on the strategy window. Everything else
# (VCD metric, pruning, opacity reset, final prune, optimizer schedule)
# is the untouched native code path. NOTE: the native budget-prune inside
# densify_and_prune_fastgs would be skipped by this split; to keep the rest
# of the native behavior we reimplement the same prune steps the native
# function performs after clone/split (opacity<0.005 / oversized prune with
# the same multinomial budget sampling, using the same native pruning_score).
#
# Eval every 500 iters: train-view sample PSNR/SSIM + test PSNR/SSIM/LPIPS,
# #GS, wall-clock. Full 30k iterations, multiple seeds.
#

import os
import sys
import json
import random
import time
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
from diagnostics.common import install_c_proxy, seed_all, native_train_one_iter  # noqa: E402

try:
    from lpipsPyTorch import lpips as lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False

OUT = "paper_b/b9_t_temporal_split_demand"
STAGES = {"early3": (1000, 5500), "middle3": (5500, 10000), "late3": (10000, 14500)}


def apply_native_densify(g, opt, it, radii, imp, pru, scene, extent, N_split):
    """Native clone + split with a configurable split cardinality, followed by
    the same prune steps as densify_and_prune_fastgs (opacity / size prune +
    native multinomial budget + 0.8 opacity truncation)."""
    # replicate native selection math (gaussian_model.py:477-494)
    grad_vars = g.xyz_gradient_accum / g.denom
    grad_vars[grad_vars.isnan()] = 0.0
    grads_abs = g.xyz_gradient_accum_abs / g.denom
    grads_abs[grads_abs.isnan()] = 0.0
    grad_qual = torch.norm(grad_vars, dim=-1) >= opt.grad_thresh
    grad_abs_qual = torch.norm(grads_abs, dim=-1) >= opt.grad_abs_thresh
    max_scale = g.get_scaling.max(dim=1).values
    clone_qual = max_scale <= opt.dense * extent
    split_qual = max_scale > opt.dense * extent
    metric_mask = imp > 5
    clone_set = metric_mask & clone_qual & grad_qual
    split_set = metric_mask & split_qual & grad_abs_qual

    g.tmp_radii = radii
    # clone (native N/A) — batched
    if clone_set.any():
        g.densify_and_clone_fastgs(clone_set, torch.ones_like(clone_set))
    # split with N
    if split_set.any():
        g.densify_and_split_fastgs(split_set, torch.ones_like(split_set), N=N_split)
    g.tmp_radii = None

    # ---- native prune steps (gaussian_model.py:499-522) ----
    min_opacity = 0.005
    size_threshold = 20 if it > opt.opacity_reset_interval else None
    prune_mask = (g.get_opacity < min_opacity).squeeze()
    if size_threshold:
        big_vs = g.max_radii2D > size_threshold
        big_ws = g.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_vs), big_ws)
    scores = 1 - pru
    to_remove = int(torch.sum(prune_mask))
    remove_budget = int(0.5 * to_remove)
    if remove_budget:
        n_now = g.get_xyz.shape[0]
        padded = torch.zeros((n_now), dtype=torch.float32, device=scores.device)
        padded[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
        sel = torch.zeros_like(padded, dtype=bool)
        idx = torch.multinomial(padded, remove_budget, replacement=False)
        sel[idx] = True
        final = torch.logical_and(prune_mask, sel)
        g.prune_points(final)
    opa_new = torch.min(g.get_opacity, torch.ones_like(g.get_opacity) * 0.8)
    from utils.general_utils import inverse_sigmoid
    optimizable = g.replace_tensor_to_optimizer(
        inverse_sigmoid(opa_new), "opacity")
    g._opacity = optimizable["opacity"]
    torch.cuda.empty_cache()


def main():
    parser = ArgumentParser("Paper B B9-T: temporal split demand")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--strategies", type=str, default="native,early3,middle3,late3,always3")
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--max_iters", type=int, default=30000)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    assert opt.optimizer_type == "default"
    strategies = args.strategies.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    curve_path = f"{OUT}/data/training_curves.csv"
    done_keys = set()
    if os.path.exists(curve_path):
        with open(curve_path) as f:
            done_keys = {(r["strategy"], r["seed"]) for r in csv.DictReader(f)}
    rows = []
    event_rows = []

    for seed in seeds:
        for strat in strategies:
            if (strat, str(seed)) in done_keys:
                print(f"[b9t] skip {strat} seed{seed} (done)")
                continue
            install_c_proxy()
            seed_all(seed)
            gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
            scene = Scene(dataset, gaussians, shuffle=True)
            gaussians.training_setup(opt)
            test_cams = scene.getTestCameras()[:20]
            train_eval_cams = scene.getTrainCameras()[:8]

            def n_split_at(it):
                if strat == "native":
                    return 2
                if strat == "always3":
                    return 3
                lo, hi = STAGES[strat]
                return 3 if lo <= it < hi else 2

            t0 = time.time()
            vp_stack = scene.getTrainCameras().copy()
            vp_idx = list(range(len(vp_stack)))
            for it in range(1, args.max_iters + 1):
                if not vp_stack:
                    vp_stack = scene.getTrainCameras().copy()
                    vp_idx = list(range(len(vp_stack)))
                r = random.randint(0, len(vp_idx) - 1)
                cam = vp_stack.pop(r)
                _ = vp_idx.pop(r)
                gaussians.update_learning_rate(it)
                if it % 1000 == 0:
                    gaussians.oneupSHdegree()
                _, vpt, vis, radii = native_train_one_iter(it, cam, gaussians, pipe, bg, opt)
                with torch.no_grad():
                    if it < opt.densify_until_iter:
                        gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis])
                        gaussians.add_densification_stats(vpt, vis)
                        if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                            n = n_split_at(it)
                            my = scene.getTrainCameras().copy()
                            cl = sampling_cameras(my)
                            imp, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt, DENSIFY=True)
                            n_before = gaussians.get_xyz.shape[0]
                            n_split_par = int((imp > 5)[
                                (gaussians.get_scaling.max(dim=1).values > opt.dense * scene.cameras_extent)
                            ].sum())
                            apply_native_densify(gaussians, opt, it, radii, imp, pru,
                                                 scene, scene.cameras_extent, N_split=n)
                            event_rows.append({
                                "strategy": strat, "seed": seed, "iteration": it,
                                "N_split": n, "n_before": n_before,
                                "n_after": gaussians.get_xyz.shape[0],
                                "split_parents": n_split_par})
                        if it % opt.opacity_reset_interval == 0:
                            gaussians.reset_opacity()
                    if it % 3000 == 0 and 15_000 < it < 30_000:
                        my = scene.getTrainCameras().copy()
                        cl = sampling_cameras(my)
                        _, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt)
                        gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pru)
                    gaussians.optimizer_step(it)

                    if it % args.eval_every == 0 or it == args.max_iters:
                        el = time.time() - t0
                        tr_psnr = tr_ssim = te_psnr = te_ssim = te_lpips = None
                        with torch.no_grad():
                            tp, tss = [], []
                            for c in train_eval_cams:
                                im = render_fastgs(c, gaussians, pipe, bg, opt.mult)["render"]
                                gt = c.original_image.cuda()
                                tp.append(float(psnr(torch.clamp(im, 0, 1), torch.clamp(gt, 0, 1)).mean()))
                                tss.append(float(fast_ssim(torch.clamp(im, 0, 1).unsqueeze(0),
                                                           torch.clamp(gt, 0, 1).unsqueeze(0)).mean()))
                            tr_psnr, tr_ssim = float(np.mean(tp)), float(np.mean(tss))
                            sp, sss, sl = [], [], []
                            for c in test_cams:
                                im = render_fastgs(c, gaussians, pipe, bg, opt.mult)["render"]
                                gt = c.original_image.cuda()
                                ic, gc = torch.clamp(im, 0, 1), torch.clamp(gt, 0, 1)
                                sp.append(float(psnr(ic, gc).mean()))
                                sss.append(float(fast_ssim(ic.unsqueeze(0), gc.unsqueeze(0)).mean()))
                                if LPIPS_OK:
                                    try:
                                        sl.append(float(lpips_fn(ic, gc, net_type="vgg").mean()))
                                    except Exception:
                                        sl.append(float("nan"))
                            te_psnr, te_ssim = float(np.mean(sp)), float(np.mean(sss))
                            te_lpips = float(np.nanmean(sl)) if sl else None
                        rows.append({"strategy": strat, "seed": seed, "iteration": it,
                                     "train_psnr": tr_psnr, "train_ssim": tr_ssim,
                                     "test_psnr": te_psnr, "test_ssim": te_ssim,
                                     "test_lpips": te_lpips,
                                     "n_gaussians": gaussians.get_xyz.shape[0],
                                     "wall_time_s": el})
                        print(f"[b9t] {strat} s{seed} it{it}: PSNR {te_psnr:.3f} "
                              f"#GS {rows[-1]['n_gaussians']} t {el:.0f}s")
                        # incremental flush
                        with open(curve_path, "w", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                            w.writeheader()
                            w.writerows(rows)
                        if event_rows:
                            with open(f"{OUT}/data/split_event_stats.csv", "w", newline="") as f:
                                w = csv.DictWriter(f, fieldnames=list(event_rows[0].keys()))
                                w.writeheader()
                                w.writerows(event_rows)
            del gaussians, scene
            torch.cuda.empty_cache()
    print(f"[b9t] all done ({len(rows)} curve rows)")


if __name__ == "__main__":
    main()
