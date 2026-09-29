#
# Paper B - B17-H1: Full-Training Historical Evidence Validation
#
# Train baseline FastGS vs FastGS + Historical Evidence in the pruning step.
# The only change: final_prune_fastgs uses EMA accumulated evidence (from
# normal training iterations) instead of / in addition to current-only
# pruning_score. All other FastGS logic is untouched.
#
# Historical evidence:
#   ema_gradn  = EMA of per-Gaussian gradient norm (from vpt.grad, free)
#   ema_radii  = EMA of screen radii (from render_fastgs, free)
#   These are updated every iteration with O(N) cost, no extra render.
#
# Pruning modification: in final_prune_fastgs (15k-30k events), a Gaussian is
# additionally preserved (not pruned) if its historical evidence is high,
# even when current pruning_score is low. This tests whether historical
# information prevents false pruning of temporarily-inactive Gaussians.
#
# No local ROI, no recovery, no Split-N. Full training to 30k.
#

import os, sys, random, time, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.image_utils import psnr
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.general_utils import inverse_sigmoid
from arguments import ModelParams, PipelineParams, OptimizationParams
from diagnostics.common import install_c_proxy, seed_all, native_train_one_iter

try:
    from lpipsPyTorch import lpips as lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False

OUT = "paper_b/b17_h1_full_training_history"
EMA_ALPHA = 0.05


@torch.no_grad()
def apply_native_densify(g, opt, it, radii, imp, pru, extent):
    grad_vars = g.xyz_gradient_accum / g.denom
    grad_vars[grad_vars.isnan()] = 0.0
    grads_abs = g.xyz_gradient_accum_abs / g.denom
    grads_abs[grads_abs.isnan()] = 0.0
    grad_qual = torch.norm(grad_vars, dim=-1) >= opt.grad_thresh
    grad_abs_qual = torch.norm(grads_abs, dim=-1) >= opt.grad_abs_thresh
    max_scale = g.get_scaling.max(dim=1).values
    metric_mask = imp > 5
    clone_set = metric_mask & (max_scale <= opt.dense * extent) & grad_qual
    split_set = metric_mask & (max_scale > opt.dense * extent) & grad_abs_qual

    g.tmp_radii = radii
    if clone_set.any():
        g.densify_and_clone_fastgs(clone_set, torch.ones_like(clone_set))
    if split_set.any():
        g.densify_and_split_fastgs(split_set, torch.ones_like(split_set), N=2)
    g.tmp_radii = None

    prune_mask = (g.get_opacity < 0.005).squeeze()
    st = 20 if it > opt.opacity_reset_interval else None
    if st:
        prune_mask = torch.logical_or(torch.logical_or(
            prune_mask, g.max_radii2D > st),
            g.get_scaling.max(dim=1).values > 0.1 * extent)
    scores = 1 - pru
    tr = int(torch.sum(prune_mask)); rb = int(0.5 * tr)
    if rb:
        n = g.get_xyz.shape[0]
        padded = torch.zeros((n), dtype=torch.float32, device=scores.device)
        padded[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
        sel = torch.zeros_like(padded, dtype=bool)
        sel[torch.multinomial(padded, rb, replacement=False)] = True
        g.prune_points(torch.logical_and(prune_mask, sel))
    g._opacity = g.replace_tensor_to_optimizer(
        inverse_sigmoid(torch.min(g.get_opacity, torch.ones_like(g.get_opacity) * 0.8)),
        "opacity")["opacity"]
    torch.cuda.empty_cache()


def run_training(label, use_historical, seed, dataset, opt, pipe, bg, max_iters):
    install_c_proxy()
    seed_all(seed)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    test_cams = scene.getTestCameras()[:20] if scene.getTestCameras() else scene.getTrainCameras()[:10]

    # historical EMA tensors
    ema_gradn = None
    ema_radii = None
    vp_stack = scene.getTrainCameras().copy()
    vp_idx = list(range(len(vp_stack)))

    curve_rows = []
    for it in range(1, max_iters + 1):
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

        n = gaussians.get_xyz.shape[0]
        # update historical EMA (O(N), free from training grads)
        if ema_gradn is None or ema_gradn.shape[0] != n:
            ema_gradn = torch.zeros(n, device="cuda")
            ema_radii = torch.zeros(n, device="cuda")
        with torch.no_grad():
            if vpt.grad is not None:
                gn = torch.norm(vpt.grad[vis][:, :2], dim=-1) if vis.any() else None
                if gn is not None and vis.any():
                    ema_gradn[vis] = (1 - EMA_ALPHA) * ema_gradn[vis] + EMA_ALPHA * gn
                inv = ~vis
                if inv.any():
                    ema_gradn[inv] *= (1 - EMA_ALPHA)  # decay for invisible
            ema_radii[vis] = (1 - EMA_ALPHA) * ema_radii[vis] + EMA_ALPHA * radii[vis].float()
            ema_radii[~vis] *= (1 - EMA_ALPHA)

        with torch.no_grad():
            if it < opt.densify_until_iter:
                gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis])
                gaussians.add_densification_stats(vpt, vis)
                if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                    my = scene.getTrainCameras().copy()
                    cl = sampling_cameras(my)
                    imp, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt, DENSIFY=True)

                    if use_historical:
                        # modified final prune: use historical EMA to override
                        # current-only pruning_score for well-established Gaussians
                        # A Gaussian with high historical gradient activity is likely
                        # important even if current score is low
                        pass  # native densify unchanged; historical used only in final prune

                    apply_native_densify(gaussians, opt, it, radii, imp, pru, scene.cameras_extent)

                    # after resize, update EMA tensors if needed
                    n_now = gaussians.get_xyz.shape[0]
                    if ema_gradn.shape[0] != n_now:
                        # resize by copying existing + zeros for new
                        old_n = ema_gradn.shape[0]
                        new_g = torch.zeros(n_now, device="cuda")
                        new_r = torch.zeros(n_now, device="cuda")
                        new_g[:old_n] = ema_gradn[:old_n]
                        new_r[:old_n] = ema_radii[:old_n]
                        ema_gradn = new_g
                        ema_radii = new_r

                if it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            if it % 3000 == 0 and 15_000 < it < 30_000:
                my = scene.getTrainCameras().copy()
                cl = sampling_cameras(my)
                _, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt)

                if use_historical:
                    # B17-H1: use historical EMA to prevent false pruning
                    # Gaussians with high historical gradient activity are likely
                    # important even if current pruning_score is low
                    # We protect the top 20% by ema_gradn from being pruned
                    n_now = gaussians.get_xyz.shape[0]
                    if ema_gradn.shape[0] == n_now:
                        grad_threshold = torch.quantile(ema_gradn, 0.8)
                        protected = ema_gradn >= grad_threshold
                        # remove protected Gaussians from pruning set
                        pru_masked = pru.clone()
                        pru_masked[protected] = 1.0  # high score = don't prune
                        gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pru_masked)
                    else:
                        gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pru)
                else:
                    gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pru)

            gaussians.optimizer_step(it)

        if it % 500 == 0 or it == max_iters:
            with torch.no_grad():
                tp, tss = [], []
                for c in test_cams[:10]:
                    im = render_fastgs(c, gaussians, pipe, bg, opt.mult)["render"]
                    gt = c.original_image.cuda()
                    ic, gc = torch.clamp(im, 0, 1), torch.clamp(gt, 0, 1)
                    tp.append(float(psnr(ic, gc).mean()))
                    tss.append(float(fast_ssim(ic.unsqueeze(0), gc.unsqueeze(0)).mean()))
                lp_ = None
                if LPIPS_OK and it % 3000 == 0:
                    lps = []
                    for c in test_cams[:10]:
                        im = render_fastgs(c, gaussians, pipe, bg, opt.mult)["render"]
                        gt = c.original_image.cuda()
                        ic, gc = torch.clamp(im, 0, 1), torch.clamp(gt, 0, 1)
                        try:
                            lps.append(float(lpips_fn(ic, gc, net_type="vgg").mean()))
                        except:
                            pass
                    lp_ = float(np.mean(lps)) if lps else None
                curve_rows.append({"label": label, "seed": seed, "iteration": it,
                                   "test_psnr": float(np.mean(tp)),
                                   "test_ssim": float(np.mean(tss)),
                                   "test_lpips": lp_,
                                   "n_gaussians": gaussians.get_xyz.shape[0],
                                   "wall_time_s": time.time() - t0})
        if it % 3000 == 0:
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  [{label} s{seed}] it={it} PSNR={curve_rows[-1]['test_psnr']:.3f} "
                  f"#GS={gaussians.get_xyz.shape[0]} t={elapsed:.0f}s peak={peak:.0f}MB", flush=True)

    torch.cuda.synchronize()
    total_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    # final eval
    with torch.no_grad():
        tp, tss, tlp = [], [], []
        for c in test_cams[:10]:
            im = render_fastgs(c, gaussians, pipe, bg, opt.mult)["render"]
            gt = c.original_image.cuda()
            ic, gc = torch.clamp(im, 0, 1), torch.clamp(gt, 0, 1)
            tp.append(float(psnr(ic, gc).mean()))
            tss.append(float(fast_ssim(ic.unsqueeze(0), gc.unsqueeze(0)).mean()))
            if LPIPS_OK:
                try:
                    tlp.append(float(lpips_fn(ic, gc, net_type="vgg").mean()))
                except:
                    pass
    final = {"label": label, "seed": seed, "psnr": float(np.mean(tp)),
             "ssim": float(np.mean(tss)),
             "lpips": float(np.mean(tlp)) if tlp else None,
             "n_gaussians": int(gaussians.get_xyz.shape[0]),
             "total_time_s": round(total_time, 1),
             "peak_memory_mb": round(peak_mem, 0)}

    del gaussians, scene
    torch.cuda.empty_cache()
    return final, curve_rows


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--scenes", type=str, required=True,
                        help="comma-separated scene paths")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max_iters", type=int, default=30000)
    args = parser.parse_args()
    dataset_base, opt_base, pipe_base = lp.extract(args), op.extract(args), pp.extract(args)
    bg = torch.tensor([1, 1, 1] if dataset_base.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    seeds = [int(s) for s in args.seeds.split(",")]
    scenes = args.scenes.split(",")
    all_final = []
    all_curves = []

    for scene_path in scenes:
        dataset_base.source_path = scene_path
        print(f"\n{'='*60}\nScene: {scene_path}\n{'='*60}")
        for seed in seeds:
            for label, use_hist in (("baseline", False), ("historical", True)):
                print(f"\n[b17h1] {label} seed={seed} scene={os.path.basename(scene_path)}")
                final, curves = run_training(
                    label, use_hist, seed, dataset_base, opt_base, pipe_base, bg, args.max_iters)
                final["scene"] = os.path.basename(scene_path)
                all_final.append(final)
                for c in curves:
                    c["scene"] = os.path.basename(scene_path)
                all_curves += curves
                print(f"  -> PSNR={final['psnr']:.3f} #GS={final['n_gaussians']} "
                      f"time={final['total_time_s']:.0f}s")

    # save
    os.makedirs(f"{OUT}/data", exist_ok=True)
    with open(f"{OUT}/data/final_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_final[0].keys()))
        w.writeheader(); w.writerows(all_final)
    with open(f"{OUT}/data/training_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_curves[0].keys()))
        w.writeheader(); w.writerows(all_curves)
    with open(f"{OUT}/data/overhead_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_final[0].keys()))
        w.writeheader(); w.writerows(all_final)

    # summary
    print("\n" + "=" * 60)
    print("B17-H1 SUMMARY")
    print("=" * 60)
    for scene_path in scenes:
        sn = os.path.basename(scene_path)
        print(f"\n{sn}:")
        for label in ("baseline", "historical"):
            sub = [r for r in all_final if r["scene"] == sn and r["label"] == label]
            if sub:
                print(f"  {label:>12s}: PSNR={np.mean([r['psnr'] for r in sub]):.3f}±"
                      f"{np.std([r['psnr'] for r in sub]):.3f} "
                      f"#GS={np.mean([r['n_gaussians'] for r in sub]):.0f} "
                      f"time={np.mean([r['total_time_s'] for r in sub]):.0f}s")


if __name__ == "__main__":
    main()
