#
# Paper B - B17-H1 Fix: Pruning Sanity + Fair Historical Comparison
#
# Root cause of B17-H1 NO-GO: pru_masked[protected] = 1.0 was interpreted as
# "protect" but in final_prune_fastgs, pruning_score > 0.9 means PRUNE.
# Setting score to 1.0 actively pruned the Gaussians we wanted to protect,
# causing 207k → 78k GS collapse and −5.6 dB PSNR.
#
# Fix:
#   1. Correct protection: set protected scores to 0.0 (low = don't prune)
#   2. Fair comparison: same prune schedule, same threshold, only changes
#      which Gaussians cross the threshold
#   3. Sanity logging at every pruning event
#
# No retraining from scratch, no method design. GPU: 0
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

OUT = "paper_b/b17_h1_full_training_history/fix"
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
    ema_gradn = None
    ema_radii = None
    vp_stack = scene.getTrainCameras().copy()
    vp_idx = list(range(len(vp_stack)))

    curve_rows = []
    prune_log = []

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
        if ema_gradn is None or ema_gradn.shape[0] != n:
            ema_gradn = torch.zeros(n, device="cuda")
            ema_radii = torch.zeros(n, device="cuda")
        with torch.no_grad():
            if vpt.grad is not None and vis.any():
                gn = torch.norm(vpt.grad[vis][:, :2], dim=-1)
                ema_gradn[vis] = (1 - EMA_ALPHA) * ema_gradn[vis] + EMA_ALPHA * gn
            inv = ~vis
            if inv.any():
                ema_gradn[inv] *= (1 - EMA_ALPHA)
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
                    apply_native_densify(gaussians, opt, it, radii, imp, pru, scene.cameras_extent)
                    # sync EMA after resize
                    n_now = gaussians.get_xyz.shape[0]
                    if ema_gradn.shape[0] != n_now:
                        new_g = torch.zeros(n_now, device="cuda")
                        new_r = torch.zeros(n_now, device="cuda")
                        cp_n = min(ema_gradn.shape[0], n_now)
                        new_g[:cp_n] = ema_gradn[:cp_n]
                        new_r[:cp_n] = ema_radii[:cp_n]
                        ema_gradn = new_g
                        ema_radii = new_r
                if it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            if it % 3000 == 0 and 15_000 < it < 30_000:
                my = scene.getTrainCameras().copy()
                cl = sampling_cameras(my)
                _, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt)

                n_before = gaussians.get_xyz.shape[0]
                protected_n = 0

                if use_historical:
                    # CORRECT: set protected Gaussians' score to 0.0 (low = don't prune)
                    # NOT 1.0 (which means prune in the > 0.9 threshold)
                    pru_masked = pru.clone()
                    # Protect Gaussians with high EMA gradient (top 20%)
                    grad_threshold = torch.quantile(ema_gradn, 0.8)
                    protected = ema_gradn >= grad_threshold
                    pru_masked[protected] = 0.0  # low score = don't prune
                    protected_n = int(protected.sum())
                    gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pru_masked)
                else:
                    gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pru)

                n_after = gaussians.get_xyz.shape[0]
                actual_pruned = n_before - n_after
                prune_log.append({
                    "label": label, "seed": seed, "iteration": it,
                    "n_before": n_before, "n_after": n_after,
                    "actual_pruned": actual_pruned, "protected_n": protected_n})

            gaussians.optimizer_step(it)

        if it % 3000 == 0 or it == max_iters:
            n_now = gaussians.get_xyz.shape[0]
            with torch.no_grad():
                tp, tss = [], []
                for c in test_cams[:10]:
                    im = render_fastgs(c, gaussians, pipe, bg, opt.mult)["render"]
                    gt = c.original_image.cuda()
                    ic, gc = torch.clamp(im, 0, 1), torch.clamp(gt, 0, 1)
                    tp.append(float(psnr(ic, gc).mean()))
                    tss.append(float(fast_ssim(ic.unsqueeze(0), gc.unsqueeze(0)).mean()))
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            curve_rows.append({"label": label, "seed": seed, "iteration": it,
                               "test_psnr": float(np.mean(tp)),
                               "test_ssim": float(np.mean(tss)),
                               "n_gaussians": n_now,
                               "wall_time_s": elapsed,
                               "peak_memory_mb": peak})
            print(f"  [{label} s{seed}] it={it} PSNR={np.mean(tp):.3f} "
                  f"#GS={n_now} t={elapsed:.0f}s peak={peak:.0f}MB", flush=True)

    torch.cuda.synchronize()
    total_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

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

    final = {"label": label, "seed": seed,
             "psnr": float(np.mean(tp)), "ssim": float(np.mean(tss)),
             "lpips": float(np.mean(tlp)) if tlp else None,
             "n_gaussians": int(gaussians.get_xyz.shape[0]),
             "total_time_s": round(total_time, 1),
             "peak_memory_mb": round(peak_mem, 0)}

    del gaussians, scene
    torch.cuda.empty_cache()
    return final, curve_rows, prune_log


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max_iters", type=int, default=30000)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    seeds = [int(s) for s in args.seeds.split(",")]

    all_final = []
    all_curves = []
    all_prune_logs = []

    for seed in seeds:
        for label, use_hist in (("baseline", False), ("historical", True)):
            print(f"\n[b17h1fix] {label} seed={seed}")
            final, curves, plogs = run_training(
                label, use_hist, seed, dataset, opt, pipe, bg, args.max_iters)
            final["seed"] = seed
            all_final.append(final)
            all_curves += curves
            all_prune_logs += plogs

    # ---- save ----
    os.makedirs(f"{OUT}/data", exist_ok=True)
    with open(f"{OUT}/data/final_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_final[0].keys()))
        w.writeheader(); w.writerows(all_final)
    with open(f"{OUT}/data/training_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_curves[0].keys()))
        w.writeheader(); w.writerows(all_curves)
    with open(f"{OUT}/data/prune_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_prune_logs[0].keys()))
        w.writeheader(); w.writerows(all_prune_logs)

    # ---- summary ----
    print("\n" + "=" * 60)
    print("B17-H1 Fix SUMMARY")
    print("=" * 60)
    for seed in seeds:
        for label in ("baseline", "historical"):
            r = next((r for r in all_final if r["seed"] == seed and r["label"] == label), None)
            if r:
                print(f"  seed={seed} {label:>12s}: PSNR={r['psnr']:.3f} "
                      f"#GS={r['n_gaussians']} t={r['total_time_s']:.0f}s")

    # ---- baseline vs historical #GS by event ----
    print("\n  #GS by prune event (baseline vs historical):")
    for seed in seeds:
        for it in (18000, 21000, 24000, 27000):
            b = next((r for r in all_prune_logs if r["seed"] == seed
                      and r["label"] == "baseline" and r["iteration"] == it), None)
            h = next((r for r in all_prune_logs if r["seed"] == seed
                      and r["label"] == "historical" and r["iteration"] == it), None)
            if b and h:
                print(f"    s{seed} it{it}: base_n_after={b['n_after']} "
                      f"hist_n_after={h['n_after']} | base_pruned={b['actual_pruned']} "
                      f"hist_pruned={h['actual_pruned']}")

    # aggregate
    ben = [r for r in all_final if r["label"] == "baseline"]
    his = [r for r in all_final if r["label"] == "historical"]
    print(f"\n  Aggregate:")
    print(f"  Baseline:  PSNR={np.mean([r['psnr'] for r in ben]):.3f}±{np.std([r['psnr'] for r in ben]):.3f} "
          f"#GS={np.mean([r['n_gaussians'] for r in ben]):.0f}")
    print(f"  Historical: PSNR={np.mean([r['psnr'] for r in his]):.3f}±{np.std([r['psnr'] for r in his]):.3f} "
          f"#GS={np.mean([r['n_gaussians'] for r in his]):.0f}")
    print(f"  ΔPSNR = {np.mean([r['psnr'] for r in his]) - np.mean([r['psnr'] for r in ben]):+.3f}")


if __name__ == "__main__":
    main()
