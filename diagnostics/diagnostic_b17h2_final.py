#
# Paper B - B17-H2 Final: Strict Historical Ranking Validation
#
# Trains Baseline and Historical from the same initialization, same seed.
# At every native final_prune event:
#   Baseline: prune where S_t > threshold (native semantics)
#   Historical: rank by H_t (EMA of S_t), prune same count
#
# Fairness: same initialization, same camera schedule, same optimizer,
# same target prune count. Only ranking source changes.
#
# S_t = compute_gaussian_score_fastgs(DENSIFY=False) output
# HIGH S_t = more redundant/error-attributed = FastGS prunes
# LOW S_t = less redundant = keep
#
# Historical EMA: H_t = 0.8*H_{t-1} + 0.2*S_t
# New Gaussians: H = current S_t (lifecycle sync)
#
# GPU: 3
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

OUT = "paper_b/b17_h2_historical_ranking/final"


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


@torch.no_grad()
def eval_full(g, test_cams, pipe, bg, mult):
    ps_, ss_, lp_ = [], [], []
    for cam in test_cams:
        img = render_fastgs(cam, g, pipe, bg, mult)["render"]
        gt = cam.original_image.cuda()
        ic, gc = torch.clamp(img, 0, 1), torch.clamp(gt, 0, 1)
        ps_.append(float(psnr(ic, gc).mean()))
        ss_.append(float(fast_ssim(ic.unsqueeze(0), gc.unsqueeze(0)).mean()))
        if LPIPS_OK:
            try:
                lp_.append(float(lpips_fn(ic, gc, net_type="vgg").mean()))
            except:
                lp_.append(float("nan"))
    return {"psnr": float(np.mean(ps_)), "ssim": float(np.mean(ss_)),
            "lpips": float(np.nanmean(lp_)) if lp_ else None}


def run_training(condition, seed, dataset, opt, pipe, bg, max_iters):
    install_c_proxy()
    seed_all(seed)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    test_cams = scene.getTestCameras()[:20] if scene.getTestCameras() else scene.getTrainCameras()[:10]
    vp_stack = scene.getTrainCameras().copy()
    vp_idx = list(range(len(vp_stack)))

    # Historical score tensor
    H_t = None  # (N,) EMA of native pruning score

    curve_rows = []
    prune_log = []
    n_final = 0

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

        with torch.no_grad():
            if it < opt.densify_until_iter:
                gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis])
                gaussians.add_densification_stats(vpt, vis)
                if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                    my = scene.getTrainCameras().copy()
                    cl = sampling_cameras(my)
                    imp, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt, DENSIFY=True)
                    apply_native_densify(gaussians, opt, it, radii, imp, pru, scene.cameras_extent)
                    # sync H_t
                    n_now = gaussians.get_xyz.shape[0]
                    if H_t is None:
                        H_t = torch.zeros(n_now, device="cuda")
                    elif H_t.shape[0] != n_now:
                        new_H = torch.zeros(n_now, device="cuda")
                        cp = min(H_t.shape[0], n_now)
                        new_H[:cp] = H_t[:cp]
                        H_t = new_H

                if it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            if it % 3000 == 0 and 15_000 < it < 30_000:
                my = scene.getTrainCameras().copy()
                cl = sampling_cameras(my)
                _, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt)
                S_t = pru.clone()

                # Update H_t
                if H_t is None:
                    H_t = S_t.clone()
                elif H_t.shape[0] != S_t.shape[0]:
                    new_H = torch.zeros(S_t.shape[0], device="cuda")
                    cp = min(H_t.shape[0], S_t.shape[0])
                    new_H[:cp] = H_t[:cp]
                    H_t = new_H
                H_t = 0.8 * H_t + 0.2 * S_t

                n_before = gaussians.get_xyz.shape[0]
                if condition == "baseline":
                    # Native: score > 0.9 → prune
                    scores_mask = S_t > 0.9
                    opacity_mask = gaussians.get_opacity < 0.1
                    final_prune = torch.logical_or(scores_mask,
                                                   opacity_mask.squeeze(-1) if opacity_mask.dim() > 1 else opacity_mask)
                    gaussians.prune_points(final_prune)
                else:
                    # Historical: rank by H_t, prune same count as baseline
                    # K = count of baseline's native threshold
                    scores_mask_b = S_t > 0.9
                    opacity_mask_b = gaussians.get_opacity < 0.1
                    baseline_prune_b = torch.logical_or(scores_mask_b,
                                                         opacity_mask_b.squeeze(-1) if opacity_mask_b.dim() > 1 else opacity_mask_b)
                    K = int(baseline_prune_b.sum())

                    # Rank by H_t, prune top K
                    hist_order = torch.argsort(H_t, descending=True)
                    hist_set = hist_order[:K]
                    hist_prune = torch.zeros(n_before, dtype=torch.bool, device="cuda")
                    hist_prune[hist_set] = True
                    opacity_prune = (gaussians.get_opacity < 0.1).squeeze(-1)
                    final_prune = torch.logical_or(hist_prune, opacity_prune)
                    gaussians.prune_points(final_prune)

                n_after = gaussians.get_xyz.shape[0]
                actual_pruned = n_before - n_after
                prune_log.append({
                    "condition": condition, "seed": seed, "iteration": it,
                    "n_before": n_before, "n_after": n_after,
                    "actual_pruned": actual_pruned})
                print(f"    [{condition} s{seed}] it={it} pruned={actual_pruned} "
                      f"#GS={n_after}", flush=True)

            gaussians.optimizer_step(it)

        if it % 3000 == 0 or it == max_iters:
            n_now = gaussians.get_xyz.shape[0]
            with torch.no_grad():
                tp, tss, tlp = [], [], []
                for c in test_cams[:10]:
                    im = render_fastgs(c, gaussians, pipe, bg, opt.mult)["render"]
                    gt = c.original_image.cuda()
                    ic, gc = torch.clamp(im, 0, 1), torch.clamp(gt, 0, 1)
                    tp.append(float(psnr(ic, gc).mean()))
                    tss.append(float(fast_ssim(ic.unsqueeze(0), gc.unsqueeze(0)).mean()))
                    if LPIPS_OK and it % 6000 == 0:
                        try:
                            tlp.append(float(lpips_fn(ic, gc, net_type="vgg").mean()))
                        except:
                            pass
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            curve_rows.append({"condition": condition, "seed": seed, "iteration": it,
                               "test_psnr": float(np.mean(tp)),
                               "test_ssim": float(np.mean(tss)),
                               "test_lpips": float(np.mean(tlp)) if tlp else None,
                               "n_gaussians": n_now,
                               "wall_time_s": elapsed,
                               "peak_memory_mb": peak})
            print(f"  [{condition} s{seed}] it={it} PSNR={np.mean(tp):.3f} "
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
                    tlp.append(float("nan"))

    final = {"condition": condition, "seed": seed,
             "psnr": float(np.mean(tp)), "ssim": float(np.mean(tss)),
             "lpips": float(np.mean(tlp)) if tlp else None,
             "n_gaussians": n_final,
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
    install_c_proxy()
    seeds = [int(s) for s in args.seeds.split(",")]

    all_final = []
    all_curves = []
    all_prune_logs = []

    for seed in seeds:
        for condition in ("baseline", "historical"):
            print(f"\n[b17h2final] {condition} seed={seed}")
            final, curves, plogs = run_training(
                condition, seed, dataset, opt, pipe, bg, args.max_iters)
            final["seed"] = seed
            all_final.append(final)
            all_curves += curves
            all_prune_logs += plogs

    os.makedirs(f"{OUT}/data", exist_ok=True)
    with open(f"{OUT}/data/final_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_final[0].keys()))
        w.writeheader(); w.writerows(all_final)
    with open(f"{OUT}/data/training_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_curves[0].keys()))
        w.writeheader(); w.writerows(all_curves)
    with open(f"{OUT}/data/prune_event_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_prune_logs[0].keys()))
        w.writeheader(); w.writerows(all_prune_logs)

    print(f"\n[b17h2final] saved {len(all_final)} finals, {len(all_prune_logs)} prune logs")


if __name__ == "__main__":
    main()
