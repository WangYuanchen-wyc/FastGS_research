#
# Paper B - B17-H0: Historical Evidence Feasibility Check
#
# Question: can FastGS accumulate per-Gaussian historical evidence at
# negligible cost during normal training, without extra rendering?
#
# Test: run original FastGS vs FastGS + historical evidence accumulation
# (EMA of per-view metric counts). Measure overhead: time, memory, extra VRAM.
#
# The historical evidence is zero-cost to collect because FastGS already
# computes per-view metric counts during normal VCD densification.
# Between densification events we simply EMA-update a running per-Gaussian
# tensor with the most recent available metric count snapshot.
#

import os, sys, random, time, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.general_utils import inverse_sigmoid
from arguments import ModelParams, PipelineParams, OptimizationParams
from diagnostics.common import install_c_proxy, seed_all, native_train_one_iter

OUT = "paper_b/b17_h0_history_feasibility"


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


def run_training(label, record_history, seed, dataset, opt, pipe, bg, max_iters):
    install_c_proxy()
    seed_all(seed)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # historical evidence tensors (EMA)
    ema_metric = None   # (N,) EMA of per-view metric count
    ema_gradn = None    # (N,) EMA of gradient norm
    ema_radii = None    # (N,) EMA of screen radii
    ema_alpha = 0.1     # EMA decay factor
    hist_update_time = 0.0
    n_hist_updates = 0

    vp_stack = scene.getTrainCameras().copy()
    vp_idx = list(range(len(vp_stack)))
    t0 = time.time()
    iter_times = []

    for it in range(1, max_iters + 1):
        torch.cuda.synchronize()
        ti = time.time()
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
                if it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            if it % 3000 == 0 and 15_000 < it < 30_000:
                my = scene.getTrainCameras().copy()
                cl = sampling_cameras(my)
                _, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt)
                gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pru)
            gaussians.optimizer_step(it)

        # ---- historical evidence recording (EMA, minimal overhead) ----
        if record_history:
            torch.cuda.synchronize()
            th0 = time.time()
            with torch.no_grad():
                gn = torch.norm(vpt.grad[vis][:, :2], dim=-1) if vpt.grad is not None else None
                n = gaussians.get_xyz.shape[0]
                if ema_metric is None or ema_metric.shape[0] != n:
                    ema_metric = torch.zeros(n, device="cuda")
                    ema_gradn = torch.zeros(n, device="cuda")
                    ema_radii = torch.zeros(n, device="cuda")
                if gn is not None and gn.shape[0] == n:
                    # Only update visible Gaussians; others keep old value
                    vmask = vis
                    if ema_metric[vmask].shape[0] > 0:
                        ema_metric[vmask] = (1 - ema_alpha) * ema_metric[vmask] + \
                                            ema_alpha * 1.0  # binary signal: was rendered
                    if gn.shape[0] == n:
                        ema_gradn[vmask] = (1 - ema_alpha) * ema_gradn[vmask] + \
                                           ema_alpha * gn[:vmask.sum()].mean().item() if gn.numel() else 0
                    ema_radii[vmask] = (1 - ema_alpha) * ema_radii[vmask] + \
                                       ema_alpha * radii[vmask].float()
            torch.cuda.synchronize()
            hist_update_time += time.time() - th0
            n_hist_updates += 1

        torch.cuda.synchronize()
        dt = time.time() - ti
        iter_times.append(dt)

        if it % 500 == 0 or it == max_iters:
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  [{label}] it={it} t={elapsed:.1f}s peak={peak:.0f}MB")

    torch.cuda.synchronize()
    total_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    del gaussians, scene
    torch.cuda.empty_cache()

    return {
        "label": label, "seed": seed,
        "total_time_s": round(total_time, 2),
        "mean_iter_time_ms": round(np.mean(iter_times[100:]) * 1000, 3),
        "std_iter_time_ms": round(np.std(iter_times[100:]) * 1000, 3),
        "peak_memory_mb": round(peak_mem, 1),
        "hist_update_time_s": round(hist_update_time, 3),
        "n_hist_updates": n_hist_updates,
        "hist_overhead_pct": round(hist_update_time / total_time * 100, 3) if total_time > 0 else 0,
    }


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--max_iters", type=int, default=6000)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    seeds = [int(s) for s in args.seeds.split(",")]

    results = []
    for seed in seeds:
        for label, rec in (("original", False), ("with_history", True)):
            print(f"\n{'='*60}")
            print(f"Running {label} seed={seed} for {args.max_iters} iters...")
            print(f"{'='*60}")
            r = run_training(label, rec, seed, dataset, opt, pipe, bg, args.max_iters)
            results.append(r)
            print(f"  -> {r}")

    # summary
    os.makedirs(OUT + "/data", exist_ok=True)
    with open(f"{OUT}/data/overhead_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    orig = [r for r in results if r["label"] == "original"]
    hist = [r for r in results if r["label"] == "with_history"]

    stats = {
        "max_iters": args.max_iters,
        "n_seeds": len(seeds),
        "original_mean_time_s": np.mean([r["total_time_s"] for r in orig]),
        "original_std_time_s": np.std([r["total_time_s"] for r in orig]),
        "history_mean_time_s": np.mean([r["total_time_s"] for r in hist]),
        "history_std_time_s": np.std([r["total_time_s"] for r in hist]),
        "time_overhead_pct": np.mean([(h["total_time_s"] - o["total_time_s"]) / o["total_time_s"] * 100
                                      for o, h in zip(orig, hist)]),
        "original_mean_peak_mb": np.mean([r["peak_memory_mb"] for r in orig]),
        "history_mean_peak_mb": np.mean([r["peak_memory_mb"] for r in hist]),
        "mem_overhead_mb": np.mean([h["peak_memory_mb"] - o["peak_memory_mb"]
                                    for o, h in zip(orig, hist)]),
        "mean_hist_update_ms": np.mean([r["hist_update_time_s"] for r in hist]) * 1000,
    }

    with open(f"{OUT}/data/b17h0_stats.txt", "w") as f:
        f.write(f"B17-H0 Historical Evidence Feasibility Check\n")
        f.write(f"{'='*60}\n")
        f.write(f"max_iters: {args.max_iters}\n")
        f.write(f"seeds: {seeds}\n\n")
        for r in results:
            f.write(json.dumps(r) + "\n")
        f.write(f"\nSummary:\n")
        f.write(f"  Original mean time: {stats['original_mean_time_s']:.2f}s\n")
        f.write(f"  History mean time:  {stats['history_mean_time_s']:.2f}s\n")
        f.write(f"  Time overhead: {stats['time_overhead_pct']:.3f}%\n")
        f.write(f"  Original peak mem: {stats['original_mean_peak_mb']:.0f}MB\n")
        f.write(f"  History peak mem:  {stats['history_mean_peak_mb']:.0f}MB\n")
        f.write(f"  Mem overhead: {stats['mem_overhead_mb']:.1f}MB\n")
    with open(f"{OUT}/data/overhead_results.csv", "a") as f:
        pass
    print(f"\n[b17h0] Summary:")
    print(f"  Original: {stats['original_mean_time_s']:.2f}s, "
          f"{stats['original_mean_peak_mb']:.0f}MB")
    print(f"  History:  {stats['history_mean_time_s']:.2f}s, "
          f"{stats['history_mean_peak_mb']:.0f}MB")
    print(f"  Time overhead: {stats['time_overhead_pct']:.3f}%")
    print(f"  Mem overhead: {stats['mem_overhead_mb']:.1f}MB")
    print(f"  -> saved to {OUT}/data/")


if __name__ == "__main__":
    main()
