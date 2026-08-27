#
# Paper B - B1: Keep / Clone / Split minimal controlled diagnostic
#
# For ONE real FastGS densification candidate (selected by FastGS's native
# VCD metric_mask + gradient + scale criteria), run three branches from the
# exact same pre-densification checkpoint:
#   keep  : do nothing
#   clone : force FastGS native clone on that single parent
#   split : force FastGS native split  on that single parent
# then continue training 100 steps with identical camera sequence / RNG /
# optimizer state, with densification, pruning and opacity reset disabled.
#
# This script DOES NOT modify any FastGS training logic. It only reuses:
#   gaussian_renderer.render_fastgs
#   utils.fast_utils.sampling_cameras / compute_gaussian_score_fastgs
#   GaussianModel.capture / restore / densify_and_clone_fastgs /
#                          densify_and_split_fastgs / optimizer_step
#
# num_rendered (Gaussian-tile pair count) is captured by a zero-invasion
# proxy of the package-level `_C` module handle (no CUDA change).
#

import os, sys, json, random, time
import numpy as np
import torch
from argparse import ArgumentParser

from gaussian_renderer import render_fastgs
from scene import Scene, GaussianModel
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from utils.image_utils import psnr
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from lpipsPyTorch import lpips as lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False


# ---------------------------------------------------------------- utils ---

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_tree(obj):
    """Deep-copy a capture() tuple (params + optimizer state dicts).

    Adam mutates params / exp_avg in place, so every branch must restore from
    its own private copy. nn.Parameter type is preserved, otherwise restored
    tensors lose requires_grad and training silently freezes.
    """
    if torch.is_tensor(obj):
        return torch.nn.Parameter(obj.detach().clone(), requires_grad=obj.requires_grad) \
            if isinstance(obj, torch.nn.Parameter) else obj.detach().clone()
    if isinstance(obj, dict):
        return {k: clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_tree(v) for v in obj)
    return obj


class CProxy:
    """Proxy for diff_gaussian_rasterization_fastgs._C that records
    num_rendered (= # of (gaussian, tile) key/value pairs) and num_buckets
    returned by each forward rasterization. No CUDA modification."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "last_num_rendered", None)
        object.__setattr__(self, "last_num_buckets", None)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def rasterize_gaussians(self, *args, **kwargs):
        real = object.__getattribute__(self, "_real")
        out = real.rasterize_gaussians(*args, **kwargs)
        object.__setattr__(self, "last_num_rendered", int(out[0]))
        object.__setattr__(self, "last_num_buckets", int(out[1]))
        return out


def install_c_proxy():
    import diff_gaussian_rasterization_fastgs as pkg
    real = pkg._C
    if not isinstance(real, CProxy):
        pkg._C = CProxy(real)  # module-global lookup inside _RasterizeGaussians.forward
    return pkg._C


# ------------------------------------------------------------ evaluation ---

@torch.no_grad()
def evaluate(gaussians, probe_views, pipe, bg, mult):
    """Fixed probe views -> mean loss / PSNR / SSIM / LPIPS (+ pairs)."""
    psnrs, ssims, lpipss, losses, pairs = [], [], [], [], []
    proxy = install_c_proxy()
    for cam in probe_views:
        out = render_fastgs(cam, gaussians, pipe, bg, mult)
        image = torch.clamp(out["render"], 0.0, 1.0)
        gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
        l1 = l1_loss(image, gt).mean().double()
        ss = fast_ssim(image.unsqueeze(0), gt.unsqueeze(0)).mean().double()
        psnrs.append(psnr(image, gt).mean().double())
        ssims.append(ss)
        losses.append(0.8 * l1 + 0.2 * (1.0 - ss))
        if LPIPS_OK:
            try:
                lpipss.append(lpips_fn(image, gt, net_type='vgg').mean().double())
            except Exception:
                lpipss.append(torch.tensor(float('nan')).double())
        pairs.append(proxy.last_num_rendered)
    res = {
        "loss": float(torch.stack([l.reshape(()) for l in losses]).mean()),
        "psnr": float(torch.stack([p.reshape(()) for p in psnrs]).mean()),
        "ssim": float(torch.stack([s.reshape(()) for s in ssims]).mean()),
        "lpips": float(torch.stack([l.reshape(()) for l in lpipss]).mean()) if lpipss else None,
        "num_gaussians": int(gaussians.get_xyz.shape[0]),
        "tile_pairs_mean": float(np.mean(pairs)),
    }
    return res


@torch.no_grad()
def measure_render_perf(gaussians, probe_views, pipe, bg, mult, warmup=3, reps=10):
    """CUDA-event render latency per probe view (ms) + FPS."""
    for _ in range(warmup):
        for cam in probe_views:
            render_fastgs(cam, gaussians, pipe, bg, mult)
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    lats = []
    for _ in range(reps):
        for cam in probe_views:
            ev0.record()
            render_fastgs(cam, gaussians, pipe, bg, mult)
            ev1.record()
            torch.cuda.synchronize()
            lats.append(ev0.elapsed_time(ev1))
    lats = torch.tensor(lats)
    return float(lats.mean()), float(lats.std())


# ------------------------------------------------------------- training ---

def native_train_one_iter(iteration, viewpoint_cam, gaussians, pipe, bg, opt):
    """One native FastGS training iteration (render -> loss -> backward)."""
    gaussians.update_learning_rate(iteration)
    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()
    render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
    image = render_pkg["render"]
    viewspace_point_tensor = render_pkg["viewspace_points"]
    visibility_filter = render_pkg["visibility_filter"]
    radii = render_pkg["radii"]
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
    loss.backward()
    return loss, viewspace_point_tensor, visibility_filter, radii


def main():
    parser = ArgumentParser("Paper B B1: Keep/Clone/Split diagnostic")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--diag_iter", type=int, default=1000,
                        help="densification event iteration used as diagnostic point")
    parser.add_argument("--diag_steps", type=int, default=100)
    parser.add_argument("--probe_count", type=int, default=5)
    parser.add_argument("--warmup_seed", type=int, default=0)
    parser.add_argument("--action_seed", type=int, default=1234)
    parser.add_argument("--train_seed", type=int, default=1234)
    parser.add_argument("--camseq_seed", type=int, default=2024)
    parser.add_argument("--prefer_action", type=str, default="split",
                        choices=["split", "clone"])
    parser.add_argument("--out_json", type=str, default="project_md/b1_results.json")
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    assert opt.optimizer_type == "default", "diagnostic assumes default optimizer (dual Adam)"
    is_event = (args.diag_iter > opt.densify_from_iter
                and args.diag_iter % opt.densification_interval == 0
                and args.diag_iter < opt.densify_until_iter)
    assert is_event, (
        f"diag_iter={args.diag_iter} is not a native densification event "
        f"(from={opt.densify_from_iter}, interval={opt.densification_interval}, "
        f"until={opt.densify_until_iter})")

    install_c_proxy()
    seed_all(args.warmup_seed)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ---------------- Phase 1: native warmup until diag_iter -------------
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    diag_ctx = None

    for iteration in range(1, args.diag_iter + 1):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = random.randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        loss, viewspace_point_tensor, visibility_filter, radii = \
            native_train_one_iter(iteration, viewpoint_cam, gaussians, pipe, bg, opt)

        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration == args.diag_iter:
                    # ---- DIAGNOSTIC HOOK: intercept before native densify ----
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)
                    importance_score, pruning_score = compute_gaussian_score_fastgs(
                        camlist, gaussians, pipe, bg, opt, DENSIFY=True)

                    # replicate native selection math (gaussian_model.py:477-494)
                    grad_vars = gaussians.xyz_gradient_accum / gaussians.denom
                    grad_vars[grad_vars.isnan()] = 0.0
                    grads_abs = gaussians.xyz_gradient_accum_abs / gaussians.denom
                    grads_abs[grads_abs.isnan()] = 0.0
                    grad_qualifiers = torch.norm(grad_vars, dim=-1) >= opt.grad_thresh
                    grad_qualifiers_abs = torch.norm(grads_abs, dim=-1) >= opt.grad_abs_thresh
                    max_scale = gaussians.get_scaling.max(dim=1).values
                    clone_qualifiers = max_scale <= opt.dense * scene.cameras_extent
                    split_qualifiers = max_scale > opt.dense * scene.cameras_extent
                    all_clones = clone_qualifiers & grad_qualifiers
                    all_splits = split_qualifiers & grad_qualifiers_abs
                    metric_mask = importance_score > 5
                    clone_set = metric_mask & all_clones
                    split_set = metric_mask & all_splits

                    n_clone, n_split = int(clone_set.sum()), int(split_set.sum())
                    order = [args.prefer_action, "clone" if args.prefer_action == "split" else "split"]
                    cand_set, native_action = None, None
                    for act in order:
                        s = split_set if act == "split" else clone_set
                        if int(s.sum()) > 0:
                            cand_set, native_action = s, act
                            break
                    assert cand_set is not None, "no native densification candidate at diag_iter"

                    imp_in_set = torch.where(cand_set, importance_score.to(torch.float32),
                                             torch.tensor(-1.0, device="cuda"))
                    parent_idx = int(torch.argmax(imp_in_set))

                    diag_ctx = {
                        "iteration": iteration,
                        "parent_index": parent_idx,
                        "native_action": native_action,
                        "importance_score": int(importance_score[parent_idx]),
                        "grad": float(torch.norm(grad_vars[parent_idx], dim=-1)),
                        "grad_abs": float(torch.norm(grads_abs[parent_idx], dim=-1)),
                        "scale": [float(v) for v in gaussians.get_scaling[parent_idx]],
                        "scale_max": float(max_scale[parent_idx]),
                        "opacity": float(gaussians.get_opacity[parent_idx]),
                        "n_clone_candidates": n_clone,
                        "n_split_candidates": n_split,
                        "n_gaussians_before": int(gaussians.get_xyz.shape[0]),
                        "pruning_score_parent": float(pruning_score[parent_idx]),
                    }
                    radii_master = radii.clone()
                    master_ckpt = clone_tree(gaussians.capture(opt.optimizer_type))
                    break  # skip native densification at the hook iteration

                elif iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)
                    importance_score, pruning_score = compute_gaussian_score_fastgs(
                        camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                    gaussians.densify_and_prune_fastgs(
                        max_screen_size=size_threshold, min_opacity=0.005,
                        extent=scene.cameras_extent, radii=radii, args=opt,
                        importance_score=importance_score, pruning_score=pruning_score)

                if iteration % opt.opacity_reset_interval == 0 or \
                        (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 3000 == 0 and 15_000 < iteration < 30_000:
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack)
                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
                gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)

            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)

    assert diag_ctx is not None, "diagnostic hook never fired"
    print("[diag] candidate context:")
    for k, v in diag_ctx.items():
        print(f"    {k}: {v}")

    # fixed camera sequence + fixed probe views (identical across branches)
    train_cams = scene.getTrainCameras()
    seq_rng = random.Random(args.camseq_seed)
    cam_seq = [seq_rng.randint(0, len(train_cams) - 1) for _ in range(args.diag_steps)]
    test_cams = scene.getTestCameras()
    probe = (test_cams[:args.probe_count] if test_cams and len(test_cams) > 0
             else train_cams[:args.probe_count])
    print(f"[diag] probe views: {len(probe)} "
          f"({'test' if test_cams and len(test_cams) > 0 else 'train'} cameras)")

    # ---------------- Phase 2: three branches from the same ckpt ----------
    n_before = diag_ctx["n_gaussians_before"]
    results = {}

    for branch in ["keep", "clone", "split"]:
        seed_all(args.action_seed)                     # deterministic split sampling
        g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        g.restore(clone_tree(master_ckpt), opt)        # private deep copy per branch

        if branch in ("clone", "split"):
            g.tmp_radii = radii_master.clone()         # native contract (gaussian_model.py:479)
            single = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            single[diag_ctx["parent_index"]] = True
            if branch == "clone":
                g.densify_and_clone_fastgs(single, torch.ones_like(single))
            else:
                g.densify_and_split_fastgs(single, torch.ones_like(single), N=2)
            g.tmp_radii = None                         # native cleanup (gaussian_model.py:523-524)

        n_after = int(g.get_xyz.shape[0])
        print(f"\n[branch {branch}] #GS {n_before} -> {n_after} (net {n_after - n_before:+d})")

        seed_all(args.train_seed)                      # identical RNG at train start
        evals = {0: evaluate(g, probe, pipe, bg, opt.mult)}

        for i in range(1, args.diag_steps + 1):
            it = args.diag_iter + i
            cam = train_cams[cam_seq[i - 1]]
            # densification / pruning / opacity reset intentionally disabled
            _, _, _, _ = native_train_one_iter(it, cam, g, pipe, bg, opt)
            with torch.no_grad():
                if opt.optimizer_type == "default":
                    g.optimizer_step(it)               # native step-frequency schedule
                if i in (args.diag_steps // 2, args.diag_steps):
                    evals[i] = evaluate(g, probe, pipe, bg, opt.mult)

        lat_ms, lat_std = measure_render_perf(g, probe, pipe, bg, opt.mult)
        final_eval = evaluate(g, probe, pipe, bg, opt.mult)
        results[branch] = {
            "n_gaussians_after_action": n_after,
            "net_delta_gs": n_after - n_before,
            "evals": {str(k): v for k, v in evals.items()},
            "render_latency_ms": lat_ms,
            "render_latency_std_ms": lat_std,
            "fps": 1000.0 / lat_ms,
            "tile_pairs_mean": final_eval["tile_pairs_mean"],
        }
        print(f"[branch {branch}] PSNR@0={evals[0]['psnr']:.4f} "
              f"PSNR@{args.diag_steps // 2}={evals[args.diag_steps // 2]['psnr']:.4f} "
              f"PSNR@{args.diag_steps}={evals[args.diag_steps]['psnr']:.4f} "
              f"lat={lat_ms:.3f}ms pairs={final_eval['tile_pairs_mean']:.0f}")

    # ---------------- Phase 3: summary ------------------------------------
    h = args.diag_steps // 2
    def p(b, t): return results[b]["evals"][str(t)]["psnr"]
    def s(b, t): return results[b]["evals"][str(t)]["ssim"]
    def l(b, t): return results[b]["evals"][str(t)]["lpips"]
    def lo(b, t): return results[b]["evals"][str(t)]["loss"]

    summary = {
        "scene": os.path.abspath(dataset.source_path),
        "diag_context": diag_ctx,
        "config": {
            "densification_interval": opt.densification_interval,
            "grad_thresh": opt.grad_thresh,
            "grad_abs_thresh": opt.grad_abs_thresh,
            "dense": opt.dense, "mult": opt.mult,
            "loss_thresh": opt.loss_thresh,
            "probe_count": len(probe), "diag_steps": args.diag_steps,
            "camseq_seed": args.camseq_seed, "action_seed": args.action_seed,
            "train_seed": args.train_seed, "warmup_seed": args.warmup_seed,
        },
        "branches": results,
        "deltas": {
            "dPSNR_clone@0": p("clone", 0) - p("keep", 0),
            "dPSNR_split@0": p("split", 0) - p("keep", 0),
            "dPSNR_clone@50": p("clone", h) - p("keep", h),
            "dPSNR_split@50": p("split", h) - p("keep", h),
            "dPSNR_clone@100": p("clone", args.diag_steps) - p("keep", args.diag_steps),
            "dPSNR_split@100": p("split", args.diag_steps) - p("keep", args.diag_steps),
            "split_minus_clone@100": p("split", args.diag_steps) - p("clone", args.diag_steps),
            "dLatency_clone_ms": results["clone"]["render_latency_ms"] - results["keep"]["render_latency_ms"],
            "dLatency_split_ms": results["split"]["render_latency_ms"] - results["keep"]["render_latency_ms"],
        },
    }

    print("\n" + "=" * 78)
    print("B1 KEEP / CLONE / SPLIT DIAGNOSTIC SUMMARY")
    print("=" * 78)
    hdr = f"{'Metric':38s}{'Keep':>12}{'Clone':>12}{'Split':>12}"
    print(hdr); print("-" * 78)
    rows = [
        ("#GS after action", [str(results[b]["n_gaussians_after_action"]) for b in ["keep", "clone", "split"]]),
        ("Loss @0", [f"{lo(b,0):.5f}" for b in ["keep", "clone", "split"]]),
        (f"PSNR @{0}", [f"{p(b,0):.4f}" for b in ["keep", "clone", "split"]]),
        (f"PSNR @{h}", [f"{p(b,h):.4f}" for b in ["keep", "clone", "split"]]),
        (f"PSNR @{args.diag_steps}", [f"{p(b,args.diag_steps):.4f}" for b in ["keep", "clone", "split"]]),
        (f"SSIM @{args.diag_steps}", [f"{s(b,args.diag_steps):.4f}" for b in ["keep", "clone", "split"]]),
        (f"LPIPS @{args.diag_steps}", [f"{l(b,args.diag_steps):.4f}" for b in ["keep", "clone", "split"]]),
        ("Render latency ms @100", [f"{results[b]['render_latency_ms']:.3f}" for b in ["keep", "clone", "split"]]),
        ("FPS @100", [f"{results[b]['fps']:.1f}" for b in ["keep", "clone", "split"]]),
        ("Tile pairs (mean/view)", [f"{results[b]['tile_pairs_mean']:.0f}" for b in ["keep", "clone", "split"]]),
    ]
    for name, vals in rows:
        print(f"{name:38s}{vals[0]:>12}{vals[1]:>12}{vals[2]:>12}")
    print("-" * 78)
    d = summary["deltas"]
    print(f"dPSNR_clone  @100 : {d['dPSNR_clone@100']:+.4f}")
    print(f"dPSNR_split  @100 : {d['dPSNR_split@100']:+.4f}")
    print(f"Split - Clone gap@100 : {d['split_minus_clone@100']:+.4f}")
    print(f"dLatency_clone : {d['dLatency_clone_ms']:+.4f} ms")
    print(f"dLatency_split : {d['dLatency_split_ms']:+.4f} ms")
    print("=" * 78)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
