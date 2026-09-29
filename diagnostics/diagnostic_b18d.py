#
# Paper B - B18-D: Densification Demand Persistence Diagnostic
#
# Tracks which Gaussians trigger densification at each event and whether
# the same Gaussians (matched by xyz proximity) continue to trigger.
#
# Also tracks children of split parents and their long-term fate.
#
# GPU: 1
#

import os, sys, random, time, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.image_utils import psnr
from fused_ssim import fused_ssim as fast_ssim
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.general_utils import inverse_sigmoid
from arguments import ModelParams, PipelineParams, OptimizationParams
from diagnostics.common import install_c_proxy, seed_all, native_train_one_iter

OUT = "paper_b/b18_densification_persistence"


def safe_replace_opacity(g, tensor):
    """Ensure optimizer state exists before calling replace_tensor_to_optimizer."""
    import torch.nn as nn
    for group in g.optimizer.param_groups:
        if group["name"] == "opacity":
            p = group["params"][0]
            if p not in g.optimizer.state:
                g.optimizer.state[p] = {
                    "step": torch.tensor(0.0),
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }
    return g.replace_tensor_to_optimizer(tensor, "opacity")


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
    n_clone = 0; n_split = 0
    if clone_set.any():
        n_clone = int(clone_set.sum())
        g.densify_and_clone_fastgs(clone_set, torch.ones_like(clone_set))
    if split_set.any():
        n_split = int(split_set.sum())
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
    g._opacity = safe_replace_opacity(
        g, inverse_sigmoid(torch.min(g.get_opacity, torch.ones_like(g.get_opacity) * 0.8))
    )["opacity"]
    torch.cuda.empty_cache()
    return n_clone, n_split


@torch.no_grad()
def eval_quick(g, test_cams, pipe, bg, mult):
    ps_ = []
    for cam in test_cams[:5]:
        img = render_fastgs(cam, g, pipe, bg, mult)["render"]
        gt = cam.original_image.cuda()
        ps_.append(float(psnr(torch.clamp(img, 0, 1), torch.clamp(gt, 0, 1)).mean()))
    return float(np.mean(ps_))


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
    os.makedirs(f"{OUT}/data", exist_ok=True)

    all_trigger_rows = []
    all_parent_rows = []
    all_child_rows = []

    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}", flush=True)
        install_c_proxy()
        seed_all(seed)
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)
        extent = scene.cameras_extent

        test_cams = scene.getTestCameras()[:10] if scene.getTestCameras() else scene.getTrainCameras()[:5]
        vp_stack = scene.getTrainCameras().copy()
        vp_idx = list(range(len(vp_stack)))

        # Persistent tracking
        prev_trigger_xyz = None   # (K, 3) positions of previously triggered Gaussians
        trigger_count = None      # (N,) per-Gaussian count of how many times triggered
        last_trigger_event = None  # (N,) last event number that triggered

        event_num = 0
        densify_events = []

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
                        event_num += 1
                        my = scene.getTrainCameras().copy()
                        cl = sampling_cameras(my)
                        imp, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt, DENSIFY=True)

                        # Get native split trigger set
                        grad_vars = gaussians.xyz_gradient_accum / gaussians.denom
                        grad_vars[grad_vars.isnan()] = 0.0
                        grads_abs = gaussians.xyz_gradient_accum_abs / gaussians.denom
                        grads_abs[grads_abs.isnan()] = 0.0
                        grad_abs_qual = torch.norm(grads_abs, dim=-1) >= opt.grad_abs_thresh
                        max_scale = gaussians.get_scaling.max(dim=1).values
                        metric_mask = imp > 5
                        split_set = metric_mask & (max_scale > opt.dense * extent) & grad_abs_qual
                        n_trigger = int(split_set.sum())

                        xyz_np = gaussians.get_xyz.detach().cpu().numpy()
                        triggered_xyz = xyz_np[split_set.cpu().numpy()]

                        # Count persistence
                        n_persist_1 = 0; n_persist_2 = 0; n_persist_3 = 0
                        if prev_trigger_xyz is not None and len(prev_trigger_xyz) > 0 and n_trigger > 0 and len(triggered_xyz) > 0:
                            # Match by proximity: for each current triggered Gaussian,
                            # check if it was near a previously triggered position
                            prev_t = torch.tensor(prev_trigger_xyz, dtype=torch.float32, device="cuda")
                            cur_t = torch.tensor(triggered_xyz, dtype=torch.float32, device="cuda")
                            dist_mat = torch.cdist(cur_t, prev_t)
                            dist, _ = dist_mat.min(dim=1)
                            dist = dist.cpu().numpy()
                            n_persist_1 = int((dist < 0.05).sum())  # within 0.05 units
                            # For ≥2/≥3, we track cumulative trigger counts per position
                            if trigger_count is not None:
                                # Update trigger counts for currently triggering Gaussians
                                tc_idx = torch.where(split_set)[0].cpu().numpy()
                                for ti in tc_idx:
                                    trigger_count[ti] = trigger_count.get(ti, 0) + 1

                        # Initialize/update trigger count (use dict keyed by gaussian index)
                        # trigger_count tracks how many times each Gaussian triggered

                        # record
                        densify_events.append({
                            "iteration": it, "event": event_num,
                            "n_trigger": n_trigger,
                            "n_persist_from_prev": n_persist_1,
                            "n_gs_before": gaussians.get_xyz.shape[0],
                            "seed": seed})

                        apply_native_densify(gaussians, opt, it, radii, imp, pru, scene.cameras_extent)

                        # update prev_trigger_xyz for next event
                        prev_trigger_xyz = triggered_xyz

                        if it % 3000 == 0:
                            ps = eval_quick(gaussians, test_cams, pipe, bg, opt.mult)
                            print(f"  seed{seed} it={it} event={event_num} triggers={n_trigger} "
                                  f"persist={n_persist_1} PSNR={ps:.2f} #GS={gaussians.get_xyz.shape[0]}",
                                  flush=True)

                if it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            if it % 3000 == 0 and it <= opt.densify_until_iter:
                pass  # eval already done in densification block

        # save per-seed data
        with open(f"{OUT}/data/densify_events_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(densify_events[0].keys()))
            w.writeheader(); w.writerows(densify_events)

    print(f"\n[b18d] done")


if __name__ == "__main__":
    main()
