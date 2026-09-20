#
# Paper B - B19-P: Actual Rendering Support of Opacity-Prune Candidates
#
# Question: do opacity<0.1 candidates of the native final prunes (18k/21k/24k/27k)
# still have real multi-view rendering contribution?
#
# Contribution definition: the per-Gaussian blend weight  w_i = alpha_i * T_i
# of the front-to-back alpha blend (the exact term the rasterizer multiplies
# into the color at forward.cu:399), summed over all pixels of a view.
# Obtained by MINIMAL renderer instrumentation: a new optional get_weights flag
# makes the rasterizer atomicAdd alpha*T into a (P,) buffer during its normal
# blend pass. No extra renders: the buffer is captured from the FIRST (plain)
# render of each view in the native final-prune VCD pass; the second render and
# the returned pruning_score are untouched, so the prune result is unchanged.
#
# Tolerance: support_view_count counts views with total contribution > 1e-6
# (float-zero filter only; a single minimal-alpha pixel contributes >= 3.9e-7).
# Fixed a priori, not tuned.
#
# Loop is native-faithful (identical to diagnostic_b19r.py): optimizer_step,
# opacity reset only below 15k, final prunes every 3000 iters in (15k, 30k).
#
# GPU: 0
#

import os, sys, random, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.image_utils import psnr
from utils.fast_utils import (compute_gaussian_score_fastgs, sampling_cameras,
                              get_loss, compute_photometric_loss)
from utils.general_utils import inverse_sigmoid
from arguments import ModelParams, PipelineParams, OptimizationParams
from diagnostics.common import install_c_proxy, seed_all, native_train_one_iter

OUT = "paper_b/b19_reliable_pruning"
TOL = 1e-6
NVIEWS = 10


def safe_replace_opacity(g, tensor):
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
    """Native event densification (no capture needed here)."""
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
    n_clone, n_split = 0, 0
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


@torch.no_grad()
def contribution_pass(camlist, gaussians, pipe, bg, args):
    """Identical to utils.fast_utils.compute_gaussian_score_fastgs(DENSIFY=False),
    additionally capturing the per-Gaussian blend-weight buffer (N, K) from the
    first (plain) render of each view. Same renders, same pruning_score."""
    full_score = None
    per_view_w = []
    for view in range(len(camlist)):
        cam = camlist[view]
        pkg = render_fastgs(cam, gaussians, pipe, bg, args.mult, get_weights=True)
        render_image = pkg["render"]
        per_view_w.append(pkg["gauss_weights"].detach().reshape(-1))
        photometric_loss = compute_photometric_loss(cam, render_image)
        gt_image = cam.original_image.cuda()
        l1_norm = get_loss(render_image, gt_image)
        metric_map = (l1_norm > args.loss_thresh).int()
        pkg2 = render_fastgs(cam, gaussians, pipe, bg, args.mult,
                             get_flag=True, metric_map=metric_map)
        acc = pkg2["accum_metric_counts"].detach()
        full_score = (photometric_loss * acc.clone() if full_score is None
                      else full_score + photometric_loss * acc)
    pruning_score = (full_score - torch.min(full_score)) / (
        torch.max(full_score) - torch.min(full_score))
    W = torch.stack(per_view_w, dim=1)  # (N, K)
    return pruning_score, W


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
    os.makedirs(args.model_path, exist_ok=True)

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

        event_num = 0
        round_num = 0
        cand_rows = []
        round_stats = []

        for it in range(1, args.max_iters + 1):
            if not vp_stack:
                vp_stack = scene.getTrainCameras().copy()
                vp_idx = list(range(len(vp_stack)))
            r = random.randint(0, len(vp_idx) - 1)
            cam = vp_stack.pop(r)
            _ = vp_idx.pop(r)
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
                        apply_native_densify(gaussians, opt, it, radii, imp, pru, extent)

                        if it % 1000 == 0:
                            ps = eval_quick(gaussians, test_cams, pipe, bg, opt.mult)
                            print(f"  s{seed} it={it} ev={event_num} PSNR={ps:.2f} "
                                  f"#GS={gaussians.get_xyz.shape[0]}", flush=True)

                if it < opt.densify_until_iter and it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

                if it % 3000 == 0 and it > 15_000 and it < 30_000:
                    round_num += 1
                    my = scene.getTrainCameras().copy()
                    cl = sampling_cameras(my)
                    pru, W = contribution_pass(cl, gaussians, pipe, bg, opt)

                    opa = gaussians.get_opacity.detach().squeeze(-1)
                    cand_mask = opa < 0.1
                    n_cand = int(cand_mask.sum())
                    pru1 = pru.reshape(-1)
                    if n_cand > 0:
                        cand_idx = torch.where(cand_mask)[0]
                        Wc = W[cand_idx]                       # (C, K) blend weights
                        opa_c = opa[cand_idx]
                        pru_c = pru1[cand_idx]
                        support = (Wc > TOL).sum(dim=1)
                        total = Wc.sum(dim=1)
                        srt, _ = torch.sort(Wc, dim=1, descending=True)
                        top1 = srt[:, 0] / total.clamp(min=1e-12)
                        top3 = srt[:, :3].sum(dim=1) / total.clamp(min=1e-12)
                        wmax = srt[:, 0]
                        for i in range(n_cand):
                            sv = int(support[i])
                            cand_rows.append({
                                "seed": seed, "round": round_num, "round_it": it,
                                "opacity": round(float(opa_c[i]), 5),
                                "pruning_score": round(float(pru_c[i]), 4),
                                "support_view_count": sv,
                                "total_contribution": round(float(total[i]), 6),
                                "mean_per_supported_view": round(
                                    float(total[i]) / sv, 6) if sv > 0 else 0.0,
                                "max_view_contribution": round(float(wmax[i]), 6),
                                "top1_share": round(float(top1[i]), 4),
                                "top3_share": round(float(top3[i]), 4),
                            })
                    round_stats.append({"seed": seed, "round": round_num, "it": it,
                                        "n_gs": gaussians.get_xyz.shape[0],
                                        "n_candidates": n_cand,
                                        "n_score_only_removed": int(
                                            ((opa >= 0.1) & (pru1 > 0.9)).sum())})
                    gaussians.final_prune_fastgs(0.1, pruning_score=pru)
                    print(f"  s{seed} it={it} round={round_num} cand(opacity<0.1)="
                          f"{n_cand} #GS->{gaussians.get_xyz.shape[0]}", flush=True)

            gaussians.optimizer_step(it)

        with open(f"{OUT}/data/prune_candidate_render_support_s{seed}.csv", "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cand_rows[0].keys()))
            w.writeheader(); w.writerows(cand_rows)
        with open(f"{OUT}/data/round_stats_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(round_stats[0].keys()))
            w.writeheader(); w.writerows(round_stats)
        print(f"  s{seed} saved: {len(cand_rows)} candidate rows", flush=True)
        torch.cuda.empty_cache()

    print(f"\n[b19p] done", flush=True)


if __name__ == "__main__":
    main()
