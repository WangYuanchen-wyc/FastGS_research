#
# Paper B - B10-R: Extra Capacity Removability
#
# Train Native / Early-3 / Always-3 to 30k (same seeds as B9-T), save final
# checkpoints, then prune each Split-3 model to Native #GS / 90% / 80% using
# FastGS's OWN pruning_score (multi-view reconstruction consistency from
# compute_gaussian_score_fastgs, DENSIFY=False), then evaluate:
#   immediate  (no further optimization)
#   +500 steps (densify/prune OFF — short recovery)
#
# Pruning rule: rank Gaussians by the native pruning_score (lower = more
# redundant) and remove the lowest-scoring (N - target) Gaussians — the same
# importance the native final_prune_fastgs uses, applied to an arbitrary
# target count instead of a fixed threshold.
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

OUT = "paper_b/b10_r_extra_capacity_removability"
STAGES = {"early3": (1000, 5500)}


def apply_native_densify(g, opt, it, radii, imp, pru, extent, N_split):
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
        g.densify_and_split_fastgs(split_set, torch.ones_like(split_set), N=N_split)
    g.tmp_radii = None

    from utils.general_utils import inverse_sigmoid
    prune_mask = (g.get_opacity < 0.005).squeeze()
    size_threshold = 20 if it > opt.opacity_reset_interval else None
    if size_threshold:
        prune_mask = torch.logical_or(torch.logical_or(
            prune_mask, g.max_radii2D > size_threshold),
            g.get_scaling.max(dim=1).values > 0.1 * extent)
    scores = 1 - pru
    to_remove = int(torch.sum(prune_mask))
    remove_budget = int(0.5 * to_remove)
    if remove_budget:
        n_now = g.get_xyz.shape[0]
        padded = torch.zeros((n_now), dtype=torch.float32, device=scores.device)
        padded[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
        sel = torch.zeros_like(padded, dtype=bool)
        sel[torch.multinomial(padded, remove_budget, replacement=False)] = True
        g.prune_points(torch.logical_and(prune_mask, sel))
    opa_new = torch.min(g.get_opacity, torch.ones_like(g.get_opacity) * 0.8)
    g._opacity = g.replace_tensor_to_optimizer(inverse_sigmoid(opa_new), "opacity")["opacity"]
    torch.cuda.empty_cache()


def main():
    parser = ArgumentParser("Paper B B10-R: extra capacity removability")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--strategies", type=str, default="native,early3,always3")
    parser.add_argument("--max_iters", type=int, default=30000)
    parser.add_argument("--recovery_steps", type=int, default=500)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    assert opt.optimizer_type == "default"
    seeds = [int(s) for s in args.seeds.split(",")]
    strategies = args.strategies.split(",")
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    import diagnostics.diagnostic_v2 as v2
    v2.G.update({"pipe": pipe, "bg": bg, "mult": opt.mult,
                 "loss_thresh": opt.loss_thresh, "sh_degree": dataset.sh_degree})

    # ---------- Phase 1: train & save checkpoints ----------
    ckpt_dir = f"{OUT}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    meta_path = f"{OUT}/data/b10r_final_models.csv"
    done_meta = set()
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            done_meta = {(r["strategy"], int(r["seed"])) for r in csv.DictReader(f)}
    meta_rows = []
    if os.path.exists(meta_path):
        meta_rows = list(csv.DictReader(open(meta_path)))

    for seed in seeds:
        for strat in strategies:
            ck = f"{ckpt_dir}/{strat}_s{seed}.pt"
            if (strat, seed) in done_meta and os.path.exists(ck):
                print(f"[b10r] skip train {strat} s{seed}")
                continue
            install_c_proxy()
            seed_all(seed)
            gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
            scene = Scene(dataset, gaussians)
            gaussians.training_setup(opt)
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
                            n = 3 if (strat == "always3" or
                                      (strat == "early3" and 1000 <= it < 5500)) else 2
                            my = scene.getTrainCameras().copy()
                            cl = sampling_cameras(my)
                            imp, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt, DENSIFY=True)
                            apply_native_densify(gaussians, opt, it, radii, imp, pru,
                                                 scene.cameras_extent, N_split=n)
                        if it % opt.opacity_reset_interval == 0:
                            gaussians.reset_opacity()
                    if it % 3000 == 0 and 15_000 < it < 30_000:
                        my = scene.getTrainCameras().copy()
                        cl = sampling_cameras(my)
                        _, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt)
                        gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pru)
                    gaussians.optimizer_step(it)
            n_gs = gaussians.get_xyz.shape[0]
            wall = time.time() - t0
            # eval + save
            test_cams = scene.getTestCameras()[:20]
            from utils.image_utils import psnr as _psnr
            with torch.no_grad():
                ps_, ss_, lp_ = [], [], []
                for cam in test_cams:
                    img = render_fastgs(cam, gaussians, pipe, bg, opt.mult)["render"]
                    gt = cam.original_image.cuda()
                    ic, gc = torch.clamp(img, 0, 1), torch.clamp(gt, 0, 1)
                    ps_.append(float(_psnr(ic, gc).mean()))
                    ss_.append(float(fast_ssim(ic.unsqueeze(0), gc.unsqueeze(0)).mean()))
                    if LPIPS_OK:
                        try:
                            lp_.append(float(lpips_fn(ic, gc, net_type="vgg").mean()))
                        except Exception:
                            lp_.append(float("nan"))
            meta_rows.append({"strategy": strat, "seed": str(seed),
                              "n_gaussians": str(n_gs), "wall_time_s": str(wall),
                              "psnr": str(np.mean(ps_)), "ssim": str(np.mean(ss_)),
                              "lpips": str(np.nanmean(lp_)) if lp_ else "NA",
                              "checkpoint": ck})
            with open(meta_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
                w.writeheader()
                w.writerows(meta_rows)
            torch.save(gaussians.capture(opt.optimizer_type), ck)
            print(f"[b10r] trained {strat} s{seed}: #GS={n_gs} PSNR={np.mean(ps_):.3f} "
                  f"({wall:.0f}s) -> {ck}")
            del gaussians, scene
            torch.cuda.empty_cache()

    # ---------- Phase 2: pruning test ----------
    results = []
    test_scene_created = False
    for seed in seeds:
        for strat in ("early3", "always3"):
            ck = f"{ckpt_dir}/{strat}_s{seed}.pt"
            nat_ck = f"{ckpt_dir}/native_s{seed}.pt"
            if not os.path.exists(ck) or not os.path.exists(nat_ck):
                continue
            seed_all(seed)
            gs_model = GaussianModel(dataset.sh_degree, opt.optimizer_type)
            scene = Scene(dataset, gs_model)
            test_cams = scene.getTestCameras()[:20]
            train_cams = scene.getTrainCameras()
            del gs_model

            captured = torch.load(ck, map_location="cuda")
            captured_nat = torch.load(nat_ck, map_location="cuda")
            n_nat = captured_nat[1].shape[0]

            for tgt_label, tgt_n in (("A=native", n_nat), ("B=90%", int(0.9 * n_nat)),
                                     ("C=80%", int(0.8 * n_nat))):
                # restore fresh copy
                from diagnostics.common import clone_tree
                g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
                g.restore(clone_tree(captured), opt)
                n_orig = g.get_xyz.shape[0]
                if tgt_n >= n_orig:
                    continue

                # native pruning score (same as final_prune_fastgs uses)
                with torch.no_grad():
                    my = scene.getTrainCameras().copy()
                    cl = sampling_cameras(my)
                    _, pru = compute_gaussian_score_fastgs(cl, g, pipe, bg, opt)
                # lower pruning_score = more redundant; remove lowest (n_orig - tgt_n)
                k = n_orig - tgt_n
                order = torch.argsort(pru)  # ascending; first k = lowest score = most redundant
                remove_mask = torch.zeros(n_orig, dtype=torch.bool, device="cuda")
                remove_mask[order[:k]] = True
                with torch.no_grad():
                    g.tmp_radii = None  # ensure attribute exists (prune_points checks it)
                    g.prune_points(remove_mask)
                n_now = g.get_xyz.shape[0]

                # immediate eval
                with torch.no_grad():
                    ps_, ss_, lp_ = [], [], []
                    for cam in test_cams:
                        img = render_fastgs(cam, g, pipe, bg, opt.mult)["render"]
                        gt = cam.original_image.cuda()
                        ic, gc = torch.clamp(img, 0, 1), torch.clamp(gt, 0, 1)
                        ps_.append(float(psnr(ic, gc).mean()))
                        ss_.append(float(fast_ssim(ic.unsqueeze(0), gc.unsqueeze(0)).mean()))
                        if LPIPS_OK:
                            try:
                                lp_.append(float(lpips_fn(ic, gc, net_type="vgg").mean()))
                            except Exception:
                                lp_.append(float("nan"))
                row = {"strategy": strat, "seed": seed, "target": tgt_label,
                       "n_orig": n_orig, "n_target": tgt_n, "n_after_prune": n_now,
                       "psnr_imm": float(np.mean(ps_)), "ssim_imm": float(np.mean(ss_)),
                       "lpips_imm": float(np.nanmean(lp_)) if lp_ else None}

                # short recovery: 500 steps, densify/prune OFF
                seed_all(1234)
                rec_seq = list(range(args.recovery_steps))
                for i in range(args.recovery_steps):
                    cam = train_cams[rec_seq[i % len(train_cams)]]
                    native_train_one_iter(30000 + i + 1, cam, g, pipe, bg, opt)
                    with torch.no_grad():
                        g.optimizer_step(30000 + i + 1)
                with torch.no_grad():
                    ps_, ss_, lp_ = [], [], []
                    for cam in test_cams:
                        img = render_fastgs(cam, g, pipe, bg, opt.mult)["render"]
                        gt = cam.original_image.cuda()
                        ic, gc = torch.clamp(img, 0, 1), torch.clamp(gt, 0, 1)
                        ps_.append(float(psnr(ic, gc).mean()))
                        ss_.append(float(fast_ssim(ic.unsqueeze(0), gc.unsqueeze(0)).mean()))
                        if LPIPS_OK:
                            try:
                                lp_.append(float(lpips_fn(ic, gc, net_type="vgg").mean()))
                            except Exception:
                                lp_.append(float("nan"))
                row.update({"psnr_rec500": float(np.mean(ps_)),
                            "ssim_rec500": float(np.mean(ss_)),
                            "lpips_rec500": float(np.nanmean(lp_)) if lp_ else None,
                            "n_after_rec": g.get_xyz.shape[0]})
                results.append(row)
                print(f"[b10r] {strat} s{seed} {tgt_label}: {n_orig}->{n_now} "
                      f"PSNR {row['psnr_imm']:.3f}->{row['psnr_rec500']:.3f}")
                del g
                torch.cuda.empty_cache()
            del scene
            torch.cuda.empty_cache()

    if results:
        with open(f"{OUT}/data/b10r_pruning_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    print(f"[b10r] done ({len(results)} pruning results)")


if __name__ == "__main__":
    main()
