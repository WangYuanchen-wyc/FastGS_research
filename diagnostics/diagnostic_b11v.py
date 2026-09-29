#
# Paper B - B11-V: Multi-view Importance Reliability
#
# Diagnostics (all read-only on the model; no method design):
#   A) View subset stability: 20 random 10-view subsets per checkpoint,
#      Spearman rho + Top-K Jaccard + per-Gaussian rank/score instability
#   B) Cross-view agreement: per-view evidence vector s_i, support_ratio,
#      effective_view_count, max_view_fraction etc.
#   C) Whole-model group removal: score-matched high/low agreement groups,
#      remove 1%/5%/10%, evaluate on full test set (no recovery)
#   D) Stability-agreement correlation
#
# Checkpoints: train native FastGS once, save at 5000/10000/15000/30000.
#

import os, sys, json, random, time, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.image_utils import psnr
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras, get_loss
from utils.general_utils import inverse_sigmoid
from arguments import ModelParams, PipelineParams, OptimizationParams
from diagnostics.common import install_c_proxy, seed_all, native_train_one_iter

try:
    from lpipsPyTorch import lpips as lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False

OUT = "paper_b/b11_v_multiview_importance_reliability"
CKPT_ITERS = [5000, 10000, 15000, 30000]
N_SUBSETS = 20


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
            prune_mask, g.max_radii2D > st), g.get_scaling.max(dim=1).values > 0.1 * extent)
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


# ------------------------------------------------ per-view importance -----

@torch.no_grad()
def per_view_importance(g, camlist, pipe, bg, opt):
    """Returns (per_view_counts: list of (N,) tensors, aggregate_score (N,))."""
    counts = []
    for cam in camlist:
        out1 = render_fastgs(cam, g, pipe, bg, opt.mult)
        l1n = get_loss(out1["render"], cam.original_image.cuda())
        mmap = (l1n > opt.loss_thresh).int()
        out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                             get_flag=True, metric_map=mmap)
        counts.append(out2["accum_metric_counts"].clone())
    agg = torch.zeros_like(counts[0], dtype=torch.float32)
    for c in counts:
        agg += c.float()
    agg = torch.div(agg, len(camlist), rounding_mode='floor')
    return counts, agg


@torch.no_grad()
def full_test_eval(g, test_cams, pipe, bg, mult):
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
            except Exception:
                lp_.append(float("nan"))
    return {"psnr": float(np.mean(ps_)), "ssim": float(np.mean(ss_)),
            "lpips": float(np.nanmean(lp_)) if lp_ else None}


def main():
    parser = ArgumentParser("Paper B B11-V: multi-view importance reliability")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_iters", type=int, default=30000)
    parser.add_argument("--n_views", type=int, default=10)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    assert opt.optimizer_type == "default"
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    # ============ Phase 1: train & save checkpoints ============
    ckpt_base = f"{OUT}/checkpoints"
    need_train = not all(os.path.exists(f"{ckpt_base}/it{i}.pt") for i in CKPT_ITERS)
    if need_train:
        print("[b11v] training native FastGS with checkpoint saves...")
        install_c_proxy()
        seed_all(args.seed)
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)
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
            if it in CKPT_ITERS:
                torch.save(gaussians.capture(opt.optimizer_type), f"{ckpt_base}/it{it}.pt")
                print(f"[b11v] checkpoint saved: it{it} (#GS={gaussians.get_xyz.shape[0]})")
        del gaussians, scene
        torch.cuda.empty_cache()

    # ============ Phase 2: diagnostics per checkpoint ============
    install_c_proxy()
    seed_all(args.seed)
    g0 = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, g0)
    del g0
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()[:20] if scene.getTestCameras() else train_cams[:10]

    subset_rows, stab_rows, agree_rows, removal_rows = [], [], [], []

    for ck_it in CKPT_ITERS:
        ck_path = f"{ckpt_base}/it{ck_it}.pt"
        if not os.path.exists(ck_path):
            continue
        captured = torch.load(ck_path, map_location="cuda")
        n_gs = captured[1].shape[0]
        print(f"\n[b11v] === checkpoint it{ck_it} (#GS={n_gs}) ===")

        # ---- A) view subset stability: 20 subsets ----
        from diagnostics.common import clone_tree
        g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        g.restore(clone_tree(captured), opt)

        scores_all = []  # (20, N)
        ranks_all = []
        for si in range(N_SUBSETS):
            rng = random.Random(1000 + ck_it + si * 17)
            cams = rng.sample(list(train_cams), args.n_views)
            _, agg = per_view_importance(g, cams, pipe, bg, opt)
            scores_all.append(agg.cpu().numpy())
            ranks_all.append(np.argsort(np.argsort(-agg.cpu().numpy())))
        scores_all = np.array(scores_all)  # (20, N)
        ranks_all = np.array(ranks_all)

        # pairwise Spearman between subsets
        def spearman_np(x, y):
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            return float(np.corrcoef(rx, ry)[0, 1])
        rhos = []
        for a in range(N_SUBSETS):
            for b in range(a + 1, N_SUBSETS):
                rhos.append(spearman_np(scores_all[a], scores_all[b]))
        mean_rho = float(np.mean(rhos))
        # Top-K Jaccard (across subset pairs)
        for k_pct, k_label in ((5, "top5"), (10, "top10"), (20, "top20")):
            k = max(1, int(n_gs * k_pct / 100))
            jacc = []
            for a in range(N_SUBSETS):
                for b in range(a + 1, N_SUBSETS):
                    ta = set(np.argsort(-scores_all[a])[:k])
                    tb = set(np.argsort(-scores_all[b])[:k])
                    jacc.append(len(ta & tb) / len(ta | tb))
            print(f"  {k_label} Jaccard: {np.mean(jacc):.4f}")

        # per-Gaussian instability
        mean_score = scores_all.mean(axis=0)
        std_score = scores_all.std(axis=0, ddof=1)
        cv_score = std_score / (np.abs(mean_score) + 1e-8)
        rank_std = ranks_all.std(axis=0)
        pct = ranks_all / n_gs
        pct_std = pct.std(axis=0)

        for i in range(n_gs):
            stab_rows.append({
                "checkpoint": ck_it, "gaussian_id": i,
                "mean_score": float(mean_score[i]), "std_score": float(std_score[i]),
                "cv_score": float(cv_score[i]), "rank_std": float(rank_std[i]),
                "pct_std": float(pct_std[i])})

        # ---- B) cross-view agreement (reference subset) ----
        rng_ref = random.Random(999 + ck_it)
        ref_cams = rng_ref.sample(list(train_cams), args.n_views)
        per_view, ref_agg = per_view_importance(g, ref_cams, pipe, bg, opt)
        # per_view: list of (N,) gpu tensors
        pv = torch.stack(per_view).float().cpu().numpy()  # (V, N)
        total = pv.sum(axis=0)
        support = (pv > 0).sum(axis=0) / pv.shape[0]
        v_mean = pv.mean(axis=0)
        v_std = pv.std(axis=0, ddof=1) if pv.shape[0] > 1 else np.zeros(n_gs)
        v_cv = v_std / (v_mean + 1e-8)
        max_frac = pv.max(axis=0) / (total + 1e-8)
        sorted_pv = np.sort(pv, axis=0)
        top2 = (sorted_pv[-1] + sorted_pv[-2]) / (total + 1e-8)
        # effective view count (entropy-based)
        p = pv / (total + 1e-8)
        H = -(p * np.log(p + 1e-10)).sum(axis=0)
        eff_vc = np.exp(H)
        eff_vc[total < 0.5] = 0.0  # no evidence → no agreement

        for i in range(n_gs):
            agree_rows.append({
                "checkpoint": ck_it, "gaussian_id": i,
                "importance_score": float(ref_agg[i]),
                "support_ratio": float(support[i]),
                "view_mean": float(v_mean[i]), "view_std": float(v_std[i]),
                "view_cv": float(v_cv[i]),
                "max_view_fraction": float(max_frac[i]),
                "top2_view_fraction": float(top2[i]),
                "effective_view_count": float(eff_vc[i])})

        # ---- C) group removal test ----
        # bin by importance percentile (use the reference aggregate)
        imp = ref_agg.cpu().numpy()
        imp_pct = np.argsort(np.argsort(imp)) / n_gs  # 0..1

        # prunable pool: non-zero importance, lower half of those
        nonzero = imp > 0
        # percentile among non-zero only
        nz_idx = np.where(nonzero)[0]
        nz_order = np.argsort(np.argsort(imp[nz_idx]))
        nz_pct = np.zeros(n_gs)
        nz_pct[nz_idx] = nz_order / max(len(nz_idx), 1)
        pool_mask = nonzero & (nz_pct < 0.5)
        pool_idx = np.where(pool_mask)[0]
        if len(pool_idx) < 100:
            pool_idx = np.where(nonzero & (nz_pct < 0.8))[0]
        if len(pool_idx) < 50:
            print(f"  WARNING: pool too small ({len(pool_idx)}), skipping removal")
            del g
            continue

        # within pool, split by effective_view_count median within each
        # importance decile (stratified) to keep distributions matched
        pool_eff = eff_vc[pool_idx]
        pool_imp_pct = nz_pct[pool_idx]
        n_pool = len(pool_idx)
        n_dec = 10
        decile = np.minimum((pool_imp_pct * n_dec).astype(int), n_dec - 1)
        hi_list, lo_list = [], []
        for d in range(n_dec):
            dm = decile == d
            if dm.sum() < 10:
                continue
            de = pool_eff[dm]
            di = np.where(dm)[0]
            med = np.median(de)
            hi_local = di[de >= med]
            lo_local = di[de < med]
            # equal counts per decile
            n_eq = min(len(hi_local), len(lo_local))
            hi_list.append(hi_local[:n_eq])
            lo_list.append(lo_local[:n_eq])
        hi_indices = np.concatenate(hi_list) if hi_list else np.array([], dtype=int)
        lo_indices = np.concatenate(lo_list) if lo_list else np.array([], dtype=int)

        n_pair = min(len(hi_indices), len(lo_indices))
        if n_pair < 50:
            print(f"  WARNING: too few pairs ({n_pair}), skipping removal")
            del g
            continue

        # baseline eval
        base = full_test_eval(g, test_cams, pipe, bg, opt.mult)
        removal_rows.append({"checkpoint": ck_it, "group": "baseline",
                            "pct_removed": 0, "n_gs": n_gs, **base})

        for pct_remove in (1, 5, 10):
            n_remove = int(n_gs * pct_remove / 100)
            n_remove = min(n_remove, n_pair)
            for group_name, group_pool in (("low_agreement", lo_indices),
                                           ("high_agreement", hi_indices)):
                # select from group, prefer lowest importance within group
                pool_local = np.array(group_pool)
                # sort by importance ascending (remove least important first)
                pool_imp_vals = pool_imp_pct[pool_local]
                sel_pool = pool_local[np.argsort(pool_imp_vals)[:n_remove]]
                sel_global = pool_idx[sel_pool]

                gg = GaussianModel(dataset.sh_degree, opt.optimizer_type)
                gg.restore(clone_tree(captured), opt)
                rm = torch.zeros(gg.get_xyz.shape[0], dtype=torch.bool, device="cuda")
                rm[torch.tensor(sel_global, dtype=torch.long)] = True
                with torch.no_grad():
                    gg.tmp_radii = None
                    gg.prune_points(rm)
                ev = full_test_eval(gg, test_cams, pipe, bg, opt.mult)
                removal_rows.append({"checkpoint": ck_it, "group": group_name,
                                     "pct_removed": pct_remove,
                                     "n_gs": int(gg.get_xyz.shape[0]), **ev})
                print(f"  remove {pct_remove}% {group_name}: PSNR {ev['psnr']:.3f} "
                      f"(drop {ev['psnr']-base['psnr']:+.3f})")
                del gg
                torch.cuda.empty_cache()

        # ---- D) stability vs agreement ----
        # (computed in analysis script; here just save the data)
        del g
        torch.cuda.empty_cache()

    # ---- save all data ----
    for name, rows in (("b11v_gaussian_stability", stab_rows),
                       ("b11v_view_agreement", agree_rows),
                       ("b11v_removal_results", removal_rows)):
        if rows:
            with open(f"{OUT}/data/{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"\n[b11v] done: {len(stab_rows)} stability, {len(agree_rows)} agreement, "
          f"{len(removal_rows)} removal rows")


if __name__ == "__main__":
    main()
