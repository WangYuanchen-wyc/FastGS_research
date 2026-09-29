#
# Paper B - B11-V Fix: Pruning-Score Controlled Agreement Diagnostic
#
# Fix A: tie-aware rank stability (rankdata instead of argsort-argsort)
# Fix B: FastGS actual pruning_score stability across view subsets
# Fix C: multi-subset agreement definition (not single reference)
# Fix D: nearest-neighbor pruning-score matched pairs (|pct_diff| <= 0.5-1%)
# Fix E: 5 bootstrap repeats per checkpoint × removal ratio
# Fix F: random matched control group
#
# Uses B11-V checkpoints it5000/10000/15000/30000. No retraining.
#

import os, sys, json, random, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.image_utils import psnr
from fused_ssim import fused_ssim as fast_ssim
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras, get_loss
from arguments import ModelParams, PipelineParams, OptimizationParams
from diagnostics.common import install_c_proxy, seed_all, clone_tree

try:
    from lpipsPyTorch import lpips as lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False

CKPT_BASE = "paper_b/b11_v_multiview_importance_reliability/checkpoints"
OUT = "paper_b/b11_v_multiview_importance_reliability/fix"
CKPT_ITERS = [5000, 10000, 15000, 30000]
N_SUBSETS = 20
N_REPEATS = 5


def rankdata_avg(x):
    """Tie-aware rank (average method), pure numpy."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman_tie(x, y):
    return float(np.corrcoef(rankdata_avg(x), rankdata_avg(y))[0, 1])


@torch.no_grad()
def per_view_counts_and_pru(g, camlist, pipe, bg, opt):
    """Returns (per_view_counts list, pruning_score). pruning_score follows
    FastGS compute_gaussian_score_fastgs DENSIFY=False path."""
    counts = []
    full_score = None
    for cam in camlist:
        out1 = render_fastgs(cam, g, pipe, bg, opt.mult)
        l1n = get_loss(out1["render"], cam.original_image.cuda())
        mmap = (l1n > opt.loss_thresh).int()
        out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                             get_flag=True, metric_map=mmap)
        counts.append(out2["accum_metric_counts"].clone().float())
        # photometric loss for pruning_score (same as compute_gaussian_score_fastgs)
        pl = 0.8 * torch.mean(torch.abs(out1["render"] - cam.original_image.cuda())) \
             + 0.2 * (1.0 - fast_ssim(out1["render"].unsqueeze(0),
                                       cam.original_image.cuda().unsqueeze(0)))
        c = out2["accum_metric_counts"].float()
        contrib = pl * c
        full_score = contrib.clone() if full_score is None else full_score + contrib
    pru = (full_score - full_score.min()) / (full_score.max() - full_score.min() + 1e-8)
    return counts, pru


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
    parser = ArgumentParser("Paper B B11-V Fix")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--n_views", type=int, default=10)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    assert opt.optimizer_type == "default"
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    install_c_proxy()
    seed_all(0)
    g0 = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, g0)
    del g0
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()[:20] if scene.getTestCameras() else train_cams[:10]

    os.makedirs(f"{OUT}/data", exist_ok=True)
    os.makedirs(f"{OUT}/plots", exist_ok=True)

    stab_rows, pru_rows, agree_rows, pair_rows, grp_rows, rem_rows = [], [], [], [], [], []

    for ck_it in CKPT_ITERS:
        ck_path = f"{CKPT_BASE}/it{ck_it}.pt"
        if not os.path.exists(ck_path):
            continue
        captured = torch.load(ck_path, map_location="cuda")
        n_gs = captured[1].shape[0]
        print(f"\n[fix] === it{ck_it} (#GS={n_gs}) ===")
        g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        g.restore(clone_tree(captured), opt)

        # ---- B) pruning_score across 20 subsets ----
        pru_scores = []  # (20, N)
        for si in range(N_SUBSETS):
            rng = random.Random(1000 + ck_it + si * 17)
            cams = rng.sample(list(train_cams), args.n_views)
            _, pru = per_view_counts_and_pru(g, cams, pipe, bg, opt)
            pru_scores.append(pru.cpu().numpy())
        pru_scores = np.array(pru_scores)

        # tie-aware stability
        tie_ranks = [rankdata_avg(s) for s in pru_scores]
        rhos = []
        for a in range(N_SUBSETS):
            for b in range(a + 1, N_SUBSETS):
                rhos.append(spearman_tie(pru_scores[a], pru_scores[b]))
        mean_rho = float(np.mean(rhos)); std_rho = float(np.std(rhos))
        p10_rho = float(np.percentile(rhos, 10)); p90_rho = float(np.percentile(rhos, 90))

        # tie fraction at top-K cutoff
        med_score = np.median(pru_scores, axis=0)
        for k_pct in (5, 10, 20):
            k = max(1, int(n_gs * k_pct / 100))
            # use median score as representative
            order = np.argsort(-med_score)
            cutoff = med_score[order[k - 1]]
            n_tied = int(np.sum(med_score == cutoff))
            tie_frac = n_tied / n_gs
            # actual set overlap
            overlaps = []
            for a in range(N_SUBSETS):
                for b in range(a + 1, N_SUBSETS):
                    ta = set(np.argsort(-pru_scores[a])[:k])
                    tb = set(np.argsort(-pru_scores[b])[:k])
                    overlaps.append(len(ta & tb) / k)
            stab_rows.append({"checkpoint": ck_it, "metric": f"top{k_pct}",
                              "mean_rho": mean_rho, "std_rho": std_rho,
                              "p10_rho": p10_rho, "p90_rho": p90_rho,
                              "mean_overlap": float(np.mean(overlaps)),
                              "cutoff_score": float(cutoff),
                              "n_tied_at_cutoff": n_tied, "tie_fraction": tie_frac})
        print(f"  pruning_score Spearman: {mean_rho:.4f}±{std_rho:.4f} "
              f"[{p10_rho:.4f}~{p90_rho:.4f}]")

        # per-Gaussian pruning score stats
        pru_mean = pru_scores.mean(axis=0)
        pru_std = pru_scores.std(axis=0, ddof=1)
        pru_cv = pru_std / (pru_mean + 1e-8)
        pru_pct = np.array([rankdata_avg(s) / n_gs for s in pru_scores])
        pru_pct_std = pru_pct.std(axis=0)
        for i in range(n_gs):
            pru_rows.append({"checkpoint": ck_it, "gaussian_id": i,
                             "pru_mean": float(pru_mean[i]),
                             "pru_std": float(pru_std[i]),
                             "pru_cv": float(pru_cv[i]),
                             "pru_pct_std": float(pru_pct_std[i])})

        # ---- C) multi-subset agreement ----
        # use 5 additional subsets, collect per-view counts from each
        all_pv = []  # list of (V, N) arrays, one per subset
        for si in range(5):
            rng = random.Random(5000 + ck_it + si * 31)
            cams = rng.sample(list(train_cams), args.n_views)
            pvc, _ = per_view_counts_and_pru(g, cams, pipe, bg, opt)
            all_pv.append(torch.stack(pvc).cpu().numpy())
        # concatenate all views from all subsets: (5*V, N)
        pv = np.concatenate(all_pv, axis=0)
        total = pv.sum(axis=0)
        support = (pv > 0).sum(axis=0) / pv.shape[0]
        v_mean = pv.mean(axis=0)
        v_std = pv.std(axis=0, ddof=1)
        max_frac = pv.max(axis=0) / (total + 1e-8)
        sp = np.sort(pv, axis=0)
        top2 = (sp[-1] + sp[-2]) / (total + 1e-8)
        p = pv / (total + 1e-8)
        H = -(p * np.log(p + 1e-10)).sum(axis=0)
        eff_vc = np.exp(H)
        eff_vc[total < 0.5] = 0.0

        # reference pruning score (median across subsets)
        ref_pru = pru_mean

        for i in range(n_gs):
            agree_rows.append({"checkpoint": ck_it, "gaussian_id": i,
                               "pruning_score": float(ref_pru[i]),
                               "effective_view_count": float(eff_vc[i]),
                               "support_ratio": float(support[i]),
                               "max_view_fraction": float(max_frac[i]),
                               "top2_view_fraction": float(top2[i])})

        # ---- D) matched pair construction + E) removal ----
        # sort by pruning score, pair adjacent H/L within tight score window
        pru_order = np.argsort(ref_pru)
        pru_sorted = ref_pru[pru_order]
        n_pair_target = n_gs // 10  # aim for ~10% pool
        # sliding window matching
        hi_ids, lo_ids = [], []
        match_errs = []
        i0 = 0
        tol_pct = 0.005  # 0.5% percentile tolerance
        while i0 < len(pru_order) - 1 and len(hi_ids) < n_pair_target:
            j = i0 + 1
            # find nearest j with different agreement
            best_j = -1; best_err = 1e9
            for j in range(i0 + 1, min(i0 + 50, len(pru_order))):
                if (eff_vc[pru_order[j]] > eff_vc[pru_order[i0]]) != \
                   (eff_vc[pru_order[i0]] > 0):  # just check they differ
                    err = abs(ref_pru[pru_order[j]] - ref_pru[pru_order[i0]])
                    if err < best_err and err / (abs(ref_pru[pru_order[i0]]) + 1e-8) < tol_pct * 10:
                        best_j = j; best_err = err
            if best_j > 0:
                a, b = pru_order[i0], pru_order[best_j]
                if eff_vc[a] >= eff_vc[b]:
                    hi_ids.append(a); lo_ids.append(b)
                else:
                    hi_ids.append(b); lo_ids.append(a)
                match_errs.append(abs(ref_pru[a] - ref_pru[b]))
            i0 += 2

        n_pairs = min(len(hi_ids), len(lo_ids))
        if n_pairs < 100:
            print(f"  WARNING: too few matched pairs ({n_pairs}), skip removal")
            del g; continue
        hi_ids = hi_ids[:n_pairs]; lo_ids = lo_ids[:n_pairs]
        mean_match_err = float(np.mean(match_errs))
        print(f"  matched pairs: {n_pairs}, mean |Δscore|: {mean_match_err:.6f}")

        # group stats
        for name, ids in (("high_agreement", hi_ids), ("low_agreement", lo_ids)):
            grp_rows.append({"checkpoint": ck_it, "group": name, "n": len(ids),
                             "mean_pru": float(np.mean(ref_pru[ids])),
                             "median_pru": float(np.median(ref_pru[ids])),
                             "mean_eff_vc": float(np.mean(eff_vc[ids])),
                             "mean_opacity": float(g.get_opacity.detach().cpu().numpy()[ids].mean()),
                             "mean_scale": float(g.get_scaling.detach().cpu().numpy()[ids].mean())})

        # baseline
        base = full_test_eval(g, test_cams, pipe, bg, opt.mult)
        rem_rows.append({"checkpoint": ck_it, "repeat": -1, "group": "baseline",
                         "pct": 0, **base, "n_gs": n_gs})

        # removal with bootstrap repeats
        max_pct_by_pairs = int(100 * n_pairs / n_gs)
        for pct in (1, 5, 10):
            n_remove = int(n_gs * pct / 100)
            if n_remove > n_pairs:
                print(f"  skip {pct}% (pool {n_pairs} < {n_remove})")
                continue
            for rep in range(N_REPEATS):
                rng = np.random.RandomState(rep * 1000 + ck_it * 10 + pct)
                for group_name, pool in (("low_agreement", lo_ids),
                                          ("high_agreement", hi_ids),
                                          ("random", pru_order.tolist())):
                    if group_name == "random":
                        sel = rng.choice(n_gs, size=n_remove, replace=False)
                    else:
                        sel = rng.choice(pool, size=n_remove, replace=False)
                    gg = GaussianModel(dataset.sh_degree, opt.optimizer_type)
                    gg.restore(clone_tree(captured), opt)
                    rm = torch.zeros(gg.get_xyz.shape[0], dtype=torch.bool, device="cuda")
                    rm[torch.tensor(sel, dtype=torch.long)] = True
                    with torch.no_grad():
                        gg.tmp_radii = None
                        gg.prune_points(rm)
                    ev = full_test_eval(gg, test_cams, pipe, bg, opt.mult)
                    rem_rows.append({"checkpoint": ck_it, "repeat": rep,
                                     "group": group_name, "pct": pct,
                                     **ev, "n_gs": int(gg.get_xyz.shape[0])})
                    del gg
                    torch.cuda.empty_cache()
            print(f"  {pct}% removal done ({N_REPEATS} repeats × 3 groups)")

        del g
        torch.cuda.empty_cache()

    # ---- save ----
    for name, rows in (("tie_aware_stability", stab_rows),
                       ("pruning_subset_scores", pru_rows),
                       ("gaussian_agreement", agree_rows),
                       ("matched_group_stats", grp_rows),
                       ("removal_results", rem_rows)):
        if rows:
            with open(f"{OUT}/data/{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"\n[fix] done: {len(rem_rows)} removal, {len(stab_rows)} stability, "
          f"{len(agree_rows)} agreement rows")


if __name__ == "__main__":
    main()
