#
# Paper B - B12-S: View-Sampling Robustness of FastGS Importance
#
# One question: does FastGS pruning importance produce meaningful noise
# across random view subsets, and does this cause wrong pruning?
#
# Uses B11-V checkpoints it10000/15000/30000 (no retraining).
# For each checkpoint: compute All-view pruning score (oracle reference),
# then 10/20/50/100-view scores (10 repeats each), measure convergence
# to All-view, and do whole-model pruning at 5%/10%/20% to compare quality.
#
# No new importance, no recovery, no Split-N. Diagnostic only.
#

import os, sys, json, random, time, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.image_utils import psnr
from fused_ssim import fused_ssim as fast_ssim
from utils.fast_utils import get_loss
from arguments import ModelParams, PipelineParams, OptimizationParams
from diagnostics.common import install_c_proxy, seed_all, clone_tree

try:
    from lpipsPyTorch import lpips as lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False

CKPT_BASE = "paper_b/b11_v_multiview_importance_reliability/checkpoints"
OUT = "paper_b/b12_s_view_sampling_robustness"
CKPT_ITERS = [10000, 15000, 30000]
VIEW_COUNTS = [10, 20, 50, 100]
N_REPEATS = 10
RATIOS = [5, 10, 20]


def rankdata_avg(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j+1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_np(x, y):
    return float(np.corrcoef(rankdata_avg(x), rankdata_avg(y))[0, 1])


@torch.no_grad()
def compute_pru(g, camlist, pipe, bg, opt):
    """FastGS pruning score from a set of views (same math as
    compute_gaussian_score_fastgs DENSIFY=False)."""
    full = None
    for cam in camlist:
        out = render_fastgs(cam, g, pipe, bg, opt.mult)
        pl = 0.8 * torch.mean(torch.abs(out["render"] - cam.original_image.cuda())) \
             + 0.2 * (1.0 - fast_ssim(out["render"].unsqueeze(0),
                                       cam.original_image.cuda().unsqueeze(0)))
        l1n = get_loss(out["render"], cam.original_image.cuda())
        mmap = (l1n > opt.loss_thresh).int()
        out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                             get_flag=True, metric_map=mmap)
        c = out2["accum_metric_counts"].float()
        contrib = pl * c
        full = contrib.clone() if full is None else full + contrib
    rng = full.max() - full.min()
    if rng < 1e-8:
        return torch.zeros_like(full)
    return (full - full.min()) / rng


@torch.no_grad()
def eval_test(g, test_cams, pipe, bg, mult):
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


def light_restore(g, captured):
    """Restore params without optimizer state (sufficient for pruning eval)."""
    g._xyz = captured[1].clone().requires_grad_(True)
    g._features_dc = captured[2].clone().requires_grad_(True)
    g._features_rest = captured[3].clone().requires_grad_(True)
    g._scaling = captured[4].clone().requires_grad_(True)
    g._rotation = captured[5].clone().requires_grad_(True)
    g._opacity = captured[6].clone().requires_grad_(True)
    g.active_sh_degree = captured[0]
    g.max_radii2D = captured[7].clone() if torch.is_tensor(captured[7]) else captured[7]
    g.xyz_gradient_accum = torch.zeros_like(g._xyz[:, :1])
    g.xyz_gradient_accum_abs = torch.zeros_like(g._xyz[:, :1])
    g.denom = torch.zeros_like(g._xyz[:, :1])


def main():
    parser = ArgumentParser("Paper B B12-S: view sampling robustness")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
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
    train_cams = list(scene.getTrainCameras())
    test_cams = scene.getTestCameras()[:20] if scene.getTestCameras() else train_cams[:10]
    n_all_views = len(train_cams)
    print(f"[b12s] train views: {n_all_views}")

    stab_rows, overlap_rows, rem_rows = [], [], []

    for ck_it in CKPT_ITERS:
        ck_path = f"{CKPT_BASE}/it{ck_it}.pt"
        if not os.path.exists(ck_path):
            print(f"[b12s] skip it{ck_it} (no checkpoint)"); continue
        captured = torch.load(ck_path, map_location="cuda")
        n_gs = captured[1].shape[0]
        print(f"\n[b12s] === it{ck_it} (#GS={n_gs}) ===")

        g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        g.training_setup(opt)
        light_restore(g, captured)

        # ---- All-view score (oracle reference) ----
        t0 = time.time()
        pru_all = compute_pru(g, train_cams, pipe, bg, opt).cpu().numpy()
        print(f"  All-view score done ({time.time()-t0:.0f}s)")
        all_order = np.argsort(pru_all)  # ascending; lowest = prune candidates

        # ---- All-view pruning eval ----
        for pct in RATIOS:
            n_rm = int(n_gs * pct / 100)
            sel = all_order[:n_rm]
            gg = GaussianModel(dataset.sh_degree, opt.optimizer_type)
            gg.training_setup(opt)
            light_restore(gg, captured)
            rm = torch.zeros(gg._xyz.shape[0], dtype=torch.bool, device="cuda")
            rm[torch.tensor(sel, dtype=torch.long)] = True
            with torch.no_grad():
                gg._xyz = gg._xyz[~rm].detach().requires_grad_(True)
                gg._features_dc = gg._features_dc[~rm].detach().requires_grad_(True)
                gg._features_rest = gg._features_rest[~rm].detach().requires_grad_(True)
                gg._opacity = gg._opacity[~rm].detach().requires_grad_(True)
                gg._scaling = gg._scaling[~rm].detach().requires_grad_(True)
                gg._rotation = gg._rotation[~rm].detach().requires_grad_(True)
            ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
            rem_rows.append({"checkpoint": ck_it, "n_views": -1, "repeat": -1,
                             "ratio": pct, "spearman_vs_all": 1.0,
                             "top10_overlap": 1.0, **ev,
                             "n_gs": int(gg._xyz.shape[0])})
            print(f"  All-view prune {pct}%: PSNR {ev['psnr']:.3f}")
            del gg; torch.cuda.empty_cache()

        # ---- Subset scores + pruning ----
        for nv in VIEW_COUNTS:
            for rep in range(N_REPEATS):
                rng = random.Random(20000 + ck_it + rep * 7 + nv * 13)
                cams = rng.sample(train_cams, min(nv, n_all_views))
                pru = compute_pru(g, cams, pipe, bg, opt).cpu().numpy()
                sp = spearman_np(pru, pru_all)
                sub_order = np.argsort(pru)

                # Top-K overlap
                for k_pct in (5, 10, 20):
                    k = max(1, int(n_gs * k_pct / 100))
                    ov = len(set(all_order[:k]) & set(sub_order[:k])) / k
                    overlap_rows.append({"checkpoint": ck_it, "n_views": nv,
                                         "repeat": rep, "top_k": k_pct,
                                         "overlap": ov})
                stab_rows.append({"checkpoint": ck_it, "n_views": nv,
                                  "repeat": rep, "spearman": sp})

                # Pruning at each ratio
                for pct in RATIOS:
                    n_rm = int(n_gs * pct / 100)
                    sel = sub_order[:n_rm]
                    gg = GaussianModel(dataset.sh_degree, opt.optimizer_type)
                    gg.training_setup(opt)
                    light_restore(gg, captured)
                    rm = torch.zeros(gg._xyz.shape[0], dtype=torch.bool, device="cuda")
                    rm[torch.tensor(sel, dtype=torch.long)] = True
                    with torch.no_grad():
                        gg._xyz = gg._xyz[~rm].detach().requires_grad_(True)
                        gg._features_dc = gg._features_dc[~rm].detach().requires_grad_(True)
                        gg._features_rest = gg._features_rest[~rm].detach().requires_grad_(True)
                        gg._opacity = gg._opacity[~rm].detach().requires_grad_(True)
                        gg._scaling = gg._scaling[~rm].detach().requires_grad_(True)
                        gg._rotation = gg._rotation[~rm].detach().requires_grad_(True)
                    ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
                    rem_rows.append({"checkpoint": ck_it, "n_views": nv,
                                     "repeat": rep, "ratio": pct,
                                     "spearman_vs_all": sp,
                                     "top10_overlap": next(
                                         r["overlap"] for r in overlap_rows
                                         if r["n_views"] == nv and r["repeat"] == rep
                                         and r["top_k"] == 10),
                                     **ev, "n_gs": int(gg._xyz.shape[0])})
                    del gg; torch.cuda.empty_cache()
            print(f"  {nv}-view done ({N_REPEATS} repeats)")

        del g, captured
        torch.cuda.empty_cache()
        # incremental save
        for name, rows in (("score_stability", stab_rows),
                           ("pruning_candidate_overlap", overlap_rows),
                           ("pruning_results", rem_rows)):
            if rows:
                with open(f"{OUT}/data/{name}.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)

    print(f"\n[b12s] done: {len(stab_rows)} stability, {len(overlap_rows)} overlap, "
          f"{len(rem_rows)} removal rows")


if __name__ == "__main__":
    main()
