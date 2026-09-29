#
# Paper B - B14-T Fix: Same-Threshold All-view Control
#
# Correction: for each threshold τ, compare 10-view@τ vs All-view@τ (not vs
# a fixed baseline). Separates "is the threshold good" from "is 10-view
# estimation accurate". Also fixes zero-evidence to use raw evidence (not
# normalized score) and radii>0 without int truncation.
#
# Reuses it30000 checkpoint + B14-T's 20 subsets. No retraining.
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
from diagnostics.common import install_c_proxy, seed_all

try:
    from lpipsPyTorch import lpips as lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False

CKPT_BASE = "paper_b/b11_v_multiview_importance_reliability/checkpoints"
OUT = "paper_b/b14_t_error_threshold/fix"
CKPT_IT = 30000
N_REPEATS = 20
RATIOS = [5, 10, 20]
THRESHOLDS = [0.025, 0.05, 0.10, 0.15, 0.20]


@torch.no_grad()
def per_view(g, cam, pipe, bg, opt, thresh):
    """Single pass: radii (true vis), raw evidence (pl × metric_count@thresh)."""
    out = render_fastgs(cam, g, pipe, bg, opt.mult)
    radii = out["radii"]  # int32 on GPU; >0 means rendered
    l1_map = get_loss(out["render"], cam.original_image.cuda())
    pl = 0.8 * torch.mean(torch.abs(out["render"] - cam.original_image.cuda())) \
         + 0.2 * (1.0 - fast_ssim(out["render"].unsqueeze(0),
                                   cam.original_image.cuda().unsqueeze(0)))
    mmap = (l1_map > thresh).int()
    out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                          get_flag=True, metric_map=mmap)
    mc = out2["accum_metric_counts"].float()
    return radii, pl * mc  # raw per-Gaussian evidence for this view


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


def light_prune(g, mask):
    keep = ~mask
    g._xyz = g._xyz[keep].detach().requires_grad_(True)
    g._features_dc = g._features_dc[keep].detach().requires_grad_(True)
    g._features_rest = g._features_rest[keep].detach().requires_grad_(True)
    g._opacity = g._opacity[keep].detach().requires_grad_(True)
    g._scaling = g._scaling[keep].detach().requires_grad_(True)
    g._rotation = g._rotation[keep].detach().requires_grad_(True)


def make_model(dataset, captured, opt):
    g = GaussianModel(dataset.sh_degree, "default")
    g.training_setup(opt)
    g._xyz = captured[1].clone(); g._features_dc = captured[2].clone()
    g._features_rest = captured[3].clone(); g._scaling = captured[4].clone()
    g._rotation = captured[5].clone(); g._opacity = captured[6].clone()
    g.active_sh_degree = captured[0]
    return g


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    install_c_proxy()
    seed_all(0)
    g0 = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, g0)
    del g0
    train_cams = list(scene.getTrainCameras())
    test_cams = scene.getTestCameras()[:20] if scene.getTestCameras() else train_cams[:10]
    n_all = len(train_cams)

    captured = torch.load(f"{CKPT_BASE}/it{CKPT_IT}.pt", map_location="cuda")
    n_gs = captured[1].shape[0]

    g = make_model(dataset, captured, opt)

    # ---- A) All-view pruning per threshold ----
    print("[fix] computing All-view scores per threshold...")
    pru_all_by_th = {}
    all_radii_by_th = {}  # for zero-evidence: raw radii (independent of threshold)
    # radii is threshold-independent; compute once
    all_radii = np.zeros((n_all, n_gs), dtype=bool)
    for th in THRESHOLDS:
        full = None
        for vi, cam in enumerate(train_cams):
            r, evi = per_view(g, cam, pipe, bg, opt, th)
            all_radii[vi] = r.cpu().numpy() > 0
            full = evi if full is None else full + evi
        pru_all_by_th[th] = full.cpu().numpy()  # RAW (un-normalized)
        print(f"  th={th}: raw score range [{pru_all_by_th[th].min():.6f}, "
              f"{pru_all_by_th[th].max():.6f}]")

    # All-view pruning eval per threshold
    rem_rows, stab_rows, fp_rows, zero_rows = [], [], [], []
    for th in THRESHOLDS:
        all_order = np.argsort(pru_all_by_th[th])
        # raw zero-evidence: raw score == 0 AND truly visible (radii>0)
        raw_zero = (pru_all_by_th[th] == 0)
        vis_any = all_radii.any(axis=0)
        vis_zero_evi = (vis_any & raw_zero).sum()
        zero_rows.append({"threshold": th, "rep": -1,
                          "visible_any": int(vis_any.sum()),
                          "visible_zero_evi": int(vis_zero_evi),
                          "zero_evi_ratio": vis_zero_evi / max(vis_any.sum(), 1),
                          "spearman": 1.0, "top10_overlap": 1.0})
        for pct in RATIOS:
            sel = all_order[:int(n_gs * pct / 100)]
            gg = make_model(dataset, captured, opt)
            rm = torch.zeros(n_gs, dtype=torch.bool, device="cuda")
            rm[torch.tensor(sel, dtype=torch.long)] = True
            with torch.no_grad():
                light_prune(gg, rm)
            ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
            rem_rows.append({"threshold": th, "rep": -1, "ratio": pct,
                             "source": "allview", **ev, "n_gs": int(gg._xyz.shape[0])})
            del gg; torch.cuda.empty_cache()
        print(f"  th={th}: All-view prune done")

    # ---- B) 10-view subsets per threshold ----
    for th in THRESHOLDS:
        all_order_th = np.argsort(pru_all_by_th[th])
        for rep in range(N_REPEATS):
            rng = random.Random(40000 + rep * 13)
            cams_10 = rng.sample(train_cams, 10)

            full_10 = None
            vis_10 = np.zeros((10, n_gs), dtype=bool)
            for ci, cam in enumerate(cams_10):
                r, evi = per_view(g, cam, pipe, bg, opt, th)
                vis_10[ci] = r.cpu().numpy() > 0
                full_10 = evi if full_10 is None else full_10 + evi
            raw_10 = full_10.cpu().numpy()
            sub_order = np.argsort(raw_10)

            # raw zero-evidence
            raw_zero_10 = (raw_10 == 0)
            vis_any_10 = vis_10.any(axis=0)
            vis_zero_10 = (vis_any_10 & raw_zero_10).sum()

            sp = spearman_np(raw_10, pru_all_by_th[th])
            k10 = max(1, int(n_gs * 0.10))
            ov10 = len(set(sub_order[:k10]) & set(all_order_th[:k10])) / k10

            zero_rows.append({"threshold": th, "rep": rep,
                              "visible_any": int(vis_any_10.sum()),
                              "visible_zero_evi": int(vis_zero_10),
                              "zero_evi_ratio": vis_zero_10 / max(vis_any_10.sum(), 1),
                              "spearman": sp, "top10_overlap": ov10})

            for pct in RATIOS:
                k = int(n_gs * pct / 100)
                P10 = set(sub_order[:k])
                Pall = set(all_order_th[:k])
                F = len(P10 - Pall)
                fp_rows.append({"threshold": th, "rep": rep, "ratio": pct,
                                "false_prune_count": F})

                gg = make_model(dataset, captured, opt)
                rm = torch.zeros(n_gs, dtype=torch.bool, device="cuda")
                rm[torch.tensor(list(P10), dtype=torch.long)] = True
                with torch.no_grad():
                    light_prune(gg, rm)
                ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
                rem_rows.append({"threshold": th, "rep": rep, "ratio": pct,
                                 "source": "10view", **ev, "n_gs": int(gg._xyz.shape[0])})
                del gg; torch.cuda.empty_cache()
        print(f"  th={th}: 10-view done ({N_REPEATS} reps)")

    for name, rows in (("same_threshold_zero_evidence", zero_rows),
                       ("same_threshold_false_prune", fp_rows),
                       ("same_threshold_pruning_results", rem_rows)):
        with open(f"{OUT}/data/{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"\n[fix] done: {len(zero_rows)} zero, {len(fp_rows)} fp, {len(rem_rows)} rem")


if __name__ == "__main__":
    main()
