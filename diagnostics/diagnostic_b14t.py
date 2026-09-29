#
# Paper B - B14-T: Error Threshold Diagnostic
#
# Does FastGS's hard error threshold (loss_thresh=0.10) cause zero-evidence
# for important Gaussians under limited-view sampling?
#
# Sweep: 0.025 / 0.05 / 0.10(baseline) / 0.15 / 0.20
# For each threshold × 20 B13-B subsets:
#   - visible-but-zero-evidence ratio (using radii>0 true visibility)
#   - Spearman + Top-10% overlap vs All-view
#   - false-prune count (|P_10 \ P_all|)
#   - whole-model pruning @5%/10%/20% + test-set eval
#
# No retraining, no method design. it30000 only.
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
OUT = "paper_b/b14_t_error_threshold"
CKPT_IT = 30000
N_REPEATS = 20
RATIOS = [5, 10, 20]
THRESHOLDS = [0.025, 0.05, 0.10, 0.15, 0.20]


@torch.no_grad()
def per_view_data(g, cam, pipe, bg, opt):
    """Returns radii (true vis), l1_map (per-pixel L1), photometric loss scalar."""
    out = render_fastgs(cam, g, pipe, bg, opt.mult)
    radii = out["radii"].clone()
    l1_map = get_loss(out["render"], cam.original_image.cuda())
    pl = 0.8 * torch.mean(torch.abs(out["render"] - cam.original_image.cuda())) \
         + 0.2 * (1.0 - fast_ssim(out["render"].unsqueeze(0),
                                   cam.original_image.cuda().unsqueeze(0)))
    return radii, l1_map, float(pl)


@torch.no_grad()
def metric_counts_at_thresh(g, cam, l1_map, thresh, pipe, bg, opt):
    """Re-render with a custom threshold on the pre-computed l1_map."""
    mmap = (l1_map > thresh).int()
    out = render_fastgs(cam, g, pipe, bg, opt.mult,
                        get_flag=True, metric_map=mmap)
    return out["accum_metric_counts"].clone().float()


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
    print(f"[b14t] it{CKPT_IT} (#GS={n_gs}), {n_all} views, {len(THRESHOLDS)} thresholds")

    g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    g.training_setup(opt)
    g._xyz = captured[1].clone(); g._features_dc = captured[2].clone()
    g._features_rest = captured[3].clone(); g._scaling = captured[4].clone()
    g._rotation = captured[5].clone(); g._opacity = captured[6].clone()
    g.active_sh_degree = captured[0]

    # Pre-compute per-view l1 maps and radii for all 311 training views
    print("[b14t] computing per-view l1_maps + radii for all views...")
    all_radii = np.zeros((n_all, n_gs), dtype=np.int32)
    all_l1maps = []  # keep on GPU, list of (H,W) tensors
    all_pl = np.zeros(n_all)
    for vi, cam in enumerate(train_cams):
        r, l1m, pl = per_view_data(g, cam, pipe, bg, opt)
        all_radii[vi] = r.cpu().numpy()
        all_l1maps.append(l1m)
        all_pl[vi] = pl

    # All-view pruning score per threshold
    print("[b14t] computing All-view scores per threshold...")
    pru_all_by_thresh = {}
    for th in THRESHOLDS:
        full = None
        for vi, cam in enumerate(train_cams):
            mc = metric_counts_at_thresh(g, cam, all_l1maps[vi], th, pipe, bg, opt)
            full = all_pl[vi] * mc if full is None else full + all_pl[vi] * mc
        rng = full.max() - full.min()
        pru = torch.zeros_like(full) if rng < 1e-8 else (full - full.min()) / rng
        pru_all_by_thresh[th] = pru.cpu().numpy()
    # use baseline threshold (0.10) as All-view ranking reference
    pru_all_ref = pru_all_by_thresh[0.10]
    all_order_ref = np.argsort(pru_all_ref)

    # All-view pruning eval (baseline threshold only, as reference)
    rem_rows, zero_rows, fp_rows = [], [], []
    for pct in RATIOS:
        sel = all_order_ref[:int(n_gs * pct / 100)]
        gg = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        gg.training_setup(opt)
        gg._xyz = captured[1].clone(); gg._features_dc = captured[2].clone()
        gg._features_rest = captured[3].clone(); gg._scaling = captured[4].clone()
        gg._rotation = captured[5].clone(); gg._opacity = captured[6].clone()
        gg.active_sh_degree = captured[0]
        rm = torch.zeros(n_gs, dtype=torch.bool, device="cuda")
        rm[torch.tensor(sel, dtype=torch.long)] = True
        with torch.no_grad():
            light_prune(gg, rm)
        ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
        rem_rows.append({"threshold": -1, "rep": -1, "ratio": pct, **ev,
                         "n_gs": int(gg._xyz.shape[0])})
        del gg; torch.cuda.empty_cache()

    # Free all-view l1 maps to save GPU memory
    del all_l1maps
    torch.cuda.empty_cache()

    # Main loop: threshold × repeat
    for th in THRESHOLDS:
        print(f"\n[b14t] === threshold {th} ===")
        for rep in range(N_REPEATS):
            rng_r = random.Random(40000 + rep * 13)
            cams_10 = rng_r.sample(train_cams, 10)
            cam_10_idx = [train_cams.index(c) for c in cams_10]

            # re-compute l1 maps for the 10 sampled views
            vis_10 = all_radii[cam_10_idx]  # (10, n_gs)
            full_10 = None
            for ci, cam in enumerate(cams_10):
                _, l1m, pl = per_view_data(g, cam, pipe, bg, opt)
                mc = metric_counts_at_thresh(g, cam, l1m, th, pipe, bg, opt)
                full_10 = pl * mc if full_10 is None else full_10 + pl * mc
            rng_ = full_10.max() - full_10.min()
            pru_10 = torch.zeros_like(full_10) if rng_ < 1e-8 else \
                     (full_10 - full_10.min()) / rng_
            pru_10_np = pru_10.cpu().numpy()
            sub_order = np.argsort(pru_10_np)

            # zero-evidence: visible (radii>0) but metric=0
            met_10 = (pru_10_np < 1e-12)  # score=0 means no evidence
            vis_mask = vis_10 > 0
            vis_zero_evi = (vis_mask.any(axis=0) & met_10).sum()
            vis_any = vis_mask.any(axis=0).sum()

            sp = spearman_np(pru_10_np, pru_all_ref)
            k10 = max(1, int(n_gs * 0.10))
            ov10 = len(set(sub_order[:k10]) & set(all_order_ref[:k10])) / k10

            zero_rows.append({"threshold": th, "rep": rep,
                              "visible_any": int(vis_any),
                              "visible_zero_evi": int(vis_zero_evi),
                              "zero_evi_ratio": vis_zero_evi / max(vis_any, 1),
                              "spearman": sp, "top10_overlap": ov10})

            for pct in RATIOS:
                k = int(n_gs * pct / 100)
                P10 = set(sub_order[:k])
                Pall = set(all_order_ref[:k])
                F = len(P10 - Pall)
                fp_rows.append({"threshold": th, "rep": rep, "ratio": pct,
                                "false_prune_count": F})

                # pruning eval
                gg = GaussianModel(dataset.sh_degree, opt.optimizer_type)
                gg.training_setup(opt)
                gg._xyz = captured[1].clone(); gg._features_dc = captured[2].clone()
                gg._features_rest = captured[3].clone(); gg._scaling = captured[4].clone()
                gg._rotation = captured[5].clone(); gg._opacity = captured[6].clone()
                gg.active_sh_degree = captured[0]
                rm = torch.zeros(n_gs, dtype=torch.bool, device="cuda")
                rm[torch.tensor(list(P10), dtype=torch.long)] = True
                with torch.no_grad():
                    light_prune(gg, rm)
                ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
                rem_rows.append({"threshold": th, "rep": rep, "ratio": pct,
                                 **ev, "n_gs": int(gg._xyz.shape[0])})
                del gg; torch.cuda.empty_cache()
        print(f"  threshold {th} done ({N_REPEATS} reps)")

    for name, rows in (("threshold_zero_evidence", zero_rows),
                       ("threshold_false_prune", fp_rows),
                       ("threshold_pruning_results", rem_rows)):
        with open(f"{OUT}/data/{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"\n[b14t] done: {len(zero_rows)} zero, {len(fp_rows)} fp, {len(rem_rows)} rem")


if __name__ == "__main__":
    main()
