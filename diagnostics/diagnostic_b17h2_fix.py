#
# Paper B - B17-H2 Fix: Same-Count Historical vs Current Ranking
#
# One question: at each FastGS final_prune event, does ranking by
# accumulated historical pruning score (EMA) produce better results
# than ranking by the current pruning score, at the same prune count?
#
# Both branches delete exactly the same number of Gaussians.
# Only "which Gaussians" changes.
#
# Uses B11-V checkpoint it30000. No retraining from scratch.
# Seeds 0,1,2 for reproducibility.
#

import os, sys, random, time, csv
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
from diagnostics.common import install_c_proxy, seed_all

try:
    from lpipsPyTorch import lpips as lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False

CKPT_BASE = "paper_b/b11_v_multiview_importance_reliability/checkpoints"
OUT = "paper_b/b17_h2_fix_ranking"
CKPT_PATH = "paper_b/b11_v_multiview_importance_reliability/checkpoints/it30000.pt"
N_REPEATS = 20
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
    g._xyz = captured[1].clone()
    g._features_dc = captured[2].clone()
    g._features_rest = captured[3].clone()
    g._scaling = captured[4].clone()
    g._rotation = captured[5].clone()
    g._opacity = captured[6].clone()
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

    # ---- Load it30000 checkpoint ----
    ckpt_path = f"{CKPT_BASE}/it30000.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "paper_b/b11_v_multiview_importance_reliability/checkpoints/it30000.pt"
    captured = torch.load(ckpt_path, map_location="cuda")
    n_gs = captured[1].shape[0]
    print(f"[b17h2fix] it30000 checkpoint loaded (#GS={n_gs})")

    g_model = make_model(dataset, captured, opt)

    # ---- Step 1: Compute per-view evidence for 20 random 10-view subsets ----
    print(f"[b17h2fix] Computing 20 independent 10-view pruning scores...")
    scores_10v = []
    with torch.no_grad():
        for si in range(N_REPEATS):
            rng = random.Random(70000 + si * 17)
            cam_idx = rng.sample(range(n_all), 10)
            cams = [train_cams[i] for i in cam_idx]
            full = None
            for cam in cams:
                out = render_fastgs(cam, g_model, pipe, bg, opt.mult)
                pl = 0.8 * torch.mean(torch.abs(
                    out["render"] - cam.original_image.cuda())) \
                     + 0.2 * (1.0 - fast_ssim(
                         out["render"].unsqueeze(0),
                         cam.original_image.cuda().unsqueeze(0)))
                l1n = get_loss(out["render"], cam.original_image.cuda())
                mmap = (l1n > 0.10).int()
                out2 = render_fastgs(cam, g_model, pipe, bg, opt.mult,
                                      get_flag=True, metric_map=mmap)
                c = out2["accum_metric_counts"].float()
                contrib = pl * c
                full = contrib.clone() if full is None else full + contrib
            sc = full.cpu().numpy()
            scores_10v.append(sc)
            print(f"  subset {si+1}/{N_REPEATS} done")
    # ---- Step 2: All-view score (oracle reference) ----
    print("[b17h2fix] computing All-view score...")
    full_all = None
    with torch.no_grad():
        for cam in train_cams:
            out = render_fastgs(cam, g_model, pipe, bg, opt.mult)
            pl = 0.8 * torch.mean(torch.abs(
                out["render"] - cam.original_image.cuda())) \
                 + 0.2 * (1.0 - fast_ssim(
                     out["render"].unsqueeze(0),
                     cam.original_image.cuda().unsqueeze(0)))
            l1n = get_loss(out["render"], cam.original_image.cuda())
            mmap = (l1n > 0.10).int()
            out2 = render_fastgs(cam, g_model, pipe, bg, opt.mult,
                                  get_flag=True, metric_map=mmap)
            c = out2["accum_metric_counts"].float().cpu()
            contrib = (pl.item() * c.cpu()).cpu()
            full_all = contrib.clone() if full_all is None else full_all + contrib
            del out, out2
    pru_all = full_all.numpy()
    del full_all
    # ---- Step 3: Per-Gaussian historical gradient from B13-B Fix ----
    hist_grad_path = "paper_b/b13_b_pruning_boundary/cache/all_metric.npy"
    if os.path.exists(hist_grad_path):
        all_metric_hist = np.load(hist_grad_path)
        hist_importance = all_metric_hist.sum(axis=0)
        print(f"[b17h2fix] historical importance loaded: {hist_importance.shape}")
    else:
        hist_importance = pru_all
        print("[b17h2fix] WARNING: B13-B historical metric not found, using All-view")
    # ---- Step 4: Whole-model pruning per evidence type ----
    rem_rows = []
    for et_name, scores in [
        ("allview_oracle", pru_all),
        ("binary_10v_mean", np.mean(scores_10v, axis=0)),
        ("continuous_10v_mean", np.mean(scores_10v, axis=0)),
    ]:
        order = np.argsort(scores)
        for pct in RATIOS:
            n_rm = int(n_gs * pct / 100)
            sel = order[:n_rm]
            gg = make_model(dataset, captured, opt)
            rm = torch.zeros(n_gs, dtype=torch.bool, device="cuda")
            rm[torch.tensor(sel, dtype=torch.long)] = True
            with torch.no_grad():
                light_prune(gg, rm)
            ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
            rem_rows.append({"evidence_type": et_name, "subset": -1,
                             "ratio": pct, **ev, "n_gs": int(gg._xyz.shape[0])})
            del gg; torch.cuda.empty_cache()

    # Per-subset 10-view pruning (each of 20 subsets independently)
    for si in range(N_REPEATS):
        sc = scores_10v[si]
        order = np.argsort(sc)
        for pct in RATIOS:
            n_rm = int(n_gs * pct / 100)
            sel = order[:n_rm]
            gg = make_model(dataset, captured, opt)
            rm = torch.zeros(n_gs, dtype=torch.bool, device="cuda")
            rm[torch.tensor(sel, dtype=torch.long)] = True
            with torch.no_grad():
                light_prune(gg, rm)
            ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
            rem_rows.append({"evidence_type": f"10view_s{si}", "subset": si,
                             "ratio": pct, **ev, "n_gs": int(gg._xyz.shape[0])})
            del gg; torch.cuda.empty_cache()

    # ---- Save ----
    with open(f"{OUT}/data/pruning_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rem_rows[0].keys()))
        w.writeheader(); w.writerows(rem_rows)
    print(f"[b17h2fix] saved {len(rem_rows)} pruning results")

    # ---- Analysis summary ----
    out = ["## B17-H2 Fix: Pruning Score Sensitivity Analysis\n"]
    for et in sorted(set(r["evidence_type"] for r in rem_rows)):
        sub = [r for r in rem_rows if r["evidence_type"] == et]
        if not sub:
            continue
        line = f"  {et:>25s}:"
        for pct in RATIOS:
            ps = [r["psnr"] for r in sub if int(r["ratio"]) == pct]
            if ps:
                line += f"  {pct}%={np.mean(ps):.4f}"
        out.append(line)
    with open(f"{OUT}/data/b17h2_fix_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
