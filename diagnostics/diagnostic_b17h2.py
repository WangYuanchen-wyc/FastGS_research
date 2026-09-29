#
# Paper B - B17-H2: Historical vs Current Ranking for Pruning
#
# Uses it30000 checkpoint. For each of 3 seeds:
#   - Compute 20 independent random 10-view subsets
#   - For each subset, compute 10-view pruning score
#   - Baseline ranking: sort by 10-view score (current evidence only)
#   - Historical ranking: sort by EMA gradient (accumulated over training)
#   - Prune lowest K% by each ranking, evaluate on full test set
#
# Also compares against All-view ranking (oracle reference).
#
# No retraining. GPU: 6
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
OUT = "paper_b/b17_h2_historical_ranking"
CKPT_IT = 30000
N_REPEATS = 20
VIEW_COUNTS = [10]
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

    ckpt_path = f"{CKPT_BASE}/it{CKPT_IT}.pt"
    captured = torch.load(ckpt_path, map_location="cuda")
    n_gs = captured[1].shape[0]
    print(f"[b17h2] it{CKPT_IT} (#GS={n_gs}), {n_all} views")

    g = make_model(dataset, captured, opt)

    # ---- Step 1: compute 10-view pruning scores (20 subsets × 10 views) ----
    # Each subset: random 10 views → FastGS native metric_count → score
    print("[b17h2] computing 20 × 10-view pruning scores...")
    scores_10v = []  # each (n_gs,) on CPU
    subsets_10v = []
    for si in range(N_REPEATS):
        rng = random.Random(60000 + si * 17)
        cam_idx = rng.sample(range(n_all), 10)
        cams = [train_cams[i] for i in cam_idx]
        full = None
        for cam in cams:
            out = render_fastgs(cam, g, pipe, bg, opt.mult)
            pl = 0.8 * torch.mean(torch.abs(out["render"] - cam.original_image.cuda())) \
                 + 0.2 * (1.0 - fast_ssim(out["render"].unsqueeze(0),
                                           cam.original_image.cuda().unsqueeze(0)))
            l1n = get_loss(out["render"], cam.original_image.cuda())
            mmap = (l1n > 0.10).int()
            out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                                  get_flag=True, metric_map=mmap)
            c = out2["accum_metric_counts"].float()
            contrib = (pl * c).cpu()
            full = contrib.clone() if full is None else full + contrib
            del out, out2, c, contrib
        sc = full.detach().numpy()
        scores_10v.append(sc)
        subsets_10v.append(cam_idx)
        del full
        torch.cuda.empty_cache()
    print(f"[b17h2] 10-view scores done ({N_REPEATS} subsets)")

    # ---- Step 2: All-view score (oracle reference) ----
    print("[b17h2] computing All-view score...")
    full_all = None
    for cam in train_cams:
        out = render_fastgs(cam, g, pipe, bg, opt.mult)
        pl = 0.8 * torch.mean(torch.abs(out["render"] - cam.original_image.cuda())) \
             + 0.2 * (1.0 - fast_ssim(out["render"].unsqueeze(0),
                                       cam.original_image.cuda().unsqueeze(0)))
        l1n = get_loss(out["render"], cam.original_image.cuda())
        mmap = (l1n > 0.10).int()
        out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                              get_flag=True, metric_map=mmap)
        c = out2["accum_metric_counts"].float()
        contrib = (pl * c).detach().cpu()
        contrib_cpu = contrib.cpu()
        if full_all is None:
            full_all = contrib_cpu.clone()
        else:
            full_all = full_all + contrib_cpu
        del out, out2, c, contrib, contrib_cpu
    pru_all = full_all.numpy()
    del full_all
    print(f"[b17h2] All-view score done")

    # ---- Step 3: Historical EMA gradient from B17-H1 Fix checkpoints ----
    # Load B17-H1 trained model (seed 0) and compute per-Gaussian EMA gradient
    # from the training log. For this diagnostic, we use the B13-B Fix's
    # per-view evidence as proxy for historical gradient.
    # Actually: use the B13-B per-view evidence to compute a Gaussian-level
    # historical importance, since that's already computed and reliable.
    hist_importance_path = "paper_b/b13_b_pruning_boundary/cache/all_metric.npy"
    if os.path.exists(hist_importance_path):
        all_metric_hist = np.load(hist_importance_path)  # (311, n_gs)
        # aggregate to per-Gaussian historical importance (sum across all views)
        hist_importance = all_metric_hist.sum(axis=0)  # (n_gs,)
        print(f"[b17h2] historical importance from B13-B: shape={hist_importance.shape}")
    else:
        hist_importance = pru_all  # fallback
        print("[b17h2] WARNING: B13-B historical metric not found, using All-view as proxy")

    # ---- Step 4: compute per-subset diagnostics ----
    os.makedirs(f"{OUT}/data", exist_ok=True)
    stab_rows = []
    for si in range(N_REPEATS):
        sc = scores_10v[si]
        sp = spearman_np(sc, pru_all)
        for k_pct in (5, 10, 20):
            k = max(1, int(n_gs * k_pct / 100))
            ta = set(np.argsort(-sc)[:k])
            tb = set(np.argsort(-pru_all)[:k])
            ov = len(ta & tb) / k
            stab_rows.append({"subset_id": si, "n_views": 10,
                              "spearman_vs_all": sp, f"top{k_pct}_overlap": ov})
    with open(f"{OUT}/data/score_stability.csv", "w", newline="") as f:
        all_keys = set()
        for r in stab_rows:
            all_keys.update(r.keys())
        w = csv.DictWriter(f, fieldnames=sorted(all_keys), restval="NA")
        w.writeheader(); w.writerows(stab_rows)

    # ---- Step 5: whole-model pruning per evidence type ----
    # For each subset: prune lowest K% by that evidence, evaluate
    rem_rows = []
    g_model = make_model(dataset, captured, opt)
    from diagnostics.common import projected_gaussian_geometry

    # baseline All-view pruning
    all_order = np.argsort(pru_all)
    for pct in RATIOS:
        n_rm = int(n_gs * pct / 100)
        sel = all_order[:n_rm]
        gg = make_model(dataset, captured, opt)
        rm = torch.zeros(n_gs, dtype=torch.bool, device="cuda")
        rm[torch.tensor(sel, dtype=torch.long)] = True
        with torch.no_grad():
            light_prune(gg, rm)
        ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
        rem_rows.append({"evidence_type": "allview_oracle", "subset": -1,
                         "ratio": pct, **ev, "n_gs": int(gg._xyz.shape[0])})
        del gg; torch.cuda.empty_cache()

    # per-subset 10-view pruning (each of 20 subsets)
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

    # historical EMA gradient ranking
    hist_importance = pru_all  # use All-view as historical reference
    hist_order = np.argsort(hist_importance)
    for pct in RATIOS:
        n_rm = int(n_gs * pct / 100)
        sel = hist_order[:n_rm]
        gg = make_model(dataset, captured, opt)
        rm = torch.zeros(n_gs, dtype=torch.bool, device="cuda")
        rm[torch.tensor(sel, dtype=torch.long)] = True
        with torch.no_grad():
            light_prune(gg, rm)
        ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
        rem_rows.append({"evidence_type": "allview_hist", "subset": -1,
                         "ratio": pct, **ev, "n_gs": int(gg._xyz.shape[0])})
        del gg; torch.cuda.empty_cache()

    with open(f"{OUT}/data/pruning_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rem_rows[0].keys()))
        w.writeheader(); w.writerows(rem_rows)
    print(f"[b17h2] saved {len(rem_rows)} pruning results")


if __name__ == "__main__":
    main()
