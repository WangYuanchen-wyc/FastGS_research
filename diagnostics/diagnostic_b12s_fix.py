#
# Paper B - B12-S Fix: Nested View Sampling Control
#
# One question: was the it30000 50-view anomaly in B12-S caused by view-count
# or by different random camera sets? Here we use NESTED sampling:
# each permutation defines 10⊂20⊂50⊂100⊂All, so the ONLY variable is
# "how many of the SAME cameras are used".
#
# 20 permutations × 5 view settings × 3 pruning ratios, it30000 only.
# No retraining, no new importance, no view selection method.
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
OUT = "paper_b/b12_s_view_sampling_robustness/fix"
N_PERMS = 20
VIEW_COUNTS = [10, 20, 50, 100]
RATIOS = [5, 10, 20]
CKPT_IT = 30000


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
        full = pl * c if full is None else full + pl * c
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


def light_prune(g, mask):
    """Prune by directly subsetting parameter tensors (no optimizer needed)."""
    keep = ~mask
    g._xyz = g._xyz[keep].detach().requires_grad_(True)
    g._features_dc = g._features_dc[keep].detach().requires_grad_(True)
    g._features_rest = g._features_rest[keep].detach().requires_grad_(True)
    g._opacity = g._opacity[keep].detach().requires_grad_(True)
    g._scaling = g._scaling[keep].detach().requires_grad_(True)
    g._rotation = g._rotation[keep].detach().requires_grad_(True)


def main():
    parser = ArgumentParser("Paper B B12-S Fix: nested view sampling")
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
    print(f"[fix] it{CKPT_IT} (#GS={n_gs}), {n_all} train views")

    g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    g.training_setup(opt)
    g._xyz = captured[1].clone().requires_grad_(True)
    g._features_dc = captured[2].clone().requires_grad_(True)
    g._features_rest = captured[3].clone().requires_grad_(True)
    g._scaling = captured[4].clone().requires_grad_(True)
    g._rotation = captured[5].clone().requires_grad_(True)
    g._opacity = captured[6].clone().requires_grad_(True)
    g.active_sh_degree = captured[0]

    # All-view score (oracle)
    pru_all = compute_pru(g, train_cams, pipe, bg, opt).cpu().numpy()
    all_order = np.argsort(pru_all)

    # All-view pruning
    stab_rows, ov_rows, rem_rows = [], [], []
    for pct in RATIOS:
        n_rm = int(n_gs * pct / 100)
        sel = all_order[:n_rm]
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
        rem_rows.append({"perm": -1, "n_views": -1, "ratio": pct, **ev,
                         "n_gs": int(gg._xyz.shape[0])})
        print(f"[fix] All-view {pct}%: PSNR {ev['psnr']:.4f}")
        del gg; torch.cuda.empty_cache()

    # Nested permutations
    for perm in range(N_PERMS):
        rng = random.Random(30000 + perm * 17)
        perm_cams = rng.sample(train_cams, n_all)  # full permutation

        for nv in VIEW_COUNTS:
            cams = perm_cams[:nv]  # nested: first nv of same permutation
            pru = compute_pru(g, cams, pipe, bg, opt).cpu().numpy()
            sp = spearman_np(pru, pru_all)
            sub_order = np.argsort(pru)

            for k_pct in (5, 10, 20):
                k = max(1, int(n_gs * k_pct / 100))
                ov = len(set(all_order[:k]) & set(sub_order[:k])) / k
                ov_rows.append({"perm": perm, "n_views": nv, "top_k": k_pct,
                                "overlap": ov})
            stab_rows.append({"perm": perm, "n_views": nv, "spearman": sp})

            for pct in RATIOS:
                n_rm = int(n_gs * pct / 100)
                sel = sub_order[:n_rm]
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
                rem_rows.append({"perm": perm, "n_views": nv, "ratio": pct,
                                 **ev, "n_gs": int(gg._xyz.shape[0])})
                del gg; torch.cuda.empty_cache()
        print(f"[fix] perm {perm+1}/{N_PERMS} done")

    for name, rows in (("nested_score_stability", stab_rows),
                       ("nested_candidate_overlap", ov_rows),
                       ("nested_pruning_results", rem_rows)):
        with open(f"{OUT}/data/{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"[fix] done: {len(stab_rows)} stability, {len(ov_rows)} overlap, "
          f"{len(rem_rows)} removal rows")


if __name__ == "__main__":
    main()
