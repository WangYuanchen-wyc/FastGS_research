#
# Paper B - B15-C: Evidence Sufficiency / Decision Confidence Diagnostic
#
# Can FastGS tell from its OWN 10-view evidence whether a pruning decision
# is stable, BEFORE seeing more views?
#
# Method: 20 nested permutations (10⊂20⊂50⊂100⊂All). For each Gaussian in
# the 10-view pruning set, track whether it STAYS pruned as views increase
# (stable) or FLIPS back to keep (unstable). Record 10-view-only confidence
# features and compare stable vs unstable.
#
# No ML predictor, no view selection, no new importance. it30000 only.
#

import os, sys, json, random, time, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.fast_utils import get_loss
from fused_ssim import fused_ssim as fast_ssim
from arguments import ModelParams, PipelineParams, OptimizationParams
from diagnostics.common import install_c_proxy, seed_all

CKPT_BASE = "paper_b/b11_v_multiview_importance_reliability/checkpoints"
OUT = "paper_b/b15_c_evidence_confidence"
CKPT_IT = 30000
N_PERMS = 20
VIEW_COUNTS = [10, 20, 50, 100]
RATIOS = [5, 10, 20]
LOSS_THRESH = 0.10


@torch.no_grad()
def per_view_evidence(g, cam, pipe, bg, opt):
    """Returns (radii bool tensor, per-Gaussian evidence = pl * metric_count)."""
    out = render_fastgs(cam, g, pipe, bg, opt.mult)
    radii = out["radii"] > 0
    l1_map = get_loss(out["render"], cam.original_image.cuda())
    pl = 0.8 * torch.mean(torch.abs(out["render"] - cam.original_image.cuda())) \
         + 0.2 * (1.0 - fast_ssim(out["render"].unsqueeze(0),
                                   cam.original_image.cuda().unsqueeze(0)))
    mmap = (l1_map > LOSS_THRESH).int()
    out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                         get_flag=True, metric_map=mmap)
    return radii, pl * out2["accum_metric_counts"].float()


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
    n_all = len(train_cams)

    captured = torch.load(f"{CKPT_BASE}/it{CKPT_IT}.pt", map_location="cuda")
    n_gs = captured[1].shape[0]
    print(f"[b15c] it{CKPT_IT} (#GS={n_gs}), {n_all} views")

    g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    g.training_setup(opt)
    g._xyz = captured[1].clone(); g._features_dc = captured[2].clone()
    g._features_rest = captured[3].clone(); g._scaling = captured[4].clone()
    g._rotation = captured[5].clone(); g._opacity = captured[6].clone()
    g.active_sh_degree = captured[0]

    # ---- Pre-compute per-view evidence for ALL views (threshold=0.10) ----
    print("[b15c] computing per-view evidence for all views...")
    all_evi = np.zeros((n_all, n_gs), dtype=np.float32)
    all_radii = np.zeros((n_all, n_gs), dtype=bool)
    for vi, cam in enumerate(train_cams):
        r, e = per_view_evidence(g, cam, pipe, bg, opt)
        all_radii[vi] = r.cpu().numpy()
        all_evi[vi] = e.cpu().numpy()

    # All-view score
    pru_all = all_evi.sum(axis=0)
    all_order = np.argsort(pru_all)
    all_pct = np.argsort(np.argsort(pru_all)) / n_gs

    conf_rows = []

    for perm in range(N_PERMS):
        rng = random.Random(50000 + perm * 17)
        perm_idx = rng.sample(range(n_all), n_all)

        # nested subsets
        scores_by_nv = {}
        for nv in VIEW_COUNTS + [n_all]:
            idx = perm_idx[:nv]
            sc = all_evi[idx].sum(axis=0)
            scores_by_nv[nv] = sc

        for pct in RATIOS:
            k = int(n_gs * pct / 100)
            P10 = set(np.argsort(scores_by_nv[10])[:k].tolist())
            Pall = set(np.argsort(scores_by_nv[n_all])[:k].tolist())

            # confidence features (10-view only) — VECTORIZED
            idx10 = perm_idx[:10]
            evi10 = all_evi[idx10]  # (10, n_gs)
            rad10 = all_radii[idx10]
            sc10 = scores_by_nv[10]
            # pre-compute rank percentile once (was inside loop = bottleneck)
            pct10_all = np.argsort(np.argsort(sc10)) / n_gs
            threshold_pct = pct / 100.0
            # pre-compute per-Gaussian stats (vectorized)
            gid_arr = np.array(list(P10))
            e_sub = evi10[:, gid_arr]  # (10, |P10|)
            r_sub = rad10[:, gid_arr]
            total_evi = e_sub.sum(axis=0)
            evi_views = (e_sub > 0).sum(axis=0)
            vis_views = r_sub.sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                max_frac = np.where(total_evi > 0,
                                     e_sub.max(axis=0) / (total_evi + 1e-30), 0.0)
                e_mean = e_sub.mean(axis=0)
                e_std = e_sub.std(axis=0, ddof=1) if e_sub.shape[0] > 1 else np.zeros_like(e_mean)
                e_cv = np.where(np.abs(e_mean) > 1e-8, e_std / (np.abs(e_mean) + 1e-8), 0.0)
            pct10_sub = pct10_all[gid_arr]
            dist_boundary = pct10_sub - threshold_pct
            stable_arr = np.array([g in Pall for g in gid_arr])
            # intermediate stability — hoist sets out of loop (was O(n²))
            Pnv_sets = {nv: set(np.argsort(scores_by_nv[nv])[:k].tolist())
                        for nv in VIEW_COUNTS + [n_all]}
            stays_arr = np.array([
                all(g in Pnv_sets[nv] for nv in Pnv_sets)
                for g in gid_arr])

            for j in range(len(gid_arr)):
                conf_rows.append({
                    "perm": perm, "ratio": pct, "gaussian_id": int(gid_arr[j]),
                    "stable_at_all": bool(stable_arr[j]),
                    "stable_throughout": bool(stays_arr[j]),
                    "vis_views_10v": int(vis_views[j]),
                    "evi_views_10v": int(evi_views[j]),
                    "total_evidence": float(total_evi[j]),
                    "max_view_fraction": float(max_frac[j]),
                    "evidence_mean": float(e_mean[j]),
                    "evidence_std": float(e_std[j]),
                    "evidence_cv": float(e_cv[j]),
                    "pct_rank_10v": float(pct10_sub[j]),
                    "dist_to_boundary": float(dist_boundary[j])})

            # simple confidence signal: split at median of dist_to_boundary
            # (done in analysis)
        print(f"[b15c] perm {perm+1}/{N_PERMS} done", flush=True)
        # incremental save after each perm
        with open(f"{OUT}/data/confidence_features.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(conf_rows[0].keys()))
            w.writeheader(); w.writerows(conf_rows)

    print(f"[b15c] done: {len(conf_rows)} conf rows", flush=True)


if __name__ == "__main__":
    main()
