#
# Paper B - B13-B Fix: Zero-Evidence Cause Diagnosis
#
# For each F (false-prune) Gaussian from B13-B, classify WHY its 10-view
# evidence is zero:
#   Type-A: truly not visible in sampled views (radii=0 in all 10)
#   Type-B: visible (radii>0) but no error evidence (metric_count=0)
#   Type-C: visible + rendering contribution but no metric evidence
#           (approximated: visible, in a rendered pixel, but metric=0)
#
# Also correct key-view definition: per-view pruning_evidence =
# photometric_loss(view) × accum_metric_counts(view, gaussian)
#
# Uses it30000 checkpoint + B13-B's 20 subsets. No new methods.
#

import os, sys, json, random, csv
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
B13B = "paper_b/b13_b_pruning_boundary"
OUT = "paper_b/b13_b_pruning_boundary/fix"
CKPT_IT = 30000
N_REPEATS = 20
RATIOS = [5, 10, 20]


@torch.no_grad()
def view_diagnostics(g, cam, pipe, bg, opt):
    """Returns per-Gaussian: radii (true visibility), metric_count (error evidence),
    photometric_loss (scalar for this view)."""
    out = render_fastgs(cam, g, pipe, bg, opt.mult)
    radii = out["radii"].clone()  # >0 means this Gaussian participated in rendering

    l1n = get_loss(out["render"], cam.original_image.cuda())
    mmap = (l1n > opt.loss_thresh).int()
    out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                         get_flag=True, metric_map=mmap)
    metric = out2["accum_metric_counts"].clone().float()

    pl = 0.8 * torch.mean(torch.abs(out["render"] - cam.original_image.cuda())) \
         + 0.2 * (1.0 - fast_ssim(out["render"].unsqueeze(0),
                                   cam.original_image.cuda().unsqueeze(0)))
    return radii, metric, float(pl)


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

    captured = torch.load(f"{CKPT_BASE}/it{CKPT_IT}.pt", map_location="cuda")
    n_gs = captured[1].shape[0]
    g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    g.training_setup(opt)
    g._xyz = captured[1].clone(); g._features_dc = captured[2].clone()
    g._features_rest = captured[3].clone(); g._scaling = captured[4].clone()
    g._rotation = captured[5].clone(); g._opacity = captured[6].clone()
    g.active_sh_degree = captured[0]
    print(f"[fix] it{CKPT_IT} (#GS={n_gs})")

    # ---- All-view per-view diagnostics (for oracle + key-view definition) ----
    print("[fix] computing all-view per-view diagnostics...")
    n_all = len(train_cams)
    all_radii = np.zeros((n_all, n_gs), dtype=np.int32)
    all_metric = np.zeros((n_all, n_gs), dtype=np.float32)
    all_pl = np.zeros(n_all)
    for vi, cam in enumerate(train_cams):
        r, m, pl = view_diagnostics(g, cam, pipe, bg, opt)
        all_radii[vi] = r.cpu().numpy()
        all_metric[vi] = m.cpu().numpy()
        all_pl[vi] = pl
    np.save(f"{OUT}/../cache/all_radii.npy", all_radii)
    np.save(f"{OUT}/../cache/all_metric.npy", all_metric)

    # All-view pruning score (from cached pru_all)
    pru_all = np.load(f"{B13B}/cache/pru_all.npy")
    all_order = np.argsort(pru_all)

    # Corrected per-view pruning evidence: pl × metric
    all_pru_evi = all_metric * all_pl[:, None]  # (n_all, n_gs)
    # Top-K key views by pruning evidence
    top1_v = np.argmax(all_pru_evi, axis=0)
    top3_v = np.argsort(-all_pru_evi, axis=0)[:3]
    top5_v = np.argsort(-all_pru_evi, axis=0)[:5]

    # ---- 20 repeats: classify F Gaussians ----
    cat_rows, grp_rows = [], []
    for rep in range(N_REPEATS):
        rng = random.Random(40000 + rep * 13)  # same seeds as B13-B
        cams_10 = rng.sample(train_cams, 10)
        cam_10_idx = [train_cams.index(c) for c in cams_10]

        # per-view diagnostics for the 10 sampled views
        r10 = all_radii[cam_10_idx]  # (10, n_gs)
        m10 = all_metric[cam_10_idx]
        pl10 = all_pl[cam_10_idx]

        # 10-view pruning score
        pru_10_scores = (m10 * pl10[:, None]).sum(axis=0)
        rng_ = pru_10_scores.max() - pru_10_scores.min()
        if rng_ > 1e-8:
            pru_10 = (pru_10_scores - pru_10_scores.min()) / rng_
        else:
            pru_10 = np.zeros(n_gs)
        sub_order = np.argsort(pru_10)

        for pct in RATIOS:
            k = int(n_gs * pct / 100)
            P10 = set(sub_order[:k])
            Pall = set(all_order[:k])
            F = P10 - Pall
            M = Pall - P10
            C = P10 & Pall

            for gname, gset in (("F", F), ("M", M), ("C", C)):
                sample = list(gset)[:2000] if gname == "C" else list(gset)
                for gid in sample:
                    vis_10v = int((r10[:, gid] > 0).sum())
                    met_10v = int((m10[:, gid] > 0).sum())
                    met_sum_10v = float(m10[:, gid].sum())
                    vis_all = int((all_radii[:, gid] > 0).sum())
                    met_all = int((all_metric[:, gid] > 0).sum())

                    # Type classification (only meaningful for F: evidence=0)
                    if gname == "F":
                        if vis_10v == 0:
                            etype = "A_not_visible"
                        elif met_10v == 0:
                            etype = "B_visible_no_evidence"
                        else:
                            etype = "other"
                    else:
                        etype = ""

                    # corrected key-view hit
                    hit_t1 = int(top1_v[gid] in cam_10_idx)
                    hit_t3 = int(any(v in cam_10_idx for v in top3_v[:, gid]))
                    hit_t5 = int(any(v in cam_10_idx for v in top5_v[:, gid]))

                    cat_rows.append({
                        "rep": rep, "ratio": pct, "gaussian_id": int(gid),
                        "group": gname, "evidence_type": etype,
                        "vis_10v": vis_10v, "met_10v": met_10v,
                        "met_sum_10v": met_sum_10v,
                        "vis_all": vis_all, "met_all": met_all,
                        "vis_ratio_10v": vis_10v / 10,
                        "vis_ratio_all": vis_all / n_all,
                        "hit_top1_corr": hit_t1,
                        "hit_top3_corr": hit_t3,
                        "hit_top5_corr": hit_t5,
                        "pru_evi_all_top1": float(all_pru_evi[top1_v[gid], gid]),
                        "pru_evi_all_sum": float(all_pru_evi[:, gid].sum()),
                    })
        print(f"[fix] rep {rep+1}/{N_REPEATS} done")

    with open(f"{OUT}/data/zero_evidence_categories.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()))
        w.writeheader(); w.writerows(cat_rows)
    print(f"[fix] done: {len(cat_rows)} rows")


if __name__ == "__main__":
    main()
