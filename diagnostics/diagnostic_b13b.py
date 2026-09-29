#
# Paper B - B13-B: Pruning Boundary Misranking & View Coverage
#
# One question: why does limited-view FastGS pruning score push some
# should-keep Gaussians into the pruning region?
#
# Uses it30000 checkpoint. For each 10-view subset (20 repeats):
#   - compute 10v + Allv pruning scores
#   - derive P_10 / P_all at 5%/10%/20% ratios
#   - define C=P∩P_all, F=P_10\P_all, M=P_all\P_10
#   - whole-model exchange test: delete C+F vs C+M (same count)
#   - per-Gaussian: boundary distance, visibility, angular coverage,
#     key-view hit rate, footprint
#   - persistent false-prune frequency across 20 repeats
#
# No view-selection method, no new importance, no recovery. Diagnostic only.
#

import os, sys, json, random, time, csv, math
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
from diagnostics.common import install_c_proxy, seed_all, project_to_pixel

try:
    from lpipsPyTorch import lpips as lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False

CKPT_BASE = "paper_b/b11_v_multiview_importance_reliability/checkpoints"
OUT = "paper_b/b13_b_pruning_boundary"
CKPT_IT = 30000
N_REPEATS = 20
RATIOS = [5, 10, 20]
N_VIEWS = 10


@torch.no_grad()
def per_view_evidence(g, camlist, pipe, bg, opt):
    """Returns list of (N,) per-view metric counts for each camera."""
    counts = []
    for cam in camlist:
        out = render_fastgs(cam, g, pipe, bg, opt.mult)
        l1n = get_loss(out["render"], cam.original_image.cuda())
        mmap = (l1n > opt.loss_thresh).int()
        out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                             get_flag=True, metric_map=mmap)
        counts.append(out2["accum_metric_counts"].clone().float())
    return counts


@torch.no_grad()
def pru_from_counts(counts, camlist, g, pipe, bg, opt):
    """FastGS pruning score from pre-computed per-view counts + photometric loss."""
    full = None
    for cam, c in zip(camlist, counts):
        out = render_fastgs(cam, g, pipe, bg, opt.mult)
        pl = 0.8 * torch.mean(torch.abs(out["render"] - cam.original_image.cuda())) \
             + 0.2 * (1.0 - fast_ssim(out["render"].unsqueeze(0),
                                       cam.original_image.cuda().unsqueeze(0)))
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
    keep = ~mask
    g._xyz = g._xyz[keep].detach().requires_grad_(True)
    g._features_dc = g._features_dc[keep].detach().requires_grad_(True)
    g._features_rest = g._features_rest[keep].detach().requires_grad_(True)
    g._opacity = g._opacity[keep].detach().requires_grad_(True)
    g._scaling = g._scaling[keep].detach().requires_grad_(True)
    g._rotation = g._rotation[keep].detach().requires_grad_(True)


def make_g(dataset, captured):
    g = GaussianModel(dataset.sh_degree, "default")
    g.training_setup(opt_holder[0])
    g._xyz = captured[1].clone()
    g._features_dc = captured[2].clone()
    g._features_rest = captured[3].clone()
    g._scaling = captured[4].clone()
    g._rotation = captured[5].clone()
    g._opacity = captured[6].clone()
    g.active_sh_degree = captured[0]
    return g


opt_holder = [None]


def main():
    parser = ArgumentParser("Paper B B13-B: pruning boundary misranking")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    opt_holder[0] = opt
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
    print(f"[b13b] it{CKPT_IT} (#GS={n_gs}), {n_all} train views")

    g = make_g(dataset, captured)

    # ---- All-view per-view evidence (oracle reference) ----
    print("[b13b] computing all-view per-view evidence...")
    all_counts = per_view_evidence(g, train_cams, pipe, bg, opt)
    all_counts_np = torch.stack(all_counts).cpu().numpy()  # (n_all, N)
    pru_all = pru_from_counts(all_counts, train_cams, g, pipe, bg, opt).cpu().numpy()
    all_order = np.argsort(pru_all)  # lowest score = prune candidates
    np.save(f"{OUT}/cache/all_counts.npy", all_counts_np)
    np.save(f"{OUT}/cache/pru_all.npy", pru_all)

    # all-view visibility (radii>0 across all views)
    all_vis = (all_counts_np > 0).sum(axis=0)  # (N,) count of views with evidence
    # per-Gaussian top-K important views (by per-view evidence)
    top1_views = np.argmax(all_counts_np, axis=0)
    top5_views = np.argsort(-all_counts_np, axis=0)[:5]  # (5, N)

    # camera centers for angular coverage
    cam_centers = np.array([c.camera_center.cpu().numpy() for c in train_cams])

    # ---- 20 repeats: 10-view subsets ----
    group_rows, exch_rows, freq_counter = [], [], {}

    for rep in range(N_REPEATS):
        rng = random.Random(40000 + rep * 13)
        cams_10 = rng.sample(train_cams, N_VIEWS)
        cam_10_idx = [train_cams.index(c) for c in cams_10]

        counts_10 = [all_counts[i] for i in cam_10_idx]
        pru_10 = pru_from_counts(counts_10, cams_10, g, pipe, bg, opt).cpu().numpy()
        sub_order = np.argsort(pru_10)

        # sampled visibility
        vis_10 = (all_counts_np[cam_10_idx] > 0).sum(axis=0)
        # key-view hit
        hit_top1 = np.isin(top1_views, cam_10_idx)  # (N,)
        top3_views = np.argsort(-all_counts_np, axis=0)[:3]  # (3, N)
        hit_top3 = np.isin(top3_views, cam_10_idx).any(axis=0)  # (N,)
        hit_top5 = np.isin(top5_views, cam_10_idx).any(axis=0)  # (N,)

        for pct in RATIOS:
            k = int(n_gs * pct / 100)
            P10 = set(sub_order[:k])
            Pall = set(all_order[:k])
            C = P10 & Pall
            F = P10 - Pall
            M = Pall - P10
            assert len(F) == len(M)

            # track false-prune frequency
            for gid in F:
                key = (pct, gid)
                freq_counter[key] = freq_counter.get(key, 0) + 1

            # exchange test: delete C+F vs C+M
            for group_name, extra in (("limited_C+F", F), ("allview_C+M", M)):
                sel = list(C) + list(extra)
                gg = make_g(dataset, captured)
                rm = torch.zeros(n_gs, dtype=torch.bool, device="cuda")
                rm[torch.tensor(list(sel), dtype=torch.long)] = True
                with torch.no_grad():
                    light_prune(gg, rm)
                ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
                exch_rows.append({"rep": rep, "ratio": pct, "group": group_name,
                                  "n_C": len(C), "n_extra": len(extra), **ev,
                                  "n_gs": int(gg._xyz.shape[0])})
                del gg; torch.cuda.empty_cache()

            # per-Gaussian group stats (sample a subset for speed: all F/M + 1000 C)
            sample_ids = list(F) + list(M) + list(C)[:1000]
            for gid in sample_ids:
                if gid in F: grp = "F"
                elif gid in M: grp = "M"
                else: grp = "C"
                # boundary distance (percentile distance to pruning cutoff)
                pct_all = np.searchsorted(np.sort(pru_all), pru_all[gid]) / n_gs
                pct_10 = np.searchsorted(np.sort(pru_10), pru_10[gid]) / n_gs
                boundary_dist = pct_all - pct / 100.0
                # angular coverage: std of view directions to this Gaussian
                pos = g._xyz[gid].detach().cpu().numpy()
                dirs = cam_centers - pos
                norms = np.linalg.norm(dirs, axis=1, keepdims=True)
                dirs_norm = dirs / (norms + 1e-8)
                sampled_dirs = dirs_norm[cam_10_idx]
                ang_spread_sampled = float(np.std(sampled_dirs, axis=0).mean())
                ang_spread_all = float(np.std(dirs_norm, axis=0).mean())
                # projected footprint on sampled vs all visible views
                vis_mask = all_counts_np[:, gid] > 0
                sampled_vis = vis_10[gid]
                total_vis = int(vis_mask.sum())
                group_rows.append({
                    "rep": rep, "ratio": pct, "gaussian_id": int(gid),
                    "group": grp,
                    "pru_10v": float(pru_10[gid]), "pru_allv": float(pru_all[gid]),
                    "pct_10v": float(pct_10), "pct_allv": float(pct_all),
                    "boundary_dist": float(boundary_dist),
                    "vis_10v": int(sampled_vis), "vis_all": int(total_vis),
                    "vis_ratio_10v": float(sampled_vis / N_VIEWS),
                    "vis_ratio_all": float(total_vis / n_all) if n_all else 0,
                    "hit_top1": bool(hit_top1[gid]),
                    "hit_top3": bool(hit_top3[gid]) if hasattr(hit_top3, '__len__') else False,
                    "hit_top5": bool(hit_top5[gid]),
                    "ang_spread_sampled": ang_spread_sampled,
                    "ang_spread_all": ang_spread_all,
                    "evidence_allv": float(all_counts_np[:, gid].sum()),
                    "evidence_10v": float(all_counts_np[cam_10_idx, gid].sum()),
                })
        print(f"[b13b] rep {rep+1}/{N_REPEATS} done")
        # incremental save
        for name, rows in (("boundary_groups", group_rows),
                           ("exchange_removal_results", exch_rows)):
            if rows:
                with open(f"{OUT}/data/{name}.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)

    # false-prune frequency
    freq_rows = []
    for (pct, gid), cnt in sorted(freq_counter.items()):
        freq_rows.append({"ratio": pct, "gaussian_id": int(gid),
                          "false_prune_count": cnt,
                          "false_prune_freq": cnt / N_REPEATS})
    with open(f"{OUT}/data/false_prune_frequency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(freq_rows[0].keys()))
        w.writeheader(); w.writerows(freq_rows)

    print(f"[b13b] done: {len(group_rows)} group, {len(exch_rows)} exchange, "
          f"{len(freq_rows)} freq rows")


if __name__ == "__main__":
    main()
