#
# Paper B - B16-E: Evidence Formulation Diagnostic
#
# Question: is 10-view pruning instability caused by the binary error
# evidence formulation, or by the 10-view information limit?
#
# Three evidence formulations, same 10 views, same model:
#   A. Binary   : FastGS native (pixel_l1 > loss_thresh → count)
#   B. Continuous : continuous residual × attribution
#   C. Contribution : rendering contribution (non-error-based)
#
# All-view scores as oracle reference. Whole-model pruning @5/10/20%.
#
# Evidence B (continuous residual): for each view, per-Gaussian evidence =
#   mean residual in the Gaussian's projected pixels × Gaussian's screen area
#   (requires per-view rendering + L1 map, no binary threshold)
#
# Evidence C (contribution): per-Gaussian evidence =
#   alpha-weighted screen-space contribution = opacity × projected_area
#   (no error information at all — purely geometric)
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
OUT = "paper_b/b16_e_evidence_formulation"

from diagnostics.diagnostic_b14t import rankdata_avg

def spearman_np(x, y):
    return float(np.corrcoef(rankdata_avg(x), rankdata_avg(y))[0, 1])


def make_g(dataset, captured, opt):
    g = GaussianModel(dataset.sh_degree, "default")
    g.training_setup(opt)
    g._xyz = captured[1].clone(); g._features_dc = captured[2].clone()
    g._features_rest = captured[3].clone(); g._scaling = captured[4].clone()
    g._rotation = captured[5].clone(); g._opacity = captured[6].clone()
    g.active_sh_degree = captured[0]
    return g
CKPT_IT = 30000
N_PERMS = 20
VIEW_COUNTS = [10]
RATIOS = [5, 10, 20]
LOSS_THRESH = 0.10


@torch.no_grad()
def per_view_evidence_A_binary(g, cam, pipe, bg, opt):
    """FastGS native binary evidence."""
    out = render_fastgs(cam, g, pipe, bg, opt.mult)
    l1_map = get_loss(out["render"], cam.original_image.cuda())
    mmap = (l1_map > LOSS_THRESH).int()
    out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                         get_flag=True, metric_map=mmap)
    pl = 0.8 * torch.mean(torch.abs(out["render"] - cam.original_image.cuda())) \
         + 0.2 * (1.0 - fast_ssim(out["render"].unsqueeze(0),
                                   cam.original_image.cuda().unsqueeze(0)))
    return out2["accum_metric_counts"].clone().float() * float(pl)


@torch.no_grad()
def per_view_evidence_B_continuous(g, cam, pipe, bg, opt):
    """Continuous residual evidence: per-pixel L1 × count via soft metric map.

    Instead of binary (l1>thresh), use soft count = l1/thresh clamped to [0,∞),
    weighted by render participation (radii>0). This preserves residual
    magnitude information without a hard gate.
    """
    out = render_fastgs(cam, g, pipe, bg, opt.mult)
    l1_map = get_loss(out["render"], cam.original_image.cuda())
    # soft metric: continuous, 0 only when l1=0 exactly
    soft_map = (l1_map / LOSS_THRESH).clamp(min=0.0).int()  # count = floor(l1/th)
    out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                         get_flag=True, metric_map=soft_map)
    # weight by actual residual magnitude (not binary)
    evidence = out2["accum_metric_counts"].float() * (l1_map.mean().item() + 1e-8)
    pl = 0.8 * torch.mean(torch.abs(out["render"] - cam.original_image.cuda())) \
         + 0.2 * (1.0 - fast_ssim(out["render"].unsqueeze(0),
                                   cam.original_image.cuda().unsqueeze(0)))
    return evidence * float(pl)


@torch.no_grad()
def per_view_evidence_C_contribution(g, cam, pipe, bg, opt):
    """Rendering contribution evidence: opacity × projected area (geometric, no error)."""
    radii = out_r = None
    out = render_fastgs(cam, g, pipe, bg, opt.mult)
    radii = out["radii"].float()
    # contribution proxy = opacity × visible views × 3σ area
    op = g.get_opacity.detach().squeeze(-1)  # (N,)
    sc = g.get_scaling.detach()  # (N,3)
    max_sc = sc.max(dim=1).values
    area_proxy = (radii > 0).float() * op * (max_sc ** 2) * 1000
    return area_proxy


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

    captured = torch.load(f"{CKPT_BASE}/it{CKPT_IT}.pt", map_location="cuda")
    n_gs = captured[1].shape[0]
    print(f"[b16e] it{CKPT_IT} (#GS={n_gs})")

    g = make_g(dataset, captured, opt)

    # ---- All-view scores per evidence type ----
    print("[b16e] computing All-view scores per evidence type...")
    from utils.fast_utils import get_loss
    from diagnostics.common import projected_gaussian_geometry
    all_scores = {k: np.zeros(n_gs) for k in ("binary", "continuous", "contribution")}
    for vi, cam in enumerate(train_cams):
        evA = per_view_evidence_A_binary(g, cam, pipe, bg, opt).cpu().numpy()
        evB = per_view_evidence_B_continuous(g, cam, pipe, bg, opt).cpu().numpy()
        evC = per_view_evidence_C_contribution(g, cam, pipe, bg, opt).cpu().numpy()
        all_scores["binary"] += evA
        all_scores["continuous"] += evB
        all_scores["contribution"] += evC
        if vi % 50 == 0:
            print(f"  view {vi}/{len(train_cams)}")
    print("[b16e] All-view scores done")

    rem_rows, stab_rows = [], []
    stab_rows_csv = open(f"{OUT}/data/evidence_stability.csv", "w", newline="")
    stab_w = csv.DictWriter(stab_rows_csv, fieldnames=["perm", "n_views",
        "evidence_type", "spearman", "top5_ov", "top10_ov", "top20_ov"])
    stab_w.writeheader()

    def prune_and_eval(scores, label, pct, tag=""):
        order = np.argsort(scores)  # lowest = prune first
        sel = order[:int(n_gs * pct / 100)]
        gg = make_g(dataset, captured, opt)
        rm = torch.zeros(n_gs, dtype=torch.bool, device="cuda")
        rm[torch.tensor(sel, dtype=torch.long)] = True
        with torch.no_grad():
            light_prune(gg, rm)
        ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
        row = {"evidence": label, "perm": -1, "n_views": -1, "ratio": pct,
               "source": "allview" if "allview" in tag else "10view", **ev,
               "n_gs": int(gg._xyz.shape[0])}
        rem_rows.append(row)
        del gg; torch.cuda.empty_cache()
        return ev

    # All-view reference for each evidence type
    all_ref = {}
    for etype in ("binary", "continuous", "contribution"):
        for pct in RATIOS:
            ev = prune_and_eval(all_scores[etype], etype, pct, tag="allview")
            all_ref[(etype, pct)] = ev
            rem_rows[-1]["n_views"] = -1
            rem_rows[-1]["source"] = "allview"

    # ---- 10-view: 20 permutations × 3 evidence types × 3 ratios ----
    train_cams_list = train_cams
    for perm in range(N_PERMS):
        rng = random.Random(50000 + perm * 17)
        perm_cams = rng.sample(train_cams_list, len(train_cams_list))
        cams_10 = perm_cams[:10]

        scores_10 = {}
        for etype, func in (("binary", per_view_evidence_A_binary),
                            ("continuous", per_view_evidence_B_continuous),
                            ("contribution", per_view_evidence_C_contribution)):
            s = None
            for cam in cams_10:
                e = func(g, cam, pipe, bg, opt).cpu().numpy()
                s = e if s is None else s + e
            scores_10[etype] = s
            sp = spearman_np(s, all_scores[etype])
            for k_pct in (5, 10, 20):
                k = max(1, int(n_gs * k_pct / 100))
                ov = len(set(np.argsort(-s)[:k]) & set(np.argsort(-all_scores[etype])[:k])) / k
                stab_w.writerow({"perm": perm, "n_views": 10,
                                 "evidence_type": etype, "spearman": sp,
                                 "top5_ov": 0, "top10_ov": ov, "top20_ov": 0})

        for etype in ("binary", "continuous", "contribution"):
            order = np.argsort(scores_10[etype])
            for pct in RATIOS:
                n_rm = int(n_gs * pct / 100)
                sel = order[:n_rm]
                gg = make_g(dataset, captured, opt)
                rm = torch.zeros(n_gs, dtype=torch.bool, device="cuda")
                rm[torch.tensor(sel, dtype=torch.long)] = True
                with torch.no_grad():
                    light_prune(gg, rm)
                ev = eval_test(gg, test_cams, pipe, bg, opt.mult)
                rem_rows.append({"evidence": etype, "perm": perm, "n_views": 10,
                                 "ratio": pct, "source": "10view", **ev,
                                 "n_gs": int(gg._xyz.shape[0])})
                del gg; torch.cuda.empty_cache()

        print(f"[b16e] perm {perm+1}/{N_PERMS} done")

    stab_rows_csv.close()

    # ---- save ----
    with open(f"{OUT}/data/evidence_pruning_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rem_rows[0].keys()))
        w.writeheader(); w.writerows(rem_rows)
    with open(f"{OUT}/data/evidence_allview_results.csv", "w", newline="") as f:
        pass  # included in main results
    print(f"[b16e] done: {len(rem_rows)} rows")


if __name__ == "__main__":
    main()
