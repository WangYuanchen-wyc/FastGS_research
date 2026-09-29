#
# Paper B - FastGS Pruning Semantics Audit: Minimal Sanity Check
#
# Uses it30000 checkpoint. No training, no removal, no recovery.
# Only computes and reports statistics.
#
import os, sys, random, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.image_utils import psnr
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras, get_loss, compute_photometric_loss
from arguments import ModelParams, PipelineParams, OptimizationParams
from diagnostics.common import install_c_proxy, seed_all

CKPT = "paper_b/b11_v_multiview_importance_reliability/checkpoints/it30000.pt"


@torch.no_grad()
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
    g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, g)
    del g
    train_cams = list(scene.getTrainCameras())
    test_cams = scene.getTestCameras()[:20] if scene.getTestCameras() else train_cams[:10]

    # Load it30000
    captured = torch.load(CKPT, map_location="cuda")
    g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    g.training_setup(opt)
    g._xyz = captured[1].clone()
    g._features_dc = captured[2].clone()
    g._features_rest = captured[3].clone()
    g._scaling = captured[4].clone()
    g._rotation = captured[5].clone()
    g._opacity = captured[6].clone()
    g.active_sh_degree = captured[0]
    n_gs = g._xyz.shape[0]

    # Compute pruning score from 10 random views
    rng = random.Random(0)
    n_all = len(train_cams)
    cam_idx = rng.sample(range(n_all), 10)
    cams = [train_cams[i] for i in cam_idx]

    # Also compute all-view score for reference
    print(f"[audit] n_gs={n_gs}, n_views={n_all}")

    # 10-view score
    full_10 = None
    for cam in cams:
        out = render_fastgs(cam, g, pipe, bg, opt.mult)
        l1_map = get_loss(out["render"], cam.original_image.cuda())
        pl = float(0.8 * torch.mean(torch.abs(out["render"] - cam.original_image.cuda())) \
             + 0.2 * (1.0 - fast_ssim(out["render"].unsqueeze(0),
                                       cam.original_image.cuda().unsqueeze(0))).item())
        mmap = (l1_map > 0.10).int()
        out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                              get_flag=True, metric_map=mmap)
        c = out2["accum_metric_counts"].float().cpu()
        contrib = (pl * c).cpu()
        full_10 = contrib.clone() if full_10 is None else full_10 + contrib

    pru_10 = full_10.numpy()
    rng_ = pru_10.max() - pru_10.min()
    pru_10_norm = (pru_10 - pru_10.min()) / rng_ if rng_ > 1e-8 else np.zeros_like(pru_10)

    # All-view score
    full_all = None
    with torch.no_grad():
        for cam in train_cams:
            out = render_fastgs(cam, g, pipe, bg, opt.mult)
            l1_map = get_loss(out["render"], cam.original_image.cuda())
            pl = float(0.8 * torch.mean(torch.abs(out["render"] - cam.original_image.cuda())) \
                 + 0.2 * (1.0 - fast_ssim(out["render"].unsqueeze(0),
                                           cam.original_image.cuda().unsqueeze(0))).item())
            mmap = (l1_map > 0.10).int()
            out2 = render_fastgs(cam, g, pipe, bg, opt.mult,
                                  get_flag=True, metric_map=mmap)
            c = out2["accum_metric_counts"].float().cpu()
            contrib = (pl * c).cpu()
            full_all = contrib.clone() if full_all is None else full_all + contrib
    pru_all = full_all.numpy()
    rng_all = pru_all.max() - pru_all.min()
    pru_all_norm = (pru_all - pru_all.min()) / rng_all if rng_all > 1e-8 else np.zeros_like(pru_all)

    # ---- Statistics ----
    print(f"\n{'='*60}")
    print(f"PRUNING SCORE STATISTICS")
    print(f"{'='*60}")

    for label, scores in (("10-view", pru_10_norm), ("All-view", pru_all_norm)):
        print(f"\n{label} normalized score:")
        for pct in (1, 10, 25, 50, 75, 90, 99):
            print(f"  p{pct:>2d}: {np.percentile(scores, pct):.4f}")
        print(f"  min:  {scores.min():.4f}")
        print(f"  max:  {scores.max():.4f}")
        print(f"  mean: {scores.mean():.4f}")
        print(f"  #score > 0.9: {np.sum(scores > 0.9)} / {len(scores)} ({100*np.mean(scores > 0.9):.1f}%)")
        print(f"  #score > 0.5: {np.sum(scores > 0.5)} / {len(scores)} ({100*np.mean(scores > 0.5):.1f}%)")

    # Cross-correlation
    from scipy.stats import spearmanr
    rho, p_val = spearmanr(pru_10_norm, pru_all_norm)
    print(f"\n10-view vs All-view Spearman: {rho:.4f} (p={p_val:.2e})")

    # score > 0.9 → what opacity?
    hi_mask = pru_10_norm > 0.9
    lo_mask = pru_10_norm < 0.1
    opacity = g.get_opacity.detach().cpu().numpy().squeeze()
    print(f"\nOpacity by score bucket (10-view):")
    print(f"  score > 0.9: opacity median = {np.median(opacity[hi_mask]):.4f}")
    print(f"  score < 0.1: opacity median = {np.median(opacity[lo_mask]):.4f}")

    # raw (unnormalized) scores
    print(f"\nRaw (unnormalized) evidence:")
    print(f"  10-view: min={pru_10.min():.2f} max={pru_10.max():.2f} mean={pru_10.mean():.2f}")
    print(f"  All-view: min={pru_all.min():.2f} max={pru_all.max():.2f} mean={pru_all.mean():.2f}")

    # ---- Save ----
    os.makedirs("paper_b/b17_h2_historical_ranking/fix/data", exist_ok=True)
    with open("paper_b/b17_h2_historical_ranking/fix/data/semantics_audit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gaussian_id", "score_10v_norm", "score_allv_norm",
                     "score_10v_raw", "score_allv_raw", "opacity"])
        for i in range(n_gs):
            w.writerow([i, pru_10_norm[i], pru_all_norm[i],
                        pru_10[i], pru_all[i], opacity[i]])

    print("\n[audit] saved semantics_audit.csv")


if __name__ == "__main__":
    main()
