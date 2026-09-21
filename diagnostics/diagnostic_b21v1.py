#
# Paper B - B21-V1: Densification Benefit Diagnostic (paired counterfactual)
#
# Question: do different native densification events yield clearly different
# actual benefits? Some events truly need more Gaussians, others nearly the
# same without densifying?
#
# Design:
#   - Native training (room, official config). ALL events recorded.
#   - Selected events, fixed a priori by training phase, blind to outcome:
#     early {1500, 2500}, middle {5000, 7000}, late {11000, 13000} (~6/seed).
#   - At each selected event, BEFORE the native event:
#       deep snapshot (params data, grads, optimizer states incl. shoptimizer,
#       RNG states, camera-pool lists) and the FIXED ROI mask.
#     ROI mask (identical in A/B, computed once pre-event): on the event's own
#     10 sampled views, high-error pixels (l1_norm > loss_thresh) AND covered
#     by >=1 triggered parent (projected footprint r = 3*sqrt(lambda1) of the
#     2D covariance, square bounding boxes, radius capped at 64 px).
#     ROI L1 = mean |render - gt| over ROI pixels.
#   - Branch A: apply the native event (densify+prune+clamp).
#     Branch B: skip exactly this event.
#     Post-decision RNG is ALIGNED: A's rng captured after the event, B set to
#     it -> identical camera sequences afterwards. Windows are 400 iters, no
#     event/reset/SH-degree change can occur inside.
#   - Horizons +0/+100/+250/+400 (all before the next event at +500). After
#     the pair, mainline restores the snapshot and applies the native event.
#   - Self-checks: restore bitwise-equality asserts, identical post-event
#     camera sequences, identical iteration counts; any failure -> INVALID.
#
# GPU: 5
#

import os, sys, random, copy, csv, math
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.image_utils import psnr
from fused_ssim import fused_ssim as fast_ssim
from utils.fast_utils import (compute_gaussian_score_fastgs, sampling_cameras,
                              get_loss, compute_photometric_loss)
from utils.general_utils import inverse_sigmoid
from arguments import ModelParams, PipelineParams, OptimizationParams
from diagnostics.common import install_c_proxy, seed_all, native_train_one_iter

OUT = "paper_b/b21_v1_densification_benefit"
SELECTED_EVENTS = {1500: "early", 2500: "early", 5000: "middle",
                   7000: "middle", 11000: "late", 13000: "late"}
HORIZONS = [0, 100, 250, 400]
ROI_RADIUS_CAP = 64.0


def safe_replace_opacity(g, tensor):
    for group in g.optimizer.param_groups:
        if group["name"] == "opacity":
            p = group["params"][0]
            if p not in g.optimizer.state:
                g.optimizer.state[p] = {
                    "step": torch.tensor(0.0),
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }
    return g.replace_tensor_to_optimizer(tensor, "opacity")


def apply_native_densify(g, opt, it, radii, imp, pru, extent):
    """Native event densification. Returns (n_clone, n_split, added_gs)."""
    grad_vars = g.xyz_gradient_accum / g.denom
    grad_vars[grad_vars.isnan()] = 0.0
    grads_abs = g.xyz_gradient_accum_abs / g.denom
    grads_abs[grads_abs.isnan()] = 0.0
    grad_qual = torch.norm(grad_vars, dim=-1) >= opt.grad_thresh
    grad_abs_qual = torch.norm(grads_abs, dim=-1) >= opt.grad_abs_thresh
    max_scale = g.get_scaling.max(dim=1).values
    metric_mask = imp > 5
    clone_set = metric_mask & (max_scale <= opt.dense * extent) & grad_qual
    split_set = metric_mask & (max_scale > opt.dense * extent) & grad_abs_qual

    g.tmp_radii = radii
    n_clone, n_split = 0, 0
    if clone_set.any():
        n_clone = int(clone_set.sum())
        g.densify_and_clone_fastgs(clone_set, torch.ones_like(clone_set))
    if split_set.any():
        n_split = int(split_set.sum())
        g.densify_and_split_fastgs(split_set, torch.ones_like(split_set), N=2)
    g.tmp_radii = None

    n_pre = g.get_xyz.shape[0] - (n_clone + 2 * n_split)
    prune_mask = (g.get_opacity < 0.005).squeeze()
    st = 20 if it > opt.opacity_reset_interval else None
    if st:
        prune_mask = torch.logical_or(torch.logical_or(
            prune_mask, g.max_radii2D > st),
            g.get_scaling.max(dim=1).values > 0.1 * extent)
    scores = 1 - pru
    tr = int(torch.sum(prune_mask)); rb = int(0.5 * tr)
    if rb:
        n = g.get_xyz.shape[0]
        padded = torch.zeros((n), dtype=torch.float32, device=scores.device)
        padded[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
        sel = torch.zeros_like(padded, dtype=bool)
        sel[torch.multinomial(padded, rb, replacement=False)] = True
        g.prune_points(torch.logical_and(prune_mask, sel))
    g._opacity = safe_replace_opacity(
        g, inverse_sigmoid(torch.min(g.get_opacity, torch.ones_like(g.get_opacity) * 0.8))
    )["opacity"]
    torch.cuda.empty_cache()
    added = g.get_xyz.shape[0] - n_pre
    return n_clone, n_split, added


# ---------------- deep snapshot / restore ----------------

def get_rng():
    return (random.getstate(), np.random.get_state(),
            torch.get_rng_state(), torch.cuda.get_rng_state())


def set_rng(s):
    random.setstate(s[0])
    np.random.set_state(s[1])
    torch.set_rng_state(s[2].cpu())
    torch.cuda.set_rng_state(s[3].cpu())


PARAM_NAMES = ["_xyz", "_features_dc", "_features_rest", "_opacity",
               "_scaling", "_rotation"]


def deep_snapshot(g):
    snap = {"params": {n: (getattr(g, n), getattr(g, n).detach().clone())
                       for n in PARAM_NAMES},
            "grads": {n: (getattr(g, n).grad.detach().clone()
                          if getattr(g, n).grad is not None else None)
                      for n in PARAM_NAMES},
            "aux": {"max_radii2D": g.max_radii2D.clone(),
                    "acc": g.xyz_gradient_accum.clone(),
                    "acc_abs": g.xyz_gradient_accum_abs.clone(),
                    "denom": g.denom.clone()},
            "opt": {}}
    for oname in ("optimizer", "shoptimizer"):
        o = getattr(g, oname)
        st = {}
        for p, s in o.state.items():
            st[id(p)] = (p, {k: (v.clone() if torch.is_tensor(v) else v)
                             for k, v in s.items()})
        snap["opt"][oname] = st
        snap.setdefault("groups", {})[oname] = [
            list(grp["params"]) for grp in o.param_groups]
    return snap


def deep_restore(g, snap):
    # densification/prune replaces Parameter objects; reattach the snapshot's
    # original objects and restore their data (branches mutate data in place).
    for n in PARAM_NAMES:
        p_obj, p_data = snap["params"][n]
        setattr(g, n, p_obj)
        p_obj.data.copy_(p_data)
        gval = snap["grads"][n]
        if gval is None:
            p_obj.grad = None
        else:
            if p_obj.grad is None:
                p_obj.grad = torch.zeros_like(p_obj)
            p_obj.grad.data.copy_(gval)
    g.max_radii2D = snap["aux"]["max_radii2D"].clone()
    g.xyz_gradient_accum = snap["aux"]["acc"].clone()
    g.xyz_gradient_accum_abs = snap["aux"]["acc_abs"].clone()
    g.denom = snap["aux"]["denom"].clone()
    for oname in ("optimizer", "shoptimizer"):
        o = getattr(g, oname)
        for grp, saved in zip(o.param_groups, snap["groups"][oname]):
            grp["params"] = list(saved)
        o.state = {}
        for p, saved in snap["opt"][oname].values():
            o.state[p] = {k: (v.clone() if torch.is_tensor(v) else v)
                          for k, v in saved.items()}


def snapshot_equal(g, snap):
    for n in PARAM_NAMES:
        p_obj = snap["params"][n][0]
        if getattr(g, n) is not p_obj:
            return False
        if not torch.equal(p_obj.detach(), snap["params"][n][1]):
            return False
    return True


# ---------------- ROI mask ----------------

@torch.no_grad()
def triggered_coverage_mask(cam, xyz, scaling, rotation, sel_idx, extent):
    """Union of triggered-parent projected footprints. Per parent: cov2d
    eigenvalue radius r = 3*sqrt(l1) (capped), square box stamped at the
    projected center. Returns bool (H*W)."""
    device = xyz.device
    H, W = int(cam.image_height), int(cam.image_width)
    mask = torch.zeros(H * W, dtype=torch.bool, device=device)
    P = sel_idx.shape[0]
    if P == 0:
        return mask
    tanfovx = math.tan(cam.FoVx * 0.5)
    tanfovy = math.tan(cam.FoVy * 0.5)
    focal_y = H / (2.0 * tanfovy)
    focal_x = W / (2.0 * tanfovx)
    view = cam.world_view_transform[:3, :3]
    proj = cam.full_proj_transform
    CH = 512
    for i0 in range(0, P, CH):
        idx = sel_idx[i0:i0 + CH]
        p = xyz[idx]
        s = scaling[idx]
        r = rotation[idx]
        # cov3d
        S = torch.zeros((s.shape[0], 3, 3), device=device)
        S[:, 0, 0] = s[:, 0] ** 2
        S[:, 1, 1] = s[:, 1] ** 2
        S[:, 2, 2] = s[:, 2] ** 2
        R = torch.zeros((r.shape[0], 3, 3), device=device)
        rn = r / r.norm(dim=1, keepdim=True).clamp(min=1e-9)
        x, y, z, w = rn[:, 0], rn[:, 1], rn[:, 2], rn[:, 3]
        R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        M = S @ R.transpose(1, 2)
        cov3 = M @ M.transpose(1, 2)
        t = (p @ view.T)
        J = torch.zeros((p.shape[0], 3, 3), device=device)
        J[:, 0, 0] = focal_x / t[:, 2].clamp(min=1e-6)
        J[:, 1, 1] = focal_y / t[:, 2].clamp(min=1e-6)
        J[:, 0, 2] = -focal_x * t[:, 0] / t[:, 2].clamp(min=1e-6) ** 2
        J[:, 1, 2] = -focal_y * t[:, 1] / t[:, 2].clamp(min=1e-6) ** 2
        Wm = view.unsqueeze(0).expand(p.shape[0], 3, 3)
        cov2 = J @ Wm @ cov3 @ Wm.transpose(1, 2) @ J.transpose(1, 2)
        det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2
        det = det.clamp(min=1e-9)
        trc = cov2[:, 0, 0] + cov2[:, 1, 1]
        mid = (trc / 2) ** 2 - det
        l1 = (trc / 2 + mid.clamp(min=0).sqrt()).clamp(min=1e-9)
        rad = (3.0 * l1.sqrt()).clamp(max=ROI_RADIUS_CAP)
        # project center
        ph = torch.cat([p, torch.ones(p.shape[0], 1, device=device)], dim=1)
        pp = ph @ proj.T
        d = pp[:, 3].clamp(min=1e-9)
        u = pp[:, 0] / d * 0.5 + 0.5
        v = pp[:, 1] / d * 0.5 + 0.5
        cx = (u * W).round().long()
        cy = (v * H).round().long()
        h = rad.ceil().long()
        for b in range(idx.shape[0]):
            r_i = int(h[b]); cxi = int(cx[b]); cyi = int(cy[b])
            x0, x1 = max(0, cxi - r_i), min(W, cxi + r_i + 1)
            y0, y1 = max(0, cyi - r_i), min(H, cyi + r_i + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            row_idx = torch.arange(y0, y1, device=device)
            col_base = row_idx * W
            mask[col_base.unsqueeze(1) + torch.arange(x0, x1, device=device)] = True
    return mask


# ---------------- evaluation ----------------

@torch.no_grad()
def global_metrics(g, test_cams, pipe, bg, mult, lpips_fn):
    ps_, ss_, lp_ = [], [], []
    for cam in test_cams:
        img = torch.clamp(render_fastgs(cam, g, pipe, bg, mult)["render"], 0, 1)
        gt = torch.clamp(cam.original_image.cuda(), 0, 1)
        ps_.append(float(psnr(img, gt).mean()))
        ss_.append(float(fast_ssim(img.unsqueeze(0), gt.unsqueeze(0)).mean()))
        lp_.append(float(lpips_fn(img.unsqueeze(0) * 2 - 1,
                                  gt.unsqueeze(0) * 2 - 1).mean()))
    return (float(np.mean(ps_)), float(np.mean(ss_)), float(np.mean(lp_)))


@torch.no_grad()
def roi_errors(g, cl, roi_masks, pipe, bg, mult):
    """Mean |render-gt| over each view's fixed ROI. Returns (mean, [per-view])."""
    vals = []
    for cam, m in zip(cl, roi_masks):
        img = render_fastgs(cam, g, pipe, bg, mult)["render"]
        gt = cam.original_image.cuda()
        l1 = (img - gt).abs().mean(dim=0).reshape(-1)
        if m.any():
            vals.append(float(l1[m].mean()))
    return (float(np.mean(vals)) if vals else float("nan")), vals


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--max_iters", type=int, default=30000)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    install_c_proxy()

    import lpips
    lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(f"{OUT}/data", exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    ev_rows = []       # all native events (mainline)
    branch_rows = []   # per branch x horizon global metrics
    roi_rows = []      # ROI benefit
    eff_rows = []      # capacity efficiency
    summary_rows = []

    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}", flush=True)
        install_c_proxy()
        seed_all(seed)
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)
        extent = scene.cameras_extent
        test_cams = scene.getTestCameras() if scene.getTestCameras() else scene.getTrainCameras()[:10]
        vp_stack = scene.getTrainCameras().copy()
        vp_idx = list(range(len(vp_stack)))

        event_num = 0
        for it in range(1, args.max_iters + 1):
            if not vp_stack:
                vp_stack = scene.getTrainCameras().copy()
                vp_idx = list(range(len(vp_stack)))
            r = random.randint(0, len(vp_idx) - 1)
            cam = vp_stack.pop(r)
            _ = vp_idx.pop(r)
            _, vpt, vis, radii = native_train_one_iter(it, cam, gaussians, pipe, bg, opt)

            with torch.no_grad():
                if it < opt.densify_until_iter:
                    gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis])
                    gaussians.add_densification_stats(vpt, vis)
                    if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                        event_num += 1
                        my = scene.getTrainCameras().copy()
                        cl = sampling_cameras(my)
                        imp, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt, DENSIFY=True)

                        n_gs_before = gaussians.get_xyz.shape[0]

                        if it in SELECTED_EVENTS:
                            phase = SELECTED_EVENTS[it]
                            # ---- trigger sets + ROI masks (pre-event) ----
                            grad_vars = gaussians.xyz_gradient_accum / gaussians.denom
                            grad_vars[grad_vars.isnan()] = 0.0
                            grads_abs = gaussians.xyz_gradient_accum_abs / gaussians.denom
                            grads_abs[grads_abs.isnan()] = 0.0
                            gq = torch.norm(grad_vars, dim=-1) >= opt.grad_thresh
                            gaq = torch.norm(grads_abs, dim=-1) >= opt.grad_abs_thresh
                            ms = gaussians.get_scaling.max(dim=1).values
                            mm = imp > 5
                            trig = (mm & (ms <= opt.dense * extent) & gq) | \
                                   (mm & (ms > opt.dense * extent) & gaq)
                            sel_idx = torch.where(trig)[0]
                            roi_masks = []
                            for cam_v in cl:
                                render_image = render_fastgs(cam_v, gaussians, pipe, bg, opt.mult)["render"]
                                gt_image = cam_v.original_image.cuda()
                                l1n = get_loss(render_image, gt_image)
                                hi = (l1n > opt.loss_thresh).reshape(-1)
                                cov = triggered_coverage_mask(
                                    cam_v, gaussians.get_xyz.detach(),
                                    gaussians.get_scaling.detach(),
                                    gaussians.get_rotation.detach(), sel_idx, extent)
                                roi_masks.append(hi & cov)
                            n_roi = sum(int(m.sum()) for m in roi_masks)
                            print(f"  [selected {phase}] it={it} triggers={sel_idx.shape[0]} "
                                  f"ROI px total={n_roi}", flush=True)

                            # ---- snapshot ----
                            snap = deep_snapshot(gaussians)
                            rng_pre = get_rng()
                            cam_snap = (list(vp_stack), list(vp_idx))

                            branch_rng = None
                            branch_cam_log = {}
                            branch_res = {}
                            for bname, do_densify in (("A_densify", True),
                                                      ("B_nodensify", False)):
                                deep_restore(gaussians, snap)
                                set_rng(rng_pre)
                                vp_stack[:] = cam_snap[0]
                                vp_idx[:] = cam_snap[1]
                                assert snapshot_equal(gaussians, snap)
                                if do_densify:
                                    apply_native_densify(gaussians, opt, it, radii,
                                                         imp, pru, extent)
                                    branch_rng = get_rng()
                                else:
                                    set_rng(branch_rng)   # align post-decision RNG
                                gaussians.optimizer_step(it)
                                cam_seq = []
                                cur = it
                                metrics_h = {}
                                for hz in HORIZONS:
                                    while cur < it + hz:
                                        cur += 1
                                        if not vp_stack:
                                            vp_stack = scene.getTrainCameras().copy()
                                            vp_idx = list(range(len(vp_stack)))
                                        rr = random.randint(0, len(vp_idx) - 1)
                                        c2 = vp_stack.pop(rr)
                                        _ = vp_idx.pop(rr)
                                        if len(cam_seq) < 3:
                                            cam_seq.append(rr)
                                        # re-enable grad: the enclosing mainline
                                        # block runs under torch.no_grad()
                                        with torch.enable_grad():
                                            _, vpt2, vis2, radii2 = native_train_one_iter(
                                                cur, c2, gaussians, pipe, bg, opt)
                                        with torch.no_grad():
                                            gaussians.max_radii2D[vis2] = torch.max(
                                                gaussians.max_radii2D[vis2], radii2[vis2])
                                            gaussians.add_densification_stats(vpt2, vis2)
                                        gaussians.optimizer_step(cur)
                                    gm = global_metrics(gaussians, test_cams, pipe, bg,
                                                        opt.mult, lpips_fn)
                                    rl, _ = roi_errors(gaussians, cl, roi_masks, pipe, bg, opt.mult)
                                    metrics_h[hz] = (gm, rl)
                                branch_cam_log[bname] = cam_seq
                                branch_res[bname] = metrics_h
                                for hz in HORIZONS:
                                    gm, rl = metrics_h[hz]
                                    branch_rows.append({
                                        "seed": seed, "event_it": it, "phase": phase,
                                        "branch": bname, "horizon": hz,
                                        "psnr": round(gm[0], 5),
                                        "ssim": round(gm[1], 6),
                                        "lpips": round(gm[2], 6),
                                        "roi_l1": round(rl, 7),
                                    })
                            # self-check: camera sequences identical
                            if branch_cam_log["A_densify"] != branch_cam_log["B_nodensify"]:
                                raise RuntimeError(
                                    f"INVALID: camera sequences diverged at it={it}")
                            # ---- mainline: restore, apply native event, step ----
                            deep_restore(gaussians, snap)
                            set_rng(rng_pre)
                            vp_stack[:] = cam_snap[0]
                            vp_idx[:] = cam_snap[1]
                            assert snapshot_equal(gaussians, snap)
                            n_clone, n_split, n_added = apply_native_densify(
                                gaussians, opt, it, radii, imp, pru, extent)
                            ev_rows.append({
                                "seed": seed, "event": event_num, "iteration": it,
                                "phase": phase, "selected": 1,
                                "trigger_split": n_split, "trigger_clone": n_clone,
                                "gs_before": n_gs_before, "gs_added": n_added,
                                "gs_after": gaussians.get_xyz.shape[0],
                            })
                            for hz in HORIZONS:
                                gmA, rlA = branch_res["A_densify"][hz]
                                gmB, rlB = branch_res["B_nodensify"][hz]
                                roi_rows.append({
                                    "seed": seed, "event_it": it, "phase": phase,
                                    "horizon": hz, "added_gs": n_added,
                                    "roi_l1_densify": round(rlA, 7),
                                    "roi_l1_nodensify": round(rlB, 7),
                                    "benefit_roi": round(rlB - rlA, 7),
                                    "dpsnr": round(gmB[0] - gmA[0], 5),
                                    "dssim": round(gmB[1] - gmA[1], 7),
                                    "dlpips": round(gmB[2] - gmA[2], 7),
                                })
                                eff_rows.append({
                                    "seed": seed, "event_it": it, "phase": phase,
                                    "horizon": hz, "added_gs": n_added,
                                    "benefit_per_gs": round((rlB - rlA) / max(n_added, 1), 12),
                                    "dpsnr_per_gs": round((gmB[0] - gmA[0]) / max(n_added, 1), 10),
                                })
                            print(f"  [{phase} it={it}] added={n_added} "
                                  f"+400: ROI benefit={roi_rows[-1]['benefit_roi']:.5f} "
                                  f"dPSNR={roi_rows[-1]['dpsnr']:+.4f}", flush=True)
                        else:
                            n_clone, n_split, n_added = apply_native_densify(
                                gaussians, opt, it, radii, imp, pru, extent)
                            ev_rows.append({
                                "seed": seed, "event": event_num, "iteration": it,
                                "phase": "", "selected": 0,
                                "trigger_split": n_split, "trigger_clone": n_clone,
                                "gs_before": n_gs_before,
                                "gs_added": n_added,
                                "gs_after": gaussians.get_xyz.shape[0],
                            })

                if it < opt.densify_until_iter and it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            gaussians.optimizer_step(it)

        # per-seed save
        with open(f"{OUT}/data/densify_events_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ev_rows[0].keys()))
            w.writeheader(); w.writerows(ev_rows)
        with open(f"{OUT}/data/paired_branch_metrics_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(branch_rows[0].keys()))
            w.writeheader(); w.writerows(branch_rows)
        with open(f"{OUT}/data/roi_benefit_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(roi_rows[0].keys()))
            w.writeheader(); w.writerows(roi_rows)
        with open(f"{OUT}/data/capacity_efficiency_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(eff_rows[0].keys()))
            w.writeheader(); w.writerows(eff_rows)
        print(f"  s{seed} saved", flush=True)
        torch.cuda.empty_cache()

    print(f"\n[b21v1] done", flush=True)


if __name__ == "__main__":
    main()
