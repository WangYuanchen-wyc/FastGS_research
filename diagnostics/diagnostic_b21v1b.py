#
# Paper B - B21-V1b: Long-Horizon & Event Disentangling
#
# Questions: (Q1) is the low benefit of late densification just a short-horizon
# artifact? (Q2) does adding Gaussians itself help? (Q3) do the event's side
# operations cancel the creation benefit?
#
# Event decomposition (from FastGS code, densify_and_prune_fastgs +
# densify_and_clone_fastgs/densify_and_split_fastgs/densification_postfix):
#   CREATION   : append clone copies; append 2x split children (scaled
#                1/(0.8N)); optimizer state extended with zero exp_avg/exp_avg_sq
#                for new rows; split parents removed (intrinsic to split);
#                xyz_gradient_accum/_abs, denom, max_radii2D reset to zero for
#                the whole model (densification_postfix, inseparable from
#                creation).
#   PRUNE      : remove 50% of candidates (opacity<0.005, + max_radii2D>20 and
#                scale>0.1*extent when it>3000), multinomial-weighted by
#                1/(1-pruning_score).
#   OPACITY    : clamp opacity to min(opacity, 0.8) via
#                replace_tensor_to_optimizer.
#
# Branches from one pre-event snapshot:
#   A Skip-All          : nothing.
#   B Add-Only          : CREATION only (native postfix semantics kept).
#   C Side-Effect-Only  : PRUNE + OPACITY only (no appends; scores map exactly).
#   D Full-Event        : native whole event.
#
# Fairness: all branches restore the same snapshot/optimizer/RNG/camera pool;
# after the branch's own event operations the RNG is reset to the pre-event
# state so every branch trains with identical camera sequences up to the first
# in-window native event (+500); in-window native events follow the identical
# schedule in every branch. Horizons +100/+400/+1000/+2000. Fixed ROI (B21-V1
# definition) computed once pre-event. Self-checks raise INVALID on divergence.
#
# Events fixed before results: 1500 (high-benefit sanity), 5000, 7000, 11000,
# 13000 (low-benefit). GPU: 5
#

import os, sys, random, csv, math
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

OUT = "paper_b/b21_v1b_event_disentangling"
SELECTED_EVENTS = {1500: "high-sanity", 5000: "low", 7000: "low",
                   11000: "low", 13000: "low"}
HORIZONS = [100, 400, 1000, 2000]
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


def event_creation(g, opt, it, radii, imp, pru, extent):
    """CREATION component only (native clone+split with postfix semantics)."""
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
    return n_clone, n_split


def event_side_effects(g, opt, it, pru, extent):
    """PRUNE + OPACITY components only (no appends; score map aligns 1:1)."""
    n_pre = g.get_xyz.shape[0]
    prune_mask = (g.get_opacity < 0.005).squeeze()
    st = 20 if it > opt.opacity_reset_interval else None
    if st:
        prune_mask = torch.logical_or(torch.logical_or(
            prune_mask, g.max_radii2D > st),
            g.get_scaling.max(dim=1).values > 0.1 * extent)
    scores = 1 - pru
    tr = int(torch.sum(prune_mask)); rb = int(0.5 * tr)
    if rb:
        padded = torch.zeros((n_pre), dtype=torch.float32, device=scores.device)
        padded[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
        sel = torch.zeros_like(padded, dtype=bool)
        sel[torch.multinomial(padded, rb, replacement=False)] = True
        g.prune_points(torch.logical_and(prune_mask, sel))
    g._opacity = safe_replace_opacity(
        g, inverse_sigmoid(torch.min(g.get_opacity, torch.ones_like(g.get_opacity) * 0.8))
    )["opacity"]


def event_full(g, opt, it, radii, imp, pru, extent):
    """Native whole event (creation then side effects), as in B21-V1."""
    n_clone, n_split = event_creation(g, opt, it, radii, imp, pru, extent)
    event_side_effects(g, opt, it, pru, extent)
    torch.cuda.empty_cache()
    return n_clone, n_split


# ---------------- deep snapshot / restore (B21-V1, object-reattaching) ------

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
            "opt": {}, "groups": {}}
    for oname in ("optimizer", "shoptimizer"):
        o = getattr(g, oname)
        st = {}
        for p, s in o.state.items():
            st[id(p)] = (p, {k: (v.clone() if torch.is_tensor(v) else v)
                             for k, v in s.items()})
        snap["opt"][oname] = st
        snap["groups"][oname] = [list(grp["params"]) for grp in o.param_groups]
    return snap


def deep_restore(g, snap):
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


# ---------------- ROI mask (B21-V1 definition) ----------------

@torch.no_grad()
def triggered_coverage_mask(cam, xyz, scaling, rotation, sel_idx):
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
        S = torch.zeros((s.shape[0], 3, 3), device=device)
        S[:, 0, 0] = s[:, 0] ** 2; S[:, 1, 1] = s[:, 1] ** 2; S[:, 2, 2] = s[:, 2] ** 2
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
        det = (cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2).clamp(min=1e-9)
        trc = cov2[:, 0, 0] + cov2[:, 1, 1]
        mid = (trc / 2) ** 2 - det
        l1 = (trc / 2 + mid.clamp(min=0).sqrt()).clamp(min=1e-9)
        rad = (3.0 * l1.sqrt()).clamp(max=ROI_RADIUS_CAP)
        ph = torch.cat([p, torch.ones(p.shape[0], 1, device=device)], dim=1)
        pp = ph @ proj.T
        d = pp[:, 3].clamp(min=1e-9)
        cx = ((pp[:, 0] / d * 0.5 + 0.5) * W).round().long()
        cy = ((pp[:, 1] / d * 0.5 + 0.5) * H).round().long()
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
    vals = []
    for cam, m in zip(cl, roi_masks):
        img = render_fastgs(cam, g, pipe, bg, mult)["render"]
        gt = cam.original_image.cuda()
        l1 = (img - gt).abs().mean(dim=0).reshape(-1)
        if m.any():
            vals.append(float(l1[m].mean()))
    return float(np.mean(vals)) if vals else float("nan")


BRANCHES = [("A_skip_all", False, False),
            ("B_add_only", True, False),
            ("C_side_only", False, True),
            ("D_full", True, True)]


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

    decomp_rows = [
        {"component": "creation_clone_split", "A_skip_all": 0, "B_add_only": 1,
         "C_side_only": 0, "D_full": 1},
        {"component": "creation_optimizer_state_extension", "A_skip_all": 0,
         "B_add_only": 1, "C_side_only": 0, "D_full": 1},
        {"component": "creation_accum_reset", "A_skip_all": 0, "B_add_only": 1,
         "C_side_only": 0, "D_full": 1},
        {"component": "split_parent_removal", "A_skip_all": 0, "B_add_only": 1,
         "C_side_only": 0, "D_full": 1},
        {"component": "prune_multinomial_half", "A_skip_all": 0, "B_add_only": 0,
         "C_side_only": 1, "D_full": 1},
        {"component": "opacity_clamp_0.8", "A_skip_all": 0, "B_add_only": 0,
         "C_side_only": 1, "D_full": 1},
    ]
    with open(f"{OUT}/data/event_decomposition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["component", "A_skip_all", "B_add_only",
                                          "C_side_only", "D_full"])
        w.writeheader(); w.writerows(decomp_rows)

    branch_rows = []
    roi_rows = []
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
                        my = scene.getTrainCameras().copy()
                        cl = sampling_cameras(my)
                        imp, pru = compute_gaussian_score_fastgs(cl, gaussians, pipe, bg, opt, DENSIFY=True)

                        if it in SELECTED_EVENTS:
                            phase = SELECTED_EVENTS[it]
                            # ---- trigger sets + fixed ROI ----
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
                                    gaussians.get_rotation.detach(), sel_idx)
                                roi_masks.append(hi & cov)

                            snap = deep_snapshot(gaussians)
                            rng_pre = get_rng()
                            cam_snap = (list(vp_stack), list(vp_idx))

                            branch_res = {}
                            cam_logs = {}
                            n_added_by = {}
                            for bname, do_create, do_side in BRANCHES:
                                deep_restore(gaussians, snap)
                                set_rng(rng_pre)
                                vp_stack[:] = cam_snap[0]
                                vp_idx[:] = cam_snap[1]
                                assert snapshot_equal(gaussians, snap)
                                n_pre = gaussians.get_xyz.shape[0]
                                if do_create:
                                    event_creation(gaussians, opt, it, radii,
                                                   imp, pru, extent)
                                if do_side:
                                    event_side_effects(gaussians, opt, it, pru, extent)
                                n_added_by[bname] = gaussians.get_xyz.shape[0] - n_pre
                                set_rng(rng_pre)   # align post-decision RNG
                                gaussians.optimizer_step(it)
                                cam_seq = []
                                cur = it
                                res_h = {}
                                for hz in HORIZONS:
                                    while cur < it + hz:
                                        cur += 1
                                        if not vp_stack:
                                            vp_stack = scene.getTrainCameras().copy()
                                            vp_idx = list(range(len(vp_stack)))
                                        rr = random.randint(0, len(vp_idx) - 1)
                                        c2 = vp_stack.pop(rr)
                                        _ = vp_idx.pop(rr)
                                        if len(cam_seq) < 100:
                                            cam_seq.append(rr)
                                        with torch.enable_grad():
                                            _, vpt2, vis2, radii2 = native_train_one_iter(
                                                cur, c2, gaussians, pipe, bg, opt)
                                        with torch.no_grad():
                                            gaussians.max_radii2D[vis2] = torch.max(
                                                gaussians.max_radii2D[vis2], radii2[vis2])
                                            gaussians.add_densification_stats(vpt2, vis2)
                                        # in-window native events (identical
                                        # schedule in every branch):
                                        if cur < opt.densify_until_iter and \
                                                cur > opt.densify_from_iter and \
                                                cur % opt.densification_interval == 0:
                                            with torch.no_grad():
                                                my2 = scene.getTrainCameras().copy()
                                                cl2 = sampling_cameras(my2)
                                                imp2, pru2 = compute_gaussian_score_fastgs(
                                                    cl2, gaussians, pipe, bg, opt, DENSIFY=True)
                                                event_full(gaussians, opt, cur, radii2,
                                                           imp2, pru2, extent)
                                        if cur < opt.densify_until_iter and \
                                                cur % opt.opacity_reset_interval == 0:
                                            with torch.no_grad():
                                                gaussians.reset_opacity()
                                        gaussians.optimizer_step(cur)
                                    gm = global_metrics(gaussians, test_cams, pipe, bg,
                                                        opt.mult, lpips_fn)
                                    rl = roi_errors(gaussians, cl, roi_masks, pipe, bg, opt.mult)
                                    res_h[hz] = (gm, rl, gaussians.get_xyz.shape[0])
                                cam_logs[bname] = cam_seq
                                branch_res[bname] = res_h
                                for hz in HORIZONS:
                                    gm, rl, ngs = res_h[hz]
                                    branch_rows.append({
                                        "seed": seed, "event_it": it, "phase": phase,
                                        "branch": bname, "horizon": hz,
                                        "added_gs": n_added_by[bname],
                                        "psnr": round(gm[0], 5),
                                        "ssim": round(gm[1], 6),
                                        "lpips": round(gm[2], 6),
                                        "roi_l1": round(rl, 7),
                                        "gs_at_hz": ngs,
                                    })
                            # self-check: first-100 camera draws identical
                            for bname, _, _ in BRANCHES[1:]:
                                if cam_logs[bname][:100] != cam_logs["A_skip_all"][:100]:
                                    raise RuntimeError(
                                        f"INVALID: camera divergence at it={it}, {bname}")
                            # mainline: restore + full native event
                            deep_restore(gaussians, snap)
                            set_rng(rng_pre)
                            vp_stack[:] = cam_snap[0]
                            vp_idx[:] = cam_snap[1]
                            assert snapshot_equal(gaussians, snap)
                            event_full(gaussians, opt, it, radii, imp, pru, extent)
                            for hz in HORIZONS:
                                row = {"seed": seed, "event_it": it, "phase": phase,
                                       "horizon": hz,
                                       "added_skip": n_added_by["A_skip_all"],
                                       "added_add": n_added_by["B_add_only"],
                                       "added_side": n_added_by["C_side_only"],
                                       "added_full": n_added_by["D_full"]}
                                for bname, _, _ in BRANCHES:
                                    gm, rl, _ = branch_res[bname][hz]
                                    row[f"roi_{bname}"] = round(rl, 7)
                                    row[f"psnr_{bname}"] = round(gm[0], 5)
                                roi_rows.append(row)
                            b400 = branch_res["B_add_only"][400][0][0]
                            d400 = branch_res["D_full"][400][0][0]
                            a400 = branch_res["A_skip_all"][400][0][0]
                            print(f"  [{phase} it={it}] added B/C/D="
                                  f"{n_added_by['B_add_only']}/{n_added_by['C_side_only']}/"
                                  f"{n_added_by['D_full']}  dPSNR vs Skip "
                                  f"B={b400-a400:+.3f} D={d400-a400:+.3f} (+400)", flush=True)
                        else:
                            event_full(gaussians, opt, it, radii, imp, pru, extent)

                if it < opt.densify_until_iter and it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            gaussians.optimizer_step(it)

        with open(f"{OUT}/data/branch_metrics_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(branch_rows[0].keys()))
            w.writeheader(); w.writerows(branch_rows)
        with open(f"{OUT}/data/long_horizon_roi_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(roi_rows[0].keys()))
            w.writeheader(); w.writerows(roi_rows)
        print(f"  s{seed} saved", flush=True)
        torch.cuda.empty_cache()

    print(f"\n[b21v1b] done", flush=True)


if __name__ == "__main__":
    main()
