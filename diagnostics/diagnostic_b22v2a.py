#
# Paper B - B22-V2A: Intra-event Densification Benefit (RAM-Safe)
#
# Per event (2500/5000/7000/11000/13000), per seed: native VCD triggered
# parents -> k-means (k=12, seeded, CHUNKED assignment, no dense NxN) ->
# 6 groups sampled uniformly (seeded, blind) -> per group, two branches from
# the SAME event snapshot:
#   A Target-Add  : native clone/split creation for this group only
#   B Target-NoAdd: no creation for this group
# The event's prune / opacity clamp are not executed in either branch.
# Post-decision RNG aligned; optimizer_step runs every window iteration and is
# verified (param-norm freeze guard -> INVALID).
#
# RAM-SAFE constraints (enforced by construction):
#   - event snapshot is saved to DISK (models/event_snapshot.pt, overwritten
#     per event); each branch torch.load()s it, restores into the live model,
#     then del + gc.collect() + torch.cuda.empty_cache().
#   - exactly one seed x event x group x branch runs at a time.
#   - k-means assignment is chunked (4096 x k), never dense NxN.
#   - ROI kept only as final per-view bool masks (10 x H*W).
#   - metric rows are appended to CSV immediately (no tensor accumulation).
#   - RAM monitor before/after every branch -> memory_usage.csv; if system RAM
#     used >= 80%, no new branch starts: partial results are written and the
#     run exits safely (RAM_STOP).
#
# GPU: 7
#

import os, sys, random, csv, gc, math
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

OUT = "paper_b/b22_v2a_intra_event_benefit"
SELECTED_EVENTS = {2500: 0, 5000: 1, 7000: 2, 11000: 3, 13000: 4}
HORIZONS = [500, 1000, 2000]
K_GROUPS = 12
N_SAMPLE = 6
MIN_GROUP = 5
ROI_RADIUS_CAP = 64.0
RAM_STOP_FRAC = 0.80
SNAP_PATH = f"{OUT}/data/event_snapshot.pt"

PARAM_NAMES = ["_xyz", "_features_dc", "_features_rest", "_opacity",
               "_scaling", "_rotation"]


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


def group_creation(g, opt, it, radii, imp, pru, extent, group_mask):
    """Native clone/split creation restricted to `group_mask` parents."""
    grad_vars = g.xyz_gradient_accum / g.denom
    grad_vars[grad_vars.isnan()] = 0.0
    grads_abs = g.xyz_gradient_accum_abs / g.denom
    grads_abs[grads_abs.isnan()] = 0.0
    grad_qual = torch.norm(grad_vars, dim=-1) >= opt.grad_thresh
    grad_abs_qual = torch.norm(grads_abs, dim=-1) >= opt.grad_abs_thresh
    max_scale = g.get_scaling.max(dim=1).values
    metric_mask = imp > 5
    clone_set = metric_mask & (max_scale <= opt.dense * extent) & grad_qual & group_mask
    split_set = metric_mask & (max_scale > opt.dense * extent) & grad_abs_qual & group_mask
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


def event_full(g, opt, it, radii, imp, pru, extent):
    """Native whole event (creation for all + prune + clamp), mainline only."""
    group_creation(g, opt, it, radii, imp, pru, extent,
                   torch.ones(g.get_xyz.shape[0], dtype=torch.bool,
                              device=g.get_xyz.device))
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


# ---------------- RAM monitor ----------------

def ram_snapshot():
    mi = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        mi[k] = int(v.strip().split()[0]) * 1024
    total = mi["MemTotal"]
    avail = mi["MemAvailable"]
    rss = 0
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
    return {
        "sys_used_frac": round(1.0 - avail / total, 4),
        "sys_avail_gb": round(avail / 2**30, 2),
        "rss_gb": round(rss / 2**30, 2),
        "gpu_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
        "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
    }


# ---------------- RNG / disk snapshot ----------------

def get_rng():
    return (random.getstate(), np.random.get_state(),
            torch.get_rng_state(), torch.cuda.get_rng_state())


def set_rng(s):
    random.setstate(s[0])
    np.random.set_state(s[1])
    torch.set_rng_state(s[2].cpu())
    torch.cuda.set_rng_state(s[3].cpu())


def save_snapshot_disk(g, path, extra):
    """Serialize model/optimizer/RNG state to disk. Parameter objects are
    keyed by id(); restore maps tensors back onto the same live objects."""
    snap = {"params": {n: getattr(g, n).detach().clone() for n in PARAM_NAMES},
            "grads": {n: (getattr(g, n).grad.detach().clone()
                          if getattr(g, n).grad is not None else None)
                      for n in PARAM_NAMES},
            "aux": {"max_radii2D": g.max_radii2D.clone(),
                    "acc": g.xyz_gradient_accum.clone(),
                    "acc_abs": g.xyz_gradient_accum_abs.clone(),
                    "denom": g.denom.clone()},
            "opt": {}, "groups": {}, "rng": get_rng(), "extra": extra,
            "idmap": {id(getattr(g, n)): n for n in PARAM_NAMES}}
    for oname in ("optimizer", "shoptimizer"):
        o = getattr(g, oname)
        snap["groups"][oname] = [[id(p) for p in grp["params"]]
                                 for grp in o.param_groups]
        st = {}
        for p, s in o.state.items():
            st[id(p)] = (p, {k: (v.clone() if torch.is_tensor(v) else v)
                             for k, v in s.items()})
        snap["opt"][oname] = st
    torch.save(snap, path)


def load_restore_disk(g, path):
    """REBUILDING restore: in-window native events replace Parameter objects,
    so we always construct fresh Parameters from the snapshot data and
    re-point the optimizer groups/states at them (position-preserving)."""
    import torch.nn as nn
    snap = torch.load(path, map_location="cuda")
    newP = {}
    for n in PARAM_NAMES:
        p = nn.Parameter(snap["params"][n].clone(), requires_grad=True)
        gval = snap["grads"][n]
        p.grad = None if gval is None else gval.clone()
        setattr(g, n, p)
        newP[n] = p
    id2new = {saved_id: newP[n] for saved_id, n in snap["idmap"].items()}
    g.max_radii2D = snap["aux"]["max_radii2D"].clone()
    g.xyz_gradient_accum = snap["aux"]["acc"].clone()
    g.xyz_gradient_accum_abs = snap["aux"]["acc_abs"].clone()
    g.denom = snap["aux"]["denom"].clone()
    for oname in ("optimizer", "shoptimizer"):
        o = getattr(g, oname)
        for grp, ids in zip(o.param_groups, snap["groups"][oname]):
            grp["params"] = [id2new[i] for i in ids]
        o.state = {}
        for saved_id, saved in snap["opt"][oname].items():
            _p, saved_state = saved   # disk format: (param, state-dict)
            o.state[id2new[saved_id]] = {
                k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in saved_state.items()}
    set_rng(snap["rng"])
    del snap
    gc.collect()
    torch.cuda.empty_cache()


def kmeans_labels(xyz, k, seed, iters=50, chunk=4096):
    """Chunked torch k-means on (N,3) cpu. Assignment computes (chunk, k)
    distances only. Returns labels (N,) cpu long."""
    xyz = xyz.cpu()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    N = xyz.shape[0]
    k = min(k, N)
    init = xyz[torch.randperm(N, generator=gen)[:k]].clone()
    labels = torch.zeros(N, dtype=torch.long)
    for _ in range(iters):
        new_labels = torch.zeros(N, dtype=torch.long)
        for i0 in range(0, N, chunk):
            d = torch.cdist(xyz[i0:i0 + chunk], init)
            new_labels[i0:i0 + chunk] = d.argmin(dim=1)
        for j in range(k):
            m = new_labels == j
            if m.any():
                init[j] = xyz[m].mean(dim=0)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
    return labels


# ---------------- ROI coverage ----------------

@torch.no_grad()
def coverage_mask(cam, xyz, scaling, rotation, sel_idx):
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
def roi_error(g, cl, roi_masks, pipe, bg, mult):
    vals = []
    for cam, m in zip(cl, roi_masks):
        img = render_fastgs(cam, g, pipe, bg, mult)["render"]
        gt = cam.original_image.cuda()
        l1 = (img - gt).abs().mean(dim=0).reshape(-1)
        if m.any():
            vals.append(float(l1[m].mean()))
    return float(np.mean(vals)) if vals else float("nan")


BENEFIT_FIELDS = ["seed", "event_it", "group", "n_parents", "horizon",
                  "added_gs", "roi_l1_add", "roi_l1_noadd",
                  "capacity_benefit", "dpsnr", "dssim", "dlpips"]


def append_row(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


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
    mem_path = f"{OUT}/data/memory_usage.csv"
    ben_path = f"{OUT}/data/paired_capacity_benefit.csv"
    ram_stop = False

    for seed in seeds:
        if ram_stop:
            break
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
        groups_path = f"{OUT}/data/groups_s{seed}.csv"
        summ_path = f"{OUT}/data/seed_event_summary_s{seed}.csv"
        group_new = not os.path.exists(groups_path)
        g_f = None
        s_f = None

        for it in range(1, args.max_iters + 1):
            if ram_stop:
                break
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
                            # ---- trigger sets + chunked grouping (blind) ----
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
                            trig_idx = torch.where(trig)[0]
                            trig_xyz = gaussians.get_xyz.detach()[trig_idx]
                            km_seed = 90000 + seed * 100 + SELECTED_EVENTS[it]
                            labels = kmeans_labels(trig_xyz, K_GROUPS, km_seed)
                            eligible = [j for j in range(K_GROUPS)
                                        if int((labels == j).sum()) >= MIN_GROUP]
                            rng = np.random.RandomState(20000 + seed * 100 + SELECTED_EVENTS[it])
                            pick = list(rng.choice(eligible, size=min(N_SAMPLE, len(eligible)),
                                                   replace=False))
                            print(f"  [event {it}] triggers={trig_idx.shape[0]} "
                                  f"eligible={len(eligible)} picked={pick}", flush=True)

                            # ---- DISK snapshot (model/optimizer/RNG) ----
                            save_snapshot_disk(gaussians, SNAP_PATH, extra=None)
                            cam_snap = (list(vp_stack), list(vp_idx))

                            for gi in pick:
                                if g_f is None:
                                    g_f = open(groups_path, "a", newline="")
                                    g_w = csv.DictWriter(g_f, fieldnames=[
                                        "seed", "event_it", "group", "n_parents"])
                                    if group_new:
                                        g_w.writeheader()
                                        group_new = False
                                s_f_open = s_f is not None
                                g_w.writerow({"seed": seed, "event_it": it,
                                              "group": gi,
                                              "n_parents": int((labels == gi).sum())})
                                g_f.flush()
                                if not s_f_open:
                                    need_header = (not os.path.exists(summ_path)) or \
                                        os.path.getsize(summ_path) == 0
                                    s_f = open(summ_path, "a", newline="")
                                    s_w = csv.DictWriter(s_f, fieldnames=[
                                        "seed", "event_it", "group", "n_parents",
                                        "added_gs", "benefit_500", "benefit_1000",
                                        "benefit_2000"])
                                    if need_header:
                                        s_w.writeheader()

                                # restore snapshot first: trigger indices and
                                # ROI masks are snapshot-based
                                load_restore_disk(gaussians, SNAP_PATH)
                                vp_stack[:] = cam_snap[0]
                                vp_idx[:] = cam_snap[1]
                                gmask = torch.zeros(gaussians.get_xyz.shape[0],
                                                    dtype=torch.bool, device="cuda")
                                gmask[trig_idx[labels == gi]] = True
                                roi_masks = []
                                for cam_v in cl:
                                    render_image = render_fastgs(cam_v, gaussians, pipe, bg, opt.mult)["render"]
                                    gt_image = cam_v.original_image.cuda()
                                    l1n = get_loss(render_image, gt_image)
                                    hi = (l1n > opt.loss_thresh).reshape(-1)
                                    cov = coverage_mask(
                                        cam_v, gaussians.get_xyz.detach(),
                                        gaussians.get_scaling.detach(),
                                        gaussians.get_rotation.detach(),
                                        trig_idx[labels == gi])
                                    roi_masks.append(hi & cov)

                                benefits = {}
                                n_added_by = {}
                                for bname, do_add in (("A_add", True),
                                                      ("B_noadd", False)):
                                    ram = ram_snapshot()
                                    append_row(mem_path, ["seed", "event_it", "group",
                                                          "branch", "stage",
                                                          "sys_used_frac", "sys_avail_gb",
                                                          "rss_gb", "gpu_alloc_gb",
                                                          "gpu_reserved_gb"],
                                               {"seed": seed, "event_it": it, "group": gi,
                                                "branch": bname, "stage": "pre",
                                                **ram})
                                    if ram["sys_used_frac"] >= RAM_STOP_FRAC:
                                        print(f"  [RAM_STOP] sys_used="
                                              f"{ram['sys_used_frac']:.2%} >= "
                                              f"{RAM_STOP_FRAC:.0%} — stopping safely",
                                              flush=True)
                                        if g_f: g_f.close()
                                        if s_f: s_f.close()
                                        sys.exit(0)
                                    load_restore_disk(gaussians, SNAP_PATH)
                                    vp_stack[:] = cam_snap[0]
                                    vp_idx[:] = cam_snap[1]
                                    n_pre_gs = gaussians.get_xyz.shape[0]
                                    if do_add:
                                        group_creation(gaussians, opt, it, radii,
                                                       imp, pru, extent, gmask)
                                    n_added = gaussians.get_xyz.shape[0] - n_pre_gs
                                    n_added_by[bname] = n_added
                                    # RNG already aligned: load_restore_disk set
                                    # it to the snapshot (pre-event) state
                                    gaussians.optimizer_step(it)
                                    cur = it
                                    xyz0 = float(gaussians._xyz.norm())
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
                                            with torch.enable_grad():
                                                _, vpt2, vis2, radii2 = native_train_one_iter(
                                                    cur, c2, gaussians, pipe, bg, opt)
                                            with torch.no_grad():
                                                gaussians.max_radii2D[vis2] = torch.max(
                                                    gaussians.max_radii2D[vis2], radii2[vis2])
                                                gaussians.add_densification_stats(vpt2, vis2)
                                                if cur < opt.densify_until_iter and \
                                                        cur > opt.densify_from_iter and \
                                                        cur % opt.densification_interval == 0:
                                                    my2 = scene.getTrainCameras().copy()
                                                    cl2 = sampling_cameras(my2)
                                                    imp2, pru2 = compute_gaussian_score_fastgs(
                                                        cl2, gaussians, pipe, bg, opt, DENSIFY=True)
                                                    event_full(gaussians, opt, cur, radii2,
                                                               imp2, pru2, extent)
                                                if cur < opt.densify_until_iter and \
                                                        cur % opt.opacity_reset_interval == 0:
                                                    gaussians.reset_opacity()
                                            gaussians.optimizer_step(cur)
                                        xyz1 = float(gaussians._xyz.norm())
                                        if abs(xyz1 - xyz0) < 1e-9:
                                            raise RuntimeError(
                                                "INVALID: frozen window "
                                                "(no optimizer_step)")
                                        gm = global_metrics(gaussians, test_cams, pipe, bg,
                                                            opt.mult, lpips_fn)
                                        rl = roi_error(gaussians, cl, roi_masks, pipe, bg,
                                                       opt.mult)
                                        res_h[hz] = (gm, rl, gaussians.get_xyz.shape[0])
                                    ram = ram_snapshot()
                                    append_row(mem_path, ["seed", "event_it", "group",
                                                          "branch", "stage",
                                                          "sys_used_frac", "sys_avail_gb",
                                                          "rss_gb", "gpu_alloc_gb",
                                                          "gpu_reserved_gb"],
                                               {"seed": seed, "event_it": it, "group": gi,
                                                "branch": bname, "stage": "post",
                                                **ram})
                                    benefits[bname] = res_h
                                    del res_h
                                    gc.collect()
                                    torch.cuda.empty_cache()

                                n_added = n_added_by["A_add"]
                                for hz in HORIZONS:
                                    gmA, rlA, _ = benefits["A_add"][hz]
                                    gmB, rlB, _ = benefits["B_noadd"][hz]
                                    row = {"seed": seed, "event_it": it, "group": gi,
                                           "n_parents": int((labels == gi).sum()),
                                           "horizon": hz, "added_gs": n_added,
                                           "roi_l1_add": round(rlA, 7),
                                           "roi_l1_noadd": round(rlB, 7),
                                           "capacity_benefit": round(rlB - rlA, 7),
                                           "dpsnr": round(gmB[0] - gmA[0], 5),
                                           "dssim": round(gmB[1] - gmA[1], 7),
                                           "dlpips": round(gmB[2] - gmA[2], 7)}
                                    append_row(ben_path, BENEFIT_FIELDS, row)
                                s_w.writerow({"seed": seed, "event_it": it, "group": gi,
                                              "n_parents": int((labels == gi).sum()),
                                              "added_gs": n_added,
                                              "benefit_500": round(benefits["B_noadd"][500][1] -
                                                                   benefits["A_add"][500][1], 7),
                                              "benefit_1000": round(benefits["B_noadd"][1000][1] -
                                                                    benefits["A_add"][1000][1], 7),
                                              "benefit_2000": round(benefits["B_noadd"][2000][1] -
                                                                    benefits["A_add"][2000][1], 7)})
                                s_f.flush()
                                print(f"  [event {it} g{gi}] added={n_added} "
                                      f"benefit +500={summary_row_b(benefits, 500):+.5f} "
                                      f"+1000={summary_row_b(benefits, 1000):+.5f} "
                                      f"+2000={summary_row_b(benefits, 2000):+.5f}",
                                      flush=True)
                                del benefits
                                gc.collect()
                                torch.cuda.empty_cache()

                            # ---- mainline: restore + full native event ----
                            load_restore_disk(gaussians, SNAP_PATH)
                            vp_stack[:] = cam_snap[0]
                            vp_idx[:] = cam_snap[1]
                            os.remove(SNAP_PATH)   # snapshot consumed
                            event_full(gaussians, opt, it, radii, imp, pru, extent)
                        else:
                            event_full(gaussians, opt, it, radii, imp, pru, extent)

                if it < opt.densify_until_iter and it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            gaussians.optimizer_step(it)

        if g_f: g_f.close(); g_f = None
        if s_f: s_f.close(); s_f = None
        print(f"  s{seed} done", flush=True)
        torch.cuda.empty_cache()

    if s_f:
        try: s_f.close()
        except Exception: pass
    print(f"\n[b22v2a] done ram_stop={ram_stop}", flush=True)


def summary_row_b(benefits, hz):
    return benefits["B_noadd"][hz][1] - benefits["A_add"][hz][1]


if __name__ == "__main__":
    main()
