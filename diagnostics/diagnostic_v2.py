#
# Paper B - B2: Multi-Candidate Local Action Oracle (diagnostic, v2)
#
# Extends the validated B1 Keep/Clone/Split protocol:
#   * multiple real FastGS candidates per densification event (stratified,
#     seeded sampling; covers native clone + native split)
#   * per-candidate fixed Local ROI (parent projected footprint + 1 tile
#     margin, defined BEFORE the action, shared by Keep/Clone/Split and by
#     steps 0/50/100)
#   * support-local and demand-local (ROI ∩ FastGS high-residual mask) quality
#   * action-specific tile cost (gaussian_tile_delta) at K=0 and K=100
#   * pre-action candidate descriptors + residual descriptors
#
# Branch protocol per candidate is exactly B1: same checkpoint (deep copy),
# same optimizer state, same camera sequence, same RNG control, 100-step
# native continuation, densification/pruning/opacity-reset disabled,
# native densify_and_clone_fastgs / densify_and_split_fastgs reused.
#
# No predictor / allocator / lineage / rollback is implemented here.
#

import os
import sys
import json
import random

import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras, get_loss  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from diagnostics.common import (seed_all, clone_tree, install_c_proxy,  # noqa: E402
                                native_train_one_iter, project_to_pixel, roi_box,
                                local_metrics, residual_descriptors,
                                projected_gaussian_geometry)

try:
    from lpipsPyTorch import lpips as lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False

TILE_MARGIN_PX = 16  # BLOCK_X = BLOCK_Y = 16 (cuda_rasterizer/config.h)

# runtime wiring (set once in main(); avoids threading pipe/bg/mult through
# every helper signature)
G = {"pipe": None, "bg": None, "mult": None, "loss_thresh": None, "sh_degree": None}


# ---------------------------------------------------------- global metrics --

@torch.no_grad()
def global_eval(gaussians, views):
    psnrs, ssims, lpipss = [], [], []
    for cam in views:
        image = torch.clamp(render_fastgs(cam, gaussians, G["pipe"], G["bg"], G["mult"])["render"], 0.0, 1.0)
        gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
        psnrs.append(float(psnr(image, gt).mean()))
        ssims.append(float(fast_ssim(image.unsqueeze(0), gt.unsqueeze(0)).mean()))
        if LPIPS_OK:
            try:
                lpipss.append(float(lpips_fn(image, gt, net_type='vgg').mean()))
            except Exception:
                lpipss.append(float("nan"))
    return {"psnr": float(np.nanmean(psnrs)) if psnrs else None,
            "ssim": float(np.nanmean(ssims)) if ssims else None,
            "lpips": float(np.nanmean(lpipss)) if lpipss else None}


# ------------------------------------------------------------ local metrics --

@torch.no_grad()
def local_eval(gaussians, cand_views):
    """Fixed per-candidate views -> support-local & demand-local metrics
    + whole-image gaussian-tile pair count (num_rendered) per view."""
    proxy = install_c_proxy()
    sup, dem, tiles = [], [], []
    for cv in cand_views:
        out = render_fastgs(cv["cam"], gaussians, G["pipe"], G["bg"], G["mult"])
        tiles.append(proxy.last_num_rendered)
        gt = cv["cam"].original_image.cuda()
        m_sup = local_metrics(out["render"], gt, cv["box"])
        if m_sup is not None:
            sup.append(m_sup)
        if cv["demand_valid"]:
            m_dem = local_metrics(out["render"], gt, cv["box"], cv["mask"])
            if m_dem is not None:
                dem.append(m_dem)

    def _agg(ms):
        if not ms:
            return {"l1": None, "mse": None, "psnr": None, "n_views": 0}
        return {"l1": float(np.mean([m["l1"] for m in ms])),
                "mse": float(np.mean([m["mse"] for m in ms])),
                "psnr": float(np.mean([m["psnr"] for m in ms])),
                "n_views": len(ms)}

    return {"support": _agg(sup), "demand": _agg(dem),
            "tile_pairs_mean": float(np.mean(tiles)) if tiles else None}


@torch.no_grad()
def latency_light(gaussians, cand_views, n_views=3, reps=3):
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    views = cand_views[:n_views]
    if not views:
        return None
    for cv in views:  # warmup
        render_fastgs(cv["cam"], gaussians, G["pipe"], G["bg"], G["mult"])
    torch.cuda.synchronize()
    lats = []
    for _ in range(reps):
        for cv in views:
            ev0.record()
            render_fastgs(cv["cam"], gaussians, G["pipe"], G["bg"], G["mult"])
            ev1.record()
            torch.cuda.synchronize()
            lats.append(ev0.elapsed_time(ev1))
    return float(np.mean(lats))


# ------------------------------------------------------- candidate sampling --

def native_selection_sets(gaussians, importance_score, opt, extent):
    """Replicates FastGS native selection math (gaussian_model.py:477-494)."""
    grad_vars = gaussians.xyz_gradient_accum / gaussians.denom
    grad_vars[grad_vars.isnan()] = 0.0
    grads_abs = gaussians.xyz_gradient_accum_abs / gaussians.denom
    grads_abs[grads_abs.isnan()] = 0.0
    grad_norm = torch.norm(grad_vars, dim=-1)
    grad_abs_norm = torch.norm(grads_abs, dim=-1)
    grad_qual = grad_norm >= opt.grad_thresh
    grad_abs_qual = grad_abs_norm >= opt.grad_abs_thresh
    max_scale = gaussians.get_scaling.max(dim=1).values
    clone_qual = max_scale <= opt.dense * extent
    split_qual = max_scale > opt.dense * extent
    metric_mask = importance_score > 5
    return (metric_mask & clone_qual & grad_qual,
            metric_mask & split_qual & grad_abs_qual,
            grad_norm, grad_abs_norm, max_scale)


def stratified_pick(indices, importance, n, rng):
    """Stratified sampling over the importance range (not all top-importance)."""
    if len(indices) <= n:
        return [int(i) for i in indices]
    order = np.argsort(importance[indices])
    sorted_idx = indices[order]
    bins = np.array_split(sorted_idx, n)
    return [int(b[rng.randint(0, len(b) - 1)]) for b in bins if len(b)]


# ------------------------------------------------------- batch construction --

def build_batch(gaussians, scene, opt, pool_cams, importance_score, pruning_score,
                iteration, n_cand, n_probe, cand_seed, force_one_each):
    """Pre-action analysis at the hook: candidate sampling, descriptors,
    per-view ROI / residual descriptors / demand masks. Read-only on model."""
    clone_set, split_set, grad_norm, grad_abs_norm, max_scale = \
        native_selection_sets(gaussians, importance_score, opt, scene.cameras_extent)
    imp = importance_score.to(torch.float32).cpu().numpy()
    n_clone_set, n_split_set = int(clone_set.sum()), int(split_set.sum())
    rng = np.random.RandomState(cand_seed * 1000 + iteration)

    if force_one_each:
        picks = {"clone": [], "split": []}
        if n_clone_set > 0:
            idx = np.where(clone_set.cpu().numpy())[0]
            picks["clone"] = [int(idx[np.argsort(imp[idx])[len(idx) // 2]])]
        if n_split_set > 0:
            idx = np.where(split_set.cpu().numpy())[0]
            picks["split"] = [int(idx[np.argsort(imp[idx])[len(idx) // 2]])]
    else:
        quota_c = min(n_cand // 2, n_clone_set)
        quota_s = min(n_cand - quota_c, n_split_set)
        if quota_c + quota_s < n_cand:  # fill from the richer side
            room_s, room_c = n_split_set - quota_s, n_clone_set - quota_c
            need = n_cand - quota_c - quota_s
            if room_s >= room_c:
                quota_s += min(need, room_s)
            else:
                quota_c += min(need, room_c)
        picks = {"clone": stratified_pick(np.where(clone_set.cpu().numpy())[0], imp, quota_c, rng),
                 "split": stratified_pick(np.where(split_set.cpu().numpy())[0], imp, quota_s, rng)}

    # ---- one render per pool view (pre-action state): radii + residual map
    #      + native FastGS high-residual mask (get_loss > loss_thresh) ----
    pool_data = []
    with torch.no_grad():
        for cam in pool_cams:
            out = render_fastgs(cam, gaussians, G["pipe"], G["bg"], G["mult"])
            gt = cam.original_image.cuda()
            l1_map = torch.mean(torch.abs(out["render"] - gt), dim=0)
            mask = (get_loss(out["render"], gt) > opt.loss_thresh).detach()
            pool_data.append({"cam": cam, "radii": out["radii"],
                              "l1_map": l1_map, "mask": mask})

    candidates = []
    for action in ("clone", "split"):
        for parent_idx in picks[action]:
            cand = build_candidate(gaussians, parent_idx, action, iteration,
                                   importance_score, pruning_score, grad_norm,
                                   grad_abs_norm, pool_data, n_probe)
            if cand is not None:
                candidates.append(cand)
            else:
                print(f"  [warn] candidate idx={parent_idx} ({action}) has no valid pool view; skipped")

    for pd in pool_data:
        pd.pop("l1_map", None)
    torch.cuda.empty_cache()

    return {"candidates": candidates, "n_clone_set": n_clone_set,
            "n_split_set": n_split_set, "iteration": iteration,
            "n_gaussians_at_event": int(gaussians.get_xyz.shape[0])}


def build_candidate(gaussians, parent_idx, action, iteration,
                    importance_score, pruning_score, grad_norm, grad_abs_norm,
                    pool_data, n_probe):
    """Fixed views + ROI + pre-action descriptors for one candidate.
    ROI rule: parent projected 3-sigma screen radius (native `radii`) + 1 tile
    margin (16 px), clamped; identical for Keep/Clone/Split and steps 0/50/100."""
    parent_xyz = gaussians.get_xyz[parent_idx].detach()
    parent_scaling = gaussians.get_scaling[parent_idx].detach()
    parent_rotation = gaussians.get_rotation[parent_idx].detach()
    views, radii_px, res_descs, n_visible = [], [], [], 0
    align_rows = []  # per-view residual<->parent geometry alignment (pre-action)

    for pd in pool_data:
        cam, view_radii = pd["cam"], pd["radii"]
        if int(view_radii[parent_idx]) <= 0:
            continue
        n_visible += 1
        u, v, _, ok = project_to_pixel(cam, parent_xyz)
        if not ok:
            continue
        r_px = float(view_radii[parent_idx])
        box = roi_box(u, v, r_px, cam.image_width, cam.image_height, TILE_MARGIN_PX)
        x0, y0, x1, y1 = box
        if (x1 - x0) < 8 or (y1 - y0) < 8:
            continue
        rd = residual_descriptors(pd["l1_map"], box)
        pg = projected_gaussian_geometry(cam, parent_xyz, parent_scaling,
                                         parent_rotation, radius_native=r_px)
        demand_valid = bool(int(pd["mask"][y0:y1, x0:x1].sum()) > 0)
        views.append({"cam": cam, "box": box, "mask": pd["mask"], "r_px": r_px,
                      "residual": rd, "parent_geom": pg,
                      "demand_valid": demand_valid})
        radii_px.append(r_px)
        if rd is not None:
            res_descs.append(rd)
            if pg is not None and not np.isnan(rd["residual_direction_deg"]):
                d_res = np.deg2rad(rd["residual_direction_deg"])
                d_par = np.deg2rad(pg["major_dir_deg"])
                cos_a = float(np.cos(d_res - d_par))
                cx_r, cy_r = rd["residual_centroid"]
                off = float(np.hypot(cx_r - pg["center"][0], cy_r - pg["center"][1])
                            / max(pg["radius_3sigma_px"], 1e-6))
                align_rows.append({
                    "residual_parent_angle_deg": float(np.degrees(np.arccos(
                        min(1.0, max(-1.0, cos_a))))),
                    "residual_parent_alignment": abs(cos_a),
                    "residual_extent_ratio": rd["residual_extent"] /
                        max(pg["radius_3sigma_px"], 1e-6),
                    "residual_centroid_offset": off,
                    "proj_anisotropy": pg["proj_anisotropy"],
                    "radius_mismatch_px": pg.get("radius_mismatch_px", None),
                })
        if len(views) >= n_probe:
            break
    if not views:
        return None

    scales = gaussians.get_scaling[parent_idx].detach().cpu().numpy()
    sc_max, sc_min = float(scales.max()), float(scales.min())
    aniso = sc_max / sc_min if sc_min > 0 else None
    n_demand_valid = sum(int(vd["demand_valid"]) for vd in views)
    energies = [d["residual_energy_mean"] for d in res_descs] or None
    anisos = [d["residual_anisotropy"] for d in res_descs
              if not np.isnan(d["residual_anisotropy"])] or None

    def _mean_std(rows, key):
        vals = [r[key] for r in rows
                if r.get(key) is not None and not np.isnan(r[key])]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    pa_m, _ = _mean_std(align_rows, "residual_parent_alignment")
    er_m, _ = _mean_std(align_rows, "residual_extent_ratio")
    co_m, _ = _mean_std(align_rows, "residual_centroid_offset")
    pani_m, _ = _mean_std(align_rows, "proj_anisotropy")
    mism = [r["radius_mismatch_px"] for r in align_rows
            if r.get("radius_mismatch_px") is not None]
    pgeoms = [vd["parent_geom"] for vd in views if vd["parent_geom"] is not None]

    return {
        "iteration": iteration,
        "parent_index": int(parent_idx),
        "native_action": action,
        "importance_score": int(importance_score[parent_idx]),
        "pruning_score": float(pruning_score[parent_idx]),
        "grad": float(grad_norm[parent_idx]),
        "grad_abs": float(grad_abs_norm[parent_idx]),
        "opacity": float(gaussians.get_opacity[parent_idx]),
        "scale_x": float(scales[0]), "scale_y": float(scales[1]), "scale_z": float(scales[2]),
        "scale_max": sc_max, "scale_min": sc_min, "scale_anisotropy": aniso,
        "projected_radius_mean": float(np.mean(radii_px)),
        "projected_radius_max": float(np.max(radii_px)),
        "footprint_mean": float(np.mean(radii_px)),  # documented: radius in px
        "footprint_max": float(np.max(radii_px)),
        "footprint_area_mean": float(np.mean([np.pi * r * r for r in radii_px])),
        "proj_anisotropy_mean": pani_m,
        "residual_energy_mean": float(np.mean(energies)) if energies else None,
        "residual_energy_std": float(np.std(energies)) if energies else None,
        "residual_anisotropy_mean": float(np.mean(anisos)) if anisos else None,
        "residual_anisotropy_std": float(np.std(anisos)) if anisos else None,
        "residual_extent_mean": float(np.mean([d["residual_extent"] for d in res_descs])) if res_descs else None,
        "residual_directions_deg": [d["residual_direction_deg"] for d in res_descs],
        "residual_parent_alignment_mean": pa_m,
        "residual_extent_ratio_mean": er_m,
        "residual_centroid_offset_mean": co_m,
        "geom_radius_mismatch_max": float(np.max(mism)) if mism else None,
        "parent_major_dirs_deg": [g["major_dir_deg"] for g in pgeoms],
        "per_view": [{"r_px": vd["r_px"], "box": list(vd["box"]),
                      "demand_valid": vd["demand_valid"],
                      "residual": vd["residual"],
                      "parent_geom": vd["parent_geom"],
                      "alignment": (align_rows[i] if i < len(align_rows) else None)}
                     for i, vd in enumerate(views)],
        "num_visible_views": n_visible,
        "num_valid_local_views": len(views),
        "num_valid_demand_views": n_demand_valid,
        "_views": views,  # runtime-only, stripped before saving
    }


# ---------------------------------------------------------- branch runner --

def run_candidate(diag_iter, ci, cand, snapshot, radii_snap, opt, train_cams,
                  cam_seq, global_probe, args):
    views = cand["_views"]
    parent_idx = cand["parent_index"]
    branch_out = {}

    for branch in ("keep", "clone", "split"):
        seed_all(args.action_seed)
        g = GaussianModel(G["sh_degree"], opt.optimizer_type)
        g.restore(clone_tree(snapshot), opt)

        if branch in ("clone", "split"):
            g.tmp_radii = radii_snap.clone()          # native contract
            single = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            single[parent_idx] = True
            if branch == "clone":
                g.densify_and_clone_fastgs(single, torch.ones_like(single))
            else:
                g.densify_and_split_fastgs(single, torch.ones_like(single), N=2)
            g.tmp_radii = None                        # native cleanup

        n_after = int(g.get_xyz.shape[0])
        seed_all(args.train_seed)
        evals = {0: local_eval(g, views)}

        for i in range(1, args.diag_steps + 1):
            it = diag_iter + i
            cam = train_cams[cam_seq[i - 1]]
            native_train_one_iter(it, cam, g, G["pipe"], G["bg"], opt)
            with torch.no_grad():
                if opt.optimizer_type == "default":
                    g.optimizer_step(it)
                if i in (args.diag_steps // 2, args.diag_steps):
                    evals[i] = local_eval(g, views)

        branch_out[branch] = {"n_after": n_after, "evals": evals,
                              "global": global_eval(g, global_probe),
                              "latency_ms": latency_light(g, views)}
        del g
        torch.cuda.empty_cache()

    def e(b, step, kind, key):
        v = branch_out[b]["evals"][step][kind][key]
        return None if v is None else float(v)

    h = args.diag_steps // 2
    rec = {k: v for k, v in cand.items() if not k.startswith("_")}
    rec["candidate_id"] = f"it{diag_iter}_c{ci:02d}"
    for step, tag in ((0, "0"), (h, str(h)), (args.diag_steps, str(args.diag_steps))):
        for b in ("keep", "clone", "split"):
            rec[f"{b}_local_error_{tag}"] = e(b, step, "support", "l1")
            rec[f"{b}_local_mse_{tag}"] = e(b, step, "support", "mse")
            rec[f"{b}_local_psnr_{tag}"] = e(b, step, "support", "psnr")
            rec[f"{b}_demand_error_{tag}"] = e(b, step, "demand", "l1")
            rec[f"{b}_demand_psnr_{tag}"] = e(b, step, "demand", "psnr")
            rec[f"tile_{b}_{tag}"] = branch_out[b]["evals"][step]["tile_pairs_mean"]
    for b in ("keep", "clone", "split"):
        rec[f"num_gaussians_{b}"] = branch_out[b]["n_after"]
        rec[f"global_PSNR_{b}_100"] = branch_out[b]["global"]["psnr"]
        rec[f"global_SSIM_{b}_100"] = branch_out[b]["global"]["ssim"]
        rec[f"global_LPIPS_{b}_100"] = branch_out[b]["global"]["lpips"]
        rec[f"latency_{b}_ms"] = branch_out[b]["latency_ms"]

    def d(ka, kb):
        a, b_ = rec.get(ka), rec.get(kb)
        return None if (a is None or b_ is None) else float(a - b_)

    rec["deltaQ_support_clone_0"] = d("keep_local_error_0", "clone_local_error_0")
    rec["deltaQ_support_split_0"] = d("keep_local_error_0", "split_local_error_0")
    rec["deltaQ_demand_clone_0"] = d("keep_demand_error_0", "clone_demand_error_0")
    rec["deltaQ_demand_split_0"] = d("keep_demand_error_0", "split_demand_error_0")
    rec["deltaQ_support_clone_100"] = d("keep_local_error_100", "clone_local_error_100")
    rec["deltaQ_support_split_100"] = d("keep_local_error_100", "split_local_error_100")
    rec["deltaQ_demand_clone_100"] = d("keep_demand_error_100", "clone_demand_error_100")
    rec["deltaQ_demand_split_100"] = d("keep_demand_error_100", "split_demand_error_100")
    for tag in ("0", "100"):
        rec[f"deltaTile_clone_{tag}"] = d(f"tile_clone_{tag}", f"tile_keep_{tag}")
        rec[f"deltaTile_split_{tag}"] = d(f"tile_split_{tag}", f"tile_keep_{tag}")
    rec["tile_action_gap_0"] = d("deltaTile_split_0", "deltaTile_clone_0")
    rec["tile_action_gap_100"] = d("deltaTile_split_100", "deltaTile_clone_100")
    qa, qb = rec["deltaQ_demand_split_100"], rec["deltaQ_demand_clone_100"]
    rec["quality_action_gap"] = None if (qa is None or qb is None) else float(qa - qb)

    errs = {b: (rec.get(f"{b}_demand_error_100")
                if rec.get(f"{b}_demand_error_100") is not None
                else rec.get(f"{b}_local_error_100"))
            for b in ("keep", "clone", "split")}
    valid = {b: v for b, v in errs.items() if v is not None}
    rec["oracle_basis"] = "demand" if rec.get("keep_demand_error_100") is not None else "support"
    rec["oracle_winner"] = min(valid, key=valid.get) if valid else None
    return rec


# ---------------------------------------------------------------- CSV out --

CSV_COLS = [
    "scene", "iteration", "candidate_id", "parent_index", "native_action",
    "importance_score", "pruning_score", "grad", "grad_abs", "opacity",
    "scale_x", "scale_y", "scale_z", "scale_max", "scale_min", "scale_anisotropy",
    "footprint_mean", "footprint_max", "footprint_area_mean",
    "projected_radius_mean", "projected_radius_max",
    "residual_energy_mean", "residual_energy_std",
    "residual_anisotropy_mean", "residual_anisotropy_std", "residual_extent_mean",
    "num_visible_views", "num_valid_local_views", "num_valid_demand_views",
    "keep_local_error_0", "clone_local_error_0", "split_local_error_0",
    "keep_local_error_50", "clone_local_error_50", "split_local_error_50",
    "keep_local_error_100", "clone_local_error_100", "split_local_error_100",
    "keep_demand_error_0", "clone_demand_error_0", "split_demand_error_0",
    "keep_demand_error_100", "clone_demand_error_100", "split_demand_error_100",
    "deltaQ_support_clone_100", "deltaQ_support_split_100",
    "deltaQ_demand_clone_100", "deltaQ_demand_split_100",
    "quality_action_gap", "oracle_winner", "oracle_basis",
    "tile_keep_0", "tile_clone_0", "tile_split_0",
    "deltaTile_clone_0", "deltaTile_split_0", "tile_action_gap_0",
    "tile_keep_100", "tile_clone_100", "tile_split_100",
    "deltaTile_clone_100", "deltaTile_split_100", "tile_action_gap_100",
    "global_PSNR_keep_100", "global_PSNR_clone_100", "global_PSNR_split_100",
    "global_SSIM_keep_100", "global_SSIM_clone_100", "global_SSIM_split_100",
    "global_LPIPS_keep_100", "global_LPIPS_clone_100", "global_LPIPS_split_100",
    "latency_keep_ms", "latency_clone_ms", "latency_split_ms",
    "num_gaussians_keep", "num_gaussians_clone", "num_gaussians_split",
]


def write_csv(records, scene, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in records:
            row = []
            for c in CSV_COLS:
                v = scene if c == "scene" else r.get(c, None)
                if v is None:
                    row.append("NA")
                elif isinstance(v, float):
                    row.append(f"{v:.8g}")
                else:
                    row.append(v)
            w.writerow(row)
    return path


# ------------------------------------------------------------------- main --

def main():
    parser = ArgumentParser("Paper B B2: multi-candidate local action diagnostic")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--diag_iters", type=str, default="1000,1500,2000")
    parser.add_argument("--n_cand", type=int, default=20)
    parser.add_argument("--pool_size", type=int, default=30)
    parser.add_argument("--n_probe", type=int, default=8)
    parser.add_argument("--diag_steps", type=int, default=100)
    parser.add_argument("--force_one_each", action="store_true")
    parser.add_argument("--warmup_seed", type=int, default=0)
    parser.add_argument("--action_seed", type=int, default=1234)
    parser.add_argument("--train_seed", type=int, default=1234)
    parser.add_argument("--camseq_seed", type=int, default=2024)
    parser.add_argument("--cand_seed", type=int, default=777)
    parser.add_argument("--out_tag", type=str, default="action_diag_v2")
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    diag_iters = sorted(int(x) for x in args.diag_iters.split(",") if x.strip())
    for it in diag_iters:
        assert it > opt.densify_from_iter and it % opt.densification_interval == 0 \
            and it < opt.densify_until_iter, f"diag_iter {it} is not a native densification event"
    assert opt.optimizer_type == "default"

    install_c_proxy()
    seed_all(args.warmup_seed)

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    G.update({"pipe": pipe, "bg": bg, "mult": opt.mult,
              "loss_thresh": opt.loss_thresh, "sh_degree": dataset.sh_degree})

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    global_probe = (test_cams[:5] if test_cams and len(test_cams) > 0 else train_cams[:5])
    global_probe_src = "test" if (test_cams and len(test_cams) > 0) else "train"
    pool_cams = train_cams[:args.pool_size]
    print(f"[v2] pool={len(pool_cams)} train cams, global probe={len(global_probe)} ({global_probe_src})")

    cam_seq_rng = random.Random(args.camseq_seed)
    cam_seq = [cam_seq_rng.randint(0, len(train_cams) - 1) for _ in range(args.diag_steps)]

    # ---------------- Phase 1: native warmup with diagnostic hooks --------
    batches = {}
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    cand_seed = args.cand_seed

    for iteration in range(1, diag_iters[-1] + 1):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = random.randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        _, vpt, vis_filter, radii = native_train_one_iter(
            iteration, viewpoint_cam, gaussians, pipe, bg, opt)

        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[vis_filter] = torch.max(
                    gaussians.max_radii2D[vis_filter], radii[vis_filter])
                gaussians.add_densification_stats(vpt, vis_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    my_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_stack)
                    importance_score, pruning_score = compute_gaussian_score_fastgs(
                        camlist, gaussians, pipe, bg, opt, DENSIFY=True)

                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if iteration in diag_iters:
                        print(f"\n[v2] === diagnostic hook at iteration {iteration} ===")
                        batch = build_batch(gaussians, scene, opt, pool_cams,
                                            importance_score, pruning_score, iteration,
                                            args.n_cand, args.n_probe, cand_seed,
                                            args.force_one_each)
                        cand_seed += 1
                        batch["snapshot"] = clone_tree(gaussians.capture(opt.optimizer_type))
                        batch["radii"] = radii.clone()
                        batches[iteration] = batch

                    # native densification always continues on the main model
                    # (identical to train.py:139-145; keeps later diag batches
                    # at the true native state)
                    gaussians.densify_and_prune_fastgs(
                        max_screen_size=size_threshold, min_opacity=0.005,
                        extent=scene.cameras_extent, radii=radii, args=opt,
                        importance_score=importance_score, pruning_score=pruning_score)

                if iteration % opt.opacity_reset_interval == 0 or \
                        (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 3000 == 0 and 15_000 < iteration < 30_000:
                my_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_stack)
                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
                gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)

            if opt.optimizer_type == "default":
                gaussians.optimizer_step(iteration)

    del gaussians
    torch.cuda.empty_cache()

    # ---------------- Phase 2: per-candidate Keep/Clone/Split -------------
    records = []
    for diag_iter in diag_iters:
        batch = batches[diag_iter]
        snapshot, radii_snap = batch["snapshot"], batch["radii"]
        n_before = int(snapshot[1].shape[0])
        print(f"\n[v2] batch iter={diag_iter}: {len(batch['candidates'])} candidates, "
              f"N={n_before} (clone set={batch['n_clone_set']}, split set={batch['n_split_set']})")

        for ci, cand in enumerate(batch["candidates"]):
            rec = run_candidate(diag_iter, ci, cand, snapshot, radii_snap,
                                opt, train_cams, cam_seq, global_probe, args)
            records.append(rec)
            f2 = (lambda x: "NA" if x is None else f"{x:+.6f}")
            print(f"  cand {ci} ({cand['native_action']}, idx={cand['parent_index']}): "
                  f"dQ_dem c={f2(rec['deltaQ_demand_clone_100'])} "
                  f"s={f2(rec['deltaQ_demand_split_100'])} | "
                  f"dTile@0 c={f2(rec['deltaTile_clone_0'])} s={f2(rec['deltaTile_split_0'])} | "
                  f"winner={rec['oracle_winner']}")

    # ---------------- Phase 3: save ---------------------------------------
    scene_name = os.path.abspath(dataset.source_path)
    out = {
        "scene": scene_name,
        "global_probe_source": global_probe_src,
        "config": {
            "densification_interval": opt.densification_interval,
            "grad_thresh": opt.grad_thresh, "grad_abs_thresh": opt.grad_abs_thresh,
            "dense": opt.dense, "mult": opt.mult, "loss_thresh": opt.loss_thresh,
            "diag_steps": args.diag_steps, "n_probe": args.n_probe,
            "pool_size": args.pool_size, "tile_margin_px": TILE_MARGIN_PX,
            "camseq_seed": args.camseq_seed, "action_seed": args.action_seed,
            "train_seed": args.train_seed, "warmup_seed": args.warmup_seed,
            "cand_seed": args.cand_seed,
        },
        "batches": {str(k): {"candidates": [{ck: cv for ck, cv in c.items() if not ck.startswith("_")}
                                            for c in v["candidates"]],
                             **{kk: vv for kk, vv in v.items()
                                if kk not in ("snapshot", "radii", "candidates")}}
                    for k, v in batches.items()},
        "records": records,
    }
    os.makedirs("project_md", exist_ok=True)
    json_path = f"project_md/{args.out_tag}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=1)
    csv_path = write_csv(records, scene_name, f"project_md/{args.out_tag}.csv")
    print(f"\n[v2] saved {json_path} , {csv_path} , {len(records)} candidate records")


if __name__ == "__main__":
    main()
