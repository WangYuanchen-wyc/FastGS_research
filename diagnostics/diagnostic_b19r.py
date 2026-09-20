#
# Paper B - B19-R: Reliable Density Control Diagnostic
#
#   A. Densification: for every native VCD trigger parent, capture the PER-VIEW
#      evidence e_1..e_10 (accum_metric_counts of that view) produced by the
#      SAME renders native compute_gaussian_score_fastgs already makes — zero
#      extra renders. Concentration stats: nonzero-support views, Top1/2/3
#      share, normalized entropy. Children fate tracked by constructive lineage
#      (clone i-th parent -> i-th appended child; split j-th parent -> slots
#      2j, 2j+1) and chained proximity checkpoints (tol=0.05).
#
#   B. Pruning: FastGS removes opacity < 0.1 in final_prune_fastgs, which native
#      train.py runs every 3000 iters in (15000, 30000). For every candidate at
#      every round, capture multi-view support from the SAME VCD pass (per-view
#      evidence + frustum visibility over the same 10 sampled views, no extra
#      renders). Classify Low-opacity+Low-support vs Low-opacity+Still-supported
#      (rule fixed in analysis from the actual distribution).
#
# Loop is native-faithful to train.py order:
#   native_train_one_iter -> [densify block if it<15000] -> [opacity reset if
#   it<15000 and it%3000==0] -> [final prune if it%3000==0 and 15000<it<30000]
#   -> gaussians.optimizer_step(it).
#
# NOTE: earlier B18-series loops omitted optimizer_step (frozen-model dynamics);
# this script steps natively. GPU: 0
#

import os, sys, random, csv, math
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.image_utils import psnr
from utils.fast_utils import (compute_gaussian_score_fastgs, sampling_cameras,
                              get_loss, compute_photometric_loss)
from utils.general_utils import inverse_sigmoid
from arguments import ModelParams, PipelineParams, OptimizationParams
from diagnostics.common import install_c_proxy, seed_all, native_train_one_iter

OUT = "paper_b/b19_reliable_density_control"
EARLY_START, EARLY_END = 1000, 3000
TOL = 0.05
NVIEWS = 10


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
    """Native event densification (0.005 opacity prune etc.).
    Returns (n_clone, n_split, clone_kids, split_kids, keep_c, keep_s)."""
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

    n_newborn = n_clone + 2 * n_split
    n_cur = g.get_xyz.shape[0]
    clone_kids = split_kids = None
    if n_newborn > 0:
        tail = g.get_xyz[n_cur - n_newborn:].detach().clone()
        clone_kids = tail[:n_clone]
        split_kids = tail[n_clone:]

    keep_c = keep_s = None
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
        mask_full = torch.logical_and(prune_mask, sel)
        if n_newborn > 0:
            keep = ~mask_full[n - n_newborn:]
            keep_c, keep_s = keep[:n_clone], keep[n_clone:]
        g.prune_points(mask_full)
    g._opacity = safe_replace_opacity(
        g, inverse_sigmoid(torch.min(g.get_opacity, torch.ones_like(g.get_opacity) * 0.8))
    )["opacity"]
    torch.cuda.empty_cache()
    return n_clone, n_split, clone_kids, split_kids, keep_c, keep_s


@torch.no_grad()
def vcd_pass_perview(camlist, gaussians, pipe, bg, args, densify):
    """Identical computation to utils.fast_utils.compute_gaussian_score_fastgs,
    additionally returning the per-view evidence matrix E (N, K)."""
    full_counts = None
    full_score = None
    per_view = []
    for view in range(len(camlist)):
        cam = camlist[view]
        render_image = render_fastgs(cam, gaussians, pipe, bg, args.mult)["render"]
        photometric_loss = compute_photometric_loss(cam, render_image)
        gt_image = cam.original_image.cuda()
        l1_loss_norm = get_loss(render_image, gt_image)
        metric_map = (l1_loss_norm > args.loss_thresh).int()
        render_pkg = render_fastgs(cam, gaussians, pipe, bg, args.mult,
                                   get_flag=True, metric_map=metric_map)
        acc = render_pkg["accum_metric_counts"].detach()
        per_view.append(acc.reshape(-1))
        full_counts = acc.clone() if full_counts is None else full_counts + acc
        full_score = (photometric_loss * acc.clone() if full_score is None
                      else full_score + photometric_loss * acc)
    pruning_score = (full_score - torch.min(full_score)) / (
        torch.max(full_score) - torch.min(full_score))
    imp = None
    if densify:
        imp = torch.div(full_counts, len(camlist), rounding_mode="floor")
    E = torch.stack(per_view, dim=1) if per_view else None  # (N, K)
    return imp, pruning_score, E


def evidence_stats(e):
    """e: (K,) tensor of per-view evidence. -> (nz, top1, top2, top3, entropy_n)."""
    tot = float(e.sum())
    nz = int((e > 0).sum())
    if tot <= 0:
        return nz, 0.0, 0.0, 0.0, 0.0
    s, _ = torch.sort(e, descending=True)
    top1 = float(s[0]) / tot
    top2 = float(s[:2].sum()) / tot
    top3 = float(s[:3].sum()) / tot
    p = e[e > 0] / tot
    H = float(-(p * torch.log(p)).sum()) / math.log(len(e))
    return nz, top1, top2, top3, H


@torch.no_grad()
def visible_view_counts(xyz, cams):
    """Frustum visibility over the given cams (projection in-bounds, w>0).
    Mirrors utils.graphics_utils.geom_transform_points convention."""
    n = xyz.shape[0]
    cnt = torch.zeros(n, device="cuda")
    ones = torch.ones(n, 1, dtype=xyz.dtype, device="cuda")
    ph = torch.cat([xyz, ones], dim=1)
    for cam in cams:
        out = ph @ cam.full_proj_transform
        denom = out[:, 3:4] + 1e-7
        pts = out[:, :3] / denom
        vis = (denom[:, 0] > 1e-6) & (pts[:, 0].abs() < 1.1) & (pts[:, 1].abs() < 1.1)
        cnt += vis.float()
    return cnt


@torch.no_grad()
def match_nearest(query, ref, chunk=8192):
    k = query.shape[0]
    best_d = torch.full((k,), float("inf"), device="cuda")
    best_i = torch.zeros(k, dtype=torch.long, device="cuda")
    for i in range(0, ref.shape[0], chunk):
        d = torch.cdist(query, ref[i:i + chunk])
        dmin, imin = d.min(dim=1)
        upd = dmin < best_d
        best_d[upd] = dmin[upd]
        best_i[upd] = imin[upd] + i
    return best_d, best_i


@torch.no_grad()
def eval_quick(g, test_cams, pipe, bg, mult):
    ps_ = []
    for cam in test_cams[:5]:
        img = render_fastgs(cam, g, pipe, bg, mult)["render"]
        gt = cam.original_image.cuda()
        ps_.append(float(psnr(torch.clamp(img, 0, 1), torch.clamp(gt, 0, 1)).mean()))
    return float(np.mean(ps_))


def is_checkpoint(it):
    return it <= 6500 or it % 1000 == 0 or it % 3000 == 0


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
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(f"{OUT}/data", exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}", flush=True)
        install_c_proxy()
        seed_all(seed)
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)
        extent = scene.cameras_extent
        test_cams = scene.getTestCameras()[:10] if scene.getTestCameras() else scene.getTrainCameras()[:5]
        vp_stack = scene.getTrainCameras().copy()
        vp_idx = list(range(len(vp_stack)))

        event_num = 0
        round_num = 0
        ev_rows = []       # densify evidence rows (per trigger parent occurrence)
        cand_rows = []     # final-prune candidate rows
        round_stats = []
        child_rows = []    # early children fate metadata (grown with tensors)
        child_ref = child_alive = child_last = None
        final_matched = None

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
                        imp, pru, E = vcd_pass_perview(cl, gaussians, pipe, bg, opt, densify=True)

                        # ---- child fate checkpoint (pre-densify state) ----
                        if child_ref is not None and child_ref.shape[0] > 0 \
                                and is_checkpoint(it) and bool(child_alive.any()):
                            alive_idx = torch.where(child_alive)[0]
                            d, j = match_nearest(child_ref[alive_idx],
                                                 gaussians.get_xyz.detach())
                            still = d < TOL
                            child_alive[alive_idx[~still]] = False
                            sur = alive_idx[still]
                            child_ref[sur] = gaussians.get_xyz.detach()[j[still]]
                            child_last[sur] = it

                        # ---- trigger parents + evidence ----
                        grad_vars = gaussians.xyz_gradient_accum / gaussians.denom
                        grad_vars[grad_vars.isnan()] = 0.0
                        grads_abs = gaussians.xyz_gradient_accum_abs / gaussians.denom
                        grads_abs[grads_abs.isnan()] = 0.0
                        grad_qual = torch.norm(grad_vars, dim=-1) >= opt.grad_thresh
                        grad_abs_qual = torch.norm(grads_abs, dim=-1) >= opt.grad_abs_thresh
                        max_scale = gaussians.get_scaling.max(dim=1).values
                        metric_mask = imp > 5
                        split_set = metric_mask & (max_scale > opt.dense * extent) & grad_abs_qual
                        clone_set = metric_mask & (max_scale <= opt.dense * extent) & grad_qual

                        event_parent_rows = {}
                        for ptype, pset in (("split", split_set), ("clone", clone_set)):
                            pidx = torch.where(pset)[0]
                            for jj in range(pidx.shape[0]):
                                gi = int(pidx[jj])
                                nz, t1, t2, t3, H = evidence_stats(E[gi])
                                row = {
                                    "seed": seed, "event": event_num, "iteration": it,
                                    "parent_id": f"s{seed}e{event_num}{ptype}{gi}",
                                    "parent_type": ptype,
                                    "nonzero_views": nz,
                                    "total_evidence": round(float(E[gi].sum()), 3),
                                    "top1_share": round(t1, 4), "top2_share": round(t2, 4),
                                    "top3_share": round(t3, 4), "entropy_norm": round(H, 4),
                                    "imp": int(imp[gi].reshape(-1)[0]),
                                    "n_children_born": 0, "n_children_kept": 0,
                                }
                                ev_rows.append(row)
                                event_parent_rows[row["parent_id"]] = row
                        n_split = int(split_set.sum()); n_clone = int(clone_set.sum())

                        # ---- densify + lineage ----
                        n_clone, n_split, clone_kids, split_kids, keep_c, keep_s = \
                            apply_native_densify(gaussians, opt, it, radii, imp, pru, extent)

                        for ptype, pidx_all, kids, keep in (
                                ("split", torch.where(split_set)[0], split_kids, keep_s),
                                ("clone", torch.where(clone_set)[0], clone_kids, keep_c)):
                            if kids is None or kids.shape[0] == 0:
                                continue
                            kids_cpu = kids.cpu()
                            keep_cpu = (torch.ones(kids.shape[0], dtype=torch.bool)
                                        if keep is None else keep.cpu())
                            for j in range(pidx_all.shape[0]):
                                pid = f"s{seed}e{event_num}{ptype}{int(pidx_all[j])}"
                                slots = [2 * j, 2 * j + 1] if ptype == "split" else [j]
                                kept_slots = [s for s in slots
                                              if s < kids_cpu.shape[0] and bool(keep_cpu[s])]
                                row = event_parent_rows.get(pid)
                                if row is not None:
                                    row["n_children_born"] = len(slots)
                                    row["n_children_kept"] = len(kept_slots)
                                if EARLY_START <= it <= EARLY_END and kept_slots:
                                    kept_pos = torch.stack(
                                        [kids_cpu[s] for s in kept_slots]).to("cuda")
                                    for s in kept_slots:
                                        child_rows.append({
                                            "child_id": f"{pid}c{s if ptype == 'split' else 0}",
                                            "parent_id": pid, "child_type": ptype,
                                            "birth_event": event_num, "birth_it": it,
                                        })
                                    new_ref = kept_pos
                                    new_alive = torch.ones(kept_pos.shape[0],
                                                           dtype=torch.bool, device="cuda")
                                    new_last = torch.full((kept_pos.shape[0],), it,
                                                          dtype=torch.long, device="cuda")
                                    if child_ref is None:
                                        child_ref, child_alive, child_last = \
                                            new_ref, new_alive, new_last
                                    else:
                                        child_ref = torch.cat([child_ref, new_ref])
                                        child_alive = torch.cat([child_alive, new_alive])
                                        child_last = torch.cat([child_last, new_last])

                        if it % 1000 == 0:
                            ps = eval_quick(gaussians, test_cams, pipe, bg, opt.mult)
                            print(f"  s{seed} it={it} ev={event_num} trig={n_split}/{n_clone} "
                                  f"children={len(child_rows)} PSNR={ps:.2f} "
                                  f"#GS={gaussians.get_xyz.shape[0]}", flush=True)

                if it < opt.densify_until_iter and it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

                if it % 3000 == 0 and it > 15_000 and it < 30_000:
                    round_num += 1
                    my = scene.getTrainCameras().copy()
                    cl = sampling_cameras(my)
                    _, pru, E = vcd_pass_perview(cl, gaussians, pipe, bg, opt, densify=False)

                    # ---- child fate checkpoint around the prune round ----
                    if child_ref is not None and child_ref.shape[0] > 0 \
                            and bool(child_alive.any()):
                        alive_idx = torch.where(child_alive)[0]
                        d, j = match_nearest(child_ref[alive_idx],
                                             gaussians.get_xyz.detach())
                        still = d < TOL
                        child_alive[alive_idx[~still]] = False
                        sur = alive_idx[still]
                        child_ref[sur] = gaussians.get_xyz.detach()[j[still]]
                        child_last[sur] = it

                    # ---- opacity<0.1 candidates: multi-view support ----
                    opa = gaussians.get_opacity.detach().squeeze(-1)
                    cand_mask = opa < 0.1
                    n_cand = int(cand_mask.sum())
                    pru1 = pru.reshape(-1)
                    n_removed_score = int(((opa >= 0.1) & (pru1 > 0.9)).sum())
                    if n_cand > 0:
                        cand_idx = torch.where(cand_mask)[0]
                        E_cand = E[cand_idx]                       # (C, K)
                        nz = (E_cand > 0).sum(dim=1)
                        tot = E_cand.sum(dim=1)
                        srt, _ = torch.sort(E_cand, dim=1, descending=True)
                        t1 = srt[:, 0] / tot.clamp(min=1e-9)
                        t3 = srt[:, :3].sum(dim=1) / tot.clamp(min=1e-9)
                        vis_cnt = visible_view_counts(
                            gaussians.get_xyz.detach()[cand_idx], cl)
                        opa_c = opa[cand_idx]
                        pru_c = pru1[cand_idx]
                        for i in range(n_cand):
                            cand_rows.append({
                                "seed": seed, "round": round_num, "round_it": it,
                                "opacity": round(float(opa_c[i]), 5),
                                "pruning_score": round(float(pru_c[i]), 4),
                                "nonzero_views": int(nz[i]),
                                "total_evidence": round(float(tot[i]), 3),
                                "top1_share": round(float(t1[i]), 4),
                                "top3_share": round(float(t3[i]), 4),
                                "visible_views": int(vis_cnt[i]),
                            })
                    round_stats.append({"seed": seed, "round": round_num, "it": it,
                                        "n_gs": gaussians.get_xyz.shape[0],
                                        "n_candidates": n_cand,
                                        "n_score_only_removed": n_removed_score})
                    gaussians.final_prune_fastgs(0.1, pruning_score=pru)
                    print(f"  s{seed} it={it} round={round_num} cand(opacity<0.1)="
                          f"{n_cand} #GS->{gaussians.get_xyz.shape[0]}", flush=True)

            gaussians.optimizer_step(it)

        # ---- final fate matching (post-training model) ----
        if child_ref is not None and child_ref.shape[0] > 0:
            final_matched = torch.full((child_ref.shape[0],), -1,
                                       dtype=torch.long, device="cuda")
            alive_idx = torch.where(child_alive)[0]
            if alive_idx.numel() > 0:
                d, j = match_nearest(child_ref[alive_idx], gaussians.get_xyz.detach())
                still = d < TOL
                child_alive[alive_idx[~still]] = False
                sur = alive_idx[still]
                child_ref[sur] = gaussians.get_xyz.detach()[j[still]]
                child_last[sur] = args.max_iters
                final_matched[sur] = j[still]

        # ---- write per-seed outputs ----
        fate_rows = []
        final_matched_cpu = final_matched.cpu().tolist() if final_matched is not None else []
        final_op_cpu = (gaussians.get_opacity.detach().squeeze(-1).cpu().tolist()
                        if final_matched is not None else [])
        for i, m in enumerate(child_rows):
            alive = bool(child_alive[i])
            last_alive = int(child_last[i])
            fidx = final_matched_cpu[i]
            fop = float(final_op_cpu[fidx]) if (alive and fidx >= 0) else float("nan")
            fate_rows.append({
                "seed": seed, "child_id": m["child_id"], "parent_id": m["parent_id"],
                "child_type": m["child_type"], "birth_event": m["birth_event"],
                "birth_it": m["birth_it"],
                "survived_500": int(last_alive >= m["birth_it"] + 500),
                "survived_1000": int(last_alive >= m["birth_it"] + 1000),
                "survived_3000": int(last_alive >= m["birth_it"] + 3000),
                "survived_30k": int(alive),
                "final_opacity": fop, "visible_01": int(alive and fop >= 0.1),
                "last_alive_it": last_alive,
                "lifetime_iters": (args.max_iters - m["birth_it"]) if alive
                                  else (last_alive - m["birth_it"]),
                "censored": int(alive),
            })
        with open(f"{OUT}/data/densify_view_evidence_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ev_rows[0].keys()))
            w.writeheader(); w.writerows(ev_rows)
        with open(f"{OUT}/data/densify_child_fate_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fate_rows[0].keys()))
            w.writeheader(); w.writerows(fate_rows)
        with open(f"{OUT}/data/prune_multiview_support_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "round", "round_it", "opacity",
                                              "pruning_score", "nonzero_views",
                                              "total_evidence", "top1_share",
                                              "top3_share", "visible_views"])
            w.writeheader()
            w.writerows(cand_rows)
        with open(f"{OUT}/data/round_stats_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(round_stats[0].keys()))
            w.writeheader(); w.writerows(round_stats)
        print(f"  s{seed} saved: {len(ev_rows)} trigger-parent evidence rows, "
              f"{len(fate_rows)} children, {len(cand_rows)} prune candidates",
              flush=True)
        torch.cuda.empty_cache()

    print(f"\n[b19r] done", flush=True)


if __name__ == "__main__":
    main()
