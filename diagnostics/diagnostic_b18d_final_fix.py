#
# Paper B - B18-D Final Fix: One-shot vs Persistent Child Fate
#
# Question: do children of one-shot vs persistent demand differ in long-term value?
#
# Design:
#   - Same config as B18-D Fix official run (room, grad_abs_thresh=0.0008,
#     highfeature_lr=0.02, interval=500) -> identical trajectory (verified
#     bit-exact in B18-D Fix). Hard-assert per-event n_split matches
#     ../fix/data/trigger_events.csv to guarantee comparability.
#   - Parent classification (chain-at-birth): at each event, a trigger's chain =
#     1 + chain of the previous event's trigger nearest within TOL=0.05 (same
#     matching rule as B18-D/Fix), else 1. Group = clamp(chain,1,3):
#     One-shot=1, Persistent-2=2, Persistent-3>=3. A birth's group is the
#     parent's chain at the birth event.
#   - Lineage (constructive, no estimation): scene/gaussian_model.py appends
#     clone children first (i-th clone parent -> i-th appended clone), then
#     split children (j-th split parent -> appended slots 2j, 2j+1), then
#     removes parents. Verified: clone child position == parent position exactly.
#     Children pruned at birth (rare; multinomial weights can in principle hit
#     newborn slots) are recorded and excluded from fate.
#   - Child fate by chained proximity tracking: at checkpoints (every event up
#     to it=6500, then every 2000 iters, plus final), alive children's reference
#     positions are matched to the current model (TOL=0.05); matched references
#     UPDATE to the matched point (tracks drift). survival@X = matched at
#     checkpoint birth_it+X. Lifetime = last matched checkpoint - birth_it
#     (censored at 30000).
#   - Visibility: final opacity >= 0.1 among survivors (documented proxy).
#
# Outputs -> paper_b/b18_densification_persistence/final_fix/data/
#   lineage_s{seed}.csv  parents_s{seed}.csv  fate_s{seed}.csv
#
# GPU: 6
#

import os, sys, random, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.image_utils import psnr
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.general_utils import inverse_sigmoid
from arguments import ModelParams, PipelineParams, OptimizationParams
from diagnostics.common import install_c_proxy, seed_all, native_train_one_iter

OUT = "paper_b/b18_densification_persistence/final_fix"
REF_TRIGGERS = "paper_b/b18_densification_persistence/fix/data/trigger_events.csv"
EARLY_START, EARLY_END = 1000, 3000
TOL = 0.05
GROUP_NAMES = {1: "One-shot", 2: "Persistent-2", 3: "Persistent-3"}


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
    """Native FastGS densification.

    Returns (n_clone, n_split, clone_kids, split_kids, keep_c, keep_s):
      clone_kids: (n_clone,3)  appended clone children, in clone_set True-order
      split_kids: (2*n_split,3) appended split children; slots 2j,2j+1 -> j-th
                  split parent (True-order)
      keep_c/keep_s: bool masks of who survived this event's final prune
                  (None = nothing pruned = all kept)
    """
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
def match_nearest(query, ref, chunk=8192):
    """Nearest ref point for each query (both cuda fp32). Chunked over ref."""
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
    return it <= 6500 or it % 1000 == 0


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

    # reference trigger counts from the verified B18-D Fix official run
    ref_trig = {}
    if os.path.exists(REF_TRIGGERS):
        with open(REF_TRIGGERS) as f:
            for r in csv.DictReader(f):
                ref_trig[(int(r["seed"]), int(r["event"]))] = int(r["n_trigger"])
    else:
        print(f"[warn] reference {REF_TRIGGERS} not found; skipping cross-check")

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

        prev_trig_xyz = None   # (M,3) cuda, previous event trigger positions
        prev_chain = None      # (M,) long, previous event trigger chains
        event_num = 0

        # child fate state (grown per event; early births only)
        child_ref = None       # (C,3) cuda reference positions (chained)
        child_alive = None     # (C,) bool cuda
        child_last_alive = None  # (C,) long cuda, last matched checkpoint it
        child_rows = []        # metadata per child (aligned with tensors)
        parent_rows = []
        lineage_rows = []

        for it in range(1, args.max_iters + 1):
            if not vp_stack:
                vp_stack = scene.getTrainCameras().copy()
                vp_idx = list(range(len(vp_stack)))
            r = random.randint(0, len(vp_idx) - 1)
            cam = vp_stack.pop(r)
            _ = vp_idx.pop(r)
            gaussians.update_learning_rate(it)
            if it % 1000 == 0:
                gaussians.oneupSHdegree()
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

                        # ---- trigger sets (pre-mutation; stats valid here) ----
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

                        trig_mask = split_set | clone_set
                        trig_xyz = gaussians.get_xyz[trig_mask].detach()
                        n_trig = int(trig_mask.sum())

                        # ---- parent chain (same matching rule as B18-D/Fix) ----
                        chain = torch.ones(n_trig, dtype=torch.long, device="cuda")
                        if prev_trig_xyz is not None and prev_trig_xyz.shape[0] > 0 \
                                and n_trig > 0:
                            d, near_idx = match_nearest(trig_xyz, prev_trig_xyz)
                            near = d < TOL
                            chain[near] = prev_chain[near_idx[near]] + 1
                        group = chain.clamp(max=3)  # 1/2/3+

                        # ---- fate checkpoint for previously born children ----
                        if child_ref is not None and child_ref.shape[0] > 0 \
                                and is_checkpoint(it) and bool(child_alive.any()):
                            alive_idx = torch.where(child_alive)[0]
                            d, j = match_nearest(child_ref[alive_idx],
                                                 gaussians.get_xyz.detach())
                            still = d < TOL
                            dead_local = alive_idx[~still]
                            child_alive[dead_local] = False
                            sur_local = alive_idx[still]
                            child_ref[sur_local] = gaussians.get_xyz.detach()[j[still]]
                            child_last_alive[sur_local] = it

                        # ---- lineage bookkeeping ----
                        split_idx = torch.where(split_set)[0]
                        clone_idx = torch.where(clone_set)[0]

                        def groups_for(mask):
                            # group of each parent in `mask`, via its trigger entry
                            if not mask.any():
                                return []
                            mpos = gaussians.get_xyz[mask].detach()
                            _, jj = match_nearest(mpos, trig_xyz)
                            return group[jj].cpu().tolist()

                        sp_groups = groups_for(split_set)
                        cl_groups = groups_for(clone_set)

                        # ---- apply native densification ----
                        n_clone, n_split, clone_kids, split_kids, keep_c, keep_s = \
                            apply_native_densify(gaussians, opt, it, radii, imp, pru, extent)

                        # hard cross-check vs reference run (trajectory identity)
                        if (seed, event_num) in ref_trig and \
                                ref_trig[(seed, event_num)] != n_split:
                            raise RuntimeError(
                                f"trajectory drift: seed {seed} ev {event_num} "
                                f"n_split={n_split} != ref {ref_trig[(seed, event_num)]}")

                        # ---- record lineage ----
                        n_pruned_at_birth = 0
                        ev_parents = {"split": (split_idx, sp_groups, "split"),
                                      "clone": (clone_idx, cl_groups, "clone")}
                        for ptype, (pidx, pgroups, _) in ev_parents.items():
                            if ptype == "split":
                                kids = split_kids; keep = keep_s
                                n_born = 2 * n_split
                            else:
                                kids = clone_kids; keep = keep_c
                                n_born = n_clone
                            if kids is None or kids.shape[0] == 0:
                                continue
                            kids_cpu = kids.cpu()
                            if keep is None:
                                keep_cpu = torch.ones(kids.shape[0], dtype=torch.bool)
                            else:
                                keep_cpu = keep.cpu()
                            for j in range(pidx.shape[0]):
                                gname = GROUP_NAMES[int(pgroups[j])]
                                pid = f"s{seed}e{event_num}{ptype}{int(pidx[j])}"
                                slots = [2 * j, 2 * j + 1] if ptype == "split" else [j]
                                n_kept_here = 0
                                for slot in slots:
                                    if slot >= kids_cpu.shape[0]:
                                        break
                                    cpos = kids_cpu[slot]
                                    cid = f"{pid}c{slot if ptype == 'split' else 0}"
                                    if not bool(keep_cpu[slot]):
                                        n_pruned_at_birth += 1
                                        continue
                                    lineage_rows.append({
                                        "seed": seed, "parent_id": pid,
                                        "parent_group": gname,
                                        "birth_event": event_num, "birth_it": it,
                                        "child_id": cid, "child_type": ptype,
                                        "child_x": round(float(cpos[0]), 5),
                                        "child_y": round(float(cpos[1]), 5),
                                        "child_z": round(float(cpos[2]), 5),
                                    })
                                    n_kept_here += 1
                                parent_rows.append({
                                    "seed": seed, "event": event_num, "iteration": it,
                                    "parent_id": pid, "group": gname,
                                    "parent_type": ptype,
                                    "n_children_born": len(slots),
                                    "n_children_kept": n_kept_here,
                                })
                                # fate tracking only for early-window births
                                if EARLY_START <= it <= EARLY_END and n_kept_here > 0:
                                    kept_slots = [s for s in slots
                                                  if s < kids_cpu.shape[0] and bool(keep_cpu[s])]
                                    kept_pos = torch.stack(
                                        [kids_cpu[s] for s in kept_slots]).to("cuda")
                                    meta = [{
                                        "child_id": f"{pid}c{s if ptype == 'split' else 0}",
                                        "parent_group": gname, "child_type": ptype,
                                        "birth_event": event_num, "birth_it": it,
                                    } for s in kept_slots]
                                    child_rows.extend(meta)
                                    new_ref = kept_pos
                                    new_alive = torch.ones(kept_pos.shape[0],
                                                           dtype=torch.bool, device="cuda")
                                    new_last = torch.full((kept_pos.shape[0],), it,
                                                          dtype=torch.long, device="cuda")
                                    if child_ref is None:
                                        child_ref, child_alive, child_last_alive = \
                                            new_ref, new_alive, new_last
                                    else:
                                        child_ref = torch.cat([child_ref, new_ref])
                                        child_alive = torch.cat([child_alive, new_alive])
                                        child_last_alive = torch.cat([child_last_alive, new_last])

                        # ---- advance chain state ----
                        prev_trig_xyz = trig_xyz if n_trig > 0 else None
                        prev_chain = chain if n_trig > 0 else None

                        if it % 1000 == 0:
                            ps = eval_quick(gaussians, test_cams, pipe, bg, opt.mult)
                            print(f"  s{seed} it={it} ev={event_num} trig={n_split} "
                                  f"P2+3={int((group > 1).sum())} children={len(child_rows)} "
                                  f"alive={int(child_alive.sum()) if child_alive is not None else 0} "
                                  f"PSNR={ps:.2f}", flush=True)

                if it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

        # ---- final fate matching ----
        final_matched = None
        if child_ref is not None and child_ref.shape[0] > 0:
            final_matched = torch.full((child_ref.shape[0],), -1,
                                       dtype=torch.long, device="cuda")
            alive_idx = torch.where(child_alive)[0]
            if alive_idx.numel() > 0:
                d, j = match_nearest(child_ref[alive_idx], gaussians.get_xyz.detach())
                still = d < TOL
                dead_local = alive_idx[~still]
                child_alive[dead_local] = False
                sur_local = alive_idx[still]
                child_ref[sur_local] = gaussians.get_xyz.detach()[j[still]]
                child_last_alive[sur_local] = args.max_iters
                final_matched[sur_local] = j[still]

        # ---- write per-seed outputs ----
        final_matched_cpu = final_matched.cpu().tolist() if final_matched is not None else []
        final_op_cpu = (gaussians.get_opacity.detach().squeeze(-1).cpu().tolist()
                        if final_matched is not None else [])
        for i, m in enumerate(child_rows):
            alive = bool(child_alive[i])
            last_alive = int(child_last_alive[i])
            fidx = final_matched_cpu[i]
            fop = float(final_op_cpu[fidx]) if (alive and fidx >= 0) else float("nan")
            child_rows[i].update({
                "seed": seed,
                "survived_500": int(last_alive >= m["birth_it"] + 500),
                "survived_1000": int(last_alive >= m["birth_it"] + 1000),
                "survived_3000": int(last_alive >= m["birth_it"] + 3000),
                "survived_30k": int(alive),
                "final_opacity": fop,
                "visible_01": int(alive and fop >= 0.1),
                "last_alive_it": last_alive,
                "lifetime_iters": (args.max_iters - m["birth_it"]) if alive
                                  else (last_alive - m["birth_it"]),
                "censored": int(alive),
            })
        # ensure 'seed' first column order stable
        fate_fields = ["seed", "child_id", "parent_id", "parent_group", "child_type",
                       "birth_event", "birth_it", "survived_500", "survived_1000",
                       "survived_3000", "survived_30k", "final_opacity", "visible_01",
                       "last_alive_it", "lifetime_iters", "censored"]
        with open(f"{OUT}/data/fate_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fate_fields, extrasaction="ignore")
            w.writeheader(); w.writerows(child_rows)
        with open(f"{OUT}/data/lineage_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(lineage_rows[0].keys()))
            w.writeheader(); w.writerows(lineage_rows)
        with open(f"{OUT}/data/parents_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(parent_rows[0].keys()))
            w.writeheader(); w.writerows(parent_rows)
        n_pruned_total = sum(pr["n_children_born"] - pr["n_children_kept"]
                             for pr in parent_rows)
        print(f"  s{seed} saved: {len(parent_rows)} parent occurrences, "
              f"{len(child_rows)} children tracked, {len(lineage_rows)} lineage rows, "
              f"pruned_at_birth={n_pruned_total}", flush=True)
        torch.cuda.empty_cache()

    print(f"\n[b18dfinalfix] done", flush=True)


if __name__ == "__main__":
    main()
