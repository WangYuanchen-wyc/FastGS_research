#
# Paper B - B18-D Fix: Early Densification Persistence & Child Fate Validation
#
# Focus on the early window (it1000-3000) where the densification burst happens.
# For each native densification event:
#   - Persistence: do the SAME spatial regions trigger again at the next event?
#     (matched by xyz proximity, torch.cdist, tol=0.05 absolute — same as B18-D)
#   - Child fate: newborns (clones + 2x split children) identified by append order
#     (scene/gaussian_model.py: clone appends, split appends children then removes
#     parents, so the last n_clone + 2*n_split points of the post-densify tensor are
#     exactly the newborns). Tracked to next event, to densify end (15k) and to 30k.
#   - Child re-trigger: how many newborns are in the trigger set of the NEXT event
#     (evidence of persistent vs one-shot demand).
#
# No scipy (not installed in container). Masks are computed BEFORE
# apply_native_densify: densification_postfix zeroes xyz_gradient_accum*, so any
# recomputation after densify would read zeros.
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

OUT = os.environ.get("B18D_OUT", "paper_b/b18_densification_persistence/fix")
EARLY_START = 1000
EARLY_END = 3000
PROXIMITY_TOL = 0.05  # xyz distance for "same" Gaussian (same definition as B18-D)


def safe_replace_opacity(g, tensor):
    """Ensure optimizer state exists before calling replace_tensor_to_optimizer."""
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
    """Native FastGS densification. Returns (n_clone, n_split, newborn_xyz, n_newborn).

    newborn_xyz: (K,3) cpu tensor of newborn positions after all pruning of this
    event. Newborns = last (n_clone + 2*n_split) points after clone+split, because
    clone appends first and split appends children then removes parents.
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
    newborn_tail = g.get_xyz[n_cur - n_newborn:].detach().clone() if n_newborn > 0 else None

    prune_mask = (g.get_opacity < 0.005).squeeze()
    st = 20 if it > opt.opacity_reset_interval else None
    keep_newborn = None
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
            keep_newborn = ~mask_full[n - n_newborn:]
        g.prune_points(mask_full)
    if n_newborn > 0:
        newborn_xyz = newborn_tail if keep_newborn is None else newborn_tail[keep_newborn]
    else:
        newborn_xyz = None
    g._opacity = safe_replace_opacity(
        g, inverse_sigmoid(torch.min(g.get_opacity, torch.ones_like(g.get_opacity) * 0.8))
    )["opacity"]
    torch.cuda.empty_cache()
    return n_clone, n_split, newborn_xyz, n_newborn


@torch.no_grad()
def match_nearest(query, ref, chunk=16384):
    """For each query point, nearest ref point. query/ref: (K,3)/(M,3) cuda fp32.
    Returns (dist (K,), idx (M->K index,)). Chunked over ref to bound memory."""
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
    os.makedirs(args.model_path, exist_ok=True)  # Scene copies input.ply into model_path

    all_trigger_rows = []
    all_child_rows = []

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

        prev_trigger_xyz = None   # (K,3) cuda, trigger positions of previous event
        prev_newborn_xyz = None   # (K,3) cuda, newborn positions of previous event
        prev_newborn_meta = None  # (birth_event, birth_it)
        born_groups = []          # [(birth_event, birth_it, (K,3) cpu), ...]
        event_num = 0
        densify_events = []

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

                        # --- trigger sets, BEFORE any mutation (stats valid here) ---
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
                        n_trigger = int(split_set.sum())

                        trig_xyz = gaussians.get_xyz[split_set].detach()
                        n_gs_before = gaussians.get_xyz.shape[0]

                        # --- persistence: current triggers near previous triggers ---
                        n_persist = 0; persist_dist = float("nan")
                        if prev_trigger_xyz is not None and len(prev_trigger_xyz) > 0 \
                                and trig_xyz.shape[0] > 0:
                            d, _ = match_nearest(trig_xyz, prev_trigger_xyz)
                            n_persist = int((d < PROXIMITY_TOL).sum())
                            persist_dist = float(d[d < PROXIMITY_TOL].mean()) if n_persist else float("nan")

                        # --- fate of previous event's newborns ---
                        n_prev_newborn = int(prev_newborn_xyz.shape[0]) if prev_newborn_xyz is not None else 0
                        n_prev_alive = 0; n_child_retrig = 0
                        if n_prev_newborn > 0:
                            cur_xyz = gaussians.get_xyz.detach()
                            d, _ = match_nearest(prev_newborn_xyz, cur_xyz)
                            alive = d < PROXIMITY_TOL
                            n_prev_alive = int(alive.sum())
                            if trig_xyz.shape[0] > 0:
                                dc, _ = match_nearest(prev_newborn_xyz[alive], trig_xyz)
                                n_child_retrig = int((dc < PROXIMITY_TOL).sum())

                        # --- apply native densification ---
                        n_clone, n_split, newborn_xyz, n_newborn = apply_native_densify(
                            gaussians, opt, it, radii, imp, pru, extent)
                        n_gs_after = gaussians.get_xyz.shape[0]

                        densify_events.append({
                            "seed": seed, "iteration": it, "event": event_num,
                            "is_early": int(EARLY_START <= it <= EARLY_END),
                            "n_trigger": n_trigger,
                            "n_clone_trig": int(clone_set.sum()),
                            "n_persist_from_prev": n_persist,
                            "persist_mean_dist": persist_dist,
                            "n_prev_newborn": n_prev_newborn,
                            "n_prev_newborn_alive_next": n_prev_alive,
                            "n_child_retrigger_next": n_child_retrig,
                            "n_gs_before": n_gs_before, "n_gs_after": n_gs_after,
                            "n_clone": n_clone, "n_split": n_split, "n_newborn": n_newborn,
                        })

                        if newborn_xyz is not None and newborn_xyz.shape[0] > 0:
                            born_groups.append((event_num, it, newborn_xyz.cpu()))
                            prev_newborn_xyz = newborn_xyz
                            prev_newborn_meta = (event_num, it)
                        else:
                            prev_newborn_xyz = None
                        prev_trigger_xyz = trig_xyz if trig_xyz.shape[0] > 0 else None

                        if it % 1000 == 0:
                            ps = eval_quick(gaussians, test_cams, pipe, bg, opt.mult)
                            print(f"  s{seed} it={it} ev={event_num} trig={n_trigger} "
                                  f"persist={n_persist} aliveNB={n_prev_alive} "
                                  f"PSNR={ps:.2f} #GS={n_gs_after}", flush=True)

                if it < opt.densify_until_iter and it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            gaussians.optimizer_step(it)  # native train.py:163 (was missing: frozen model)

        # --- child fate at densify end (15k) and final (30k) ---
        final_xyz = gaussians.get_xyz.detach()
        final_op = gaussians.get_opacity.detach().squeeze(-1)
        print(f"  s{seed} matching {len(born_groups)} born groups to final model "
              f"({final_xyz.shape[0]} GS)...", flush=True)
        for b_ev, b_it, b_xyz in born_groups:
            q = b_xyz.to("cuda")
            n_born = q.shape[0]
            if b_it < opt.densify_until_iter:
                # survival to densify end is unknown post-hoc (indices/positions at
                # 15k not stored) — final only; kept column for schema stability
                n_alive_15k = -1
            else:
                n_alive_15k = -1
            d, idx = match_nearest(q, final_xyz)
            alive = d < PROXIMITY_TOL
            n_alive_final = int(alive.sum())
            mean_op = float(final_op[idx[alive]].mean()) if n_alive_final else float("nan")
            all_child_rows.append({
                "seed": seed, "birth_event": b_ev, "birth_it": b_it,
                "is_early": int(EARLY_START <= b_it <= EARLY_END),
                "n_born": n_born, "n_alive_15k": n_alive_15k,
                "n_alive_final": n_alive_final,
                "alive_frac_final": n_alive_final / max(n_born, 1),
                "mean_opacity_final_alive": mean_op,
            })
            del q

        # per-seed incremental save (child rows: this seed only)
        seed_child_rows = [r for r in all_child_rows if r["seed"] == seed]
        with open(f"{OUT}/data/trigger_events_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(densify_events[0].keys()))
            w.writeheader(); w.writerows(densify_events)
        with open(f"{OUT}/data/child_fate_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_child_rows[0].keys()))
            w.writeheader(); w.writerows(seed_child_rows)
        print(f"  s{seed} saved ({len(densify_events)} events, "
              f"{len(all_child_rows)} cumulative child rows)", flush=True)
        all_trigger_rows.extend(densify_events)
        torch.cuda.empty_cache()

    with open(f"{OUT}/data/trigger_events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_trigger_rows[0].keys()))
        w.writeheader(); w.writerows(all_trigger_rows)
    print(f"\n[b18dfix] done: {len(all_trigger_rows)} trigger rows, "
          f"{len(all_child_rows)} child-fate rows", flush=True)


if __name__ == "__main__":
    main()
