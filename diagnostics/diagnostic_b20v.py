#
# Paper B - B20-V: Rendering Support Validity (matched deletion)
#
# Question: for opacity<0.1 prune candidates, do multi-view-supported Gaussians
# actually deserve retention more than low-support ones?
#
#   - Contribution: B19-P's real blend weight w_i = alpha_i*T_i per view
#     (instrumented rasterizer, zero extra renders), tol=1e-6.
#   - Temporal support: after 15k there is no densification and prunes keep
#     order, so cohort identity across prune rounds is EXACT (orig-id array
#     compacted by each round's removed mask). Each candidate's support history
#     over past rounds is looked up by orig id. Birth iteration is tracked
#     exactly through the event-time densification/prunes (birth_it array).
#   - Matched deletion (core): at each round, from candidates build
#     A = low-support (sv<=2), B = multi-view (sv>=3); greedy opacity-matched
#     pairs, K = min(|A|,|B|); Random control = K uniform candidates
#     opacity-matched to B's list. Branches: restore(capture) -> delete exactly
#     K -> eval PSNR/SSIM/LPIPS on the full test set -> restore. Immediate
#     impact only; no recovery, no re-optimization, prune results unchanged.
#
# Pre-registered verdict (in analyze_b20v.py): GO iff supported contribution is
# not noise (V1), temporal persistence exists (V2), matching fair (V3), and
# deleting B hurts more than deleting A in 3/3 seeds with >=8/12 cells and
# mean dPSNR >= 0.1 dB (V4). GPU: 7
#

import os, sys, random, csv
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

OUT = "paper_b/b20_rendering_support_validity"
TOL = 1e-6
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


def apply_native_densify(g, opt, it, radii, imp, pru, extent, birth_it=None):
    """Native event densification.
    Returns (n_clone, n_split, removed_mask, birth_it) where birth_it tracks
    each Gaussian's birth iteration through clone/split appends, split-parent
    removal and the multinomial prune (exact lifecycle bookkeeping)."""
    n_pre = g.get_xyz.shape[0]
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

    if birth_it is not None:
        birth_it = torch.cat([birth_it,
                              torch.full((n_clone + 2 * n_split,), it,
                                         dtype=torch.long)])
        # densify_and_split removed its parents from the OLD block: compact it
        birth_it = torch.cat([birth_it[:n_pre][~split_set.cpu()], birth_it[n_pre:]])

    removed = None
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
        removed = torch.logical_and(prune_mask, sel)
        g.prune_points(removed)
        if birth_it is not None:
            birth_it = birth_it[~removed.cpu()]
    g._opacity = safe_replace_opacity(
        g, inverse_sigmoid(torch.min(g.get_opacity, torch.ones_like(g.get_opacity) * 0.8))
    )["opacity"]
    torch.cuda.empty_cache()
    return n_clone, n_split, removed, birth_it


@torch.no_grad()
def contribution_pass(camlist, gaussians, pipe, bg, args):
    """Native final-prune pass (DENSIFY=False) + per-Gaussian blend weights."""
    full_score = None
    per_view_w = []
    for view in range(len(camlist)):
        cam = camlist[view]
        pkg = render_fastgs(cam, gaussians, pipe, bg, args.mult, get_weights=True)
        render_image = pkg["render"]
        per_view_w.append(pkg["gauss_weights"].detach().reshape(-1))
        photometric_loss = compute_photometric_loss(cam, render_image)
        gt_image = cam.original_image.cuda()
        l1_norm = get_loss(render_image, gt_image)
        metric_map = (l1_norm > args.loss_thresh).int()
        pkg2 = render_fastgs(cam, gaussians, pipe, bg, args.mult,
                             get_flag=True, metric_map=metric_map)
        acc = pkg2["accum_metric_counts"].detach()
        full_score = (photometric_loss * acc.clone() if full_score is None
                      else full_score + photometric_loss * acc)
    pruning_score = (full_score - torch.min(full_score)) / (
        torch.max(full_score) - torch.min(full_score))
    W = torch.stack(per_view_w, dim=1)
    return pruning_score, W


@torch.no_grad()
def eval_metrics(g, test_cams, pipe, bg, mult, lpips_fn):
    ps_, ss_, lp_ = [], [], []
    for cam in test_cams:
        img = torch.clamp(render_fastgs(cam, g, pipe, bg, mult)["render"], 0, 1)
        gt = torch.clamp(cam.original_image.cuda(), 0, 1)
        ps_.append(float(psnr(img, gt).mean()))
        ss_.append(float(fast_ssim(img.unsqueeze(0), gt.unsqueeze(0)).mean()))
        lp_.append(float(lpips_fn(img.unsqueeze(0) * 2 - 1,
                                  gt.unsqueeze(0) * 2 - 1).mean()))
    return float(np.mean(ps_)), float(np.mean(ss_)), float(np.mean(lp_))


def greedy_opacity_match(op_a, op_b, caliper=0.01):
    """Two-pointer caliper matching on sorted opacity: pair the smaller value
    when within caliper, else skip it. Maximal cardinality under the caliper,
    every index used at most once (distinct, fair pairs)."""
    order_a = np.argsort(op_a)
    order_b = np.argsort(op_b)
    sa = op_a[order_a]
    sb = op_b[order_b]
    i = j = 0
    pairs = []
    while i < len(sa) and j < len(sb):
        if abs(sa[i] - sb[j]) <= caliper:
            pairs.append((int(order_a[i]), int(order_b[j])))
            i += 1
            j += 1
        elif sa[i] < sb[j]:
            i += 1
        else:
            j += 1
    return pairs


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

        birth_it = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.long)
        event_num = 0
        round_num = 0
        strength_rows = []
        temporal_rows = []
        del_rows = []
        summary_rows = []
        orig = None            # cohort orig-ids aligned with current indices
        W_hist = {}            # round -> (W (N_r,10) cpu, orig_r (N_r,))

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
                        _, _, _, birth_it = apply_native_densify(
                            gaussians, opt, it, radii, imp, pru, extent, birth_it)
                        assert birth_it.shape[0] == gaussians.get_xyz.shape[0]

                if it < opt.densify_until_iter and it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

                if it % 3000 == 0 and it > 15_000 and it < 30_000:
                    round_num += 1
                    my = scene.getTrainCameras().copy()
                    cl = sampling_cameras(my)
                    pru, W = contribution_pass(cl, gaussians, pipe, bg, opt)
                    Wc = W.cpu()
                    pru1 = pru.reshape(-1)

                    if orig is None:
                        orig = torch.arange(Wc.shape[0])
                    W_hist[round_num] = (Wc, orig.clone())

                    opa = gaussians.get_opacity.detach().squeeze(-1)
                    scale = gaussians.get_scaling.max(dim=1).values.detach()
                    cand_mask = opa < 0.1
                    cand_idx = torch.where(cand_mask)[0]
                    cand_idx_n = cand_idx.cpu().numpy()
                    n_cand = int(cand_mask.sum())
                    removed_mask = cand_mask | (pru1 > 0.9)

                    # ---- per-candidate strength rows (vectorized) ----
                    W_cand = Wc[cand_idx]                    # (C, K)
                    sv = (W_cand > TOL).sum(dim=1)
                    tot = W_cand.sum(dim=1)
                    srt, _ = torch.sort(W_cand, dim=1, descending=True)
                    t1 = srt[:, 0] / tot.clamp(min=1e-12)
                    t3 = srt[:, :3].sum(dim=1) / tot.clamp(min=1e-12)
                    oids_all = orig[cand_idx].numpy()
                    opa_n = opa.cpu().numpy()
                    scale_n = scale.cpu().numpy()
                    birth_n = birth_it.numpy()
                    sv_n = sv.numpy(); tot_n = tot.numpy()
                    srt_n = srt.numpy(); t1_n = t1.numpy(); t3_n = t3.numpy()
                    for i in range(n_cand):
                        svi = int(sv_n[i])
                        strength_rows.append({
                            "seed": seed, "round": round_num, "round_it": it,
                            "orig_id": int(oids_all[i]),
                            "opacity": round(float(opa_n[cand_idx[i]]), 5),
                            "scale": round(float(scale_n[cand_idx[i]]), 6),
                            "birth_it": int(birth_n[cand_idx[i]]),
                            "support_view_count": svi,
                            "total_contribution": round(float(tot_n[i]), 6),
                            "mean_per_supported_view": round(float(tot_n[i]) / svi, 6)
                                if svi > 0 else 0.0,
                            "max_view_contribution": round(float(srt_n[i, 0]), 6),
                            "top1_share": round(float(t1_n[i]), 4),
                            "top3_share": round(float(t3_n[i]), 4),
                            "group": "A_low" if svi <= 2 else "B_multi",
                        })
                    groups = ["A_low" if int(v) <= 2 else "B_multi" for v in sv]

                    # ---- temporal support history (batched orig-id lookup) ----
                    oids_t = torch.from_numpy(oids_all)
                    multi_by_round = []
                    for rr in range(1, round_num + 1):
                        W_r, orig_r = W_hist[rr]
                        pos = torch.searchsorted(orig_r, oids_t)
                        pos_c = pos.clamp(max=max(orig_r.numel() - 1, 0))
                        ok = (orig_r[pos_c] == oids_t) & (pos < orig_r.numel())
                        sv_r = torch.full((n_cand,), -1.0)
                        if ok.any():
                            sv_r[ok] = (W_r[pos_c[ok]] > TOL).sum(dim=1).float()
                        multi_by_round.append(sv_r)
                    run = torch.zeros(n_cand)
                    for rr in range(round_num - 1, -1, -1):
                        cont = (multi_by_round[rr] >= 3).float()
                        run = (run + 1) * cont
                    for i in range(n_cand):
                        if int(sv[i]) >= 3:
                            tclass = ("Persistent-supported" if int(run[i]) >= 2
                                      else "Transient-supported")
                        else:
                            tclass = "Low-support"
                        hist = ";".join(
                            "-" if multi_by_round[rr][i] < 0
                            else str(int(multi_by_round[rr][i]))
                            for rr in range(round_num))
                        temporal_rows.append({
                            "seed": seed, "death_round": round_num,
                            "death_it": it, "orig_id": int(oids_all[i]),
                            "support_history": hist,
                            "temporal_class": tclass,
                        })

                    # ---- matched deletion (core) ----
                    a_local = [i for i in range(n_cand) if groups[i] == "A_low"]
                    b_local = [i for i in range(n_cand) if groups[i] == "B_multi"]
                    K = min(len(a_local), len(b_local))
                    if K == 0:
                        print(f"  s{seed} it={it} round={round_num}: no matchable "
                              f"groups (A={len(a_local)}, B={len(b_local)})", flush=True)
                    else:
                        op_arr = np.asarray(opa_n)
                        rng = np.random.RandomState(10000 + seed * 100 + round_num)
                        a_op = op_arr[cand_idx_n[a_local]]
                        b_op = op_arr[cand_idx_n[b_local]]
                        pairs_ab = greedy_opacity_match(a_op, b_op)[:K]
                        del_a = [int(cand_idx_n[a_local[pi]]) for pi, _ in pairs_ab]
                        del_b = [int(cand_idx_n[b_local[pj]]) for _, pj in pairs_ab]
                        # random control: K uniform candidates, opacity-matched to B list
                        r_pool = list(range(n_cand))
                        r_sel = rng.choice(r_pool, size=min(K * 3, n_cand), replace=False)
                        r_op = op_arr[cand_idx_n[r_sel]]
                        pairs_rb = greedy_opacity_match(r_op, b_op)[:K]
                        del_r = [int(cand_idx_n[r_sel[pi]]) for pi, _ in pairs_rb]

                        d_op = np.mean([abs(op_arr[ia] - op_arr[ib])
                                        for ia, ib in zip(del_a, del_b)])
                        snap = gaussians.capture(opt.optimizer_type)
                        base = eval_metrics(gaussians, test_cams, pipe, bg,
                                            opt.mult, lpips_fn)
                        branch_res = {}
                        for bname, didx in (("A_low", del_a), ("B_multi", del_b),
                                            ("R_random", del_r)):
                            gaussians.restore(snap, opt)
                            n_before = gaussians.get_xyz.shape[0]
                            m = torch.zeros(n_before,
                                            dtype=torch.bool, device="cuda")
                            m[torch.tensor(didx, dtype=torch.long, device="cuda")] = True
                            gaussians.prune_points(m)
                            n_after = gaussians.get_xyz.shape[0]
                            res = eval_metrics(gaussians, test_cams, pipe, bg,
                                               opt.mult, lpips_fn)
                            branch_res[bname] = res
                            del_rows.append({
                                "seed": seed, "round": round_num, "round_it": it,
                                "branch": bname, "K": K,
                                "mean_abs_opacity_gap": round(float(d_op), 6),
                                "psnr_base": round(base[0], 4),
                                "ssim_base": round(base[1], 5),
                                "lpips_base": round(base[2], 5),
                                "psnr_del": round(res[0], 4),
                                "ssim_del": round(res[1], 5),
                                "lpips_del": round(res[2], 5),
                                "dpsnr": round(res[0] - base[0], 4),
                                "dssim": round(res[1] - base[1], 6),
                                "dlpips": round(res[2] - base[2], 6),
                            })
                            print(f"    branch {bname}: N {n_before}->{n_after} "
                                  f"PSNR {res[0]:.4f} (base {base[0]:.4f})", flush=True)
                        gaussians.restore(snap, opt)
                        print(f"  s{seed} it={it} round={round_num} K={K} "
                              f"|dOp|={d_op:.4f} dPSNR A/B/R="
                              f"{del_rows[-3]['dpsnr']:.3f}/"
                              f"{del_rows[-2]['dpsnr']:.3f}/"
                              f"{del_rows[-1]['dpsnr']:.3f}", flush=True)

                    # ---- native prune + cohort/birth compaction ----
                    gaussians.final_prune_fastgs(0.1, pruning_score=pru)
                    keep = ~removed_mask.cpu()
                    birth_it = birth_it[keep]
                    orig = orig[keep]
                    print(f"  s{seed} it={it} round={round_num} cand={n_cand} "
                          f"#GS->{gaussians.get_xyz.shape[0]}", flush=True)

            gaussians.optimizer_step(it)

        # ---- per-seed save ----
        with open(f"{OUT}/data/contribution_strength_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(strength_rows[0].keys()))
            w.writeheader(); w.writerows(strength_rows)
        with open(f"{OUT}/data/temporal_support_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(temporal_rows[0].keys()))
            w.writeheader(); w.writerows(temporal_rows)
        with open(f"{OUT}/data/matched_deletion_results_s{seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(del_rows[0].keys()))
            w.writeheader(); w.writerows(del_rows)
        print(f"  s{seed} saved: {len(strength_rows)} strength rows, "
              f"{len(temporal_rows)} temporal rows, {len(del_rows)} deletion rows",
              flush=True)
        torch.cuda.empty_cache()

    print(f"\n[b20v] done", flush=True)


if __name__ == "__main__":
    main()
