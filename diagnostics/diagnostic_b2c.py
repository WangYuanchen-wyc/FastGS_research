#
# Paper B - B2-C: Personalized Oracle-Mix Scalability Diagnostic
#
# Core question: does assigning each real FastGS candidate its own
# short-horizon Clone-vs-Split oracle action (Oracle-Mix) beat Native /
# All-Clone / All-Split / Shuffled-Mix (same action composition, randomly
# reassigned)?
#
# Fairness backbone: ONE master snapshot at a real densification iteration
# (2000, before native densification). BOTH the single-candidate oracle and
# all group policies derive from this same snapshot / population, so oracle
# labels and group members refer to identical Gaussians (fixes the cross-run
# mismatch of B2-A/B2-B).
#
# Oracle = binary Clone-vs-Split (Keep excluded as label, kept as control):
#   clone_quality_i = dQ_demand_clone@100           (1 run)
#   split_quality_i = mean of 5 candidate-specific seeded repeats
#   oracle_action_i = Clone iff clone_quality > split_quality
#   oracle_gap_i    = split_quality - clone_quality
#   split_sem = std/ sqrt(5); confident = |gap| > sem; high = |gap| > max(sem,1e-4)
#
# Groups: per seed one 300-candidate pool sampled naturally; K=30/100/300 are
# nested prefixes. Policies: keep / native / all_clone / all_split /
# oracle_mix / shuffled x5 (identical action counts, permuted assignment).
# Group split realization uses the SAME draw as oracle repeat 0
# (seed = SPLIT_BASE + parent_index*100 + 0), applied in descending original
# index order so every split sees its snapshot parent values.
#
# Oracle results are cached per candidate (immediate flush, resumable).
# No predictor / allocator / budget / interaction model is implemented.
#

import os
import sys
import json
import random
import csv

import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene, GaussianModel  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402
from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras, get_loss  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from diagnostics.common import (seed_all, clone_tree, install_c_proxy,  # noqa: E402
                                native_train_one_iter, project_to_pixel, roi_box)
import diagnostics.diagnostic_v2 as v2  # noqa: E402

OUT = "paper_b/b2_c_oracle_scalability"
TILE_MARGIN_PX = 16
SPLIT_BASE = 300000


def split_seed_for(parent_index, repeat):
    return SPLIT_BASE + int(parent_index) * 100 + int(repeat)


def restore_from(snapshot, opt):
    g = GaussianModel(v2.G["sh_degree"], opt.optimizer_type)
    g.restore(clone_tree(snapshot), opt)
    return g


def apply_single_action(g, action, parent_idx, radii_snap, seed):
    g.tmp_radii = radii_snap.clone()
    single = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    single[parent_idx] = True
    ones = torch.ones_like(single)
    if action == "clone":
        g.densify_and_clone_fastgs(single, ones)
    else:
        seed_all(seed)
        g.densify_and_split_fastgs(single, ones, N=2)
    g.tmp_radii = None


def apply_assignment(g, assignment, radii_snap):
    """assignment: list of (parent_idx, 'clone'|'split'). Clones batched first,
    splits sequential in DESCENDING index (parent rows still hold snapshot
    values -> children identical to the solo oracle runs). Split draw = oracle
    repeat 0 seed."""
    g.tmp_radii = radii_snap.clone()
    clones = [i for i, a in assignment if a == "clone"]
    if clones:
        mask = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        mask[torch.tensor(clones, dtype=torch.long, device="cuda")] = True
        g.densify_and_clone_fastgs(mask, torch.ones_like(mask))
    for idx in sorted([i for i, a in assignment if a == "split"], reverse=True):
        single = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        single[idx] = True
        seed_all(split_seed_for(idx, 0))
        g.densify_and_split_fastgs(single, torch.ones_like(single), N=2)
    g.tmp_radii = None


def train_100(g, opt, train_cams, cam_seq, diag_iter, steps, train_seed):
    seed_all(train_seed)
    for i in range(1, steps + 1):
        cam = train_cams[cam_seq[i - 1]]
        native_train_one_iter(diag_iter + i, cam, g, v2.G["pipe"], v2.G["bg"], opt)
        with torch.no_grad():
            if opt.optimizer_type == "default":
                g.optimizer_step(diag_iter + i)


# ----------------------------------------------------------- metrics -------

@torch.no_grad()
def demand_l1_on_views(g, cand_views):
    """Candidate-level demand-local L1 (fixed views/ROI, B2-A convention)."""
    vals = []
    for cv in cand_views:
        if not cv["demand_valid"]:
            continue
        x0, y0, x1, y1 = cv["box"]
        out = render_fastgs(cv["cam"], g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])
        m = cv["mask"][y0:y1, x0:x1]
        ri = out["render"][:, y0:y1, x0:x1][:, m]
        gi = cv["cam"].original_image.cuda()[:, y0:y1, x0:x1][:, m]
        vals.append(float((ri - gi).abs().mean()))
    return float(np.mean(vals)) if vals else None


@torch.no_grad()
def group_eval(g, views, support_masks, demand_masks, global_probe, with_lpips=False):
    proxy = install_c_proxy()
    sup, dem, tiles = [], [], []
    for cam, smask, dmask in zip(views, support_masks, demand_masks):
        out = render_fastgs(cam, g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])
        tiles.append(proxy.last_num_rendered)
        gt = cam.original_image.cuda()
        for mask, acc in ((smask, sup), (dmask, dem)):
            if int(mask.sum()) == 0:
                continue
            ri, gi = out["render"][:, mask], gt[:, mask]
            diff = (ri - gi).abs()
            mse = float((diff ** 2).mean())
            acc.append({"l1": float(diff.mean()), "mse": mse,
                        "psnr": 10.0 * np.log10(1.0 / max(mse, 1e-10))})

    def agg(a):
        if not a:
            return {"l1": None, "mse": None, "psnr": None}
        return {"l1": float(np.mean([x["l1"] for x in a])),
                "mse": float(np.mean([x["mse"] for x in a])),
                "psnr": float(np.mean([x["psnr"] for x in a]))}

    psnrs, ssims, losses, lpipss = [], [], [], []
    for cam in global_probe:
        image = render_fastgs(cam, g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])["render"]
        gt = cam.original_image.cuda()
        Ll1 = float(l1_loss(image, gt))
        ss = float(fast_ssim(image.unsqueeze(0), gt.unsqueeze(0)))
        losses.append(0.8 * Ll1 + 0.2 * (1.0 - ss))
        image_c, gt_c = torch.clamp(image, 0, 1), torch.clamp(gt, 0, 1)
        psnrs.append(float(psnr(image_c, gt_c).mean()))
        ssims.append(ss)
        if with_lpips:
            try:
                from lpipsPyTorch import lpips as lpips_fn
                lpipss.append(float(lpips_fn(image_c, gt_c, net_type='vgg').mean()))
            except Exception:
                lpipss.append(float("nan"))
    glob = {"loss": float(np.mean(losses)), "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims))}
    if with_lpips:
        glob["lpips"] = float(np.nanmean(lpipss)) if lpipss else None
    return {"support": agg(sup), "demand": agg(dem), "global": glob,
            "tile_pairs_mean": float(np.mean(tiles))}


# --------------------------------------------------- candidate views -------

def build_cand_views(parent_idx, pool_data, xyz_snap, n_probe):
    views = []
    for pd in pool_data:
        cam, vr = pd["cam"], pd["radii"]
        if int(vr[parent_idx]) <= 0:
            continue
        u, v, _, ok = project_to_pixel(cam, xyz_snap[parent_idx])
        if not ok:
            continue
        box = roi_box(u, v, float(vr[parent_idx]), cam.image_width,
                      cam.image_height, TILE_MARGIN_PX)
        x0, y0, x1, y1 = box
        if (x1 - x0) < 8 or (y1 - y0) < 8:
            continue
        views.append({"cam": cam, "box": box, "mask": pd["mask"],
                      "demand_valid": bool(int(pd["mask"][y0:y1, x0:x1].sum()) > 0)})
        if len(views) >= n_probe:
            break
    return views


# ------------------------------------------------------------------ main ---

def run_warmup(gaussians, scene, opt, pipe, bg, dataset, args, diag_iter,
               train_cams, pool_cams, sampling_cameras,
               compute_gaussian_score_fastgs, render_fastgs, v2):
    """Native FastGS warmup to diag_iter; at the event, capture the master
    snapshot BEFORE the native densification (read-only hook on the model)."""
    from utils.fast_utils import get_loss
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    hook = None
    for iteration in range(1, diag_iter + 1):
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
                    imp, pru = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                    if iteration == diag_iter:
                        clone_set, split_set, _, _, _ = v2.native_selection_sets(
                            gaussians, imp, opt, scene.cameras_extent)
                        pool_data = []
                        with torch.no_grad():
                            for cam in pool_cams:
                                out = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
                                pool_data.append({"cam": cam, "radii": out["radii"],
                                                  "mask": (get_loss(out["render"], cam.original_image.cuda())
                                                           > opt.loss_thresh).detach()})
                        hook = {"imp": imp, "clone_set": clone_set, "split_set": split_set,
                                "pool_data": pool_data, "radii": radii.clone(),
                                "snapshot": clone_tree(gaussians.capture(opt.optimizer_type))}
                    gaussians.densify_and_prune_fastgs(
                        max_screen_size=(20 if iteration > opt.opacity_reset_interval else None),
                        min_opacity=0.005, extent=scene.cameras_extent, radii=radii, args=opt,
                        importance_score=imp, pruning_score=pru)
                if iteration % opt.opacity_reset_interval == 0 or \
                        (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            if iteration % 3000 == 0 and 15_000 < iteration < 30_000:
                my_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_stack)
                _, pru = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
                gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pru)
            if opt.optimizer_type == "default":
                gaussians.optimizer_step(iteration)
    assert hook is not None, "warmup finished without hitting the diag hook"
    return hook


def main():
    parser = ArgumentParser("Paper B B2-C: personalized oracle-mix scalability")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--diag_iter", type=int, default=2000)
    parser.add_argument("--pool_K", type=int, default=300)
    parser.add_argument("--Ks", type=str, default="30,100,300")
    parser.add_argument("--n_group_seeds", type=int, default=3)
    parser.add_argument("--shuffled_repeats", type=int, default=5)
    parser.add_argument("--view_pool", type=int, default=30)
    parser.add_argument("--n_cost_views", type=int, default=8)
    parser.add_argument("--n_probe", type=int, default=8)
    parser.add_argument("--diag_steps", type=int, default=100)
    parser.add_argument("--warmup_seed", type=int, default=0)
    parser.add_argument("--train_seed", type=int, default=1234)
    parser.add_argument("--camseq_seed", type=int, default=2024)
    parser.add_argument("--group_base_seed", type=int, default=4711)
    parser.add_argument("--shuffle_base", type=int, default=50000)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    diag_iter = args.diag_iter
    assert diag_iter > opt.densify_from_iter and diag_iter % opt.densification_interval == 0 \
        and diag_iter < opt.densify_until_iter
    assert opt.optimizer_type == "default"
    Ks = sorted({min(int(k), args.pool_K) for k in args.Ks.split(",")})
    protocol_base = {"scene": os.path.abspath(dataset.source_path), "diag_iter": diag_iter,
                     "diag_steps": args.diag_steps, "n_probe": args.n_probe,
                     "train_seed": args.train_seed, "camseq_seed": args.camseq_seed,
                     "split_base": SPLIT_BASE, "n_cost_views": args.n_cost_views}
    protocol = None  # bound to the snapshot fingerprint after the hook

    os.makedirs(f"{OUT}/cache", exist_ok=True)
    os.makedirs(f"{OUT}/data/groups", exist_ok=True)

    install_c_proxy()
    seed_all(args.warmup_seed)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    v2.G.update({"pipe": pipe, "bg": bg, "mult": opt.mult,
                 "loss_thresh": opt.loss_thresh, "sh_degree": dataset.sh_degree})

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    global_probe = (test_cams[:5] if test_cams and len(test_cams) > 0 else train_cams[:5])
    pool_cams = train_cams[:args.view_pool]
    cost_views = pool_cams[:args.n_cost_views]
    cam_seq_rng = random.Random(args.camseq_seed)
    cam_seq = [cam_seq_rng.randint(0, len(train_cams) - 1) for _ in range(args.diag_steps)]

    # ---------------- master snapshot (persisted, reused on rerun) ---------
    # A fresh warmup differs through CUDA atomic nondeterminism, so the whole
    # experiment (oracle + groups) must derive from ONE snapshot. Persisting
    # it makes the per-candidate oracle cache resumable across processes.
    snap_path = f"{OUT}/cache/master_snapshot.pt"
    static_proto = dict(protocol_base)
    static_proto.update({"warmup_seed": args.warmup_seed, "view_pool": args.view_pool})
    hook = None
    if os.path.exists(snap_path):
        try:
            blob = torch.load(snap_path)
            if blob.get("static_proto") == static_proto:
                hook = {"imp": blob["imp"], "clone_set": blob["clone_set"],
                        "split_set": blob["split_set"],
                        "radii": blob["radii"], "snapshot": blob["snapshot"],
                        "pool_data": [{"cam": cam, "radii": blob["pool_radii"][i],
                                       "mask": blob["pool_masks"][i]}
                                      for i, cam in enumerate(pool_cams)]}
                print("[b2c] loaded persisted master snapshot (warmup skipped)")
        except Exception as e:
            print(f"[b2c] snapshot load failed ({e}); running fresh warmup")
            hook = None

    if hook is None:
        hook = run_warmup(gaussians, scene, opt, pipe, bg, dataset, args, diag_iter,
                          train_cams, pool_cams, sampling_cameras,
                          compute_gaussian_score_fastgs, render_fastgs, v2)
        blob = {"static_proto": static_proto, "imp": hook["imp"],
                "clone_set": hook["clone_set"], "split_set": hook["split_set"],
                "radii": hook["radii"], "snapshot": hook["snapshot"],
                "pool_radii": [pd["radii"] for pd in hook["pool_data"]],
                "pool_masks": [pd["mask"] for pd in hook["pool_data"]]}
        torch.save(blob, snap_path)
        print("[b2c] master snapshot persisted to cache/master_snapshot.pt")

    del gaussians
    torch.cuda.empty_cache()

    snapshot, radii_snap = hook["snapshot"], hook["radii"].clone()
    xyz_snap = snapshot[1]
    n_before = int(xyz_snap.shape[0])
    # snapshot fingerprint: cache keys are only valid within THIS snapshot
    # (fresh warmups differ through CUDA atomic nondeterminism)
    protocol = dict(protocol_base)
    protocol["snapshot_fp"] = "%d_%.4f_%.4f" % (
        n_before, float(xyz_snap.sum()), float(snapshot[6].sum()))
    with open(f"{OUT}/cache/protocol.json", "w") as f:
        json.dump(protocol, f, indent=1)
    clone_idx_all = np.where(hook["clone_set"].cpu().numpy())[0]
    split_idx_all = np.where(hook["split_set"].cpu().numpy())[0]
    population = np.concatenate([clone_idx_all, split_idx_all])
    native_action_of = {int(i): "clone" for i in clone_idx_all}
    native_action_of.update({int(i): "split" for i in split_idx_all})
    imp = hook["imp"].to(torch.float32).cpu().numpy()
    print(f"[b2c] iter={diag_iter} population={len(population)} "
          f"(clone {len(clone_idx_all)} / split {len(split_idx_all)}), N={n_before}")
    with open(f"{OUT}/cache/population.json", "w") as f:
        json.dump({"iteration": diag_iter, "n_before": n_before,
                   "population": [int(i) for i in population]}, f)
    # (gaussians already deleted after the snapshot block above)

    # ---------------- group pools (nested) ---------------------------------
    pools = {}
    for seed_id in range(args.n_group_seeds):
        rng = np.random.RandomState(args.group_base_seed + seed_id)
        pool = [int(i) for i in rng.choice(population, size=min(args.pool_K, len(population)),
                                           replace=False)]
        pools[seed_id] = pool
        for K in Ks:
            members = pool[:K]
            with open(f"{OUT}/data/groups/group_K{K}_seed{seed_id}.json", "w") as f:
                json.dump({"K": K, "seed": seed_id,
                           "members": [{"parent_index": i,
                                        "native_action": native_action_of[i],
                                        "importance_score": int(imp[i]),
                                        "split_seed_group": split_seed_for(i, 0)}
                                       for i in members]}, f, indent=1)
    needed = sorted({i for pool in pools.values() for i in pool})
    print(f"[b2c] {args.n_group_seeds} pools x {args.pool_K} -> {len(needed)} unique candidates")

    # ---------------- shared Keep reference (trained ONCE) -----------------
    keep_path = f"{OUT}/cache/keep_ref.json"
    keep_err = {}
    if os.path.exists(keep_path):
        d = json.load(open(keep_path))
        if d.get("protocol") == protocol:
            keep_err = {int(k): v for k, v in d["keep"].items()}
    gk = None
    todo = [i for i in needed if i not in keep_err]
    if todo:
        print(f"[b2c] training shared Keep reference model ({args.diag_steps} steps) ...")
        gk = restore_from(snapshot, opt)
        train_100(gk, opt, train_cams, cam_seq, diag_iter, args.diag_steps, args.train_seed)
    for i in todo:
        views = build_cand_views(i, hook["pool_data"], xyz_snap, args.n_probe)
        keep_err[i] = demand_l1_on_views(gk, views)
    if todo:
        with open(keep_path, "w") as f:
            json.dump({"protocol": protocol,
                       "keep": {str(k): v for k, v in keep_err.items()}}, f)
        del gk
        torch.cuda.empty_cache()

    # ---------------- single-candidate oracles (cached, resumable) ---------
    oracle = {}
    for fname in os.listdir(f"{OUT}/cache"):
        if fname.startswith("cand_") and fname.endswith(".json"):
            d = json.load(open(f"{OUT}/cache/{fname}"))
            if d.get("protocol") == protocol:
                oracle[d["parent_index"]] = d
    print(f"[b2c] oracle cache: {len(oracle)} candidates already done")

    for n_done, idx in enumerate([i for i in needed if i not in oracle]):
        views = build_cand_views(idx, hook["pool_data"], xyz_snap, args.n_probe)
        e_keep = keep_err[idx]

        g = restore_from(snapshot, opt)
        apply_single_action(g, "clone", idx, radii_snap, 0)
        train_100(g, opt, train_cams, cam_seq, diag_iter, args.diag_steps, args.train_seed)
        e_clone = demand_l1_on_views(g, views)
        del g

        e_splits = []
        for r in range(5):
            g = restore_from(snapshot, opt)
            apply_single_action(g, "split", idx, radii_snap, split_seed_for(idx, r))
            train_100(g, opt, train_cams, cam_seq, diag_iter, args.diag_steps, args.train_seed)
            e_splits.append(demand_l1_on_views(g, views))
            del g
        torch.cuda.empty_cache()

        dq_clone = None if (e_keep is None or e_clone is None) else e_keep - e_clone
        dq_splits = [e_keep - e for e in e_splits if e is not None and e_keep is not None]
        split_mean = float(np.mean(dq_splits)) if dq_splits else None
        split_std = float(np.std(dq_splits)) if dq_splits else None
        if dq_clone is None or split_mean is None:
            oracle_action, gap, sem = None, None, None
        else:
            gap = split_mean - dq_clone
            sem = split_std / np.sqrt(len(dq_splits)) if dq_splits else None
            oracle_action = "clone" if dq_clone > split_mean else "split"
        rec = {"parent_index": int(idx), "native_action": native_action_of[idx],
               "importance_score": int(imp[idx]), "keep_demand_l1_100": e_keep,
               "clone_demand_l1_100": e_clone, "split_demand_l1_100_repeats": e_splits,
               "clone_dQ_100": dq_clone, "split_dQ_mean_100": split_mean,
               "split_dQ_std_100": split_std, "oracle_action": oracle_action,
               "oracle_gap": gap, "split_sem": sem,
               "oracle_confident": None if gap is None or sem is None else bool(abs(gap) > sem),
               "oracle_high_confident": None if gap is None else bool(abs(gap) > max(sem or 0, 1e-4)),
               "split_seeds": [split_seed_for(idx, r) for r in range(5)],
               "protocol": protocol}
        with open(f"{OUT}/cache/cand_{idx}.json", "w") as f:
            json.dump(rec, f)
        oracle[idx] = rec
        if (n_done + 1) % 25 == 0:
            print(f"[b2c] oracle progress: {n_done + 1}/{len(needed)}")

    # oracle summary files
    with open(f"{OUT}/data/b2c_candidate_oracles.json", "w") as f:
        json.dump({"protocol": protocol, "oracles": [oracle[i] for i in needed]}, f, indent=1)
    ocols = ["parent_index", "native_action", "importance_score", "keep_demand_l1_100",
             "clone_demand_l1_100", "clone_dQ_100", "split_dQ_mean_100", "split_dQ_std_100",
             "oracle_action", "oracle_gap", "split_sem", "oracle_confident",
             "oracle_high_confident"]
    with open(f"{OUT}/data/b2c_candidate_oracles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ocols)
        for i in needed:
            r = oracle[i]
            w.writerow(["NA" if r.get(c) is None else (f"{r[c]:.8g}" if isinstance(r[c], float)
                        else r[c]) for c in ocols])
    print(f"[b2c] oracle table saved ({len(needed)} candidates)")

    # ---------------- group phase ------------------------------------------
    records = []
    h = args.diag_steps // 2
    for K in Ks:
        for seed_id in range(args.n_group_seeds):
            members = pools[seed_id][:K]
            # candidates without a measurable demand oracle (no demand-valid
            # view, 33/873) fall back to their NATIVE action so that every
            # policy processes exactly K candidates (dN = K); oracle and
            # shuffled share the same defaulted list -> composition matched.
            assignment_oracle = [(i, oracle[i]["oracle_action"] or native_action_of[i])
                                 for i in members]
            n_oc = sum(1 for _, a in assignment_oracle if a == "clone")
            n_os = K - n_oc
            n_hc = sum(1 for i in members if oracle[i].get("oracle_high_confident"))
            print(f"\n[b2c] K={K} seed={seed_id}: oracle clone/split = {n_oc}/{n_os} "
                  f"(high-conf {n_hc}), native clone = "
                  f"{sum(1 for i in members if native_action_of[i]=='clone')}")

            # fixed Group Support/Demand ROI (pre-action union, shared by all policies)
            support_masks, demand_masks = [], []
            for vi, cam in enumerate(cost_views):
                sm = torch.zeros(int(cam.image_height), int(cam.image_width),
                                 dtype=torch.bool, device="cuda")
                for idx in members:
                    vr = hook["pool_data"][vi]["radii"]
                    if int(vr[idx]) <= 0:
                        continue
                    u, v, _, ok = project_to_pixel(cam, xyz_snap[idx])
                    if not ok:
                        continue
                    x0, y0, x1, y1 = roi_box(u, v, float(vr[idx]), cam.image_width,
                                             cam.image_height, TILE_MARGIN_PX)
                    sm[y0:y1, x0:x1] = True
                dm = sm & hook["pool_data"][vi]["mask"]
                support_masks.append(sm)
                demand_masks.append(dm)

            policies = [("keep", None), ("native", "native"), ("all_clone", "all_clone"),
                        ("all_split", "all_split"), ("oracle_mix", assignment_oracle)]
            shuffles = []
            for rep in range(args.shuffled_repeats):
                rng = np.random.RandomState(args.shuffle_base + seed_id * 100 + rep)
                acts = [a for _, a in assignment_oracle]
                rng.shuffle(acts)
                shuffles.append([(i, a) for (i, _), a in zip(assignment_oracle, acts)])
            branches = [(p, a) for p, a in policies] + \
                       [("shuffled", s) for s in shuffles]

            for b_idx, (pname, assign) in enumerate(branches):
                g = restore_from(snapshot, opt)
                if pname == "native":
                    apply_assignment(g, [(i, native_action_of[i]) for i in members], radii_snap)
                elif pname == "all_clone":
                    apply_assignment(g, [(i, "clone") for i in members], radii_snap)
                elif pname == "all_split":
                    apply_assignment(g, [(i, "split") for i in members], radii_snap)
                elif pname in ("oracle_mix", "shuffled"):
                    apply_assignment(g, assign, radii_snap)
                n_after = int(g.get_xyz.shape[0])

                seed_all(args.train_seed)
                ev = {0: group_eval(g, cost_views, support_masks, demand_masks, global_probe)}
                for i in range(1, args.diag_steps + 1):
                    cam = train_cams[cam_seq[i - 1]]
                    native_train_one_iter(diag_iter + i, cam, g, pipe, bg, opt)
                    with torch.no_grad():
                        if opt.optimizer_type == "default":
                            g.optimizer_step(diag_iter + i)
                        if i in (h, args.diag_steps):
                            ev[i] = group_eval(g, cost_views, support_masks, demand_masks,
                                               global_probe, with_lpips=(i == args.diag_steps))
                del g
                torch.cuda.empty_cache()

                rec = {"K": K, "group_seed": seed_id, "policy": pname,
                       "shuffle_repeat": (b_idx - len(policies)) if pname == "shuffled" else None,
                       "num_candidates": K, "oracle_clone_count": n_oc, "oracle_split_count": n_os,
                       "high_confident_count": n_hc,
                       "native_clone_count": sum(1 for i in members if native_action_of[i] == "clone"),
                       "num_gaussians_before": n_before, "num_gaussians_after": n_after,
                       "delta_num_gaussians": n_after - n_before,
                       "group_demand_l1_0": ev[0]["demand"]["l1"],
                       "group_demand_l1_50": ev[h]["demand"]["l1"],
                       "group_demand_l1_100": ev[args.diag_steps]["demand"]["l1"],
                       "group_demand_psnr_100": ev[args.diag_steps]["demand"]["psnr"],
                       "group_support_l1_100": ev[args.diag_steps]["support"]["l1"],
                       "group_support_psnr_100": ev[args.diag_steps]["support"]["psnr"],
                       "global_loss_100": ev[args.diag_steps]["global"]["loss"],
                       "global_psnr_0": ev[0]["global"]["psnr"],
                       "global_psnr_100": ev[args.diag_steps]["global"]["psnr"],
                       "global_ssim_100": ev[args.diag_steps]["global"].get("ssim"),
                       "global_lpips_100": ev[args.diag_steps]["global"].get("lpips"),
                       "tile_0": ev[0]["tile_pairs_mean"],
                       "tile_100": ev[args.diag_steps]["tile_pairs_mean"]}
                records.append(rec)
                dl1 = rec["group_demand_l1_100"]
                dps = rec["group_demand_psnr_100"]
                dl1_s = "NA" if dl1 is None else "%.6f" % dl1
                dps_s = "NA" if dps is None else "%.3f" % dps
                print(f"    {pname:10s}: N{rec['delta_num_gaussians']:+d} "
                      f"dL1@100={dl1_s} dPSNR@100={dps_s} tile0={rec['tile_0']:.0f}")

            # keep reference for deltas
            keep_rec = next(r for r in records if r["K"] == K and r["group_seed"] == seed_id
                            and r["policy"] == "keep")
            for r in [x for x in records if x["K"] == K and x["group_seed"] == seed_id]:
                r["dTile_0"] = r["tile_0"] - keep_rec["tile_0"]
                r["dTile_100"] = r["tile_100"] - keep_rec["tile_100"]

    with open(f"{OUT}/data/b2c_group_results.json", "w") as f:
        n_cl = sum(1 for i in needed if oracle[i]["oracle_action"] == "clone")
        n_sp = sum(1 for i in needed if oracle[i]["oracle_action"] == "split")
        n_na = len(needed) - n_cl - n_sp
        json.dump({"protocol": protocol, "records": records,
                   "oracle_summary": {"n": len(needed), "n_clone": n_cl,
                                      "n_split": n_sp, "n_no_oracle": n_na,
                                      "clone_pct": n_cl / len(needed) * 100}}, f, indent=1)
    cols = list(records[0].keys())
    with open(f"{OUT}/data/b2c_group_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow(["NA" if r[c] is None else (f"{r[c]:.8g}" if isinstance(r[c], float) else r[c])
                        for c in cols])
    print(f"\n[b2c] saved {OUT}/data/b2c_group_results.json / .csv ({len(records)} records)")


if __name__ == "__main__":
    main()
