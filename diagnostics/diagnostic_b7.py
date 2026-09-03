#
# Paper B - B7: Dynamic Capacity Signal Diagnostic
#
# Does the optimization history of a candidate Gaussian over the last W=100
# iterations BEFORE the densification snapshot predict q_best better than the
# B6 static features?  Who-reachability only (no Clone/Split predictor, no
# allocator, no FastGS modification).
#
# collect : one native warmup to 12000. At every 10th iteration inside each
#           target's pre-snapshot window (10 samples / 100 steps) record
#           per-Gaussian instantaneous gradient (norm + dx/dy), activated
#           params (xyz/scale/opacity) and screen radii, plus residual maps
#           of the 8 fixed cost views. At each target event the snapshot
#           (pre-densification), pool radii/masks, scores and camera identity
#           are persisted. All index-stable: no densification event happens
#           inside any window (events every 500, windows 100 wide).
# process : camera-asserted. Per snapshot: oversample -> 150 valid
#           candidates, B2C-protocol oracles (Clone x1 + Split x5, cached),
#           static features (B6 set) and dynamic features (window statistics
#           + DemandPersistence / OptimizationExposure / Inefficiency).
# replay  : only run when the LOSO gate passes (see analyze_b7.py).
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
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras, get_loss  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from diagnostics.common import (clone_tree, install_c_proxy, seed_all,  # noqa: E402
                                project_to_pixel, roi_box, native_train_one_iter)
import diagnostics.diagnostic_v2 as v2  # noqa: E402
from diagnostics.diagnostic_b2c import (restore_from, apply_assignment, group_eval,  # noqa: E402
                                        split_seed_for, build_cand_views, demand_l1_on_views)

OUT = "paper_b/b7_dynamic_capacity_signal"
B2C = "paper_b/b2_c_oracle_scalability"
TILE_MARGIN_PX = 16
W = 100
SAMP_EVERY = 10


def _cam_id(c):
    return [str(c.image_name), int(c.uid)]


# ------------------------------------------------------------- collect -----

def collect(dataset, opt, pipe, args, targets):
    seed_all(0)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    train_cams = scene.getTrainCameras()
    pool_cams = train_cams[:args.view_pool]
    cost_views = pool_cams[:args.n_cost_views]

    ident = {"train": [_cam_id(c) for c in train_cams],
             "test": [_cam_id(c) for c in scene.getTestCameras()],
             "pool": [_cam_id(c) for c in pool_cams]}
    json.dump(ident, open(f"{OUT}/cache/camera_identity.json", "w"))

    # window membership: iteration -> target T (windows do not overlap)
    win_of = {}
    for T in targets:
        for k in range(1, W // SAMP_EVERY + 1):          # samples T-90 .. T
            win_of[T - W + k * SAMP_EVERY] = T

    hist = {T: {"iters": [], "xyz": [], "scale": [], "opacity": [],
                "gradn": [], "grada": [], "gdx": [], "gdy": [], "radii": [],
                "l1": []} for T in targets}
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    for iteration in range(1, targets[-1] + 1):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = random.randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)
        _, vpt, vis_filter, radii = native_train_one_iter(
            iteration, viewpoint_cam, gaussians, pipe, bg, opt)

        if iteration in win_of and iteration % SAMP_EVERY == 0:
            T = win_of[iteration]
            g = vpt.grad.detach()
            gn = torch.norm(g[:, :2], dim=-1)
            ga = torch.norm(g[:, 2:], dim=-1)
            h = hist[T]
            h["iters"].append(iteration)
            h["xyz"].append(gaussians.get_xyz.detach().cpu())
            h["scale"].append(gaussians.get_scaling.detach().cpu())
            h["opacity"].append(gaussians.get_opacity.detach().cpu().squeeze(-1))
            h["gradn"].append(gn.cpu())
            h["grada"].append(ga.cpu())
            h["gdx"].append(g[:, 0].cpu())
            h["gdy"].append(g[:, 1].cpu())
            h["radii"].append(radii.cpu().float())
            l1s = []
            with torch.no_grad():
                for cam in cost_views:
                    out = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
                    l1s.append(torch.mean(torch.abs(out["render"] - cam.original_image.cuda()),
                                          dim=0).half().cpu())
            h["l1"].append(torch.stack(l1s))
            del g

        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[vis_filter] = torch.max(
                    gaussians.max_radii2D[vis_filter], radii[vis_filter])
                gaussians.add_densification_stats(vpt, vis_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    my_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_stack)
                    imp, pru = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                    if iteration in targets:
                        clone_set, split_set, _, _, _ = v2.native_selection_sets(
                            gaussians, imp, opt, scene.cameras_extent)
                        pool_radii, pool_masks = [], []
                        with torch.no_grad():
                            for cam in pool_cams:
                                out = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
                                pool_radii.append(out["radii"])
                                pool_masks.append((get_loss(out["render"], cam.original_image.cuda())
                                                   > opt.loss_thresh).detach())
                        torch.save({"snapshot": clone_tree(gaussians.capture(opt.optimizer_type)),
                                    "radii": radii.clone(), "imp": imp,
                                    "clone_set": clone_set, "split_set": split_set,
                                    "pool_radii": pool_radii, "pool_masks": pool_masks},
                                   f"{OUT}/cache/snap7_{iteration}.pt")
                        h = hist[iteration]
                        torch.save({"iters": h["iters"],
                                    "xyz": torch.stack(h["xyz"]),
                                    "scale": torch.stack(h["scale"]),
                                    "opacity": torch.stack(h["opacity"]),
                                    "gradn": torch.stack(h["gradn"]),
                                    "grada": torch.stack(h["grada"]),
                                    "gdx": torch.stack(h["gdx"]),
                                    "gdy": torch.stack(h["gdy"]),
                                    "radii": torch.stack(h["radii"]),
                                    "l1": torch.stack(h["l1"])},
                                   f"{OUT}/cache/dyn_{iteration}.pt")
                        for k in list(h.keys()):
                            h[k] = [] if isinstance(h[k], list) else h[k]
                        n_c, n_s = int(clone_set.sum()), int(split_set.sum())
                        print(f"[b7-collect] it={iteration}: snapshot + dyn saved, "
                              f"cands={n_c + n_s} (clone {n_c})")
                    gaussians.densify_and_prune_fastgs(
                        max_screen_size=(20 if iteration > opt.opacity_reset_interval else None),
                        min_opacity=0.005, extent=scene.cameras_extent, radii=radii, args=opt,
                        importance_score=imp, pruning_score=pru)
                if iteration % opt.opacity_reset_interval == 0 or \
                        (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            if opt.optimizer_type == "default":
                gaussians.optimizer_step(iteration)
    print("[b7-collect] done")


# ------------------------------------------------------------- process -----

def process(dataset, opt, pipe, args, iters, train_cams, test_cams, pool_cams,
            global_probe, cam_seq):
    bg = v2.G["bg"]
    h = args.diag_steps // 2
    feats = []
    for it in iters:
        snap = torch.load(f"{OUT}/cache/snap7_{it}.pt")
        dyn = torch.load(f"{OUT}/cache/dyn_{it}.pt")
        snapshot, radii_snap = snap["snapshot"], snap["radii"].clone()
        xyz_snap = snapshot[1]
        n0 = int(xyz_snap.shape[0])
        clone_idx = set(int(i) for i in np.where(snap["clone_set"].cpu().numpy())[0])
        gaussians = GaussianModel(v2.G["sh_degree"], opt.optimizer_type)
        gaussians.restore(clone_tree(snapshot), opt)

        pool_data = []
        with torch.no_grad():
            for i, cam in enumerate(pool_cams):
                out = render_fastgs(cam, gaussians, pipe, bg, opt.mult)
                l1_map = torch.mean(torch.abs(out["render"] - cam.original_image.cuda()), dim=0)
                pool_data.append({"cam": cam, "radii": snap["pool_radii"][i],
                                  "l1_map": l1_map, "mask": snap["pool_masks"][i]})
        gv = gaussians.xyz_gradient_accum / gaussians.denom
        gv[gv.isnan()] = 0.0
        ga = gaussians.xyz_gradient_accum_abs / gaussians.denom
        ga[ga.isnan()] = 0.0
        grad_norm, grad_abs_norm = torch.norm(gv, dim=-1), torch.norm(ga, dim=-1)
        imp, dummy = snap["imp"], torch.zeros_like(snap["imp"], dtype=torch.float32)

        # --- candidate selection: oversample to N_VALID valid (seed 0) ---
        population = np.concatenate([np.where(snap["clone_set"].cpu().numpy())[0],
                                     np.where(snap["split_set"].cpu().numpy())[0]])
        rng = np.random.RandomState(777 + it * 10 + 0)
        order = rng.permutation(population)

        # keep reference model (B5 definition): trained 100 steps from snapshot
        gk = restore_from(snapshot, opt)
        seed_all(args.train_seed)
        for i in range(1, args.diag_steps + 1):
            native_train_one_iter(it + i, train_cams[cam_seq[i - 1]], gk, pipe, bg, opt)
            with torch.no_grad():
                gk.optimizer_step(it + i)

        def oracle_of(idx):
            cp = f"{OUT}/cache/oracle_{it}_{idx}.json"
            if os.path.exists(cp):
                return json.load(open(cp))
            views = build_cand_views(int(idx), pool_data, xyz_snap, args.n_probe)
            e_keep = demand_l1_on_views(gk, views)
            if e_keep is None:
                o = {"parent_index": int(idx), "valid": False}
                json.dump(o, open(cp, "w"))
                return o
            e_clone, e_splits = None, []
            g = restore_from(snapshot, opt)
            g.tmp_radii = radii_snap.clone()
            s = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            s[idx] = True
            g.densify_and_clone_fastgs(s, torch.ones_like(s))
            g.tmp_radii = None
            seed_all(args.train_seed)
            for i in range(1, args.diag_steps + 1):
                native_train_one_iter(it + i, train_cams[cam_seq[i - 1]], g, pipe, bg, opt)
                with torch.no_grad():
                    g.optimizer_step(it + i)
            e_clone = demand_l1_on_views(g, views)
            del g
            for r in range(5):
                g = restore_from(snapshot, opt)
                g.tmp_radii = radii_snap.clone()
                s = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
                s[idx] = True
                seed_all(split_seed_for(int(idx), r))
                g.densify_and_split_fastgs(s, torch.ones_like(s), N=2)
                g.tmp_radii = None
                seed_all(args.train_seed)
                for i in range(1, args.diag_steps + 1):
                    native_train_one_iter(it + i, train_cams[cam_seq[i - 1]], g, pipe, bg, opt)
                    with torch.no_grad():
                        g.optimizer_step(it + i)
                e_splits.append(demand_l1_on_views(g, views))
                del g
            torch.cuda.empty_cache()
            q_c, q_s = e_keep - e_clone, float(np.mean([e_keep - e for e in e_splits]))
            o = {"parent_index": int(idx), "valid": True,
                 "q_clone": q_c, "q_split_mean": q_s,
                 "q_best": max(q_c, q_s), "oracle_action": "clone" if q_c > q_s else "split"}
            json.dump(o, open(cp, "w"))
            return o

        members, tried = [], 0
        for idx in order:
            if len(members) >= args.n_valid:
                break
            tried += 1
            o = oracle_of(int(idx))
            if o.get("valid"):
                members.append(o)
        print(f"[b7] it={it}: {len(members)} valid ({tried} tried) oracles")

        # --- dynamic features per member ---
        dxyz = dyn["xyz"].float()                       # (S,N,3)
        dsc = dyn["scale"].float()                      # (S,N,3)
        dop = dyn["opacity"].float()                    # (S,N)
        gn = dyn["gradn"].float()                       # (S,N)
        ga_ = dyn["grada"].float()
        gdx = dyn["gdx"].float()
        gdy = dyn["gdy"].float()
        rad = dyn["radii"].float()                      # (S,N)
        l1maps = dyn["l1"].float().cuda()               # (S,8,H,W)
        S = dxyz.shape[0]
        xs = np.arange(S, dtype=np.float64)

        def series_stats(v):                            # v: (S,)
            if np.any(np.isnan(v)):
                v = v[~np.isnan(v)]
            if len(v) < 2:
                return {"mean": float(np.mean(v)) if len(v) else None,
                        "std": 0.0, "slope": 0.0, "last_first": 0.0, "cv": 0.0}
            slope = float(np.polyfit(xs[:len(v)], v, 1)[0])
            return {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "slope": slope, "last_first": float(v[-1] - v[0]),
                    "cv": float(np.std(v) / (abs(np.mean(v)) + 1e-8))}

        cost_idx = list(range(len(pool_cams)))[:args.n_cost_views]
        rows = []
        for o in members:
            idx = o["parent_index"]
            # static features (B6 set) via v2.build_candidate
            cand = v2.build_candidate(gaussians, idx, "clone" if idx in clone_idx else "split",
                                      it, imp, dummy, grad_norm, grad_abs_norm,
                                      pool_data, args.n_probe)
            views = cand["_views"] if cand is not None else []
            f = {"iteration": it, "parent_index": int(idx),
                 "q_best": o["q_best"], "q_clone": o["q_clone"],
                 "q_split": o["q_split_mean"], "oracle_action": o["oracle_action"]}
            if cand is not None:
                f.update({"importance_score": cand["importance_score"], "grad": cand["grad"],
                          "grad_abs": cand["grad_abs"], "opacity": cand["opacity"],
                          "scale_max": cand["scale_max"], "scale_min": cand["scale_min"],
                          "scale_anisotropy": cand["scale_anisotropy"],
                          "projected_radius_mean": cand["projected_radius_mean"],
                          "footprint_mean": cand["footprint_mean"],
                          "footprint_area_mean": cand["footprint_area_mean"],
                          "visibility_count": cand["num_visible_views"],
                          "residual_energy_mean": cand["residual_energy_mean"],
                          "residual_energy_std": cand["residual_energy_std"],
                          "residual_anisotropy_mean": cand["residual_anisotropy_mean"],
                          "residual_anisotropy_std": cand["residual_anisotropy_std"],
                          "residual_extent_mean": cand["residual_extent_mean"],
                          "residual_centroid_offset_mean": cand["residual_centroid_offset_mean"],
                          "demand_pixel_ratio_mean":
                              (float(np.mean([float(int(vd["mask"][vd["box"][1]:vd["box"][3],
                                                        vd["box"][0]:vd["box"][2]].sum()))
                                              / max((vd["box"][2]-vd["box"][0])*(vd["box"][3]-vd["box"][1]), 1)
                                              for vd in views])) if views else None),
                          "residual_parent_alignment_mean": cand["residual_parent_alignment_mean"],
                          "residual_extent_ratio_mean": cand["residual_extent_ratio_mean"],
                          "proj_anisotropy_mean": cand["proj_anisotropy_mean"]})
            # dynamic series
            xyz_i = dxyz[:, idx].numpy()
            sc_i = np.max(dsc[:, idx].numpy(), axis=1)
            op_i = dop[:, idx].numpy()
            gn_i = gn[:, idx].numpy()
            ga_i = ga_[:, idx].numpy()
            gdx_i, gdy_i = gdx[:, idx].numpy(), gdy[:, idx].numpy()
            rad_i = rad[:, idx].numpy()
            vis_i = (rad_i > 0).astype(np.float64)
            step_xyz = np.linalg.norm(np.diff(xyz_i, axis=0), axis=1) if S > 1 else np.array([0.0])
            path_xyz = float(np.sum(step_xyz))
            net_xyz = float(np.linalg.norm(xyz_i[-1] - xyz_i[0])) if S > 1 else 0.0
            step_sc = np.abs(np.diff(sc_i)) if S > 1 else np.array([0.0])
            step_op = np.abs(np.diff(op_i)) if S > 1 else np.array([0.0])

            # demand history: mean ROI residual energy per sample (fixed ROI)
            dem = []
            for si in range(S):
                vals = []
                for vd in views:
                    x0, y0, x1, y1 = vd["box"]
                    try:
                        vi = pool_cams.index(vd["cam"])
                    except ValueError:
                        continue
                    if vi < args.n_cost_views:
                        vals.append(float(l1maps[si, vi, y0:y1, x0:x1].mean()))
                if vals:
                    dem.append(float(np.mean(vals)))
            ds = series_stats(np.array(dem)) if dem else \
                {"mean": None, "std": None, "slope": None, "last_first": None, "cv": None}

            a2 = np.deg2rad(np.arctan2(gdy_i, gdx_i + 1e-12)) * 2.0
            gdc = float(np.hypot(np.mean(np.cos(a2)), np.mean(np.sin(a2))))

            param_resp = path_xyz + float(np.sum(step_sc)) + float(np.sum(step_op)) + 1e-8
            f.update({
                "dyn_gradn_mean": series_stats(gn_i)["mean"],
                "dyn_gradn_std": series_stats(gn_i)["std"],
                "dyn_gradn_slope": series_stats(gn_i)["slope"],
                "dyn_gradn_cv": series_stats(gn_i)["cv"],
                "dyn_grada_mean": float(np.mean(ga_i)),
                "dyn_grada_slope": series_stats(ga_i)["slope"],
                "dyn_xyz_path": path_xyz,
                "dyn_xyz_net": net_xyz,
                "dyn_xyz_net_over_path": net_xyz / (path_xyz + 1e-8),
                "dyn_scale_path": float(np.sum(step_sc)),
                "dyn_scale_slope": series_stats(sc_i)["slope"],
                "dyn_opacity_net": float(op_i[-1] - op_i[0]) if S > 1 else 0.0,
                "dyn_opacity_path": float(np.sum(step_op)),
                "dyn_vis_persistence": float(np.mean(vis_i)),
                "dyn_radius_mean": float(np.mean(rad_i)),
                "dyn_radius_std": series_stats(rad_i)["std"],
                # core constructed quantities
                "DemandPersistence": ds["mean"],
                "DemandPersistence_cv": ds["cv"],
                "DemandPersistence_slope": ds["slope"],
                "OptimizationExposure": float(np.mean(gn_i) * np.sum(vis_i)),
                "OptimizationInefficiency":
                    (ds["mean"] / param_resp) if ds["mean"] is not None else None,
                "dyn_grad_dir_consistency": gdc,
            })
            rows.append(f)
        feats += rows
        del gaussians, snap, dyn, l1maps, gk
        torch.cuda.empty_cache()
        print(f"[b7] it={it}: features for {len(rows)} candidates")

    with open(f"{OUT}/data/b7_dynamic_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(feats[0].keys()))
        w.writeheader()
        for r in feats:
            w.writerow({k: ("NA" if v is None else (f"{v:.8g}" if isinstance(v, float) else v))
                        for k, v in r.items()})
    json.dump(feats, open(f"{OUT}/data/b7_dynamic_features.json", "w"))
    print(f"[b7] saved {len(feats)} rows")


def main():
    parser = ArgumentParser("Paper B B7: dynamic capacity signal")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--mode", type=str, default="collect",
                        choices=["collect", "process", "replay"])
    parser.add_argument("--iters", type=str, default="1000,1500,2000,5000,12000")
    parser.add_argument("--n_valid", type=int, default=150)
    parser.add_argument("--diag_steps", type=int, default=100)
    parser.add_argument("--view_pool", type=int, default=30)
    parser.add_argument("--n_cost_views", type=int, default=8)
    parser.add_argument("--n_probe", type=int, default=8)
    parser.add_argument("--train_seed", type=int, default=1234)
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    assert opt.optimizer_type == "default"
    iters = sorted(int(x) for x in args.iters.split(","))
    install_c_proxy()
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    v2.G.update({"pipe": pipe, "bg": bg, "mult": opt.mult,
                 "loss_thresh": opt.loss_thresh, "sh_degree": dataset.sh_degree})

    if args.mode == "collect":
        collect(dataset, opt, pipe, args, iters)
        return

    # process/replay: camera-asserted Scene (B5-Fix lesson)
    seed_all(0)
    g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, g)
    del g
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    pool_cams = train_cams[:args.view_pool]
    ident = json.load(open(f"{OUT}/cache/camera_identity.json"))
    for name, current, saved in (("train", [_cam_id(c) for c in train_cams], ident["train"]),
                                 ("test", [_cam_id(c) for c in test_cams], ident["test"]),
                                 ("pool", [_cam_id(c) for c in pool_cams], ident["pool"])):
        assert current == saved, f"[b7] camera identity FAIL on {name}"
    print("[b7] camera identity assertion PASS")
    global_probe = (test_cams[:5] if test_cams and len(test_cams) > 0 else train_cams[:5])
    cam_seq_rng = random.Random(2024)
    cam_seq = [cam_seq_rng.randint(0, len(train_cams) - 1) for _ in range(args.diag_steps)]
    if args.mode == "process":
        process(dataset, opt, pipe, args, iters, train_cams, test_cams, pool_cams,
                global_probe, cam_seq)
    else:
        from diagnostics.diagnostic_b7_replay import run_replay
        run_replay(dataset, opt, pipe, args, iters, train_cams, test_cams, pool_cams,
                   global_probe, cam_seq)


if __name__ == "__main__":
    main()
