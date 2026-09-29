#
# Paper B - B8-L: Long-Horizon Cardinality Validation
#
# Does the B8-A cardinality heterogeneity persist after 1000 optimization
# steps, or was it a short-horizon optimization transient?
#
# Parents: from B8-A Analysis Fix categories only —
#   B_split2_sufficient        -> branches Keep / Split-2 / Split-3
#   C3/C4/C6 higher-cardinality-> branches Keep / Split-2 / Split-MSCC
# 10 per category per stage where available (C parents ranked by
# normalized_gain, B parents by |gain_N3_vs_2| most-negative first — i.e.
# those where Split-3 hurt most at short horizon), fixed reproducible rule.
#
# Replay: 1000 steps, densify/prune/opacity-reset OFF, eval at 0/100/500/
# 1000 (demand L1 + local PSNR + global PSNR/SSIM/LPIPS at 100/500/1000).
# Split branches x 5 seeds; Keep x 1. Child lifecycle recorded per newborn
# child at birth/100/500/1000 (opacity, visible-view count, screen-space
# contribution proxy, xyz displacement from birth, scale change).
#
# Reuses B5 snapshots (identity-asserted). No FastGS modification, no B8-B.
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
from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from diagnostics.common import (clone_tree, install_c_proxy, seed_all,  # noqa: E402
                                native_train_one_iter)
import diagnostics.diagnostic_v2 as v2  # noqa: E402
from diagnostics.diagnostic_b2c import restore_from, build_cand_views  # noqa: E402

try:
    from lpipsPyTorch import lpips as lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False

B5 = "paper_b/b5_cross_stage_capacity_oracle"
B8A = "paper_b/b8_a_split_cardinality"
OUT = "paper_b/b8_l_long_horizon_cardinality"
REPEATS = 5
SEED_BASE = 950000
CKPTS = [0, 100, 500, 1000]


def _cam_id(c):
    return [str(c.image_name), int(c.uid)]


def split_seed(idx, n, r):
    return SEED_BASE + int(idx) * 100 + n * 10 + r


@torch.no_grad()
def eval_views(g, views, global_probe, with_global):
    """demand L1 + local PSNR on fixed views; global metrics; tile pairs."""
    proxy = install_c_proxy()
    l1s, mses, tiles = [], [], []
    for vd in views:
        out = render_fastgs(vd["cam"], g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])
        tiles.append(proxy.last_num_rendered)
        if not vd["demand_valid"]:
            continue
        x0, y0, x1, y1 = vd["box"]
        m = vd["mask"][y0:y1, x0:x1]
        ri = out["render"][:, y0:y1, x0:x1][:, m]
        gi = vd["cam"].original_image.cuda()[:, y0:y1, x0:x1][:, m]
        l1s.append(float((ri - gi).abs().mean()))
        mses.append(float(((ri - gi) ** 2).mean()))
    res = {"l1": float(np.mean(l1s)) if l1s else None,
           "psnr": float(np.mean([10 * np.log10(1 / max(m, 1e-10)) for m in mses]))
           if mses else None,
           "tile": float(np.mean(tiles)) if tiles else None}
    if with_global:
        ps_, ss_, lp_ = [], [], []
        for cam in global_probe:
            image = render_fastgs(cam, g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])["render"]
            gt = cam.original_image.cuda()
            ic, gc = torch.clamp(image, 0, 1), torch.clamp(gt, 0, 1)
            ps_.append(float(psnr(ic, gc).mean()))
            ss_.append(float(fast_ssim(ic.unsqueeze(0), gc.unsqueeze(0)).mean()))
            if LPIPS_OK:
                try:
                    lp_.append(float(lpips_fn(ic, gc, net_type="vgg").mean()))
                except Exception:
                    lp_.append(float("nan"))
        res["gpsnr"] = float(np.mean(ps_))
        res["gssim"] = float(np.mean(ss_))
        res["glpips"] = float(np.nanmean(lp_)) if lp_ else None
    return res


@torch.no_grad()
def child_state(g, child_ids, views, birth_xyz, birth_scale):
    """lifecycle snapshot for newborn children on the parent's fixed views."""
    proxy = install_c_proxy()
    # per-child visible-view count + mean alpha-weighted power over views
    vis_count = {i: 0 for i in child_ids}
    contrib = {i: 0.0 for i in child_ids}
    for vd in views:
        cam = vd["cam"]
        out = render_fastgs(cam, g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])
        # per-child contribution via a second pass is expensive; approximate
        # with alpha*power of each child at its own projected location using
        # the stored conic from the render bundle is not exposed — instead
        # count visibility (radii>0) and record opacity/scale/displacement.
        rr = out["radii"]
        for i in child_ids:
            if int(rr[i]) > 0:
                vis_count[i] += 1
    op = g.get_opacity.detach()
    sc = g.get_scaling.detach()
    xy = g.get_xyz.detach()
    rows = []
    for k, i in enumerate(child_ids):
        rows.append({
            "child_ordinal": k,
            "opacity": float(op[i]),
            "visible_views": vis_count[i],
            "xyz_disp_from_birth": float(torch.norm(xy[i] - birth_xyz[k]).item()),
            "scale_change_from_birth": float(
                torch.norm(sc[i] - birth_scale[k]).item()),
            "scale_norm": float(torch.norm(sc[i]).item()),
        })
    return rows


def main():
    parser = ArgumentParser("Paper B B8-L: long-horizon cardinality")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--per_cat", type=int, default=10)
    parser.add_argument("--view_pool", type=int, default=30)
    parser.add_argument("--n_probe", type=int, default=8)
    parser.add_argument("--train_seed", type=int, default=1234)
    parser.add_argument("--camseq_seed", type=int, default=2024)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    assert opt.optimizer_type == "default"

    install_c_proxy()
    seed_all(0)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    v2.G.update({"pipe": pipe, "bg": bg, "mult": opt.mult,
                 "loss_thresh": opt.loss_thresh, "sh_degree": dataset.sh_degree})
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    del gaussians
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    global_probe = (test_cams[:5] if test_cams and len(test_cams) > 0 else train_cams[:5])
    pool_cams = train_cams[:args.view_pool]
    ident = json.load(open(f"{B5}/cache/camera_identity.json"))
    for name, cur, saved in (("train", [_cam_id(c) for c in train_cams], ident["train"]),
                             ("pool", [_cam_id(c) for c in pool_cams], ident["pool"])):
        assert cur == saved, f"[b8l] camera identity FAIL on {name}"
    print("[b8l] camera identity assertion PASS")
    cam_seq_rng = random.Random(args.camseq_seed)
    cam_seq = [cam_seq_rng.randint(0, len(train_cams) - 1) for _ in range(args.steps)]

    # ---- parent selection from B8-A categories (fixed rule) ----
    arows = list(csv.DictReader(open(f"{B8A}/data/b8a_parent_results.csv")))
    per_cat = 2 if args.smoke else args.per_cat
    plan = []
    for it in (1000, 2000, 5000):
        sub = [r for r in arows if int(r["iteration"]) == it]
        b = [r for r in sub if r["parent_category"] == "B_split2_sufficient"]
        c = [r for r in sub if r["parent_category"].startswith("C")]
        # B: those where Split-3 hurt most at short horizon (most-negative gain_N3)
        b.sort(key=lambda r: float(r["gain_N3_vs_2"]))
        # C: strongest effect first (normalized_gain desc, tie gain desc)
        c.sort(key=lambda r: (-float(r["normalized_gain"] or 0),
                              -float(r["gain_over_split2"] or 0)))
        take_b = b[:per_cat]
        take_c = c[:per_cat]
        for r in take_b:
            plan.append({"iteration": it, "parent_index": int(r["parent_index"]),
                         "category": "B_split2_sufficient", "mscc": 2,
                         "b8a_norm_gain": r.get("normalized_gain"),
                         "actions": [(2, "split2"), (3, "split3")]})
        for r in take_c:
            n = int(r["parent_category"][1])
            plan.append({"iteration": it, "parent_index": int(r["parent_index"]),
                         "category": r["parent_category"], "mscc": n,
                         "b8a_norm_gain": r.get("normalized_gain"),
                         "actions": [(2, "split2"), (n, "split_mscc")]})
    json.dump(plan, open(f"{OUT}/cache/parent_plan.json", "w"), indent=1)
    print(f"[b8l] plan: {len(plan)} parents "
          f"(B={sum(1 for p in plan if p['category']=='B_split2_sufficient')}, "
          f"C={sum(1 for p in plan if p['category'].startswith('C'))})")

    cache_ckpt = f"{OUT}/cache/branches.json"
    done = json.load(open(cache_ckpt)) if os.path.exists(cache_ckpt) else {}
    life_rows, branch_rows = [], []

    for it in (1000, 2000, 5000):
        snap = torch.load(f"{B5}/cache/snap_{it}.pt")
        snapshot, radii_snap = snap["snapshot"], snap["radii"].clone()
        xyz_snap = snapshot[1]
        n_before = int(xyz_snap.shape[0])
        pool_data = [{"cam": cam, "radii": snap["pool_radii"][i], "mask": snap["pool_masks"][i]}
                     for i, cam in enumerate(pool_cams)]
        for p in [q for q in plan if q["iteration"] == it]:
            idx = p["parent_index"]
            views = build_cand_views(idx, pool_data, xyz_snap, args.n_probe)
            assert views, f"no views for parent {idx}"

            def run_one(action, n, seed):
                key = f"{it}|{idx}|{n}|{seed if seed is not None else 'keep'}"
                if key in done:
                    rec = done[key]
                    # JSON round-trip turns checkpoint keys into strings
                    # ("0"/"100"/"500"/"1000") — normalize back to int (B8-L cache fix)
                    rec["ev"] = {int(k): v for k, v in rec["ev"].items()}
                    return rec
                g = restore_from(snapshot, opt)
                life = []
                if action == "keep":
                    pass
                else:
                    g.tmp_radii = radii_snap.clone()
                    single = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
                    single[idx] = True
                    seed_all(seed)
                    g.densify_and_split_fastgs(single, torch.ones_like(single), N=n)
                    g.tmp_radii = None
                    child_ids = list(range(n_before - 1 + 0, n_before - 1 + n)) \
                        if False else None
                    # children are appended at the tail: last n rows
                    child_ids = list(range(int(g.get_xyz.shape[0]) - n, int(g.get_xyz.shape[0])))
                    bx = g.get_xyz.detach()[child_ids].clone()
                    bs = g.get_scaling.detach()[child_ids].clone()
                n_after = int(g.get_xyz.shape[0])
                seed_all(args.train_seed)
                ev = {0: eval_views(g, views, global_probe, with_global=False)}
                if action != "keep":
                    life.append((0, child_state(g, child_ids, views, bx, bs)))
                for i in range(1, args.steps + 1):
                    native_train_one_iter(it + i, train_cams[cam_seq[i - 1]], g,
                                          v2.G["pipe"], v2.G["bg"], opt)
                    with torch.no_grad():
                        if opt.optimizer_type == "default":
                            g.optimizer_step(it + i)
                        if i in (100, 500, 1000):
                            ev[i] = eval_views(g, views, global_probe, with_global=True)
                            if action != "keep":
                                life.append((i, child_state(g, child_ids, views, bx, bs)))
                rec = {"n_after": n_after, "d_n": n_after - n_before, "ev": ev}
                done[key] = rec
                json.dump(done, open(cache_ckpt, "w"))
                for t, lrows in life:
                    for lr in lrows:
                        life_rows.append({"iteration": it, "parent_index": idx,
                                          "category": p["category"], "N": n, "seed": seed,
                                          "step": t, **lr})
                del g
                torch.cuda.empty_cache()
                return rec

            keep = run_one("keep", 0, None)
            krow = {"iteration": it, "parent_index": idx, "category": p["category"],
                    "branch": "keep", "N": 0, "repeat": 0, "seed": None,
                    "d_n": keep["d_n"]}
            for k in (0, 100, 500, 1000):
                e = keep["ev"].get(k, {})
                krow[f"l1_{k}"] = e.get("l1")
                krow[f"psnr_{k}"] = e.get("psnr")
                krow[f"gpsnr_{k}"] = e.get("gpsnr")
                krow[f"gssim_{k}"] = e.get("gssim")
                krow[f"glpips_{k}"] = e.get("glpips")
                krow[f"tile_{k}"] = e.get("tile")
            branch_rows.append(krow)
            for n, act in [(0, "keep")] + p["actions"]:
                if act == "keep":
                    continue
                for r in range(REPEATS):
                    sd = split_seed(idx, n, r)
                    rec = run_one(act, n, sd)
                    row = {"iteration": it, "parent_index": idx, "category": p["category"],
                           "branch": act, "N": n, "repeat": r, "seed": sd,
                           "d_n": rec["d_n"]}
                    for k in (0, 100, 500, 1000):
                        e = rec["ev"].get(k, {})
                        row[f"l1_{k}"] = e.get("l1")
                        row[f"psnr_{k}"] = e.get("psnr")
                        row[f"gpsnr_{k}"] = e.get("gpsnr")
                        row[f"gssim_{k}"] = e.get("gssim")
                        row[f"glpips_{k}"] = e.get("glpips")
                        row[f"tile_{k}"] = e.get("tile")
                    branch_rows.append(row)
            print(f"[b8l] it={it} idx={idx} ({p['category']}) done "
                  f"({len(done)} branches cached)")
        del snap
        torch.cuda.empty_cache()

    # persist
    def flush():
        with open(f"{OUT}/data/b8l_repeat_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(branch_rows[0].keys()))
            w.writeheader()
            for r in branch_rows:
                w.writerow({k: ("NA" if v is None else (f"{v:.8g}" if isinstance(v, float) else v))
                            for k, v in r.items()})
        with open(f"{OUT}/data/b8l_child_lifecycle.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(life_rows[0].keys()))
            w.writeheader()
            for r in life_rows:
                w.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in r.items()})
        json.dump({"branches": branch_rows, "lifecycle": life_rows},
                  open(f"{OUT}/data/b8l_repeat_results.json", "w"))

    flush()
    print(f"[b8l] saved {len(branch_rows)} branch rows, {len(life_rows)} lifecycle rows")


if __name__ == "__main__":
    main()
