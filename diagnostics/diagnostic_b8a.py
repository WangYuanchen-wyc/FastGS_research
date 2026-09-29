#
# Paper B - B8-A: Split Cardinality Oracle
#
# Question: does a FastGS parent that truly needs Split need exactly TWO
# children, or does the minimal sufficient child count vary by parent?
#
# Per native-split parent, controlled branches from the SAME B5
# pre-densification snapshot (iterations 1000/2000/5000):
#     Keep x1
#     Split-N x 5 seeds, N in {2, 3, 4, 6}
# 100-step replay per branch (densify/prune/reset OFF), identical camera
# sequence / training seed / ROI; only N (and the split sampling seed)
# varies. Split-N calls the ORIGINAL densify_and_split_fastgs with N=k —
# its children construction is already parametrized by N:
#     children = parent.repeat(N) with independent normal offsets sampled
#     from the parent covariance; scale shrink = log(s / (0.8*N));
#     rotation/SH/opacity copied; parent row removed; child Adam state
#     zero-initialized (identical to native Split-2 for N=2).
# Δ#GS = N-1 per parent, verified per branch.
#
# Results are cached per branch (resumable). No FastGS source modification.
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
from arguments import ModelParams, PipelineParams, OptimizationParams  # noqa: E402
from diagnostics.common import (clone_tree, install_c_proxy, seed_all,  # noqa: E402
                                native_train_one_iter)
import diagnostics.diagnostic_v2 as v2  # noqa: E402
from diagnostics.diagnostic_b2c import restore_from, build_cand_views  # noqa: E402
from diagnostics.diagnostic_b2c import split_seed_for  # noqa: E402

B5 = "paper_b/b5_cross_stage_capacity_oracle"
OUT = "paper_b/b8_a_split_cardinality"
NS = [2, 3, 4, 6]
REPEATS = 5
SEED_BASE = 900000  # split seed: SEED_BASE + it*100000 + idx*100 + N*10 + r


def _cam_id(c):
    return [str(c.image_name), int(c.uid)]


def split_seed(idx, it, n, r):
    return SEED_BASE + it * 100000 + int(idx) * 100 + n * 10 + r


@torch.no_grad()
def eval_full(g, views, global_probe, step_tag, with_global=True):
    proxy = install_c_proxy()
    vals, mses, tiles = [], [], []
    for vd in views:
        out_r = render_fastgs(vd["cam"], g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])
        tiles.append(proxy.last_num_rendered)
        if not vd["demand_valid"]:
            continue
        x0, y0, x1, y1 = vd["box"]
        m = vd["mask"][y0:y1, x0:x1]
        ri = out_r["render"][:, y0:y1, x0:x1][:, m]
        gi = vd["cam"].original_image.cuda()[:, y0:y1, x0:x1][:, m]
        diff = (ri - gi).abs()
        vals.append(float(diff.mean()))
        mses.append(float((diff ** 2).mean()))
    psnrs = [10.0 * np.log10(1.0 / max(m, 1e-10)) for m in mses]
    if with_global:
        gpsnrs = []
        for cam in global_probe:
            image = render_fastgs(cam, g, v2.G["pipe"], v2.G["bg"], v2.G["mult"])["render"]
            gt = cam.original_image.cuda()
            from utils.image_utils import psnr as _psnr
            gpsnrs.append(float(_psnr(torch.clamp(image, 0, 1), torch.clamp(gt, 0, 1)).mean()))
    else:
        gpsnrs = []
    return {"demand_l1_" + step_tag: float(np.mean(vals)) if vals else None,
            "demand_psnr_" + step_tag: float(np.mean(psnrs)) if psnrs else None,
            "global_psnr_" + step_tag: float(np.mean(gpsnrs)) if gpsnrs else None,
            "tile_pairs_" + step_tag: float(np.mean(tiles)) if tiles else None}


def apply_split_n(g, parent_idx, n, seed, radii_snap):
    g.tmp_radii = radii_snap.clone()
    single = torch.zeros(g.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    single[parent_idx] = True
    seed_all(seed)
    g.densify_and_split_fastgs(single, torch.ones_like(single), N=n)
    g.tmp_radii = None


def run_branch(snapshot, radii_snap, opt, train_cams, cam_seq, it, steps, train_seed,
               parent_idx, action, n, seed, views, global_probe, n_before):
    g = restore_from(snapshot, opt)
    if action == "split":
        apply_split_n(g, parent_idx, n, seed, radii_snap)
    n_after = int(g.get_xyz.shape[0])
    seed_all(train_seed)
    h = steps // 2
    ev = {**eval_full(g, views, global_probe, "0", with_global=False)}
    for i in range(1, steps + 1):
        native_train_one_iter(it + i, train_cams[cam_seq[i - 1]], g, v2.G["pipe"], v2.G["bg"], opt)
        with torch.no_grad():
            if opt.optimizer_type == "default":
                g.optimizer_step(it + i)
            if i == h:
                ev.update(eval_full(g, views, global_probe, "50", with_global=False))
            if i == steps:
                ev.update(eval_full(g, views, global_probe, "100"))
    ev["n_after"] = n_after
    ev["d_n"] = n_after - n_before
    del g
    torch.cuda.empty_cache()
    return ev


def main():
    parser = ArgumentParser("Paper B B8-A: split cardinality oracle")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iters", type=str, default="1000,2000,5000")
    parser.add_argument("--n_parents", type=int, default=50)
    parser.add_argument("--diag_steps", type=int, default=100)
    parser.add_argument("--view_pool", type=int, default=30)
    parser.add_argument("--n_probe", type=int, default=8)
    parser.add_argument("--train_seed", type=int, default=1234)
    parser.add_argument("--camseq_seed", type=int, default=2024)
    parser.add_argument("--cand_seed", type=int, default=888)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    assert opt.optimizer_type == "default"
    iters = sorted(int(x) for x in args.iters.split(","))
    n_par = 2 if args.smoke else args.n_parents

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
                             ("test", [_cam_id(c) for c in test_cams], ident["test"]),
                             ("pool", [_cam_id(c) for c in pool_cams], ident["pool"])):
        assert cur == saved, f"[b8a] camera identity FAIL on {name}"
    print("[b8a] camera identity assertion PASS")
    cam_seq_rng = random.Random(args.camseq_seed)
    cam_seq = [cam_seq_rng.randint(0, len(train_cams) - 1) for _ in range(args.diag_steps)]

    rows = []
    cache_ckpt = f"{OUT}/cache/branches.json"
    done = {}
    if os.path.exists(cache_ckpt):
        done = json.load(open(cache_ckpt))
        print(f"[b8a] resume: {len(done)} branch results cached")

    def key(it, idx, act, n, r):
        return f"{it}|{idx}|{act}|{n}|{r}"

    for it in iters:
        snap = torch.load(f"{B5}/cache/snap_{it}.pt")
        snapshot, radii_snap = snap["snapshot"], snap["radii"].clone()
        xyz_snap = snapshot[1]
        n_before = int(xyz_snap.shape[0])
        split_set = snap["split_set"]
        split_idx_all = np.where(split_set.cpu().numpy())[0]
        imp = snap["imp"]
        rng = np.random.RandomState(args.cand_seed + it)
        order = rng.permutation(split_idx_all)
        pool_data = [{"cam": cam, "radii": snap["pool_radii"][i], "mask": snap["pool_masks"][i]}
                     for i, cam in enumerate(pool_cams)]

        selected, invalid_log = [], []
        for idx in order:
            if len(selected) >= n_par:
                break
            views = build_cand_views(int(idx), pool_data, xyz_snap, args.n_probe)
            n_dem = sum(1 for v in views if v["demand_valid"])
            if not views or n_dem == 0:
                invalid_log.append({"parent_index": int(idx),
                                    "reason": "no_valid_views" if not views else "no_demand_valid_views"})
                continue
            selected.append({"parent_index": int(idx), "views": views,
                             "importance": int(imp[int(idx)])})
        json.dump({"iteration": it, "selection_seed": args.cand_seed + it,
                   "parents": [{"parent_index": p["parent_index"], "native_action": "split",
                                "importance_score": p["importance"]} for p in selected],
                   "invalid_skipped": invalid_log},
                  open(f"{OUT}/cache/selection_it{it}.json", "w"), indent=1)
        print(f"[b8a] it={it}: {len(selected)} parents selected "
              f"({len(invalid_log)} invalid skipped, population={len(split_idx_all)})")

        for pi, p in enumerate(selected):
            idx = p["parent_index"]
            views = p["views"]
            spec = [(0, 0, "keep", None)] + [(n, r, "split", split_seed(idx, it, n, r))
                                             for n in NS for r in range(REPEATS)]
            for n, r, act, sd in spec:
                k = key(it, idx, act, n, r)
                if k not in done:
                    done[k] = run_branch(snapshot, radii_snap, opt, train_cams, cam_seq, it,
                                         args.diag_steps, args.train_seed, idx, act, n, sd,
                                         views, global_probe, n_before)
                    json.dump(done, open(cache_ckpt, "w"))
                rows.append({"iteration": it, "parent_index": idx, "branch": act,
                             "N": n, "repeat": r, "seed": sd,
                             **{kk: vv for kk, vv in done[k].items()}})
            # Δ#GS verification
            dns = [done[key(it, idx, "split", n, 0)]["d_n"] for n in NS]
            ok = all(d == n - 1 for d, n in zip(dns, NS))
            print(f"[b8a] it={it} parent {pi + 1}/{len(selected)} idx={idx}: "
                  f"dN={dns} (expect {[n - 1 for n in NS]}) {'OK' if ok else 'MISMATCH!'}")
        del snap
        torch.cuda.empty_cache()

    # save raw rows
    os.makedirs(f"{OUT}/data", exist_ok=True)
    with open(f"{OUT}/data/b8a_repeat_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("NA" if v is None else (f"{v:.8g}" if isinstance(v, float) else v))
                        for k, v in r.items()})
    json.dump(rows, open(f"{OUT}/data/b8a_repeat_results.json", "w"))
    print(f"[b8a] saved {len(rows)} branch rows")


if __name__ == "__main__":
    main()
