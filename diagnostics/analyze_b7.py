#
# Paper B - B7 analysis (host, sklearn): Static vs Dynamic vs Static+Dynamic
# Ridge LOSO on q_best.  python3 diagnostics/analyze_b7.py
#

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

BASE = "paper_b/b7_dynamic_capacity_signal"

STATIC = ["importance_score", "grad", "grad_abs", "opacity", "scale_max", "scale_min",
          "scale_anisotropy", "projected_radius_mean", "footprint_mean",
          "footprint_area_mean", "visibility_count", "residual_energy_mean",
          "residual_energy_std", "residual_anisotropy_mean", "residual_anisotropy_std",
          "residual_extent_mean", "residual_centroid_offset_mean",
          "demand_pixel_ratio_mean", "residual_parent_alignment_mean",
          "residual_extent_ratio_mean", "proj_anisotropy_mean"]
DYNAMIC = ["dyn_gradn_mean", "dyn_gradn_std", "dyn_gradn_slope", "dyn_gradn_cv",
           "dyn_grada_mean", "dyn_grada_slope", "dyn_xyz_path", "dyn_xyz_net",
           "dyn_xyz_net_over_path", "dyn_scale_path", "dyn_scale_slope",
           "dyn_opacity_net", "dyn_opacity_path", "dyn_vis_persistence",
           "dyn_radius_mean", "dyn_radius_std", "DemandPersistence",
           "DemandPersistence_cv", "DemandPersistence_slope", "OptimizationExposure",
           "OptimizationInefficiency", "dyn_grad_dir_consistency"]
SETS = {"static": STATIC, "dynamic": DYNAMIC, "static+dynamic": STATIC + DYNAMIC}


def main():
    rows = json.load(open(f"{BASE}/data/b7_dynamic_features.json"))
    iters = sorted({r["iteration"] for r in rows})
    print(f"[b7a] rows={len(rows)} iters={iters}")

    def feat(r, keys):
        return np.array([np.nan if r.get(k) is None else float(r[k]) for k in keys])

    results = []
    preds_store = {str(it): {} for it in iters}
    for held in iters:
        train = [r for r in rows if r["iteration"] != held]
        test = [r for r in rows if r["iteration"] == held]
        for setname, keys in SETS.items():
            tr_med = np.nan_to_num(np.nanmedian(np.array([feat(r, keys) for r in train]), axis=0), nan=0.0)
            Xtr = np.nan_to_num(np.array([feat(r, keys) for r in train]), nan=tr_med)
            Xte = np.nan_to_num(np.array([feat(r, keys) for r in test]), nan=tr_med)
            ytr = np.array([float(r["q_best"]) for r in train])
            yte = np.array([float(r["q_best"]) for r in test])
            m = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))]).fit(Xtr, ytr)
            p = m.predict(Xte)
            n = len(yte)
            o_order, p_order = np.argsort(-yte), np.argsort(-p)
            rec = {"held_out": held, "feature_set": setname, "n": n,
                   "spearman": float(spearmanr(p, yte)[0])}
            for frac, tag in ((0.25, "top25"), (0.5, "top50")):
                mt = int(round(frac * n))
                rec[f"{tag}_overlap"] = len(set(o_order[:mt]) & set(p_order[:mt])) / mt
                rec[f"{tag}_overlap_random"] = mt / n
            for rho in (0.5, 0.75):
                M = int(round(rho * n))
                denom = max(yte[o_order[:M]].sum(), 1e-9)
                rng = np.random.RandomState(0)
                rec[f"vc@{rho}"] = float(yte[p_order[:M]].sum() / denom)
                rec[f"vc_random@{rho}"] = float(np.mean(
                    [yte[rng.permutation(n)[:M]].sum() / denom for _ in range(200)]))
            results.append(rec)
            for r, v in zip(test, p):
                preds_store[str(r["iteration"])].setdefault(setname, {})[str(r["parent_index"])] = float(v)

    import csv as _csv
    with open(f"{BASE}/data/b7_loso_results.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()})

    out = ["## B7 LOSO（Ridge 固定，target=q_best）\n```text"]
    out.append(f"{'held-out':>9s} {'set':>16s} {'spearman':>9s} {'top25(rand)':>12s} "
               f"{'top50(rand)':>12s} {'VC@.5(rand)':>12s} {'VC@.75(rand)':>12s}")
    for r in results:
        out.append(f"{r['held_out']:>9d} {r['feature_set']:>16s} {r['spearman']:>9.3f} "
                   f"{r['top25_overlap']:>6.3f}({r['top25_overlap_random']:.3f}) "
                   f"{r['top50_overlap']:>6.3f}({r['top50_overlap_random']:.3f}) "
                   f"{r['vc@0.5']:>6.3f}({r['vc_random@0.5']:.3f}) "
                   f"{r['vc@0.75']:>6.3f}({r['vc_random@0.75']:.3f})")
    out.append("```\n")
    out.append("```text")
    for s in SETS:
        sp = [r["spearman"] for r in results if r["feature_set"] == s]
        vc5 = [r["vc@0.5"] - r["vc_random@0.5"] for r in results if r["feature_set"] == s]
        out.append(f"{s:>16s}: spearman mean {np.mean(sp):+.3f} | VC@.5−rand mean {np.mean(vc5):+.3f} "
                   f"(positive stages {sum(v > 0 for v in vc5)}/{len(vc5)})")
    out.append("```\n")

    os.makedirs(f"{BASE}/plots", exist_ok=True)
    x = np.arange(len(iters))
    # 1 spearman comparison
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for s, col in (("static", "#cc8800"), ("dynamic", "#4477aa"), ("static+dynamic", "#228833")):
        ys = [next(r["spearman"] for r in results if r["held_out"] == it and r["feature_set"] == s)
              for it in iters]
        ax.plot(x, ys, marker="o", label=s, color=col)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(iters)
    ax.set_ylabel("Spearman(pred, q_best)")
    ax.set_title("dynamic vs static (Ridge LOSO)")
    ax.legend()
    fig.savefig(f"{BASE}/plots/dynamic_vs_static_spearman.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # 2 value capture
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for s, col in (("static", "#cc8800"), ("dynamic", "#4477aa"), ("static+dynamic", "#228833")):
        ys = [next(r["vc@0.5"] for r in results if r["held_out"] == it and r["feature_set"] == s)
              for it in iters]
        rc = [next(r["vc_random@0.5"] for r in results if r["held_out"] == it and r["feature_set"] == s)
              for it in iters]
        ax.plot(x, ys, marker="o", color=col, label=s)
    ax.plot(x, rc, marker="x", color="gray", ls=":", label="random")
    ax.set_xticks(x); ax.set_xticklabels(iters)
    ax.set_ylabel("ValueCapture@0.5")
    ax.set_title("value capture (Ridge LOSO)")
    ax.legend(fontsize=8)
    fig.savefig(f"{BASE}/plots/dynamic_value_capture.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    json.dump(preds_store, open(f"{BASE}/cache/b7_predictions.json", "w"))
    with open(f"{BASE}/data/b7_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
