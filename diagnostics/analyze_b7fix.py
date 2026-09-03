#
# Paper B - B7-Fix analysis (host): LOSO comparison of
#   Static / Old-Dynamic / Fixed-Dynamic / Static+Fixed-Dynamic (Ridge, q_best)
# with per-stage metrics and the two key increment questions.
#   python3 diagnostics/analyze_b7fix.py
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

BASE = "paper_b/b7_fix_dynamic_signal"

STATIC = ["importance_score", "grad", "grad_abs", "opacity", "scale_max", "scale_min",
          "scale_anisotropy", "projected_radius_mean", "footprint_mean",
          "footprint_area_mean", "visibility_count", "residual_energy_mean",
          "residual_energy_std", "residual_anisotropy_mean", "residual_anisotropy_std",
          "residual_extent_mean", "residual_centroid_offset_mean",
          "demand_pixel_ratio_mean", "residual_parent_alignment_mean",
          "residual_extent_ratio_mean", "proj_anisotropy_mean"]
OLD_DYN = ["dyn_gradn_mean", "dyn_gradn_std", "dyn_gradn_slope", "dyn_gradn_cv",
           "dyn_grada_mean", "dyn_grada_slope", "dyn_xyz_path", "dyn_xyz_net",
           "dyn_xyz_net_over_path", "dyn_scale_path", "dyn_scale_slope",
           "dyn_opacity_net", "dyn_opacity_path", "dyn_vis_persistence",
           "dyn_radius_mean", "dyn_radius_std", "DemandPersistence",
           "DemandPersistence_cv", "DemandPersistence_slope", "OptimizationExposure",
           "OptimizationInefficiency", "dyn_grad_dir_consistency"]
FX_DYN = ["fx_gradn_mean", "fx_gradn_std", "fx_gradn_slope", "fx_gradn_cv",
          "fx_grad_dir_consistency", "fx_xyz_path", "fx_xyz_net",
          "fx_xyz_net_over_path", "fx_scale_path", "fx_scale_slope",
          "fx_opacity_path", "fx_opacity_net", "fx_demand_mean", "fx_demand_std",
          "fx_demand_slope", "fx_demand_last_first", "fx_demand_cv",
          "fx_demand_high_fraction", "fx_OptimizationExposure", "fx_Ineff_xyz",
          "fx_Ineff_scale", "fx_Ineff_opacity", "fx_fixedview_radius_mean",
          "fx_fixedview_radius_slope", "fx_fixedview_visibility_fraction"]
SETS = {"static": STATIC, "old_dynamic": OLD_DYN,
        "fixed_dynamic": FX_DYN, "static+fixed_dynamic": STATIC + FX_DYN}


def main():
    rows = json.load(open(f"{BASE}/data/b7fix_dynamic_features.json"))
    iters = sorted({r["iteration"] for r in rows})
    print(f"[b7fixa] rows={len(rows)} iters={iters}")

    def feat(r, keys):
        return np.array([np.nan if r.get(k) is None else float(r[k]) for k in keys])

    results = []
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

    import csv as _csv
    with open(f"{BASE}/data/b7fix_loso_results.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()})

    def get(it, s, k):
        return next(r[k] for r in results if r["held_out"] == it and r["feature_set"] == s)

    out = ["## B7-Fix LOSO（Ridge 固定，target=q_best）\n```text"]
    out.append(f"{'held-out':>9s} {'set':>20s} {'spearman':>9s} {'top25(rand)':>12s} "
               f"{'top50(rand)':>12s} {'VC@.5(rand)':>12s} {'VC@.75(rand)':>12s}")
    for r in results:
        out.append(f"{r['held_out']:>9d} {r['feature_set']:>20s} {r['spearman']:>9.3f} "
                   f"{r['top25_overlap']:>6.3f}({r['top25_overlap_random']:.3f}) "
                   f"{r['top50_overlap']:>6.3f}({r['top50_overlap_random']:.3f}) "
                   f"{r['vc@0.5']:>6.3f}({r['vc_random@0.5']:.3f}) "
                   f"{r['vc@0.75']:>6.3f}({r['vc_random@0.75']:.3f})")
    out.append("```\n")

    out.append("### 汇总\n```text")
    for s in SETS:
        sp = [get(it, s, "spearman") for it in iters]
        v5 = [get(it, s, "vc@0.5") - get(it, s, "vc_random@0.5") for it in iters]
        out.append(f"{s:>20s}: spearman mean {np.mean(sp):+.3f} | VC@.5−rand mean {np.mean(v5):+.3f} "
                   f"(positive {sum(v > 0 for v in v5)}/{len(v5)})")
    out.append("```\n")

    out.append("### 关键增量（per stage）\n```text")
    out.append(f"{'held-out':>9s} {'FX−OLD spear':>13s} {'S+FX−S spear':>13s} "
               f"{'FX−OLD VC.5':>12s} {'S+FX−S VC.5':>12s}")
    inc_sp, inc_vc = [], []
    for it in iters:
        d1 = get(it, "fixed_dynamic", "spearman") - get(it, "old_dynamic", "spearman")
        d2 = get(it, "static+fixed_dynamic", "spearman") - get(it, "static", "spearman")
        c1 = (get(it, "fixed_dynamic", "vc@0.5") - get(it, "fixed_dynamic", "vc_random@0.5")) - \
             (get(it, "old_dynamic", "vc@0.5") - get(it, "old_dynamic", "vc_random@0.5"))
        c2 = (get(it, "static+fixed_dynamic", "vc@0.5") - get(it, "static+fixed_dynamic", "vc_random@0.5")) - \
             (get(it, "static", "vc@0.5") - get(it, "static", "vc_random@0.5"))
        inc_sp.append((d1, d2)); inc_vc.append((c1, c2))
        out.append(f"{it:>9d} {d1:>+13.3f} {d2:>+13.3f} {c1:>+12.3f} {c2:>+12.3f}")
    out.append(f"mean     {np.mean([a for a,_ in inc_sp]):>+13.3f} "
               f"{np.mean([b for _,b in inc_sp]):>+13.3f} "
               f"{np.mean([a for a,_ in inc_vc]):>+12.3f} {np.mean([b for _,b in inc_vc]):>+12.3f}")
    out.append(f"positive {sum(a>0 for a,_ in inc_sp)}/5 {sum(b>0 for _,b in inc_sp)}/5 "
               f"{sum(a>0 for a,_ in inc_vc)}/5 {sum(b>0 for _,b in inc_vc)}/5")
    out.append("```\n")

    # plots
    os.makedirs(f"{BASE}/plots", exist_ok=True)
    x = np.arange(len(iters))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for s, col in (("static", "#cc8800"), ("old_dynamic", "#aa3377"),
                   ("fixed_dynamic", "#4477aa"), ("static+fixed_dynamic", "#228833")):
        ys = [get(it, s, "spearman") for it in iters]
        ax.plot(x, ys, marker="o", label=s, color=col)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(iters)
    ax.set_ylabel("Spearman(pred, q_best)")
    ax.set_title("B7-Fix: dynamic vs static (Ridge LOSO)")
    ax.legend(fontsize=8)
    fig.savefig(f"{BASE}/plots/b7fix_dynamic_vs_static.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for s, col in (("static", "#cc8800"), ("old_dynamic", "#aa3377"),
                   ("fixed_dynamic", "#4477aa"), ("static+fixed_dynamic", "#228833")):
        ys = [get(it, s, "vc@0.5") for it in iters]
        ax.plot(x, ys, marker="o", label=s, color=col)
    rc = [get(it, "static", "vc_random@0.5") for it in iters]
    ax.plot(x, rc, marker="x", color="gray", ls=":", label="random")
    ax.set_xticks(x); ax.set_xticklabels(iters)
    ax.set_ylabel("ValueCapture@0.5")
    ax.set_title("B7-Fix: value capture")
    ax.legend(fontsize=8)
    fig.savefig(f"{BASE}/plots/b7fix_value_capture.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(f"{BASE}/data/b7fix_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
