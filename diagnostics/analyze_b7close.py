#
# Paper B - B7-Closeout analysis (host): family ablation LOSO.
#   Static vs Static+{Demand, Gradient, Parameter, Response, Geometry,
#   AllFixedDynamic} (Ridge, q_best, leave-one-iteration-out), with family
#   deltas and the formal closeout verdict.
#   python3 diagnostics/analyze_b7close.py
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

BASE = "paper_b/b7_closeout_dynamic_ablation"

STATIC = ["importance_score", "grad", "grad_abs", "opacity", "scale_max", "scale_min",
          "scale_anisotropy", "projected_radius_mean", "footprint_mean",
          "footprint_area_mean", "visibility_count", "residual_energy_mean",
          "residual_energy_std", "residual_anisotropy_mean", "residual_anisotropy_std",
          "residual_extent_mean", "residual_centroid_offset_mean",
          "demand_pixel_ratio_mean", "residual_parent_alignment_mean",
          "residual_extent_ratio_mean", "proj_anisotropy_mean"]
DEMAND = ["cl_demand_mean", "cl_demand_std", "cl_demand_slope_trueidx",
          "cl_demand_last_first", "cl_demand_cv", "cl_demand_high_fraction"]
GRADIENT = ["cl_gradn_mean_valid", "cl_gradn_std_valid", "cl_gradn_slope_valid",
            "cl_gradn_cv_valid", "cl_grad_dir_consistency", "cl_Exposure_valid"]
PARAMETER = ["fx_xyz_path", "fx_xyz_net", "fx_xyz_net_over_path", "fx_scale_path",
             "fx_scale_slope", "fx_opacity_path", "fx_opacity_net"]
RESPONSE = ["fx_Ineff_xyz", "fx_Ineff_scale", "fx_Ineff_opacity", "cl_Exposure_valid"]
GEOMETRY = ["cl_fixedview_radius_mean", "cl_fixedview_radius_slope_trueidx",
            "cl_fixedview_visibility_fraction"]
FAMILIES = {"Demand": DEMAND, "Gradient": GRADIENT, "Parameter": PARAMETER,
            "Response": RESPONSE, "Geometry": GEOMETRY}
ALLDYN = sum(FAMILIES.values(), [])
SETS = {"Static": STATIC}
for name, fam in FAMILIES.items():
    SETS[f"Static+{name}"] = STATIC + fam
SETS["Static+AllDynamic"] = STATIC + ALLDYN
for name, fam in FAMILIES.items():          # family-only (secondary)
    SETS[f"{name}-only"] = fam


def main():
    rows = json.load(open(f"{BASE}/data/b7close_features.json"))
    iters = sorted({r["iteration"] for r in rows})
    print(f"[b7closea] rows={len(rows)} iters={iters}")

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
                rec[f"vc{rho}_minus_random"] = rec[f"vc@{rho}"] - rec[f"vc_random@{rho}"]
            results.append(rec)

    import csv as _csv
    with open(f"{BASE}/data/b7close_loso_results.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in r.items()})

    def get(it, s, k):
        return next(r[k] for r in results if r["held_out"] == it and r["feature_set"] == s)

    # ---------------- family summary ----------------
    fam_rows = []
    for fam in list(FAMILIES) + ["AllDynamic"]:
        sname = f"Static+{fam}"
        d_sp = [get(it, sname, "spearman") - get(it, "Static", "spearman") for it in iters]
        d_v5 = [get(it, sname, "vc0.5_minus_random") - get(it, "Static", "vc0.5_minus_random")
                for it in iters]
        d_v75 = [get(it, sname, "vc0.75_minus_random") - get(it, "Static", "vc0.75_minus_random")
                 for it in iters]
        fam_rows.append({"family": fam, "mean_dSpearman": float(np.mean(d_sp)),
                         "positive_stages_spearman": int(sum(x > 0 for x in d_sp)),
                         "mean_dVC05": float(np.mean(d_v5)),
                         "positive_stages_vc05": int(sum(x > 0 for x in d_v5)),
                         "mean_dVC075": float(np.mean(d_v75)),
                         "positive_stages_vc075": int(sum(x > 0 for x in d_v75))})
    with open(f"{BASE}/data/b7close_family_summary.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(fam_rows[0].keys()))
        w.writeheader()
        w.writerows(fam_rows)

    out = ["## B7-Closeout 家族消融（Static+Family − Static）\n```text"]
    out.append(f"{'family':>11s} {'mean ΔSpear':>12s} {'pos stages':>11s} {'mean ΔVC@.5':>12s} "
               f"{'pos stages':>11s} {'mean ΔVC@.75':>12s} {'pos stages':>11s}")
    for fr in fam_rows:
        out.append(f"{fr['family']:>11s} {fr['mean_dSpearman']:>+12.3f} "
                   f"{fr['positive_stages_spearman']:>8d}/5     "
                   f"{fr['mean_dVC05']:>+12.3f} {fr['positive_stages_vc05']:>8d}/5     "
                   f"{fr['mean_dVC075']:>+12.3f} {fr['positive_stages_vc075']:>8d}/5")
    out.append("```\n")
    out.append("### Static / Static+AllDynamic 汇总\n```text")
    for s in ("Static", "Static+AllDynamic"):
        sp = [get(it, s, "spearman") for it in iters]
        v5 = [get(it, s, "vc0.5_minus_random") for it in iters]
        out.append(f"{s:>20s}: spearman mean {np.mean(sp):+.3f} | VC@.5−rand mean {np.mean(v5):+.3f}")
    out.append("```\n")

    verdicts = []
    for fr in fam_rows:
        ok = (fr["positive_stages_spearman"] >= 4 and fr["positive_stages_vc05"] >= 4
              and abs(fr["mean_dSpearman"]) > 0.02 and abs(fr["mean_dVC05"]) > 0.02)
        verdicts.append((fr["family"], ok, fr))
    any_pass = any(v for _, v, _ in verdicts)
    out.append("### 判定（ΔSpearman>0 且 ΔVC@.5>0 于 ≥4/5 阶段，且均值非零波动）\n```text")
    for fam, ok, fr in verdicts:
        out.append(f"{fam:>11s}: {'PASS' if ok else 'FAIL'} "
                   f"(ΔSpear {fr['positive_stages_spearman']}/5, ΔVC.5 {fr['positive_stages_vc05']}/5, "
                   f"mean {fr['mean_dSpearman']:+.3f}/{fr['mean_dVC05']:+.3f})")
    out.append("```\n")

    # plots
    os.makedirs(f"{BASE}/plots", exist_ok=True)
    x = np.arange(len(iters))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for fam, col in (("Demand", "#cc3311"), ("Gradient", "#4477aa"), ("Parameter", "#228833"),
                     ("Response", "#cc8800"), ("Geometry", "#aa3377"), ("AllDynamic", "#000000")):
        ys = [get(it, f"Static+{fam}", "spearman") - get(it, "Static", "spearman") for it in iters]
        ax.plot(x, ys, marker="o", label=fam, color=col)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(iters)
    ax.set_ylabel("ΔSpearman vs Static")
    ax.set_title("family delta spearman (LOSO)")
    ax.legend(fontsize=8)
    fig.savefig(f"{BASE}/plots/family_delta_spearman.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for fam, col in (("Demand", "#cc3311"), ("Gradient", "#4477aa"), ("Parameter", "#228833"),
                     ("Response", "#cc8800"), ("Geometry", "#aa3377"), ("AllDynamic", "#000000")):
        ys = [get(it, f"Static+{fam}", "vc0.5_minus_random") - get(it, "Static", "vc0.5_minus_random")
              for it in iters]
        ax.plot(x, ys, marker="o", label=fam, color=col)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(iters)
    ax.set_ylabel("ΔVC@0.5 vs Static")
    ax.set_title("family delta value capture (LOSO)")
    ax.legend(fontsize=8)
    fig.savefig(f"{BASE}/plots/family_delta_value_capture.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(f"{BASE}/data/b7close_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nVERDICT: {'SOME FAMILY PASS' if any_pass else 'ALL FAMILIES FAIL'}")


if __name__ == "__main__":
    main()
