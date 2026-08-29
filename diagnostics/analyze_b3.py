#
# Paper B - B3 analysis: pre-action predictability of the B2-C Clone/Split
# oracle. Host python3 (numpy/matplotlib/scikit-learn).
#
#   python3 diagnostics/analyze_b3.py
#
# Leakage policy: oracle_gap / split_sem / split_std / oracle_action are used
# ONLY as labels, regret weights and eval-subset definitions — never inputs.
# Features are strictly pre-action (see b3_candidate_dataset columns).
#

import json
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix)

BASE = "paper_b/b3_action_predictability"

GROUPS = {
    "scale": ["scale_max", "scale_anisotropy"],
    "state": ["importance_score", "grad", "grad_abs", "opacity", "scale_min",
              "projected_radius_mean", "footprint_area_mean", "visibility_count"],
    "residual": ["residual_energy_mean", "residual_energy_std", "residual_anisotropy_mean",
                 "residual_anisotropy_std", "residual_extent_mean", "residual_extent_std",
                 "residual_centroid_offset_mean", "residual_centroid_offset_std",
                 "demand_pixel_ratio_mean"],
    "geometry": ["residual_parent_alignment_mean", "residual_extent_ratio_mean",
                 "proj_anisotropy_mean"],
    "consistency": ["residual_energy_cv", "residual_anisotropy_cv", "residual_extent_cv",
                    "residual_direction_consistency"],
}
SETS = {"A_scale": GROUPS["scale"],
        "B_state": GROUPS["scale"] + GROUPS["state"],
        "C_residual": GROUPS["scale"] + GROUPS["state"] + GROUPS["residual"],
        "D_geometry": GROUPS["scale"] + GROUPS["state"] + GROUPS["residual"] + GROUPS["geometry"],
        "E_consistency": (GROUPS["scale"] + GROUPS["state"] + GROUPS["residual"]
                          + GROUPS["geometry"] + GROUPS["consistency"])}


def metrics(y, pred, score=None):
    out = {"accuracy": accuracy_score(y, pred),
           "balanced_accuracy": balanced_accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0)}
    out["roc_auc"] = roc_auc_score(y, score) if score is not None and len(set(y)) == 2 else None
    return out


def regret_stats(y, pred, gap):
    r = np.where(pred != y, np.abs(gap), 0.0)
    return {"mean_regret": float(np.mean(r)), "median_regret": float(np.median(r)),
            "p95_regret": float(np.percentile(r, 95))}


def main():
    with open(f"{BASE}/data/b3_candidate_dataset.json") as f:
        data = json.load(f)
    rows = [r for r in data["rows"] if r.get("oracle_action") in ("clone", "split")]
    n_all = len(rows)
    hc = [r for r in rows if r.get("oracle_high_confident")]
    print(f"[b3a] candidates: {n_all} valid oracle ({data['meta']['n_candidates']} total), "
          f"{len(hc)} high-confidence")
    print(f"[b3a] clone% = {np.mean([r['oracle_action']=='clone' for r in rows])*100:.1f} "
          f"(high-conf {np.mean([r['oracle_action']=='clone' for r in hc])*100:.1f})")

    def pack(subset):
        y = np.array([1 if r["oracle_action"] == "clone" else 0 for r in subset])
        gap = np.array([r["oracle_gap"] for r in subset], float)
        nat = np.array([1 if r["native_action"] == "clone" else 0 for r in subset])
        xyz = np.array([[r["x"], r["y"], r["z"]] for r in subset], float)
        return y, gap, nat, xyz

    subsets = {"all_valid": pack(rows), "high_conf": pack(hc)}

    def make_model(kind):
        if kind == "lr":
            return Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(max_iter=3000, C=1.0))])
        return DecisionTreeClassifier(max_depth=3, random_state=0)

    cv_results = []
    oof_store = {}

    for eval_name, (y, gap, nat, xyz) in subsets.items():
        kmeans = KMeans(n_clusters=10, n_init=10, random_state=0).fit(xyz)
        groups = kmeans.labels_
        cv_defs = {"spatial_group": GroupKFold(n_splits=5),
                   "stratified": StratifiedKFold(n_splits=5, shuffle=True, random_state=0)}
        for cv_name, cv in cv_defs.items():
            sub = rows if eval_name == "all_valid" else hc
            y, gap, nat, xyz = subsets[eval_name]
            # native baseline (heuristic; hard labels, no CV needed)
            oof_store[(eval_name, cv_name, "native", "A_scale")] = nat
            m = metrics(y, nat)
            rg = regret_stats(y, nat, gap)
            cv_results.append({"eval_set": eval_name, "cv": cv_name, "model": "native",
                               "feature_set": "A_scale", "n": len(y), **m, **rg})
            for set_name, feats in SETS.items():
                # rows with any NA in this feature set are dropped (n reported)
                ok = [i for i, r in enumerate(sub)
                      if all(r.get(f) is not None and not np.isnan(float(r[f])) for f in feats)]
                X = np.array([[float(r[f]) for f in feats]
                              for i, r in enumerate(sub) if i in set(ok)])
                y_s, gap_s = y[ok], gap[ok]
                g_s = groups[ok] if cv_name == "spatial_group" else None
                for kind in ("lr", "tree"):
                    model = make_model(kind)
                    try:
                        score = cross_val_predict(model, X, y_s, groups=g_s, cv=cv,
                                                  method="predict_proba")[:, 1]
                    except Exception:
                        score = None
                    pred = cross_val_predict(model, X, y_s, groups=g_s, cv=cv)
                    m = metrics(y_s, pred, score)
                    rg = regret_stats(y_s, pred, gap_s)
                    cv_results.append({"eval_set": eval_name, "cv": cv_name,
                                       "model": kind, "feature_set": set_name,
                                       "n": len(ok), **m, **rg})
                    oof_store[(eval_name, cv_name, kind, set_name)] = (ok, pred)

    import csv as _csv
    with open(f"{BASE}/data/b3_cv_results.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(cv_results[0].keys()))
        w.writeheader()
        for r in cv_results:
            w.writerow({k: ("NA" if v is None else (f"{v:.6g}" if isinstance(v, float) else v))
                        for k, v in r.items()})
    with open(f"{BASE}/data/b3_cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=1)

    # ---------------- main comparison table (spatial_group CV) -------------
    def get(eval_set, model, fset, key):
        for r in cv_results:
            if r["eval_set"] == eval_set and r["cv"] == "spatial_group" \
                    and r["model"] == model and r["feature_set"] == fset:
                return r[key]
        return None

    out = ["## B3 统计输出（自动生成；主 CV = spatial group 5-fold）\n"]
    for eval_name in ("all_valid", "high_conf"):
        out.append(f"### {eval_name}\n")
        out.append("| Model | Features | Balanced Acc | F1 | ROC-AUC | Mean Regret | Median Regret | P95 Regret |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|")
        table_rows = [("FastGS Native", "native", "A_scale")]
        for disp, fset in (("Scale-only", "A_scale"), ("FastGS-state", "B_state"),
                           ("Residual", "C_residual"), ("Residual+Geometry", "D_geometry"),
                           ("Full(+consistency)", "E_consistency")):
            best = max(("lr", "tree"),
                       key=lambda k: get(eval_name, k, fset, "balanced_accuracy") or 0)
            table_rows.append((disp + f" [{best}]", best, fset))
        for disp, model, fset in table_rows:
            out.append(f"| {disp} | {fset} | "
                       f"{get(eval_name, model, fset, 'balanced_accuracy'):.4f} | "
                       f"{get(eval_name, model, fset, 'f1'):.4f} | "
                       f"{get(eval_name, model, fset, 'roc_auc') if get(eval_name, model, fset, 'roc_auc') is None else format(get(eval_name, model, fset, 'roc_auc'), '.4f')} | "
                       f"{get(eval_name, model, fset, 'mean_regret'):.6f} | "
                       f"{get(eval_name, model, fset, 'median_regret'):.6f} | "
                       f"{get(eval_name, model, fset, 'p95_regret'):.6f} |")
        out.append("")

    # ablation deltas
    out.append("### Feature ablation（Balanced Accuracy，spatial CV，每个 set 取 lr/tree 较优）\n")
    ab_rows = []
    for eval_name in ("all_valid", "high_conf"):
        vals = {}
        for fset in SETS:
            vals[fset] = max(get(eval_name, k, fset, "balanced_accuracy") or 0 for k in ("lr", "tree"))
            ab_rows.append({"eval_set": eval_name, "feature_set": fset,
                            "balanced_accuracy": vals[fset]})
        out.append(f"- {eval_name}: " + " → ".join(
            f"{k}={v:.4f}" for k, v in vals.items()))
        out.append(f"  - +residual: {vals['C_residual']-vals['B_state']:+.4f} | "
                   f"+geometry: {vals['D_geometry']-vals['C_residual']:+.4f} | "
                   f"+consistency: {vals['E_consistency']-vals['D_geometry']:+.4f} | "
                   f"full vs scale: {vals['E_consistency']-vals['A_scale']:+.4f}")
    with open(f"{BASE}/data/b3_feature_ablation.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["eval_set", "feature_set", "balanced_accuracy"])
        w.writeheader()
        for r in ab_rows:
            w.writerow(r)

    # ---------------- confidence buckets -----------------------------------
    y, gap, nat, xyz = subsets["all_valid"]
    qs = np.quantile(np.abs(gap), [1 / 3, 2 / 3])
    buckets = np.digitize(np.abs(gap), qs)  # 0 low, 1 med, 2 high
    out.append("### Confidence buckets（|oracle_gap| 三分位, all-valid, spatial CV OOF）\n")
    out.append("| bucket | n | Native BAcc | Scale BAcc | Full BAcc | Native regret | Full regret |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")

    def oof_pred(eval_name, model, fset, ylen):
        v = oof_store.get((eval_name, "spatial_group", model, fset))
        if v is None:
            return np.zeros(ylen, dtype=int)
        if isinstance(v, tuple):
            ok, pred = v
            full = np.zeros(ylen, dtype=int)
            full[np.array(ok)] = pred
            return full
        return np.asarray(v)

    pred_scale = oof_pred("all_valid", "tree", "A_scale", len(y))
    # pick best model kind per set for bucket table
    kind_scale = max(("lr", "tree"), key=lambda k: get("all_valid", k, "A_scale", "balanced_accuracy") or 0)
    kind_full = max(("lr", "tree"), key=lambda k: get("all_valid", k, "E_consistency", "balanced_accuracy") or 0)
    pred_scale = oof_pred("all_valid", kind_scale, "A_scale", len(y))
    pred_full = oof_pred("all_valid", kind_full, "E_consistency", len(y))
    for b, name in ((0, "low"), (1, "medium"), (2, "high")):
        m = buckets == b
        rg_n = float(np.mean(np.where(nat[m] != y[m], np.abs(gap[m]), 0)))
        rg_f = float(np.mean(np.where(pred_full[m] != y[m], np.abs(gap[m]), 0)))
        out.append(f"| {name} | {int(m.sum())} | "
                   f"{balanced_accuracy_score(y[m], nat[m]):.4f} | "
                   f"{balanced_accuracy_score(y[m], pred_scale[m]):.4f} | "
                   f"{balanced_accuracy_score(y[m], pred_full[m]):.4f} | "
                   f"{rg_n:.6f} | {rg_f:.6f} |")
    out.append("")

    # ---------------- predictions file --------------------------------------
    pred_rows = []
    for i, r in enumerate(rows):
        pred_rows.append({"parent_index": r["parent_index"],
                          "oracle_action": r["oracle_action"], "oracle_gap": r["oracle_gap"],
                          "high_conf": r.get("oracle_high_confident"),
                          "native_pred": "clone" if nat[i] == 1 else "split",
                          "scale_pred": "clone" if pred_scale[i] == 1 else "split",
                          "full_pred": "clone" if pred_full[i] == 1 else "split"})
    with open(f"{BASE}/data/b3_predictions.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        w.writeheader()
        w.writerows(pred_rows)

    # ---------------- models + importance -----------------------------------
    feats_full = SETS["E_consistency"]
    okf = [i for i, r in enumerate(rows)
           if all(r.get(f) is not None and not np.isnan(float(r[f])) for f in feats_full)]
    Xf = np.array([[float(r[f]) for f in feats_full] for i, r in enumerate(rows) if i in set(okf)])
    yf = y[okf]
    lr = make_model("lr").fit(Xf, yf)
    tree = make_model("tree").fit(Xf, yf)
    os.makedirs(f"{BASE}/models", exist_ok=True)
    pickle.dump(lr, open(f"{BASE}/models/logistic_full.pkl", "wb"))
    pickle.dump(tree, open(f"{BASE}/models/tree_full.pkl", "wb"))
    coefs = lr.named_steps["lr"].coef_[0]
    order = np.argsort(-np.abs(coefs))
    out.append("### Feature importance（LR |standardized coef|，前 12）\n```text")
    for i in order[:12]:
        out.append(f"{feats_full[i]:36s} {coefs[i]:+.4f}")
    out.append("```\n")
    timp = tree.feature_importances_
    torder = np.argsort(-timp)
    out.append("### Tree(depth=3) importance（前 8）\n```text")
    for i in torder[:8]:
        if timp[i] > 0:
            out.append(f"{feats_full[i]:36s} {timp[i]:.4f}")
    out.append("```\n")

    # ---------------- plots --------------------------------------------------
    def save(fig, name):
        fig.savefig(f"{BASE}/plots/{name}", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 1 balanced accuracy by model/set (spatial CV)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    labels = list(SETS.keys())
    x = np.arange(len(labels))
    for j, (eval_name, col) in enumerate((("all_valid", "#4477aa"), ("high_conf", "#228833"))):
        vals = [max(get(eval_name, k, f, "balanced_accuracy") or 0 for k in ("lr", "tree"))
                for f in labels]
        ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, label=eval_name, color=col)
    nat_all = balanced_accuracy_score(y, nat)
    nat_hc = balanced_accuracy_score(subsets["high_conf"][0], subsets["high_conf"][2])
    for j, v in enumerate((nat_all, nat_hc)):
        ax.axhline(v, color=["#4477aa", "#228833"][j], ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["A scale", "B +state", "C +residual", "D +geometry", "E +consistency"], fontsize=9)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("predictability ablation (spatial group CV; dashed = FastGS native)")
    ax.legend()
    save(fig, "model_balanced_accuracy.png")

    # 2 feature ablation deltas
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for eval_name, col, mk in (("all_valid", "#4477aa", "o"), ("high_conf", "#228833", "s")):
        vals = [max(get(eval_name, k, f, "balanced_accuracy") or 0 for k in ("lr", "tree"))
                for f in labels]
        ax.plot(range(5), vals, marker=mk, color=col, label=eval_name)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["A", "B", "C(+res)", "D(+geo)", "E(+cons)"])
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("feature ablation (cumulative)")
    ax.legend()
    save(fig, "feature_ablation.png")

    # 3 regret by model
    fig, ax = plt.subplots(figsize=(7, 4.2))
    models = [("native", "A_scale", "FastGS Native"), ("lr", "A_scale", "Scale LR"),
              ("tree", "A_scale", "Scale Tree"), ("lr", "E_consistency", "Full LR"),
              ("tree", "E_consistency", "Full Tree")]
    names, means, meds = [], [], []
    for mk, fs, disp in models:
        r = next(r for r in cv_results if r["eval_set"] == "all_valid"
                 and r["cv"] == "spatial_group" and r["model"] == mk and r["feature_set"] == fs)
        names.append(disp); means.append(r["mean_regret"]); meds.append(r["median_regret"])
    xx = np.arange(len(names))
    ax.bar(xx - 0.2, means, 0.4, label="mean regret", color="#cc3311")
    ax.bar(xx + 0.2, meds, 0.4, label="median regret", color="#ee7766")
    ax.set_xticks(xx)
    ax.set_xticklabels(names, fontsize=8, rotation=15)
    ax.set_ylabel("oracle regret (|gap| of wrong picks)")
    ax.legend()
    save(fig, "oracle_regret_by_model.png")

    # 4 accuracy vs oracle gap buckets
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, pred, col in (("Native", nat, "#cc8800"), ("Scale", pred_scale, "#4477aa"),
                            ("Full", pred_full, "#228833")):
        bas = [balanced_accuracy_score(y[buckets == b], pred[buckets == b]) for b in (0, 1, 2)]
        ax.plot([0, 1, 2], bas, marker="o", color=col, label=name)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["low |gap|", "medium", "high |gap|"])
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("accuracy vs oracle-gap bucket (all-valid)")
    ax.legend()
    save(fig, "accuracy_vs_oracle_gap.png")

    # 5 coefficients
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sel = order[:14][::-1]
    ax.barh([feats_full[i] for i in sel], [coefs[i] for i in sel], color="#4477aa")
    ax.set_xlabel("standardized LR coefficient (full feature set)")
    ax.set_title("feature coefficients (clone = positive class)")
    save(fig, "feature_coefficients.png")

    # 6 confusion scale vs full
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax_, (pred, ttl) in zip(axes, ((pred_scale, "Scale-only"), (pred_full, "Full (E)"))):
        cm = confusion_matrix(y, pred)
        im = ax_.imshow(cm, cmap="Blues")
        ax_.set_xticks([0, 1]); ax_.set_xticklabels(["pred split", "pred clone"])
        ax_.set_yticks([0, 1]); ax_.set_yticklabels(["oracle split", "oracle clone"])
        for i in range(2):
            for j in range(2):
                ax_.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax_.set_title(ttl)
    save(fig, "confusion_scale_vs_full.png")

    with open(f"{BASE}/data/b3_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[b3a] outputs -> {BASE}/data , {BASE}/plots , {BASE}/models")


if __name__ == "__main__":
    main()
