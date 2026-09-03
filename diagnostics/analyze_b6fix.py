#
# Paper B - B6-Fix analysis (host, sklearn):
#  1) FIXED-predictor LOSO (Who=Ridge, How=Tree) from existing b6_features
#     -> b6fix_predictions.json (used by the audit replay)
#  2) corrected How diagnostics: balanced accuracy vs majority-class
#     baseline, predicted/oracle/native Clone ratios (no high-|gap| bucket
#     as primary evidence)
#  3) final: action-prior comparison table + gains + plot + stats
#
#   python3 diagnostics/analyze_b6fix.py loso
#   python3 diagnostics/analyze_b6fix.py final
#

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import balanced_accuracy_score, accuracy_score

B6 = "paper_b/b6_practical_oracle_approximation"
OUT = "paper_b/b6_action_prior_audit"
FEATURES = ["importance_score", "grad", "grad_abs", "opacity",
            "scale_max", "scale_min", "scale_anisotropy",
            "projected_radius_mean", "footprint_mean", "footprint_area_mean",
            "visibility_count",
            "residual_energy_mean", "residual_energy_std",
            "residual_anisotropy_mean", "residual_anisotropy_std",
            "residual_extent_mean", "residual_centroid_offset_mean",
            "demand_pixel_ratio_mean",
            "residual_parent_alignment_mean", "residual_extent_ratio_mean",
            "proj_anisotropy_mean",
            "residual_energy_cv", "residual_anisotropy_cv", "residual_extent_cv",
            "residual_direction_consistency"]


def feat_vals(r):
    return np.array([np.nan if r.get(f) is None else float(r[f]) for f in FEATURES])


def impute(rows_X, medians):
    return np.nan_to_num(rows_X, nan=medians)


def stage_loso():
    rows = [r for r in json.load(open(f"{B6}/data/b6_features.json"))
            if (r.get("valid_views", 1) or 0) > 0 and r.get("q_best") is not None]
    iters = sorted({r["iteration"] for r in rows})
    print(f"[b6fix] rows={len(rows)} iters={iters}  FIXED predictors: Who=Ridge How=Tree")

    preds = {str(it): {"pred_q_best": {}, "pred_q_gap": {}} for it in iters}
    how_diag = []
    for held in iters:
        train = [r for r in rows if r["iteration"] != held]
        test = [r for r in rows if r["iteration"] == held]
        medians = np.nan_to_num(np.nanmedian(np.array([feat_vals(r) for r in train]), axis=0), nan=0.0)

        def pack(sub, target):
            ok = [r for r in sub if r.get(target) is not None]
            X = impute(np.array([feat_vals(r) for r in ok]), medians)
            y = np.array([float(r[target]) for r in ok])
            ids = [r["parent_index"] for r in ok]
            return X, y, ids, ok

        # ---- Who = Ridge (fixed) ----
        Xtr, ytr, _, _ = pack(train, "q_best")
        Xte, yte, ids, _ = pack(test, "q_best")
        ridge = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))]).fit(Xtr, ytr)
        p_best = ridge.predict(Xte)
        for i, v in zip(ids, p_best):
            preds[str(held)]["pred_q_best"][str(i)] = float(v)
        who_sp = float(spearmanr(p_best, yte)[0])

        # ---- How = Tree (fixed) ----
        Xtr, ytr, _, ok_tr = pack(train, "q_gap")
        Xte, yte, ids, ok_te = pack(test, "q_gap")
        tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(Xtr, ytr)
        p_gap = tree.predict(Xte)
        for i, v in zip(ids, p_gap):
            preds[str(held)]["pred_q_gap"][str(i)] = float(v)

        pred_pos = p_gap > 0
        true_pos = yte > 0
        train_pos = ytr > 0
        # majority-class baseline = majority sign of the TRAIN fold
        majority_clone = bool(np.mean(train_pos) > 0.5)
        maj_pred = np.full_like(pred_pos, majority_clone)
        native_pos = np.array([r["native_action"] == "clone" for r in ok_te])
        how_diag.append({
            "held_out": held, "n_test": len(yte),
            "how_spearman": float(spearmanr(p_gap, yte)[0]),
            "sign_accuracy": float(accuracy_score(true_pos, pred_pos)),
            "balanced_accuracy": float(balanced_accuracy_score(true_pos, pred_pos)),
            "majority_class_baseline": float(balanced_accuracy_score(true_pos, maj_pred)),
            "pred_clone_ratio": float(np.mean(pred_pos)),
            "oracle_clone_ratio": float(np.mean(true_pos)),
            "native_clone_ratio": float(np.mean(native_pos)),
            "who_ridge_spearman": who_sp,
        })

    os.makedirs(f"{OUT}/cache", exist_ok=True)
    json.dump({"preds": preds, "who": "ridge", "how": "tree"},
              open(f"{OUT}/cache/b6fix_predictions.json", "w"))
    import csv as _csv
    with open(f"{OUT}/data/b6fix_how_diagnostics.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(how_diag[0].keys()))
        w.writeheader()
        w.writerows(how_diag)

    out = ["## B6-Fix 固定预测器 LOSO 与 How 诊断（修正口径）\n"]
    out.append("### How = Tree(depth 3)，target = q_gap（clone 为正）\n```text")
    out.append(f"{'held-out':>9s} {'spearman':>9s} {'signAcc':>8s} {'balAcc':>7s} "
               f"{'majority balAcc':>15s} {'predClone%':>10s} {'oracleClone%':>12s} {'nativeClone%':>12s}")
    for d in how_diag:
        out.append(f"{d['held_out']:>9d} {d['how_spearman']:>9.3f} {d['sign_accuracy']:>8.3f} "
                   f"{d['balanced_accuracy']:>7.3f} {d['majority_class_baseline']:>15.3f} "
                   f"{d['pred_clone_ratio']*100:>9.1f}% {d['oracle_clone_ratio']*100:>11.1f}% "
                   f"{d['native_clone_ratio']*100:>11.1f}%")
    out.append("```\n")
    out.append("### Who = Ridge（固定），Spearman(pred_q_best, q_best): " +
               ", ".join(f"it{d['held_out']}={d['who_ridge_spearman']:+.3f}" for d in how_diag) + "\n")
    with open(f"{OUT}/data/b6fix_stats_loso.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


def stage_final():
    rows = json.load(open(f"{OUT}/data/b6fix_action_results.json"))
    iters = sorted({r["iteration"] for r in rows})

    def get(it, p, key="demand_l1_100"):
        return next(r for r in rows if r["iteration"] == it and r["policy"] == p)[key]

    out = ["## B6-Fix Action-Prior Audit（同一 RW repeat-0 subset，Δ#GS=50）\n"]
    out.append("```text")
    out.append(f"{'iter':>6s} {'NativeHow':>10s} {'AllClone':>10s} {'AllSplit':>10s} "
               f"{'PredHow':>10s} {'OracleHow':>10s} | {'P−Nat':>9s} {'P−AllC':>9s} "
               f"{'P−AllS':>9s} {'O−P':>9s} {'predClone%':>10s}")
    pn, pc, ps, op = [], [], [], []
    for it in iters:
        nat, ac, asp = get(it, "rw_nativehow"), get(it, "rw_allclone"), get(it, "rw_allsplit")
        p, o = get(it, "rw_predhow"), get(it, "rw_oraclehow")
        cr = next(r for r in rows if r["iteration"] == it and r["policy"] == "rw_predhow")["pred_clone_ratio"]
        d_nat, d_ac, d_asp, d_op = nat - p, ac - p, asp - p, p - o
        pn.append(d_nat); pc.append(d_ac); ps.append(d_asp); op.append(d_op)
        out.append(f"{it:>6d} {nat:>10.6f} {ac:>10.6f} {asp:>10.6f} {p:>10.6f} {o:>10.6f} | "
                   f"{d_nat:>+9.6f} {d_ac:>+9.6f} {d_asp:>+9.6f} {d_op:>+9.6f} {cr*100:>9.1f}%")
    out.append("```\n")
    out.append("```text")
    out.append(f"gains (>0 = PredHow 更优): vs Native mean {np.mean(pn):+.6f} | "
               f"vs AllClone mean {np.mean(pc):+.6f} | vs AllSplit mean {np.mean(ps):+.6f}")
    out.append(f"positive stages: vs Native {sum(x>0 for x in pn)}/{len(pn)} | "
               f"vs AllClone {sum(x>0 for x in pc)}/{len(pc)} | "
               f"vs AllSplit {sum(x>0 for x in ps)}/{len(ps)}")
    out.append(f"Oracle-vs-Pred gap (mean, >0 = oracle 仍更好): {np.mean(op):+.6f}\n")
    # global PSNR
    out.append("### Global PSNR@100\n```text")
    for it in iters:
        vals = {p: get(it, p, "global_psnr_100") for p in
                ("rw_nativehow", "rw_allclone", "rw_allsplit", "rw_predhow", "rw_oraclehow")}
        out.append(f"it={it:>5d}: " + " ".join(f"{p.split('_')[1]}={v:.4f}" for p, v in vals.items()))
    out.append("```\n")

    # plot
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = np.arange(len(iters))
    cols = {"rw_nativehow": "#cc8800", "rw_allclone": "#cc3311", "rw_allsplit": "#4477aa",
            "rw_predhow": "#228833", "rw_oraclehow": "#000000"}
    labs = {"rw_nativehow": "RW-NativeHow", "rw_allclone": "RW-AllClone",
            "rw_allsplit": "RW-AllSplit", "rw_predhow": "RW-PredHow",
            "rw_oraclehow": "RW-OracleHow"}
    for p in cols:
        ys = [get(it, p) for it in iters]
        ax.plot(x, ys, marker="o", color=cols[p], label=labs[p],
                ls="--" if p == "rw_oraclehow" else "-")
    ax.set_xticks(x)
    ax.set_xticklabels(iters)
    ax.set_ylabel("group demand L1 @100 (lower better)")
    ax.set_title("action-prior audit (same RW repeat-0 subset, K=100 rho=0.5)")
    ax.legend(fontsize=8)
    fig.savefig(f"{OUT}/plots/action_prior_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    merged = ""
    for f in ("b6fix_stats_loso.txt", "b6fix_stats_action.txt"):
        p = f"{OUT}/data/{f}"
        if os.path.exists(p):
            merged += open(p).read() + "\n"
    with open(f"{OUT}/data/b6fix_stats.txt", "w") as fh:
        fh.write(merged)
    with open(f"{OUT}/data/b6fix_stats_action.txt", "w") as fh:
        fh.write("\n".join(out) + "\n")
    with open(f"{OUT}/data/b6fix_stats.txt", "w") as fh:
        fh.write(merged + "\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "loso"
    if stage == "loso":
        stage_loso()
    else:
        stage_final()
