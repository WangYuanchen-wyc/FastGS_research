#
# Paper B - B6 analysis (host, sklearn):
#   model stage : LOSO (leave-one-stage-out) evaluation of Who-ranking
#                 (q_best) and How-gap (q_clone - q_split) prediction from
#                 pre-action features; exports the held-out predictions used
#                 by the B6 replay; 4 plots.
#   final stage : replay retention stats + practical_vs_oracle plot.
#
#   python3 diagnostics/analyze_b6.py model
#   python3 diagnostics/analyze_b6.py final
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
from sklearn.metrics import balanced_accuracy_score

BASE = "paper_b/b6_practical_oracle_approximation"

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


def load_rows():
    rows = [r for r in json.load(open(f"{BASE}/data/b6_features.json"))
            if (r.get("valid_views", 1) or 0) > 0 and r.get("q_best") is not None]
    return rows


def make_model(kind):
    if kind == "ridge":
        return Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    return DecisionTreeRegressor(max_depth=3, random_state=0)


def stage_model():
    rows = load_rows()
    iters = sorted({r["iteration"] for r in rows})
    print(f"[b6a] rows={len(rows)} iters={iters}")

    def feat_vals(r):
        return np.array([np.nan if r.get(f) is None else float(r[f]) for f in FEATURES])

    def xy(sub, target, medians):
        ok = [r for r in sub if r.get(target) is not None]
        X = np.array([np.nan_to_num(feat_vals(r), nan=medians) for r in ok])
        y = np.array([float(r[target]) for r in ok])
        ids = [r["parent_index"] for r in ok]
        return X, y, ids, ok

    who_rows, how_rows = [], []
    preds = {str(it): {"pred_q_best": {}, "pred_q_gap": {}} for it in iters}
    best_who, best_how = {}, {}

    for held in iters:
        train = [r for r in rows if r["iteration"] != held]
        test = [r for r in rows if r["iteration"] == held]
        for task, target in (("who", "q_best"), ("how", "q_gap")):
            # median imputation from the TRAIN fold only (test rows never
            # influence their own imputation)
            tr_feats = np.array([feat_vals(r) for r in train])
            medians = np.nanmedian(np.where(np.isnan(tr_feats), np.nan, tr_feats), axis=0)
            medians = np.nan_to_num(medians, nan=0.0)
            Xtr, ytr, _, _ = xy(train, target, medians)
            Xte, yte, ids, ok_te = xy(test, target, medians)
            for kind in ("ridge", "tree"):
                m = make_model(kind).fit(Xtr, ytr)
                p = m.predict(Xte)
                sp = float(spearmanr(p, yte)[0])
                rec = {"held_out_iteration": held, "model": kind, "n_test": len(yte)}
                if task == "who":
                    n = len(yte)
                    o_order = np.argsort(-yte)
                    p_order = np.argsort(-p)
                    for frac, tag in ((0.25, "top25"), (0.5, "top50")):
                        mtop = int(round(frac * n))
                        ov = len(set(o_order[:mtop]) & set(p_order[:mtop])) / mtop
                        rec[f"{tag}_overlap"] = ov
                        rec[f"{tag}_overlap_random"] = mtop / n
                    for rho in (0.5, 0.75):
                        M = int(round(rho * n))
                        vc = float(yte[p_order[:M]].sum() / max(yte[o_order[:M]].sum(), 1e-9))
                        # random capture baseline (expectation over draws)
                        rng = np.random.RandomState(0)
                        rvc = np.mean([yte[rng.permutation(n)[:M]].sum()
                                       / max(yte[o_order[:M]].sum(), 1e-9) for _ in range(200)])
                        rec[f"value_capture@{rho}"] = vc
                        rec[f"value_capture_random@{rho}"] = rvc
                    rec["spearman"] = sp
                    who_rows.append(rec)
                else:
                    pred_pos = p > 0
                    true_pos = yte > 0
                    rec["spearman"] = sp
                    rec["sign_accuracy"] = float(np.mean(pred_pos == true_pos))
                    rec["balanced_accuracy"] = float(balanced_accuracy_score(true_pos, pred_pos))
                    # tercile buckets of |q_gap|
                    qs = np.quantile(np.abs(yte), [1 / 3, 2 / 3])
                    bucket = np.digitize(np.abs(yte), qs)
                    for b, name in ((0, "low"), (1, "medium"), (2, "high")):
                        msk = bucket == b
                        rec[f"sign_acc_{name}|gap|"] = float(np.mean(
                            pred_pos[msk] == true_pos[msk])) if msk.sum() else None
                    how_rows.append(rec)
            # per-stage best model kind for THIS task (replay later uses one
            # fixed kind per task = majority across stages)
            task_rows = who_rows if task == "who" else how_rows
            best = max(("ridge", "tree"),
                       key=lambda k: [r for r in task_rows
                                      if r["held_out_iteration"] == held
                                      and r["model"] == k][-1]["spearman"])
            if task == "who":
                best_who[held] = best
            else:
                best_how[held] = best
            p = make_model(best).fit(Xtr, ytr).predict(Xte)
            key = "pred_q_best" if task == "who" else "pred_q_gap"
            for i, v in zip(ids, p):
                preds[str(held)][key][str(i)] = float(v)

    # fixed model kind per task = majority across stages (report)
    who_kind = max(set(best_who.values()), key=list(best_who.values()).count)
    how_kind = max(set(best_how.values()), key=list(best_how.values()).count)
    print(f"[b6a] selected model kinds: who={who_kind} how={how_kind}")

    # rebuild predictions with the FIXED kinds (consistency for replay)
    for held in iters:
        train = [r for r in rows if r["iteration"] != held]
        test = [r for r in rows if r["iteration"] == held]
        for task, target, kind, key in (("who", "q_best", who_kind, "pred_q_best"),
                                        ("how", "q_gap", how_kind, "pred_q_gap")):
            tr_feats = np.array([feat_vals(r) for r in train])
            medians = np.nan_to_num(np.nanmedian(tr_feats, axis=0), nan=0.0)
            Xtr, ytr, _, _ = xy(train, target, medians)
            Xte, yte, ids, _ = xy(test, target, medians)
            p = make_model(kind).fit(Xtr, ytr).predict(Xte)
            preds[str(held)][key] = {str(i): float(v) for i, v in zip(ids, p)}

    os.makedirs(f"{BASE}/cache", exist_ok=True)
    json.dump({"preds": preds, "who_kind": who_kind, "how_kind": how_kind},
              open(f"{BASE}/cache/b6_predictions.json", "w"))

    import csv as _csv
    for name, rr in (("b6_who_results", who_rows), ("b6_how_results", how_rows)):
        with open(f"{BASE}/data/{name}.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rr[0].keys()))
            w.writeheader()
            for r in rr:
                w.writerow({k: ("NA" if v is None else f"{v:.6g}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    out = ["## B6 建模结果（LOSO leave-one-stage-out）\n"]
    out.append("### Who ranking（target=q_best）\n```text")
    out.append(f"{'held-out':>9s} {'model':>6s} {'spearman':>9s} {'top25ov':>8s} {'rand':>6s} "
               f"{'top50ov':>8s} {'rand':>6s} {'VC@.5':>7s} {'rand':>7s} {'VC@.75':>8s} {'rand':>7s}")
    for r in who_rows:
        out.append(f"{r['held_out_iteration']:>9d} {r['model']:>6s} {r['spearman']:>9.3f} "
                   f"{r['top25_overlap']:>8.3f} {r['top25_overlap_random']:>6.3f} "
                   f"{r['top50_overlap']:>8.3f} {r['top50_overlap_random']:>6.3f} "
                   f"{r['value_capture@0.5']:>7.3f} {r['value_capture_random@0.5']:>7.3f} "
                   f"{r['value_capture@0.75']:>8.3f} {r['value_capture_random@0.75']:>7.3f}")
    out.append("```\n")
    out.append("### How gap（target=q_gap）\n```text")
    out.append(f"{'held-out':>9s} {'model':>6s} {'spearman':>9s} {'signAcc':>8s} {'balAcc':>7s} "
               f"{'acc low':>8s} {'acc med':>8s} {'acc high':>8s}")
    for r in how_rows:
        out.append(f"{r['held_out_iteration']:>9d} {r['model']:>6s} {r['spearman']:>9.3f} "
                   f"{r['sign_accuracy']:>8.3f} {r['balanced_accuracy']:>7.3f} "
                   f"{r['sign_acc_low|gap|']:>8.3f} {r['sign_acc_medium|gap|']:>8.3f} "
                   f"{r['sign_acc_high|gap|']:>8.3f}")
    out.append("```\n")

    # ---------------- plots ----------------
    os.makedirs(f"{BASE}/plots", exist_ok=True)

    def save(fig, name):
        fig.savefig(f"{BASE}/plots/{name}", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 1 who value capture
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(iters))
    for i, (rho, col) in enumerate(((0.5, "#4477aa"), (0.75, "#228833"))):
        vc = [next(r for r in who_rows if r["held_out_iteration"] == it
                   and r["model"] == who_kind)[f"value_capture@{rho}"] for it in iters]
        rc = [next(r for r in who_rows if r["held_out_iteration"] == it
                   and r["model"] == who_kind)[f"value_capture_random@{rho}"] for it in iters]
        ax.plot(x, vc, marker="o", color=col, label=f"pred rho={rho}")
        ax.plot(x, rc, marker="x", color=col, ls=":", alpha=0.7, label=f"random rho={rho}")
    ax.set_xticks(x)
    ax.set_xticklabels(iters)
    ax.set_xlabel("held-out iteration")
    ax.set_ylabel("ValueCapture")
    ax.set_title(f"Who value capture ({who_kind}, LOSO)")
    ax.legend(fontsize=8)
    save(fig, "who_value_capture.png")

    # 2 who spearman by iteration
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, col in (("ridge", "#cc8800"), ("tree", "#4477aa")):
        sp = [next(r for r in who_rows if r["held_out_iteration"] == it
                   and r["model"] == kind)["spearman"] for it in iters]
        ax.plot(x, sp, marker="o", label=kind, color=col)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(iters)
    ax.set_ylabel("Spearman(pred, q_best)")
    ax.set_title("Who ranking quality by held-out stage")
    ax.legend()
    save(fig, "who_spearman_by_iteration.png")

    # 3 how gap prediction (spearman + sign acc)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for kind, col in (("ridge", "#cc8800"), ("tree", "#4477aa")):
        sp = [next(r for r in how_rows if r["held_out_iteration"] == it
                   and r["model"] == kind)["spearman"] for it in iters]
        ax.plot(x, sp, marker="o", label=f"spearman ({kind})", color=col)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(iters)
    ax.set_ylabel("Spearman(pred_gap, q_gap)")
    ax.set_title("How gap prediction by held-out stage")
    ax.legend()
    save(fig, "how_gap_prediction.png")

    # 4 accuracy vs gap bucket
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for kind, col in (("ridge", "#cc8800"), ("tree", "#4477aa")):
        accs = [[next(r for r in how_rows if r["held_out_iteration"] == it
                      and r["model"] == kind)[f"sign_acc_{b}|gap|"] for it in iters]
                for b in ("low", "medium", "high")]
        for b, acc, mk in zip(("low", "medium", "high"), accs, ("v", "s", "D")):
            ax.plot(x, acc, marker=mk, color=col, alpha=0.8,
                    label=f"{kind} {b}|gap|" if kind == "tree" else None)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(iters)
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel("sign accuracy")
    ax.set_title("How accuracy vs |q_gap| bucket (tree; ridge similar)")
    ax.legend(fontsize=7)
    save(fig, "how_accuracy_vs_gap.png")

    with open(f"{BASE}/data/b6_stats_model.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


def stage_final():
    recs = json.load(open(f"{BASE}/data/b6_replay_results.json"))
    iters = sorted({r["iteration"] for r in recs})
    out = ["## B6 Replay 与 Retention\n"]

    def get(it, p):
        sub = [r for r in recs if r["iteration"] == it and r["policy"] == p]
        if p in ("rw_nh", "rw_oh"):
            return float(np.mean([r["demand_l1_100"] for r in sub]))
        return sub[0]["demand_l1_100"]

    out.append("### Replay（K=100, rho=0.5, seed=0；RW 为 5 repeats 均值）\n```text")
    out.append(f"{'iter':>6s} {'NativeFull':>11s} {'RW-NH':>9s} {'PWho-NHow':>10s} "
               f"{'RWho-PHow':>10s} {'PWho-PHow':>10s} {'OW-OH':>9s} {'retention':>9s}")
    rets = []
    for it in iters:
        nf, rw = get(it, "native_full"), get(it, "rw_nh")
        pwnh, rwph = get(it, "predwho_nativehow"), get(it, "randomwho_predhow")
        ppph, oo = get(it, "predwho_predhow"), get(it, "ow_oh")
        ret = (rw - ppph) / (rw - oo) if abs(rw - oo) > 1e-9 else float("nan")
        rets.append(ret)
        out.append(f"{it:>6d} {nf:>11.6f} {rw:>9.6f} {pwnh:>10.6f} {rwph:>10.6f} "
                   f"{ppph:>10.6f} {oo:>9.6f} {ret:>+9.1%}")
    out.append("```\n")
    out.append(f"mean retention = {np.nanmean(rets):+.1%}\n")

    # global PSNR table
    out.append("### Global PSNR@100\n```text")
    for it in iters:
        vals = {p: (float(np.mean([r["global_psnr_100"] for r in recs
                                   if r["iteration"] == it and r["policy"] == p]))
                    if any(r["iteration"] == it and r["policy"] == p for r in recs) else None)
                for p in ("native_full", "rw_nh", "predwho_nativehow", "randomwho_predhow",
                          "predwho_predhow", "ow_oh")}
        out.append(f"it={it:>5d}: " + " ".join(f"{p}={v:.4f}" if v is not None else f"{p}=NA"
                                               for p, v in vals.items()))
    out.append("```\n")

    # plot 5
    os.makedirs(f"{BASE}/plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(len(iters))
    for p, col, mk in (("rw_nh", "#4477aa", "o"), ("predwho_predhow", "#228833", "D"),
                       ("ow_oh", "#cc3311", "*"), ("native_full", "#cc8800", "s"),
                       ("predwho_nativehow", "#66ccee", "^"), ("randomwho_predhow", "#aa3377", "v")):
        ys = [get(it, p) for it in iters]
        ax.plot(x, ys, marker=mk, color=col, label=p)
    ax.set_xticks(x)
    ax.set_xticklabels(iters)
    ax.set_ylabel("group demand L1 @100 (lower better)")
    ax.set_title("practical vs oracle replay (K=100 rho=0.5 seed=0)")
    ax.legend(fontsize=7)
    fig.savefig(f"{BASE}/plots/practical_vs_oracle_replay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # merge stats
    merged = ""
    for f in ("b6_stats_model.txt", "b6_stats_replay.txt"):
        p = f"{BASE}/data/{f}"
        if os.path.exists(p):
            merged += open(p).read() + "\n"
    with open(f"{BASE}/data/b6_stats.txt", "w") as f:
        f.write(merged)
    print("\n".join(out))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "model"
    if stage == "model":
        stage_model()
    else:
        # write replay text then merge
        stage_final()
