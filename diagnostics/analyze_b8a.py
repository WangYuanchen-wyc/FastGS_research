#
# Paper B - B8-A analysis (FIXED, host, analysis-only — no GPU reruns):
#  * std/SEM use ddof=1 (sample std), SEM = std/sqrt(5)
#  * MSCC main rule = +-1 SEM overlap (mean_N <= best + SEM_N + SEM_best);
#    sensitivity vs max-rule (old) and RSS rule
#  * Split-beneficial (conservative): mean_best_split + SEM_best_split < Keep_L1
#  * main cardinality tables conditioned on Split-beneficial parents
#  * effect-size analysis for MSCC>2 (gain thresholds + uncertainty-normalized)
#  * parent_category A/B/C(3/4/6)
#  * diminishing returns reported as curves + per-additional-child marginal
#    (no boolean formula)
#
#   python3 diagnostics/analyze_b8a.py
#

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b8_a_split_cardinality"
NS = [2, 3, 4, 6]


def sample_std(v):
    return float(np.std(v, ddof=1)) if len(v) >= 2 else 0.0


def sem_of(v):
    return sample_std(v) / np.sqrt(len(v)) if len(v) >= 2 else 0.0


def mscc_of(means, sems, rule):
    """smallest N whose mean is statistically indistinguishable from best.
    rule: 'max' | 'sum' | 'rss' — how the two SEMs combine."""
    valid = [n for n in NS if means[n] is not None]
    best_n = min(valid, key=lambda n: means[n])
    bm, bs = means[best_n], sems[best_n]
    if rule == "max":
        comb = lambda s: max(s, bs)
    elif rule == "rss":
        comb = lambda s: float(np.hypot(s, bs))
    else:  # sum (main)
        comb = lambda s: s + bs
    suff = [n for n in valid if means[n] <= bm + comb(sems[n])]
    return min(suff) if suff else best_n, best_n, suff


def main():
    rows = json.load(open(f"{BASE}/data/b8a_repeat_results.json"))
    parents = defaultdict(dict)
    for r in rows:
        k = (r["iteration"], r["parent_index"])
        if r["branch"] == "keep":
            parents[k]["keep"] = r
        else:
            parents[k].setdefault(int(r["N"]), []).append(r)

    prow = []
    for (it, idx), d in sorted(parents.items()):
        rec = {"iteration": it, "parent_index": idx,
               "keep_demand_l1_100": d["keep"]["demand_l1_100"]}
        means, sems, stds = {}, {}, {}
        for n in NS:
            l1 = [b["demand_l1_100"] for b in d[n] if b["demand_l1_100"] is not None]
            means[n] = float(np.mean(l1)) if l1 else None
            stds[n] = sample_std(l1)
            sems[n] = sem_of(l1)
            rec[f"mean_l1_N{n}"] = means[n]
            rec[f"std_l1_N{n}"] = stds[n]        # sample std (ddof=1)
            rec[f"sem_l1_N{n}"] = sems[n]        # std / sqrt(5)
            rec[f"mean_tile_N{n}"] = float(np.mean([b["tile_pairs_100"] for b in d[n]]))
            rec[f"mean_gpsnr_N{n}"] = float(np.mean([b["global_psnr_100"] for b in d[n]]))
        # MSCC under three rules (main = sum)
        for rule, tag in (("max", "mscc_max"), ("sum", "mscc"), ("rss", "mscc_rss")):
            m, b, _ = mscc_of(means, sems, rule)
            rec[tag] = m
            if rule == "sum":
                rec["best_quality_N"] = b
        # Split-beneficial (conservative, main): best split's +1 SEM upper
        # bound still better than Keep; loose auxiliary: mean only
        best_n = rec["best_quality_N"]
        rec["split_beneficial"] = bool(
            means[best_n] + sems[best_n] < rec["keep_demand_l1_100"])
        rec["split_beneficial_loose"] = bool(means[best_n] < rec["keep_demand_l1_100"])
        # effect size for higher-cardinality parents
        higher = [n for n in (3, 4, 6) if means[n] is not None]
        if higher:
            hb_n = min(higher, key=lambda n: means[n])
            rec["higher_best_N"] = hb_n
            rec["gain_over_split2"] = means[2] - means[hb_n]
            rec["normalized_gain"] = (means[2] - means[hb_n]) / float(
                np.hypot(sems[2], sems[hb_n])) if (sems[2] + sems[hb_n]) > 0 else None
        else:
            rec["higher_best_N"] = None
            rec["gain_over_split2"] = None
            rec["normalized_gain"] = None
        for n in NS:
            if means[n] is not None and means.get(2) is not None:
                rec[f"gain_N{n}_vs_2"] = means[2] - means[n]
        # parent category
        if not rec["split_beneficial"]:
            rec["parent_category"] = "A_split_not_confirmed"
        elif rec["mscc"] == 2:
            rec["parent_category"] = "B_split2_sufficient"
        else:
            rec["parent_category"] = f"C{rec['mscc']}_higher_cardinality_required"
        prow.append(rec)

    import csv as _csv
    with open(f"{BASE}/data/b8a_parent_results.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(prow[0].keys()))
        w.writeheader()
        for r in prow:
            w.writerow({k: ("NA" if v is None else (f"{v:.8g}" if isinstance(v, float) else v))
                        for k, v in r.items()})

    iters = sorted({r["iteration"] for r in prow})
    ben = [r for r in prow if r["split_beneficial"]]
    out = ["## B8-A Split Cardinality Oracle（Analysis Fix）\n"]
    out.append("std = sample std (ddof=1)，SEM = std/sqrt(5)；MSCC 主规则 = ±1 SEM overlap"
               "（mean_N ≤ best_mean + SEM_N + SEM_best，非严格显著性检验）\n")

    # ---------------- split-beneficial breakdown ----------------
    out.append("### Split-beneficial 判定（主定义：best Split 的 mean+SEM 上界仍优于 Keep）\n```text")
    sumrows = []
    for scope, sub in [("ALL", prow)] + [(f"it{it}", [r for r in prow if r["iteration"] == it])
                                         for it in iters]:
        n = len(sub)
        b = sum(1 for r in sub if r["split_beneficial"])
        bl = sum(1 for r in sub if r["split_beneficial_loose"])
        out.append(f"{scope:>7s} (n={n}): Split-beneficial {b}({b/n*100:.0f}%) | "
                   f"not-confirmed/uncertain {n-b}({(n-b)/n*100:.0f}%) | loose(辅助) {bl}({bl/n*100:.0f}%)")
        sumrows.append({"scope": scope, "n": n, "split_beneficial": b,
                        "split_beneficial_ratio": b / n, "not_confirmed": n - b,
                        "loose": bl})
    # ---------------- conditioned MSCC (main) ----------------
    out.append("```\n\n### 主表：Split-beneficial parents 的 MSCC 分布（主规则 SEM-sum）\n```text")
    for scope, sub in [("ALL-beneficial", ben)] + \
                      [(f"it{it}-beneficial", [r for r in ben if r["iteration"] == it])
                       for it in iters]:
        n = len(sub)
        if n == 0:
            out.append(f"{scope:>20s}: n=0")
            continue
        dist = {nn: sum(1 for r in sub if r["mscc"] == nn) for nn in NS}
        out.append(f"{scope:>20s} (n={n}): " +
                   " ".join(f"N{nn}={dist[nn]}({dist[nn]/n*100:.0f}%)" for nn in NS) +
                   f" | MSCC>2 {sum(dist[n_] for n_ in (3, 4, 6))}({(n-dist[2])/n*100:.0f}%)"
                   f" | Split-2 sufficient {dist[2]}({dist[2]/n*100:.0f}%)")
        sumrows.append({"scope": scope, "n": n,
                        **{f"mscc{nn}": dist[nn] for nn in NS},
                        "mscc_gt2": n - dist[2], "mscc2_sufficient": dist[2]})
    # auxiliary all-parents
    out.append("```\n\n辅助（全 150 parents，不作主结论）：\n```text")
    for scope, sub in [("ALL", prow)] + [(f"it{it}", [r for r in prow if r["iteration"] == it])
                                         for it in iters]:
        n = len(sub)
        dist = {nn: sum(1 for r in sub if r["mscc"] == nn) for nn in NS}
        out.append(f"{scope:>7s} (n={n}): " +
                   " ".join(f"N{nn}={dist[nn]}({dist[nn]/n*100:.0f}%)" for nn in NS) +
                   f" | MSCC>2 {(n-dist[2])/n*100:.0f}%")
    out.append("```\n")

    # ---------------- MSCC sensitivity ----------------
    out.append("### MSCC sensitivity（三种 SEM 合成规则）\n```text")
    sens_rows = []
    for label, sub in (("all_parents", prow), ("split_beneficial", ben)):
        for field, disp in (("mscc_max", "old-max-SEM"), ("mscc", "SEM-sum"),
                            ("mscc_rss", "RSS-SEM")):
            n = len(sub)
            dist = {nn: sum(1 for r in sub if r[field] == nn) for nn in NS}
            gt2 = n - dist[2]
            row = {"population": label, "criterion": disp, "n": n,
                   **{f"N{nn}_pct": 100 * dist[nn] / n for nn in NS},
                   "MSCC_gt2_pct": 100 * gt2 / n}
            sens_rows.append(row)
            out.append(f"{label:>17s} {disp:>11s}: " +
                       " ".join(f"N{nn}={100*dist[nn]/n:4.0f}%" for nn in NS) +
                       f" | MSCC>2 {100*gt2/n:4.0f}%")
    out.append("```\n")

    # ---------------- effect size (Split-beneficial & MSCC>2) ----------------
    eff = [r for r in ben if r["mscc"] > 2]
    out.append(f"### Effect-size（Split-beneficial ∧ MSCC>2，n={len(eff)}）\n```text")
    gains = [r["gain_over_split2"] for r in eff if r["gain_over_split2"] is not None]
    ngains = [r["normalized_gain"] for r in eff if r["normalized_gain"] is not None]
    if gains:
        out.append(f"gain_over_split2 = mean_L1(N2) − min(mean_L1 N3/N4/N6)："
                   f"mean {np.mean(gains):+.6f} median {np.median(gains):+.6f} "
                   f"p25 {np.percentile(gains, 25):+.6f} p75 {np.percentile(gains, 75):+.6f}")
        for th in (1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3):
            c = sum(1 for g in gains if g > th)
            out.append(f"  gain > {th:.0e}: {c}/{len(gains)} ({c/len(gains)*100:.0f}%)")
        out.append(f"normalized_gain = gain / sqrt(SEM_N2²+SEM_higher²)（effect-vs-repeat-noise diagnostic，非 z-test）："
                   f">1: {sum(1 for x in ngains if x > 1)}/{len(ngains)} "
                   f"({100*sum(1 for x in ngains if x > 1)/len(ngains):.0f}%) | "
                   f">2: {sum(1 for x in ngains if x > 2)}/{len(ngains)} "
                   f"({100*sum(1 for x in ngains if x > 2)/len(ngains):.0f}%)")
    out.append("```\n")

    # ---------------- diminishing returns (curves, no boolean) ----------------
    out.append("### Quality vs N 曲线与 per-additional-child 边际（分组）\n```text")

    def curve(sub):
        return {n: float(np.mean([r[f"mean_l1_N{n}"] for r in sub])) for n in NS}

    def report(name, sub):
        c = curve(sub)
        marg = {("2→3"): c[2] - c[3],
                ("3→4"): c[3] - c[4],
                ("4→6/child"): (c[4] - c[6]) / 2}
        out.append(f"{name:>22s} (n={len(sub)}): keep "
                   f"{np.mean([r['keep_demand_l1_100'] for r in sub]):.6f} | " +
                   " ".join(f"N{n}={c[n]:.6f}" for n in NS) + " | 边际/child " +
                   " ".join(f"{k}:{v:+.6f}" for k, v in marg.items()))

    report("beneficial-ALL", ben)
    report("beneficial-MSCC=2", [r for r in ben if r["mscc"] == 2])
    report("beneficial-MSCC>2", [r for r in ben if r["mscc"] > 2])
    for it in iters:
        report(f"it{it}-beneficial", [r for r in ben if r["iteration"] == it])
    out.append("```\n")

    # ---------------- categories ----------------
    out.append("### parent_category 分布\n```text")
    cats = ["A_split_not_confirmed", "B_split2_sufficient",
            "C3_higher_cardinality_required", "C4_higher_cardinality_required",
            "C6_higher_cardinality_required"]
    for scope, sub in [("ALL", prow)] + [(f"it{it}", [r for r in prow if r["iteration"] == it])
                                         for it in iters]:
        n = len(sub)
        out.append(f"{scope:>7s}: " + " ".join(
            f"{c.split('_')[0]}={sum(1 for r in sub if r['parent_category'] == c)}"
            for c in cats) + f"  (n={n})")
    out.append("```\n")

    # ---------------- GO/NO-GO (conditioned, 7 conditions) ----------------
    n_b = len(ben)
    dist = {nn: sum(1 for r in ben if r["mscc"] == nn) for nn in NS}
    gt2 = n_b - dist[2]
    g_nonnoise = sum(1 for x in ngains if x > 1) if ngains else 0
    cross = all(len({r["mscc"] for r in ben if r["iteration"] == it} & {2, 3, 4, 6}) >= 2
                and any(r["mscc"] > 2 for r in ben if r["iteration"] == it)
                and any(r["mscc"] == 2 for r in ben if r["iteration"] == it) for it in iters)
    sens_ok = all(0.15 < (sum(1 for r in sub if r[col] > 2) / len(sub)) < 0.85
                  for sub in (prow, ben) for col in ("mscc_max", "mscc", "mscc_rss"))
    monotone_bad = all(curve(ben)[n] > curve(ben)[6] for n in (2, 3, 4))
    checks = [
        ("1. MSCC 非单一 N 主导（beneficial）", max(dist.values()) / n_b < 0.8),
        (f"2. MSCC>2 稳定比例（{gt2}/{n_b} = {gt2/n_b*100:.0f}%）", gt2 / n_b >= 0.15),
        (f"3. MSCC=2 大量存在（{dist[2]}/{n_b} = {dist[2]/n_b*100:.0f}%）", dist[2] / n_b >= 0.3),
        (f"4. MSCC>2 中非噪声级 gain（normalized>1: {g_nonnoise}/{len(ngains) if ngains else 0}）",
         len(ngains) > 0 and g_nonnoise / len(ngains) >= 0.3),
        ("5. 三 stage 均存在 heterogeneity", cross),
        ("6. 三种 criterion 稳健（MSCC>2 ∈ 15–85%）", sens_ok),
        ("7. quality 非随 N 单调变好", not monotone_bad),
    ]
    out.append("### GO/NO-GO（修正后，Split-beneficial 条件下）\n```text")
    for name, ok in checks:
        out.append(f"{name} -> {'Y' if ok else 'N'}")
    verdict = all(ok for _, ok in checks)
    out.append(f"\n判定: {'B8-A GO' if verdict else 'B8-A NO-GO'}")
    out.append("```")

    # ---------------- CSVs ----------------
    all_fields = []
    for r_ in sumrows:
        for k_ in r_:
            if k_ not in all_fields:
                all_fields.append(k_)
    with open(f"{BASE}/data/b8a_summary.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=all_fields, restval="")
        w.writeheader()
        w.writerows(sumrows)
    with open(f"{BASE}/data/b8a_mscc_sensitivity.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(sens_rows[0].keys()))
        w.writeheader()
        w.writerows(sens_rows)
    with open(f"{BASE}/data/b8a_effect_size.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["iteration", "parent_index", "mscc",
                                           "higher_best_N", "gain_over_split2", "normalized_gain"])
        w.writeheader()
        for r in eff:
            w.writerow({"iteration": r["iteration"], "parent_index": r["parent_index"],
                        "mscc": r["mscc"], "higher_best_N": r["higher_best_N"],
                        "gain_over_split2": r["gain_over_split2"],
                        "normalized_gain": r["normalized_gain"]})
    # conditioned summary
    with open(f"{BASE}/data/b8a_conditioned_summary.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["scope", "n", "mscc2", "mscc3", "mscc4",
                                           "mscc6", "mscc_gt2"])
        w.writeheader()
        for scope, sub in [("ALL", ben)] + [(f"it{it}", [r for r in ben if r["iteration"] == it])
                                            for it in iters]:
            w.writerow({"scope": scope, "n": len(sub),
                        **{f"mscc{nn}": sum(1 for r in sub if r["mscc"] == nn) for nn in NS},
                        "mscc_gt2": sum(1 for r in sub if r["mscc"] > 2)})

    # ---------------- plots ----------------
    os.makedirs(f"{BASE}/plots", exist_ok=True)

    def save(fig, name):
        fig.savefig(f"{BASE}/plots/{name}", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # conditioned MSCC distribution
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(NS))
    for it, col, off in zip(iters + ["all"], ["#4477aa", "#228833", "#cc8800", "#666666"],
                            np.linspace(-0.3, 0.3, 4)):
        sub = ben if it == "all" else [r for r in ben if r["iteration"] == it]
        ys = [100 * sum(1 for r in sub if r["mscc"] == n) / len(sub) if sub else 0 for n in NS]
        ax.bar(x + off, ys, 0.22, label=f"{'ALL' if it == 'all' else it}", color=col)
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in NS])
    ax.set_ylabel("% split-beneficial parents")
    ax.set_title("conditioned MSCC distribution")
    ax.legend()
    save(fig, "conditioned_mscc_distribution.png")

    # sensitivity
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    w_ = 0.12
    for i, (col, lab) in enumerate((("mscc_max", "old-max-SEM"), ("mscc", "SEM-sum"),
                                    ("mscc_rss", "RSS-SEM"))):
        ys = [100 * sum(1 for r in ben if r[col] == n) / len(ben) for n in NS]
        ax.bar(x + (i - 1) * w_, ys, w_, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in NS])
    ax.set_ylabel("% split-beneficial parents")
    ax.set_title("MSCC criterion sensitivity")
    ax.legend(fontsize=8)
    save(fig, "mscc_sensitivity.png")

    # gain distribution
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(gains, bins=30, color="#4477aa", edgecolor="white")
    for th, c in ((1e-5, "gray"), (1e-4, "#cc8800")):
        ax.axvline(th, color=c, ls="--", lw=1, label=f"gain={th:.0e}")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("gain_over_split2 (L1, >0 = higher-N better)")
    ax.set_ylabel("parents (beneficial & MSCC>2)")
    ax.set_title("Split-2 insufficiency effect size")
    ax.legend(fontsize=8)
    save(fig, "split2_gain_distribution.png")

    # conditioned quality vs child count
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for name, sub, col, mk in (("MSCC=2", [r for r in ben if r["mscc"] == 2], "#4477aa", "o"),
                               ("MSCC=3", [r for r in ben if r["mscc"] == 3], "#228833", "s"),
                               ("MSCC=4", [r for r in ben if r["mscc"] == 4], "#cc8800", "D"),
                               ("MSCC=6", [r for r in ben if r["mscc"] == 6], "#cc3311", "^")):
        if not sub:
            continue
        c = curve(sub)
        ax.plot(NS, [c[n] - np.mean([r["keep_demand_l1_100"] for r in sub]) for n in NS],
                marker=mk, color=col, label=f"{name} (n={len(sub)})")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("child count N"); ax.set_ylabel("L1 improvement vs Keep")
    ax.set_title("conditioned quality vs child count (by MSCC group)")
    ax.legend(fontsize=8)
    save(fig, "conditioned_quality_vs_child_count.png")

    with open(f"{BASE}/data/b8a_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
