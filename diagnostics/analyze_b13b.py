#
# Paper B - B13-B analysis (host).  python3 diagnostics/analyze_b13b.py
#
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b13_b_pruning_boundary"
RATIOS = [5, 10, 20]
N_REPEATS = 20


def main():
    grp = list(csv.DictReader(open(f"{BASE}/data/boundary_groups.csv")))
    exch = list(csv.DictReader(open(f"{BASE}/data/exchange_removal_results.csv")))
    freq = list(csv.DictReader(open(f"{BASE}/data/false_prune_frequency.csv")))
    for r in grp + exch + freq:
        for k, v in r.items():
            if k not in ("group",):
                try: r[k] = float(v)
                except: pass
    out = ["## B13-B Pruning Boundary Misranking & View Coverage (it30000)\n"]

    # ---- exchange test ----
    out.append("### Exchange removal（C+F vs C+M，同数量，20 repeats mean±std）\n```text")
    for pct in RATIOS:
        lf = [r["psnr"] for r in exch if int(r["ratio"]) == pct and r["group"] == "limited_C+F"]
        am = [r["psnr"] for r in exch if int(r["ratio"]) == pct and r["group"] == "allview_C+M"]
        if lf and am:
            out.append(f"  {pct:>2d}%: C+F(10v) {np.mean(lf):.4f}±{np.std(lf):.4f} | "
                       f"C+M(All) {np.mean(am):.4f}±{np.std(am):.4f} | "
                       f"diff(10v−All) {np.mean(lf)-np.mean(am):+.4f} | "
                       f"10v worse: {sum(1 for a,b in zip(lf,am) if a<b)}/{len(lf)}")
    out.append("```\n")

    # ---- group statistics ----
    metrics = ["pru_10v", "pru_allv", "pct_10v", "pct_allv", "boundary_dist",
               "vis_10v", "vis_all", "vis_ratio_10v", "vis_ratio_all",
               "ang_spread_sampled", "ang_spread_all",
               "evidence_10v", "evidence_allv"]
    out.append("### F/M/C group statistics（@10% ratio，across reps）\n```text")
    for pct in [10]:  # focus on 10% for readability
        by_g = {g: [r for r in grp if int(r["ratio"]) == pct and r["group"] == g]
                for g in ("F", "M", "C")}
        out.append(f"{'metric':>20s} {'F(mean)':>10s} {'M(mean)':>10s} {'C(mean)':>10s} "
                   f"{'F−M':>10s} {'effect(d)':>9s}")
        for m in metrics:
            vals = {g: [r[m] for r in by_g[g]] for g in ("F", "M", "C")}
            f_m, m_m = np.mean(vals["F"]), np.mean(vals["M"])
            pooled_sd = np.sqrt((np.std(vals["F"])**2 + np.std(vals["M"])**2) / 2)
            d = (f_m - m_m) / pooled_sd if pooled_sd > 0 else 0
            out.append(f"{m:>20s} {f_m:>10.4f} {m_m:>10.4f} {np.mean(vals['C']):>10.4f} "
                       f"{f_m-m_m:>+10.4f} {d:>+9.2f}")
        # hit rates
        for hit in ("hit_top1", "hit_top3", "hit_top5"):
            vals = {g: np.mean([r[hit] for r in by_g[g]]) for g in ("F", "M", "C")}
            out.append(f"{hit:>20s} {vals['F']:>10.3f} {vals['M']:>10.3f} {vals['C']:>10.3f} "
                       f"{vals['F']-vals['M']:>+10.3f}")
    out.append("```\n")

    # ---- persistent false-prune ----
    out.append("### Persistent false-prune（20 repeats 中进入 F 的频率）\n```text")
    for pct in RATIOS:
        sub = [r for r in freq if int(r["ratio"]) == pct]
        n_total = len(sub)
        p30 = sum(1 for r in sub if r["false_prune_freq"] >= 0.3)
        p50 = sum(1 for r in sub if r["false_prune_freq"] >= 0.5)
        p80 = sum(1 for r in sub if r["false_prune_freq"] >= 0.8)
        out.append(f"  {pct:>2d}%: total in F ≥1 time: {n_total} | "
                   f"freq≥30%: {p30} | freq≥50%: {p50} | freq≥80%: {p80}")
    out.append("```\n")

    # ---- GO/NO-GO ----
    out.append("### GO/NO-GO\n```text")
    for pct in (10, 20):
        lf = [r["psnr"] for r in exch if int(r["ratio"]) == pct and r["group"] == "limited_C+F"]
        am = [r["psnr"] for r in exch if int(r["ratio"]) == pct and r["group"] == "allview_C+M"]
        d = np.mean(lf) - np.mean(am) if lf and am else 0
        worse = sum(1 for a, b in zip(lf, am) if a < b)
        out.append(f"  {pct}%: F vs M PSNR diff {d:+.4f} | F worse in {worse}/{len(lf)} repeats")
    out.append("```")

    # ---- plots ----
    os.makedirs(f"{BASE}/plots", exist_ok=True)

    def save(fig, name):
        fig.savefig(f"{BASE}/plots/{name}", dpi=130, bbox_inches="tight"); plt.close(fig)

    # 1 boundary rank shift
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for g, col in (("F", "#cc3311"), ("M", "#4477aa"), ("C", "#228833")):
        sub = [r for r in grp if int(r["ratio"]) == 10 and r["group"] == g]
        if sub:
            ax.hist([r["pct_allv"] for r in sub], bins=50, alpha=0.5,
                    label=g, color=col, density=True)
    ax.axvline(0.10, color="k", ls="--", lw=1, label="10% cutoff")
    ax.set_xlabel("All-view percentile"); ax.set_ylabel("density")
    ax.set_title("boundary distribution (F/M/C @10%)")
    ax.legend()
    save(fig, "boundary_rank_shift.png")

    # 2 exchange
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for g, col, mk in (("limited_C+F", "#cc3311", "o"), ("allview_C+M", "#4477aa", "s")):
        xs, ys, es = [], [], []
        for pct in RATIOS:
            sub = [r["psnr"] for r in exch if int(r["ratio"]) == pct and r["group"] == g]
            if sub: xs.append(pct); ys.append(np.mean(sub)); es.append(np.std(sub))
        ax.errorbar(xs, ys, yerr=es, marker=mk, color=col, label=g)
    ax.set_xlabel("pruning ratio %"); ax.set_ylabel("PSNR")
    ax.set_title("exchange: C+F vs C+M")
    ax.legend()
    save(fig, "false_vs_missed_prune.png")

    # 3 visibility coverage
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for g, col in (("F", "#cc3311"), ("M", "#4477aa"), ("C", "#228833")):
        sub = [r for r in grp if int(r["ratio"]) == 10 and r["group"] == g]
        if sub:
            ax.hist([r["vis_ratio_all"] for r in sub], bins=30, alpha=0.5,
                    label=g, color=col, density=True)
    ax.set_xlabel("all-view visibility ratio"); ax.set_ylabel("density")
    ax.set_title("visibility coverage (F/M/C @10%)")
    ax.legend()
    save(fig, "visibility_coverage.png")

    # 4 key view hit rate
    fig, ax = plt.subplots(figsize=(6, 4.2))
    hits = ["hit_top1", "hit_top3", "hit_top5"]
    x = np.arange(len(hits))
    for i, (g, col) in enumerate((("F", "#cc3311"), ("M", "#4477aa"), ("C", "#228833"))):
        ys = [np.mean([r[h] for r in grp if int(r["ratio"]) == 10 and r["group"] == g])
              for h in hits]
        ax.bar(x + (i-1)*0.25, ys, 0.25, label=g, color=col)
    ax.set_xticks(x); ax.set_xticklabels(hits)
    ax.set_ylabel("hit rate"); ax.set_title("key-view hit rate @10%")
    ax.legend()
    save(fig, "key_view_hit_rate.png")

    # 5 angular coverage
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for g, col in (("F", "#cc3311"), ("M", "#4477aa"), ("C", "#228833")):
        sub = [r for r in grp if int(r["ratio"]) == 10 and r["group"] == g]
        if sub:
            ax.hist([r["ang_spread_sampled"] for r in sub], bins=30, alpha=0.5,
                    label=g, color=col, density=True)
    ax.set_xlabel("sampled angular spread"); ax.set_ylabel("density")
    ax.set_title("angular coverage (F/M/C @10%)")
    ax.legend()
    save(fig, "angular_coverage.png")

    # 6 false prune frequency
    fig, ax = plt.subplots(figsize=(6, 4))
    for pct, col in zip(RATIOS, ["#4477aa", "#228833", "#cc3311"]):
        sub = [r["false_prune_freq"] for r in freq if int(r["ratio"]) == pct]
        if sub:
            ax.hist(sub, bins=20, alpha=0.5, label=f"{pct}%", color=col, density=True)
    ax.set_xlabel("false-prune frequency (across 20 reps)")
    ax.set_ylabel("density"); ax.set_title("persistent false-prune")
    ax.legend()
    save(fig, "false_prune_frequency.png")

    with open(f"{BASE}/data/b13b_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
