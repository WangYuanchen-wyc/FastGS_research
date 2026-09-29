#
# Paper B - B12-S Fix analysis (host).  python3 diagnostics/analyze_b12s_fix.py
#
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b12_s_view_sampling_robustness/fix"
VCS = [10, 20, 50, 100]
RATIOS = [5, 10, 20]
N_PERMS = 20


def main():
    stab = list(csv.DictReader(open(f"{BASE}/data/nested_score_stability.csv")))
    ov = list(csv.DictReader(open(f"{BASE}/data/nested_candidate_overlap.csv")))
    rem = list(csv.DictReader(open(f"{BASE}/data/nested_pruning_results.csv")))
    for r in stab + ov + rem:
        for k, v in r.items():
            try: r[k] = float(v)
            except: pass
    out = ["## B12-S Fix: Nested View Sampling Control (it30000)\n"]

    # Spearman convergence
    out.append("### Spearman vs All-view（nested，20 perms mean±std）\n```text")
    for v in VCS:
        sub = [r["spearman"] for r in stab if int(r["n_views"]) == v]
        out.append(f"{v:>4d}-view: {np.mean(sub):.4f} ± {np.std(sub):.4f}")
    out.append("```\n")

    # Overlap
    out.append("### Top-K overlap（nested，20 perms mean）\n```text")
    for k_pct in (5, 10, 20):
        line = f"Top-{k_pct:>2d}%: "
        for v in VCS:
            sub = [r["overlap"] for r in ov if int(r["n_views"]) == v
                   and int(r["top_k"]) == k_pct]
            line += f"{v}v={np.mean(sub):.3f} "
        out.append(line)
    out.append("```\n")

    # Pruning quality
    out.append("### Pruning quality（nested，20 perms mean±std）\n```text")
    for pct in RATIOS:
        a = next((r["psnr"] for r in rem if int(r["n_views"]) == -1
                  and int(r["ratio"]) == pct), None)
        line = f"  {pct:>2d}%: All={a:.4f}"
        for v in VCS:
            sub = [r["psnr"] for r in rem if int(r["n_views"]) == v
                   and int(r["ratio"]) == pct]
            line += f" | {v}v={np.mean(sub):.4f}±{np.std(sub):.4f}"
        out.append(line)
    out.append("```\n")

    # Per-permutation trajectory & monotonicity
    out.append("### Per-permutation monotonicity\n```text")
    mono_counts = {pct: 0 for pct in RATIOS}
    for pct in RATIOS:
        wins = {"10<20": 0, "20<50": 0, "50<100": 0, "100<=All": 0}
        for perm in range(N_PERMS):
            ps = {}
            for v in VCS:
                r = next((x for x in rem if int(x["perm"]) == perm
                          and int(x["n_views"]) == v and int(x["ratio"]) == pct), None)
                if r: ps[v] = r["psnr"]
            a = next((r["psnr"] for r in rem if int(r["n_views"]) == -1
                      and int(r["ratio"]) == pct), None)
            if len(ps) == 4 and a:
                if ps[10] <= ps[20]: wins["10<20"] += 1
                if ps[20] <= ps[50]: wins["20<50"] += 1
                if ps[50] <= ps[100]: wins["50<100"] += 1
                if ps[100] <= a: wins["100<=All"] += 1
                if ps[10] <= ps[20] <= ps[50] <= ps[100] <= a:
                    mono_counts[pct] += 1
        out.append(f"  {pct:>2d}%: 10v≤20v {wins['10<20']}/{N_PERMS} | "
                   f"20v≤50v {wins['20<50']}/{N_PERMS} | "
                   f"50v≤100v {wins['50<100']}/{N_PERMS} | "
                   f"100v≤All {wins['100<=All']}/{N_PERMS} | "
                   f"full monotone {mono_counts[pct]}/{N_PERMS}")
    out.append("```\n")

    # 50-view anomaly check
    out.append("### 50-view anomaly check\n```text")
    for pct in RATIOS:
        c50_worse = 0; c50_vs = {10: 0, 20: 0, 100: 0}
        for perm in range(N_PERMS):
            ps = {}
            for v in VCS:
                r = next((x for x in rem if int(x["perm"]) == perm
                          and int(x["n_views"]) == v and int(x["ratio"]) == pct), None)
                if r: ps[v] = r["psnr"]
            if len(ps) == 4:
                for v in (10, 20, 100):
                    if ps[50] < ps[v]: c50_vs[v] += 1
                if ps[50] < ps[10] and ps[50] < ps[20]:
                    c50_worse += 1
        out.append(f"  {pct:>2d}%: 50v worse than 10v: {c50_vs[10]}/{N_PERMS} | "
                   f"worse than 20v: {c50_vs[20]}/{N_PERMS} | "
                   f"worse than both 10v&20v: {c50_worse}/{N_PERMS}")
    out.append("```\n")

    # Gap summary
    out.append("### Gap summary（正确方向：PSNR_10view − PSNR_Allview）\n```text")
    for pct in RATIOS:
        a = next((r["psnr"] for r in rem if int(r["n_views"]) == -1
                  and int(r["ratio"]) == pct), None)
        gaps = {v: [] for v in VCS}
        for perm in range(N_PERMS):
            for v in VCS:
                r = next((x for x in rem if int(x["perm"]) == perm
                          and int(x["n_views"]) == v and int(x["ratio"]) == pct), None)
                if r and a: gaps[v].append(r["psnr"] - a)
        line = f"  {pct:>2d}%: "
        for v in VCS:
            line += f"{v}v−All={np.mean(gaps[v]):+.4f} "
        out.append(line)
    out.append("```\n")

    # GO/NO-GO
    mono_10 = mono_counts[10]
    gap10 = np.mean([next(r["psnr"] for r in rem if int(r["perm"]) == p
                          and int(r["n_views"]) == 10 and int(r["ratio"]) == 10)
                     for p in range(N_PERMS)]) - \
            next(r["psnr"] for r in rem if int(r["n_views"]) == -1 and int(r["ratio"]) == 10)
    gap20 = np.mean([next(r["psnr"] for r in rem if int(r["perm"]) == p
                          and int(r["n_views"]) == 10 and int(r["ratio"]) == 20)
                     for p in range(N_PERMS)]) - \
            next(r["psnr"] for r in rem if int(r["n_views"]) == -1 and int(r["ratio"]) == 20)
    out.append(f"### GO/NO-GO\n```text\n10v−All @10%: {gap10:+.4f} | @20%: {gap20:+.4f} | "
               f"full monotone @10%: {mono_10}/{N_PERMS}\n```\n")

    # Plots
    os.makedirs(f"{BASE}/plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    xs = VCS; ys = []; es = []
    for v in VCS:
        sub = [r["spearman"] for r in stab if int(r["n_views"]) == v]
        ys.append(np.mean(sub)); es.append(np.std(sub))
    ax.errorbar(xs, ys, yerr=es, marker="o", color="#4477aa")
    ax.set_xlabel("n_views (nested)"); ax.set_ylabel("Spearman vs All-view")
    ax.set_title("nested score convergence")
    fig.savefig(f"{BASE}/plots/nested_spearman_vs_views.png", dpi=130, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for k_pct, col in ((5, "#cc3311"), (10, "#4477aa"), (20, "#228833")):
        ys = [np.mean([r["overlap"] for r in ov if int(r["n_views"]) == v
                       and int(r["top_k"]) == k_pct]) for v in VCS]
        ax.plot(VCS, ys, marker="o", color=col, label=f"Top-{k_pct}%")
    ax.set_xlabel("n_views"); ax.set_ylabel("overlap vs All")
    ax.set_title("nested overlap")
    ax.legend()
    fig.savefig(f"{BASE}/plots/nested_overlap_vs_views.png", dpi=130, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for pct, col, mk in ((5, "#228833", "o"), (10, "#4477aa", "s"), (20, "#cc3311", "^")):
        xs = VCS + [311]; ys = []
        for v in VCS:
            sub = [r["psnr"] for r in rem if int(r["n_views"]) == v and int(r["ratio"]) == pct]
            ys.append(np.mean(sub))
        a = next((r["psnr"] for r in rem if int(r["n_views"]) == -1
                  and int(r["ratio"]) == pct), None)
        ys.append(a)
        ax.plot(xs, ys, marker=mk, color=col, label=f"{pct}% prune")
    ax.set_xlabel("n_views (311=All)"); ax.set_ylabel("PSNR")
    ax.set_title("nested pruning quality")
    ax.legend()
    fig.savefig(f"{BASE}/plots/nested_psnr_vs_views.png", dpi=130, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    pcts = list(RATIOS)
    mono_rates = [mono_counts[p] / N_PERMS * 100 for p in pcts]
    ax.bar(range(len(pcts)), mono_rates, 0.5, color="#4477aa")
    for i, r in enumerate(mono_rates):
        ax.text(i, r + 1, f"{mono_counts[pcts[i]]}/{N_PERMS}", ha="center")
    ax.set_xticks(range(len(pcts))); ax.set_xticklabels([f"{p}%" for p in pcts])
    ax.set_ylabel("% full monotone (10≤20≤50≤100≤All)")
    ax.set_title("monotonicity rate")
    fig.savefig(f"{BASE}/plots/monotonicity_rate.png", dpi=130, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for perm in range(min(10, N_PERMS)):
        ps = []
        for v in VCS:
            r = next((x for x in rem if int(x["perm"]) == perm
                      and int(x["n_views"]) == v and int(x["ratio"]) == 10), None)
            ps.append(r["psnr"] if r else np.nan)
        ax.plot(VCS, ps, alpha=0.4, lw=0.8, color="#4477aa")
    a = next((r["psnr"] for r in rem if int(r["n_views"]) == -1
              and int(r["ratio"]) == 10), None)
    ax.axhline(a, color="red", ls="--", lw=1.5, label="All-view")
    ax.set_xlabel("n_views"); ax.set_ylabel("PSNR @ 10% prune")
    ax.set_title("per-permutation trajectory (first 10)")
    ax.legend()
    fig.savefig(f"{BASE}/plots/per_repeat_trajectory.png", dpi=130, bbox_inches="tight"); plt.close()

    with open(f"{BASE}/data/b12s_fix_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
