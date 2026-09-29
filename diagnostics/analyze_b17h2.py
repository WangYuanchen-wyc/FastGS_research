#
# Paper B - B17-H2 analysis (host).  python3 diagnostics/analyze_b17h2.py
#
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b17_h2_historical_ranking"
RATIOS = [5, 10, 20]
TYPES = ["allview_oracle", "10view", "allview_hist"]


def main():
    rows = list(csv.DictReader(open(f"{BASE}/data/pruning_results.csv")))
    for r in rows:
        for k, v in r.items():
            if k not in ("evidence_type",):
                try: r[k] = float(v)
                except: pass

    out = ["## B17-H2 Historical vs Current Ranking\n"]

    # group results
    groups = defaultdict(list)
    for r in rows:
        groups[r["evidence_type"]].append(r)

    # mean PSNR per type per ratio
    out.append("### Pruning quality (10-view score, 20 subsets mean)\n```text")
    out.append(f"{'evidence':>16s} {'5% PSNR':>8s} {'10% PSNR':>8s} {'20% PSNR':>8s} {'#GS':>8s}")
    for et in sorted(groups.keys()):
        line = f"{et:>16s}"
        for pct in RATIOS:
            sub = [r for r in groups[et] if int(r["ratio"]) == pct]
            if sub:
                line += f" {np.mean([r['psnr'] for r in sub]):>8.4f}"
            else:
                line += f" {'NA':>8s}"
        ng = np.mean([r["n_gs"] for r in groups[et]]) if groups[et] else 0
        line += f" {ng:>8.0f}"
        out.append(line)
    out.append("```\n")

    # All-view reference
    out.append("### All-view oracle\n```text")
    for et in sorted(groups.keys()):
        sub = [r for r in groups[et] if int(r["subset"]) == -1]
        if sub:
            r = sub[0]
            out.append(f"  {et}: PSNR={r['psnr']:.4f} #GS={int(r['n_gs'])}")
    out.append("```\n")

    # score stability
    if os.path.exists(f"{BASE}/data/score_stability.csv"):
        ss = list(csv.DictReader(open(f"{BASE}/data/score_stability.csv")))
        out.append("### 10-view score stability (Spearman vs All-view)\n```text")
        for r in ss:
            out.append(f"  subset {int(r['subset_id'])}: Spearman={float(r['spearman_vs_all']):.4f} "
                       f"Top10%ov={r['top10_overlap']}")
        sps = [float(r["spearman_vs_all"]) for r in ss]
        out.append(f"  mean={np.mean(sps):.4f} ± {np.std(sps):.4f}")
        out.append("```\n")

    # GO/NO-GO
    out.append("### GO/NO-GO\n```text")
    # compare mean 10-view quality vs All-view quality at each ratio
    for pct in RATIOS:
        # allview oracle
        a = [r for r in rows if r["evidence_type"] == "allview_oracle"
             and int(r["ratio"]) == pct]
        if a:
            out.append(f"  {pct}%: All-view oracle PSNR = {float(a[0]['psnr']):.4f}")
    out.append("```\n")

    # plots
    os.makedirs(f"{BASE}/plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for et, col, mk in (("binary", "#777777", "o"), ("continuous", "#228833", "s"),
                         ("contribution", "#4477aa", "^")):
        xs, ys = [], []
        for pct in RATIOS:
            sub = [r for r in rows if r["evidence_type"] == et
                   and int(r["ratio"]) == pct and r["source"] == "10view"]
            if sub:
                xs.append(pct); ys.append(np.mean([r["psnr"] for r in sub]))
        if xs:
            ax.plot(xs, ys, marker=mk, color=col, label=et)
    ax.set_xlabel("pruning ratio %"); ax.set_ylabel("PSNR")
    ax.set_title("evidence formulation comparison")
    ax.legend()
    fig.savefig(f"{BASE}/plots/evidence_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()

    with open(f"{BASE}/data/b17h2_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
