#
# Paper B - B14-T Fix analysis (host).  python3 diagnostics/analyze_b14t_fix.py
#
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b14_t_error_threshold/fix"
THRESHOLDS = [0.025, 0.05, 0.10, 0.15, 0.20]
RATIOS = [5, 10, 20]


def main():
    ze = list(csv.DictReader(open(f"{BASE}/data/same_threshold_zero_evidence.csv")))
    fp = list(csv.DictReader(open(f"{BASE}/data/same_threshold_false_prune.csv")))
    rem = list(csv.DictReader(open(f"{BASE}/data/same_threshold_pruning_results.csv")))
    for r in ze + fp + rem:
        for k, v in r.items():
            if v not in ("allview", "10view"):
                try: r[k] = float(v)
                except: pass

    out = ["## B14-T Fix: Same-Threshold All-view Control\n"]

    # A) All-view quality per threshold
    out.append("### A) All-view pruning quality per threshold\n```text")
    out.append(f"{'thresh':>7s} {'5%PSNR':>8s} {'10%PSNR':>9s} {'20%PSNR':>9s} "
               f"{'5%SSIM':>8s} {'10%LPIPS':>9s}")
    for th in THRESHOLDS:
        line = f"{th:>7.3f}"
        for pct in RATIOS:
            a = next((r["psnr"] for r in rem if r["threshold"] == th
                      and int(r["ratio"]) == pct and r["source"] == "allview"), None)
            line += f" {a:>8.4f}" if pct == 5 else f" {a:>9.4f}"
        for pct, m in ((5, "ssim"), (10, "lpips")):
            a = next((r[m] for r in rem if r["threshold"] == th
                      and int(r["ratio"]) == pct and r["source"] == "allview"), None)
            line += f" {a:>8.4f}" if m == "ssim" else f" {a:>9.4f}"
        out.append(line)
    out.append("```\n")

    # B) 10v vs Allv@same-threshold gap
    out.append("### B) 10-view gap vs All-view@same-threshold\n```text")
    out.append(f"{'thresh':>7s} {'Spearman':>10s} {'Top10%ov':>10s} {'zero-evi%':>10s}")
    for th in THRESHOLDS:
        sub = [r for r in ze if r["threshold"] == th and r["rep"] >= 0]
        sp = [r["spearman"] for r in sub]
        ov = [r["top10_overlap"] for r in sub]
        zr = [r["zero_evi_ratio"] for r in sub]
        out.append(f"{th:>7.3f} {np.mean(sp):>10.4f} {np.mean(ov):>10.4f} {np.mean(zr):>10.4f}")
    out.append("```\n")

    # C) 10v quality and gap
    out.append("### C) 10-view quality and gap\n```text")
    for pct in RATIOS:
        out.append(f"\n  {pct}%:")
        for th in THRESHOLDS:
            a = next((r["psnr"] for r in rem if r["threshold"] == th
                      and int(r["ratio"]) == pct and r["source"] == "allview"), None)
            v10 = [r["psnr"] for r in rem if r["threshold"] == th
                   and int(r["ratio"]) == pct and r["source"] == "10view"]
            ss = [r["ssim"] for r in rem if r["threshold"] == th
                  and int(r["ratio"]) == pct and r["source"] == "10view"]
            lp = [r["lpips"] for r in rem if r["threshold"] == th
                  and int(r["ratio"]) == pct and r["source"] == "10view"]
            out.append(f"    th={th:>5.3f}: All={a:.4f} | 10v={np.mean(v10):.4f}±{np.std(v10):.4f} "
                       f"gap={np.mean(v10)-a:+.4f} | SSIM={np.mean(ss):.4f} LPIPS={np.mean(lp):.4f}")
    out.append("```\n")

    # D) false-prune
    out.append("### D) False-prune count\n```text")
    for th in THRESHOLDS:
        line = f"  th={th:>5.3f}:"
        for pct in RATIOS:
            sub = [r["false_prune_count"] for r in fp
                   if r["threshold"] == th and int(r["ratio"]) == pct]
            line += f" {pct}%={np.mean(sub):.0f}"
        out.append(line)
    out.append("```\n")

    # GO/NO-GO
    out.append("### GO/NO-GO\n```text")
    for pct in (10, 20):
        gaps = {}
        for th in THRESHOLDS:
            a = next((r["psnr"] for r in rem if r["threshold"] == th
                      and int(r["ratio"]) == pct and r["source"] == "allview"), None)
            v10 = [r["psnr"] for r in rem if r["threshold"] == th
                   and int(r["ratio"]) == pct and r["source"] == "10view"]
            gaps[th] = np.mean(v10) - a if v10 and a else None
        out.append(f"  {pct}% gap: " + " ".join(f"th={th}:{g:+.4f}" for th, g in gaps.items()))
    out.append("```")

    # plots
    os.makedirs(f"{BASE}/plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for pct, col in ((10, "#4477aa"), (20, "#cc3311")):
        all_p = [next(r["psnr"] for r in rem if r["threshold"] == th
                      and int(r["ratio"]) == pct and r["source"] == "allview")
                 for th in THRESHOLDS]
        v10_p = [np.mean([r["psnr"] for r in rem if r["threshold"] == th
                           and int(r["ratio"]) == pct and r["source"] == "10view"])
                 for th in THRESHOLDS]
        axes[0].plot(THRESHOLDS, all_p, "o-", color=col, label=f"All {pct}%")
        axes[0].plot(THRESHOLDS, v10_p, "s--", color=col, label=f"10v {pct}%")
    axes[0].set_xlabel("threshold"); axes[0].set_ylabel("PSNR")
    axes[0].set_title("A) All-view vs 10-view quality")
    axes[0].legend(fontsize=7)
    gaps_10 = []
    for th in THRESHOLDS:
        a = next(r["psnr"] for r in rem if r["threshold"] == th
                 and int(r["ratio"]) == 10 and r["source"] == "allview")
        v = np.mean([r["psnr"] for r in rem if r["threshold"] == th
                     and int(r["ratio"]) == 10 and r["source"] == "10view"])
        gaps_10.append(v - a)
    axes[1].bar(range(len(THRESHOLDS)), gaps_10, 0.5, color="#4477aa")
    axes[1].set_xticks(range(len(THRESHOLDS))); axes[1].set_xticklabels(THRESHOLDS)
    axes[1].set_ylabel("10v − All (dB)"); axes[1].set_title("B) gap @10%")
    fig.savefig(f"{BASE}/plots/same_threshold_analysis.png", dpi=130, bbox_inches="tight")
    plt.close()

    with open(f"{BASE}/data/b14t_fix_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
