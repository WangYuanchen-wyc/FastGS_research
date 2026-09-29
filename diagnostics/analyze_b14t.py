#
# Paper B - B14-T analysis (host).  python3 diagnostics/analyze_b14t.py
#
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b14_t_error_threshold"
THRESHOLDS = [0.025, 0.05, 0.10, 0.15, 0.20]
RATIOS = [5, 10, 20]


def main():
    ze = list(csv.DictReader(open(f"{BASE}/data/threshold_zero_evidence.csv")))
    fp = list(csv.DictReader(open(f"{BASE}/data/threshold_false_prune.csv")))
    rem = list(csv.DictReader(open(f"{BASE}/data/threshold_pruning_results.csv")))
    for r in ze + fp + rem:
        for k, v in r.items():
            try: r[k] = float(v)
            except: pass

    out = ["## B14-T Error Threshold Diagnostic\n"]

    # zero-evidence
    out.append("### 1) Visible-but-zero-evidence ratio\n```text")
    for th in THRESHOLDS:
        sub = [r["zero_evi_ratio"] for r in ze if r["threshold"] == th]
        out.append(f"  thresh={th:>5.3f}: {np.mean(sub):.4f} ± {np.std(sub):.4f}")
    out.append("```\n")

    # Spearman + overlap
    out.append("### 2) Spearman + Top-10% overlap vs All-view\n```text")
    for th in THRESHOLDS:
        sp = [r["spearman"] for r in ze if r["threshold"] == th]
        ov = [r["top10_overlap"] for r in ze if r["threshold"] == th]
        out.append(f"  thresh={th:>5.3f}: Spearman={np.mean(sp):.4f}±{np.std(sp):.4f} | "
                   f"Top10% overlap={np.mean(ov):.4f}±{np.std(ov):.4f}")
    out.append("```\n")

    # false-prune
    out.append("### 3) False-prune count\n```text")
    for th in THRESHOLDS:
        line = f"  thresh={th:>5.3f}:"
        for pct in RATIOS:
            sub = [r["false_prune_count"] for r in fp
                   if r["threshold"] == th and int(r["ratio"]) == pct]
            line += f" {pct}%={np.mean(sub):.0f}±{np.std(sub):.0f}"
        out.append(line)
    out.append("```\n")

    # pruning quality
    out.append("### 4) Pruning quality（10-view，20 reps mean±std）\n```text")
    for pct in RATIOS:
        a = next((r["psnr"] for r in rem if r["threshold"] == -1
                  and int(r["ratio"]) == pct), None)
        out.append(f"  {pct}%: All-view={a:.4f}")
        for th in THRESHOLDS:
            sub = [r["psnr"] for r in rem if r["threshold"] == th
                   and int(r["ratio"]) == pct]
            ss = [r["ssim"] for r in rem if r["threshold"] == th
                  and int(r["ratio"]) == pct]
            lp = [r["lpips"] for r in rem if r["threshold"] == th
                  and int(r["ratio"]) == pct]
            out.append(f"    th={th:>5.3f}: PSNR={np.mean(sub):.4f}±{np.std(sub):.4f} "
                       f"SSIM={np.mean(ss):.4f} LPIPS={np.mean(lp):.4f} "
                       f"(ΔPSNR vs All={np.mean(sub)-a:+.4f})")
    out.append("```\n")

    # GO/NO-GO
    out.append("### 5) Best threshold analysis\n```text")
    for pct in (10, 20):
        best_th = None; best_psnr = -999
        for th in THRESHOLDS:
            sub = [r["psnr"] for r in rem if r["threshold"] == th
                   and int(r["ratio"]) == pct]
            if np.mean(sub) > best_psnr:
                best_psnr = np.mean(sub); best_th = th
        a = next((r["psnr"] for r in rem if r["threshold"] == -1
                  and int(r["ratio"]) == pct), None)
        base = np.mean([r["psnr"] for r in rem if r["threshold"] == 0.10
                        and int(r["ratio"]) == pct])
        out.append(f"  {pct}%: best={best_th} ({best_psnr:.4f}) | baseline 0.10 ({base:.4f}) | "
                   f"All ({a:.4f}) | best−baseline={best_psnr-base:+.4f}")
    out.append("```")

    # plots
    os.makedirs(f"{BASE}/plots", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ys1 = [np.mean([r["zero_evi_ratio"] for r in ze if r["threshold"] == th])
           for th in THRESHOLDS]
    axes[0].plot(THRESHOLDS, ys1, "o-"); axes[0].set_xlabel("threshold"); axes[0].set_ylabel("zero-evi ratio")
    axes[0].set_title("visible-but-zero-evidence")
    ys2 = [np.mean([r["spearman"] for r in ze if r["threshold"] == th])
           for th in THRESHOLDS]
    axes[1].plot(THRESHOLDS, ys2, "s-", color="#228833")
    axes[1].set_xlabel("threshold"); axes[1].set_ylabel("Spearman")
    axes[1].set_title("Spearman vs All-view")
    ys3 = [np.mean([r["false_prune_count"] for r in fp
                    if r["threshold"] == th and int(r["ratio"]) == 10])
           for th in THRESHOLDS]
    axes[2].plot(THRESHOLDS, ys3, "^-", color="#cc3311")
    axes[2].set_xlabel("threshold"); axes[2].set_ylabel("|F| @10%")
    axes[2].set_title("false-prune count")
    fig.savefig(f"{BASE}/plots/threshold_effect.png", dpi=130, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for pct, col, mk in ((5, "#228833", "o"), (10, "#4477aa", "s"), (20, "#cc3311", "^")):
        xs = THRESHOLDS
        ys = [np.mean([r["psnr"] for r in rem if r["threshold"] == th
                       and int(r["ratio"]) == pct]) for th in THRESHOLDS]
        a = next((r["psnr"] for r in rem if r["threshold"] == -1
                  and int(r["ratio"]) == pct), None)
        ax.plot(xs, ys, marker=mk, color=col, label=f"{pct}% prune")
        ax.axhline(a, color=col, ls=":", alpha=0.5)
    ax.set_xlabel("threshold"); ax.set_ylabel("PSNR (10-view)")
    ax.set_title("pruning quality vs threshold (dotted=All-view)")
    ax.legend()
    fig.savefig(f"{BASE}/plots/psnr_vs_threshold.png", dpi=130, bbox_inches="tight"); plt.close()

    with open(f"{BASE}/data/b14t_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
