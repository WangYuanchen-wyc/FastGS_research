#
# Paper B - B12-S analysis (host).  python3 diagnostics/analyze_b12s.py
#
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b12_s_view_sampling_robustness"
VCS = [10, 20, 50, 100]
RATIOS = [5, 10, 20]


def main():
    stab = list(csv.DictReader(open(f"{BASE}/data/score_stability.csv")))
    ov = list(csv.DictReader(open(f"{BASE}/data/pruning_candidate_overlap.csv")))
    rem = list(csv.DictReader(open(f"{BASE}/data/pruning_results.csv")))
    for r in stab + ov + rem:
        for k, v in r.items():
            try: r[k] = float(v)
            except: pass
    cks = sorted({int(r["checkpoint"]) for r in stab})
    out = ["## B12-S View-Sampling Robustness\n"]

    # ---- score stability vs view count ----
    out.append("### Spearman vs All-view（10 repeats mean±std）\n```text")
    out.append(f"{'ckpt':>7s} " + " ".join(f"{'%d-view' % v:>14s}" for v in VCS))
    for ck in cks:
        line = f"it{int(ck):>5d}"
        for v in VCS:
            sub = [r["spearman"] for r in stab if int(r["checkpoint"]) == ck
                   and int(r["n_views"]) == v]
            line += f" {np.mean(sub):>7.4f}±{np.std(sub):.3f}"
        out.append(line)
    out.append("```\n")

    # ---- Top-K overlap ----
    out.append("### Top-K overlap vs All-view\n```text")
    for k_pct in (5, 10, 20):
        out.append(f"Top-{k_pct}%:")
        out.append(f"{'ckpt':>7s} " + " ".join(f"{'%d-v' % v:>10s}" for v in VCS))
        for ck in cks:
            line = f"it{int(ck):>5d}"
            for v in VCS:
                sub = [r["overlap"] for r in ov if int(r["checkpoint"]) == ck
                       and int(r["n_views"]) == v and int(r["top_k"]) == k_pct]
                line += f" {np.mean(sub):>10.4f}"
            out.append(line)
    out.append("```\n")

    # ---- pruning quality ----
    out.append("### Pruning quality（10 repeats mean±std，+ All-view）\n```text")
    for ck in cks:
        out.append(f"\nit{int(ck)}:")
        for pct in RATIOS:
            all_r = next((r for r in rem if int(r["checkpoint"]) == ck
                          and int(r["n_views"]) == -1 and int(r["ratio"]) == pct), None)
            line = f"  {pct:>2d}%: All={all_r['psnr']:.4f}"
            for v in VCS:
                sub = [r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                       and int(r["n_views"]) == v and int(r["ratio"]) == pct]
                if sub:
                    line += f" | {v}v={np.mean(sub):.4f}±{np.std(sub):.4f}"
            out.append(line)
            # 10-view best/worst
            sub10 = [r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                     and int(r["n_views"]) == 10 and int(r["ratio"]) == pct]
            if sub10:
                out.append(f"        10v best={max(sub10):.4f} worst={min(sub10):.4f} "
                           f"range={max(sub10)-min(sub10):.4f}")
    out.append("```\n")

    # ---- trend check ----
    out.append("### View-count trend（PSNR 随 view 数变化，@10% pruning）\n```text")
    for ck in cks:
        all_r = next((r for r in rem if int(r["checkpoint"]) == ck
                      and int(r["n_views"]) == -1 and int(r["ratio"]) == 10), None)
        vals = [all_r["psnr"]]
        for v in VCS:
            sub = [r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                   and int(r["n_views"]) == v and int(r["ratio"]) == 10]
            vals.append(np.mean(sub) if sub else np.nan)
        monotone_up = all(vals[i] <= vals[i+1] + 0.01 for i in range(len(vals)-1))
        gap_10_all = vals[0] - vals[-1]
        out.append(f"it{int(ck):>5d}: All={vals[0]:.4f} → " +
                   " ".join(f"{v}v={vals[i+1]:.4f}" for i, v in enumerate(VCS)) +
                   f" | gap(10v−All)={gap_10_all:+.4f} | monotone↑: {'Y' if monotone_up else 'N'}")
    out.append("```\n")

    # ---- GO/NO-GO ----
    out.append("### GO/NO-GO\n```text")
    for ck in cks:
        for pct in (10, 20):
            a = next((r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                      and int(r["n_views"]) == -1 and int(r["ratio"]) == pct), None)
            v10 = [r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                   and int(r["n_views"]) == 10 and int(r["ratio"]) == pct]
            v100 = [r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                    and int(r["n_views"]) == 100 and int(r["ratio"]) == pct]
            if a and v10:
                d10 = np.mean(v10) - a
                d10_std = np.std(v10)
                d100 = np.mean(v100) - a if v100 else None
                out.append(f"it{int(ck)} {pct}%: 10v−All={d10:+.4f}±{d10_std:.4f} | "
                           f"100v−All={d100:+.4f}" if d100 is not None else "")
    out.append("```")

    # ---- plots ----
    os.makedirs(f"{BASE}/plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ck, col in zip(cks, ["#4477aa", "#228833", "#cc3311"]):
        xs = VCS; ys = []; es = []
        for v in VCS:
            sub = [r["spearman"] for r in stab if int(r["checkpoint"]) == ck
                   and int(r["n_views"]) == v]
            ys.append(np.mean(sub)); es.append(np.std(sub))
        ax.errorbar(xs, ys, yerr=es, marker="o", color=col, label=f"it{ck}")
    ax.set_xlabel("n_views"); ax.set_ylabel("Spearman vs All-view")
    ax.set_title("score convergence to all-view")
    ax.legend()
    fig.savefig(f"{BASE}/plots/spearman_vs_num_views.png", dpi=130, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k_pct, col in ((5, "#cc3311"), (10, "#4477aa"), (20, "#228833")):
        xs = VCS; ys = []
        for v in VCS:
            sub = [r["overlap"] for r in ov if int(r["n_views"]) == v
                   and int(r["top_k"]) == k_pct]
            # average across checkpoints
            ys.append(np.mean(sub))
        ax.plot(xs, ys, marker="o", color=col, label=f"Top-{k_pct}%")
    ax.set_xlabel("n_views"); ax.set_ylabel("overlap vs All-view")
    ax.set_title("pruning candidate overlap")
    ax.legend()
    fig.savefig(f"{BASE}/plots/overlap_vs_num_views.png", dpi=130, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ck, col in zip(cks, ["#4477aa", "#228833", "#cc3311"]):
        xs = VCS + [300]; ys = []
        for v in VCS:
            sub = [r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                   and int(r["n_views"]) == v and int(r["ratio"]) == 10]
            ys.append(np.mean(sub) if sub else np.nan)
        a = next((r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                  and int(r["n_views"]) == -1 and int(r["ratio"]) == 10), None)
        ys.append(a if a else np.nan)
        ax.plot(xs, ys, marker="o", color=col, label=f"it{ck}")
    ax.set_xlabel("n_views (300=All)"); ax.set_ylabel("PSNR @ 10% prune")
    ax.set_title("pruning quality vs view count")
    ax.legend()
    fig.savefig(f"{BASE}/plots/psnr_vs_num_views.png", dpi=130, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ck, col in zip(cks, ["#4477aa", "#228833", "#cc3311"]):
        for v, mk in ((10, "o"), (50, "s"), (300, "*")):
            xs = []; ys = []
            for pct in RATIOS:
                if v == 300:
                    a = next((r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                              and int(r["n_views"]) == -1 and int(r["ratio"]) == pct), None)
                    if a: xs.append(pct); ys.append(a)
                else:
                    sub = [r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                           and int(r["n_views"]) == v and int(r["ratio"]) == pct]
                    if sub: xs.append(pct); ys.append(np.mean(sub))
            ax.plot(xs, ys, marker=mk, color=col, alpha=0.7,
                    label=f"it{ck} {v}v")
    ax.set_xlabel("pruning ratio %"); ax.set_ylabel("PSNR")
    ax.set_title("pruning quality vs ratio")
    ax.legend(fontsize=6)
    fig.savefig(f"{BASE}/plots/pruning_quality_vs_ratio.png", dpi=130, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ck, col in zip(cks, ["#4477aa", "#228833", "#cc3311"]):
        vals = [r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                and int(r["n_views"]) == 10 and int(r["ratio"]) == 10]
        if vals:
            ax.hist(vals, bins=15, alpha=0.5, color=col, label=f"it{ck}", density=True)
    ax.set_xlabel("PSNR (10-view @ 10% prune)"); ax.set_ylabel("density")
    ax.set_title("10-view repeat variance")
    ax.legend()
    fig.savefig(f"{BASE}/plots/ten_view_repeat_variance.png", dpi=130, bbox_inches="tight"); plt.close()

    with open(f"{BASE}/data/b12s_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
