#
# Paper B - B11-V Fix analysis (host).  python3 diagnostics/analyze_b11v_fix.py
#
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b11_v_multiview_importance_reliability/fix"


def main():
    stab = list(csv.DictReader(open(f"{BASE}/data/tie_aware_stability.csv")))
    rem = list(csv.DictReader(open(f"{BASE}/data/removal_results.csv")))
    grp = list(csv.DictReader(open(f"{BASE}/data/matched_group_stats.csv")))
    for r in stab + rem + grp:
        for k, v in r.items():
            if k not in ("group", "metric"):
                try: r[k] = float(v)
                except: pass
    cks = sorted({int(r["checkpoint"]) for r in stab})
    out = ["## B11-V Fix: Pruning-Score Controlled Agreement\n"]

    # A) tie-aware stability
    out.append("### A) Tie-aware pruning_score stability\n```text")
    for r in stab:
        out.append(f"it{int(r['checkpoint']):>5d} {r['metric']:>6s}: Spearman "
                   f"{r['mean_rho']:.4f}±{r['std_rho']:.4f} [{r['p10_rho']:.4f}~{r['p90_rho']:.4f}] | "
                   f"overlap {r['mean_overlap']:.4f} | cutoff={r['cutoff_score']:.4f} "
                   f"tied={int(r['n_tied_at_cutoff'])} ({r['tie_fraction']:.2%})")
    out.append("```\n")

    # C) group stats (matching quality)
    out.append("### C) Matched group stats\n```text")
    for r in grp:
        out.append(f"it{int(r['checkpoint']):>5d} {r['group']:>16s}: n={int(r['n'])} "
                   f"mean_pru={r['mean_pru']:.6f} eff_vc={r['mean_eff_vc']:.2f}")
    out.append("```\n")

    # removal results
    out.append("### Removal (mean±std over 5 repeats)\n```text")
    out.append(f"{'ckpt':>6s} {'pct':>4s} {'PSNR_base':>10s} "
               f"{'PSNR_lo':>16s} {'PSNR_rand':>16s} {'PSNR_hi':>16s} "
               f"{'Δlo−Δhi':>8s} {'Δlo−Δrand':>9s}")
    for ck in cks:
        base = next((r for r in rem if int(r["checkpoint"]) == ck and r["group"] == "baseline"), None)
        if not base: continue
        for pct in (1, 5, 10):
            line = f"{int(ck):>6d} {pct:>3d}% {base['psnr']:>10.3f}"
            vals = {}
            for gname in ("low_agreement", "random", "high_agreement"):
                sub = [r for r in rem if int(r["checkpoint"]) == ck
                       and r["group"] == gname and int(r["pct"]) == pct]
                if sub:
                    m = np.mean([r["psnr"] for r in sub])
                    s = np.std([r["psnr"] for r in sub])
                    vals[gname] = m
                    line += f" {m:>8.3f}±{s:.3f}"
                else:
                    line += f" {'N/A':>16s}"
            if len(vals) == 3:
                d_lo = vals["low_agreement"] - base["psnr"]
                d_hi = vals["high_agreement"] - base["psnr"]
                d_ra = vals["random"] - base["psnr"]
                line += f" {(d_lo-d_hi):>+8.3f} {(d_lo-d_ra):>+9.3f}"
            out.append(line)
    out.append("```\n")

    # GO/NO-GO
    diffs = []
    for ck in cks:
        base = next((r for r in rem if int(r["checkpoint"]) == ck and r["group"] == "baseline"), None)
        if not base: continue
        for pct in (5, 10):
            lo = [r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                  and r["group"] == "low_agreement" and int(r["pct"]) == pct]
            hi = [r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                  and r["group"] == "high_agreement" and int(r["pct"]) == pct]
            ra = [r["psnr"] for r in rem if int(r["checkpoint"]) == ck
                  and r["group"] == "random" and int(r["pct"]) == pct]
            if lo and hi:
                d = np.mean(lo) - np.mean(hi)
                lo_wins = sum(1 for a, b in zip(lo, hi) if a > b)
                diffs.append((ck, pct, d, lo_wins, len(lo),
                              np.mean(ra) if ra else None))
    out.append("### GO/NO-GO\n```text")
    out.append("diff = PSNR_low − PSNR_high (>0 = low-agreement 更可删):")
    for ck, pct, d, w, n, ra in diffs:
        extra = f" | rand={ra:.3f}" if ra else ""
        out.append(f"  it{ck} {pct}%: {d:+.4f} (low wins {w}/{n} repeats){extra}")
    pos = sum(1 for _,_,d,_,_,_ in diffs if d > 0)
    out.append(f"  positive: {pos}/{len(diffs)}\n```")

    # plots
    os.makedirs(f"{BASE}/plots", exist_ok=True)
    for ck, col in zip(cks, ["#4477aa", "#228833", "#cc8800", "#cc3311"]):
        pass
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for ck, col in zip(cks, ["#4477aa", "#228833", "#cc8800", "#cc3311"]):
        base = next((r for r in rem if int(r["checkpoint"]) == ck and r["group"] == "baseline"), None)
        if not base: continue
        for gname, mk, ls in (("low_agreement", "o", "-"), ("random", "x", ":"),
                              ("high_agreement", "s", "--")):
            pcts, drops = [], []
            for pct in (1, 5, 10):
                sub = [r for r in rem if int(r["checkpoint"]) == ck
                       and r["group"] == gname and int(r["pct"]) == pct]
                if sub:
                    pcts.append(pct)
                    drops.append(np.mean([r["psnr"] for r in sub]) - base["psnr"])
            if pcts:
                ax.plot(pcts, drops, marker=mk, ls=ls, color=col,
                        label=f"it{ck} {gname.split('_')[0]}", alpha=0.7)
    ax.set_xlabel("% removed"); ax.set_ylabel("PSNR drop (dB)")
    ax.set_title("removal: low vs random vs high agreement")
    ax.legend(fontsize=5, ncol=3)
    fig.savefig(f"{BASE}/plots/removal_low_random_high.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for pct, mk in ((5, "o"), (10, "s")):
        xs, ys = [], []
        for ck, pct_, d, w, n, ra in diffs:
            if pct_ == pct:
                xs.append(ck); ys.append(d)
        ax.plot(xs, ys, marker=mk, label=f"{pct}% removed")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("checkpoint"); ax.set_ylabel("PSNR_low − PSNR_high (dB)")
    ax.set_title("removal difference across checkpoints")
    ax.legend()
    fig.savefig(f"{BASE}/plots/removal_difference_across_checkpoints.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    with open(f"{BASE}/data/b11v_fix_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
