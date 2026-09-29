#
# Paper B - B11-V analysis (host).  python3 diagnostics/analyze_b11v.py
#

import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = "paper_b/b11_v_multiview_importance_reliability"


def main():
    stab = list(csv.DictReader(open(f"{BASE}/data/b11v_gaussian_stability.csv")))
    agree = list(csv.DictReader(open(f"{BASE}/data/b11v_view_agreement.csv")))
    removal = list(csv.DictReader(open(f"{BASE}/data/b11v_removal_results.csv")))
    for r in stab + agree + removal:
        for k, v in r.items():
            if k not in ("group",):
                try: r[k] = float(v)
                except: pass
    cks = sorted({int(r["checkpoint"]) for r in stab})
    out = ["## B11-V Multi-view Importance Reliability\n"]

    # ---- A) stability ----
    out.append("### A) View Subset Stability\n```text")
    for ck in cks:
        sub = [r for r in stab if int(r["checkpoint"]) == ck]
        n = len(sub)
        if not n: continue
        rhos = []; j10 = []
        # reconstruct per-subset scores from stab file is impossible (only summary)
        # use per-Gaussian stats directly
        pct_std = [r["pct_std"] for r in sub]
        rank_std = [r["rank_std"] for r in sub]
        cv = [r["cv_score"] for r in sub]
        out.append(f"it{ck:>5d} (n={n}): percentile_std median {np.median(pct_std):.4f} "
                   f"p90 {np.percentile(pct_std,90):.4f} | rank_std median {np.median(rank_std):.0f} "
                   f"| score CV median {np.median(cv):.3f}")
    out.append("```\n")

    # ---- B) agreement heterogeneity within same importance ----
    out.append("### B) Agreement heterogeneity within same importance bin\n```text")
    for ck in cks:
        sub = [r for r in agree if int(r["checkpoint"]) == ck]
        if not sub: continue
        n = len(sub)
        imp = np.array([r["importance_score"] for r in sub])
        # bin by importance percentile (deciles)
        imp_order = np.argsort(np.argsort(imp))
        bins = imp_order // (n // 10)
        eff = np.array([r["effective_view_count"] for r in sub])
        sup = np.array([r["support_ratio"] for r in sub])
        maxf = np.array([r["max_view_fraction"] for r in sub])
        out.append(f"it{ck:>5d}: within-decile effective_view_count range (median):")
        for b in range(10):
            m = bins == b
            if m.sum() > 5:
                out.append(f"  decile {b}: eff_vc [{np.percentile(eff[m],25):.1f} ~ "
                           f"{np.percentile(eff[m],75):.1f}] median {np.median(eff[m]):.1f} | "
                           f"support_ratio median {np.median(sup[m]):.2f} | "
                           f"max_frac median {np.median(maxf[m]):.2f}")
    out.append("```\n")

    # ---- D) stability vs agreement correlation ----
    out.append("### D) Stability vs Agreement\n```text")
    for ck in cks:
        ss = {r["gaussian_id"]: r for r in stab if int(r["checkpoint"]) == ck}
        aa = [r for r in agree if int(r["checkpoint"]) == ck]
        ids = [r["gaussian_id"] for r in aa]
        pct_std = np.array([ss[i]["pct_std"] for i in ids])
        eff = np.array([r["effective_view_count"] for r in aa])
        sup = np.array([r["support_ratio"] for r in aa])
        maxf = np.array([r["max_view_fraction"] for r in aa])
        r1 = spearmanr(pct_std, sup)[0]
        r2 = spearmanr(pct_std, eff)[0]
        r3 = spearmanr(pct_std, maxf)[0]
        out.append(f"it{ck:>5d}: Spearman(pct_std, support_ratio) = {r1:+.3f} | "
                   f"(pct_std, eff_vc) = {r2:+.3f} | (pct_std, max_frac) = {r3:+.3f}")
    out.append("```\n")

    # ---- C) removal ----
    out.append("### C) Group Removal (whole-model, no recovery)\n```text")
    out.append(f"{'ckpt':>6s} {'pct':>4s} {'PSNR_base':>10s} {'PSNR_lo':>9s} {'PSNR_hi':>9s} "
               f"{'Δ_lo':>8s} {'Δ_hi':>8s} {'diff':>8s} {'SSIM_lo':>8s} {'SSIM_hi':>8s} {'LPIPS_lo':>8s} {'LPIPS_hi':>8s}")
    for ck in cks:
        base = next((r for r in removal if int(r["checkpoint"]) == ck and r["group"] == "baseline"), None)
        if not base: continue
        for pct in (1, 5, 10):
            lo = next((r for r in removal if int(r["checkpoint"]) == ck
                       and r["group"] == "low_agreement" and int(r["pct_removed"]) == pct), None)
            hi = next((r for r in removal if int(r["checkpoint"]) == ck
                       and r["group"] == "high_agreement" and int(r["pct_removed"]) == pct), None)
            if not lo or not hi: continue
            d_lo = lo["psnr"] - base["psnr"]
            d_hi = hi["psnr"] - base["psnr"]
            out.append(f"{int(ck):>6d} {pct:>3d}% {base['psnr']:>10.3f} {lo['psnr']:>9.3f} {hi['psnr']:>9.3f} "
                       f"{d_lo:>+8.3f} {d_hi:>+8.3f} {d_lo-d_hi:>+8.3f} "
                       f"{lo['ssim']:>8.4f} {hi['ssim']:>8.4f} {lo['lpips']:>8.4f} {hi['lpips']:>8.4f}")
    out.append("```\n")

    # GO/NO-GO
    out.append("### GO/NO-GO\n```text")
    # check removal difference across all checkpoints & percentages
    diffs = []
    for ck in cks:
        base = next((r for r in removal if int(r["checkpoint"]) == ck and r["group"] == "baseline"), None)
        if not base: continue
        for pct in (5, 10):  # use 5% and 10% as main criteria
            lo = next((r for r in removal if int(r["checkpoint"]) == ck
                       and r["group"] == "low_agreement" and int(r["pct_removed"]) == pct), None)
            hi = next((r for r in removal if int(r["checkpoint"]) == ck
                       and r["group"] == "high_agreement" and int(r["pct_removed"]) == pct), None)
            if lo and hi:
                d = (lo["psnr"] - base["psnr"]) - (hi["psnr"] - base["psnr"])
                diffs.append((ck, pct, d))
    if diffs:
        out.append("removal diff (ΔPSNR_low − ΔPSNR_high, >0 = low-agreement 更可删):")
        for ck, pct, d in diffs:
            out.append(f"  it{ck} {pct}%: {d:+.4f}")
        pos = sum(1 for _,_,d in diffs if d > 0)
        out.append(f"  positive: {pos}/{len(diffs)}")
    out.append("```")

    # ---- plots ----
    os.makedirs(f"{BASE}/plots", exist_ok=True)

    # 1 rank stability
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ck, col in zip(cks, ["#4477aa", "#228833", "#cc8800", "#cc3311"]):
        sub = [r for r in stab if int(r["checkpoint"]) == ck]
        ps = [r["pct_std"] for r in sub]
        ax.hist(ps, bins=40, alpha=0.5, label=f"it{ck}", color=col, density=True)
    ax.set_xlabel("percentile_std across 20 subsets"); ax.set_ylabel("density")
    ax.set_title("rank stability distribution")
    ax.legend()
    fig.savefig(f"{BASE}/plots/rank_stability.png", dpi=130, bbox_inches="tight"); plt.close()

    # 2 agreement distribution
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ck, col in zip(cks, ["#4477aa", "#228833", "#cc8800", "#cc3311"]):
        sub = [r for r in agree if int(r["checkpoint"]) == ck]
        eff = [r["effective_view_count"] for r in sub]
        ax.hist(eff, bins=40, alpha=0.5, label=f"it{ck}", color=col, density=True)
    ax.set_xlabel("effective_view_count"); ax.set_ylabel("density")
    ax.set_title("agreement distribution")
    ax.legend()
    fig.savefig(f"{BASE}/plots/agreement_distribution.png", dpi=130, bbox_inches="tight"); plt.close()

    # 3 score vs agreement
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ck_last = cks[-1]
    sub = [r for r in agree if int(r["checkpoint"]) == ck_last]
    imp = [r["importance_score"] for r in sub]
    eff = [r["effective_view_count"] for r in sub]
    ax.scatter(imp, eff, s=3, alpha=0.3, color="#4477aa")
    ax.set_xlabel("FastGS importance score"); ax.set_ylabel("effective_view_count")
    ax.set_title(f"score vs agreement (it{ck_last})")
    fig.savefig(f"{BASE}/plots/score_vs_agreement.png", dpi=130, bbox_inches="tight"); plt.close()

    # 4 stability vs agreement
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ck_last = cks[-1]
    ss = {r["gaussian_id"]: r for r in stab if int(r["checkpoint"]) == ck_last}
    aa = [r for r in agree if int(r["checkpoint"]) == ck_last]
    ids = [r["gaussian_id"] for r in aa]
    ps = [ss[i]["pct_std"] for i in ids]
    sup = [r["support_ratio"] for r in aa]
    ax.scatter(ps, sup, s=3, alpha=0.3, color="#228833")
    ax.set_xlabel("percentile_std (instability)"); ax.set_ylabel("support_ratio")
    ax.set_title(f"stability vs agreement (it{ck_last})")
    fig.savefig(f"{BASE}/plots/stability_vs_agreement.png", dpi=130, bbox_inches="tight"); plt.close()

    # 5 removal quality drop
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for ck, col in zip(cks, ["#4477aa", "#228833", "#cc8800", "#cc3311"]):
        base = next((r for r in removal if int(r["checkpoint"]) == ck and r["group"] == "baseline"), None)
        if not base: continue
        for group, mk, ls in (("low_agreement", "o", "-"), ("high_agreement", "s", "--")):
            pcts, drops = [], []
            for pct in (1, 5, 10):
                r = next((x for x in removal if int(x["checkpoint"]) == ck
                          and x["group"] == group and int(x["pct_removed"]) == pct), None)
                if r:
                    pcts.append(pct); drops.append(r["psnr"] - base["psnr"])
            if pcts:
                ax.plot(pcts, drops, marker=mk, ls=ls, color=col,
                        label=f"it{ck} {group.split('_')[0]}", alpha=0.8)
    ax.set_xlabel("% removed"); ax.set_ylabel("PSNR drop (dB)")
    ax.set_title("removal quality drop")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(f"{BASE}/plots/removal_quality_drop.png", dpi=130, bbox_inches="tight"); plt.close()

    # 6 topk overlap
    fig, ax = plt.subplots(figsize=(6, 4))
    # recompute jaccard from raw scores isn't possible (only summary stats saved)
    # plot per-Gaussian CV distribution as proxy
    for ck, col in zip(cks, ["#4477aa", "#228833", "#cc8800", "#cc3311"]):
        sub = [r for r in stab if int(r["checkpoint"]) == ck]
        cv = sorted([r["cv_score"] for r in sub])
        ax.plot(np.arange(len(cv)) / len(cv), cv, color=col, label=f"it{ck}")
    ax.set_xlabel("CDF"); ax.set_ylabel("score CV")
    ax.set_title("per-Gaussian score CV across subsets")
    ax.legend()
    fig.savefig(f"{BASE}/plots/topk_overlap.png", dpi=130, bbox_inches="tight"); plt.close()

    with open(f"{BASE}/data/b11v_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
