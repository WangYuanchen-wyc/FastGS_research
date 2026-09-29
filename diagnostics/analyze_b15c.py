#
# Paper B - B15-C analysis (host).  python3 diagnostics/analyze_b15c.py
#
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

BASE = "paper_b/b15_c_evidence_confidence"
RATIOS = [5, 10, 20]
N_PERMS = 20
VCS = [10, 20, 50, 100, 311]


def main():
    conf = list(csv.DictReader(open(f"{BASE}/data/confidence_features.csv")))
    for r in conf:
        for k, v in r.items():
            if v == "True": r[k] = 1.0
            elif v == "False": r[k] = 0.0
            else:
                try: r[k] = float(v)
                except: pass
    out = ["## B15-C Evidence Sufficiency / Decision Confidence\n"]

    # ---- flip rate ----
    out.append("### 1) Flip rate（10-view prune → All-view keep）\n```text")
    for pct in RATIOS:
        sub = [r for r in conf if int(r["ratio"]) == pct]
        n = len(sub)
        flips = sum(1 for r in sub if r["stable_at_all"] == 0.0)
        out.append(f"  {pct}%: {flips}/{n} = {flips/n*100:.1f}% flip")
    out.append("```\n")

    # ---- stable vs unstable features ----
    feats = ["vis_views_10v", "evi_views_10v", "total_evidence", "max_view_fraction",
             "evidence_mean", "evidence_std", "evidence_cv",
             "pct_rank_10v", "dist_to_boundary"]
    out.append("### 2) Stable vs Unstable confidence features（@10%）\n```text")
    out.append(f"{'feature':>20s} {'Stable':>10s} {'Unstable':>10s} {'diff':>10s} {'effect_d':>9s}")
    sub10 = [r for r in conf if int(r["ratio"]) == 10]
    S = [r for r in sub10 if r["stable_at_all"] == 1.0]
    U = [r for r in sub10 if r["stable_at_all"] == 0.0]
    for f in feats:
        sv = [r[f] for r in S if isinstance(r.get(f), (int, float))]
        uv = [r[f] for r in U if isinstance(r.get(f), (int, float))]
        if sv and uv:
            m_s, m_u = np.mean(sv), np.mean(uv)
            sd = np.sqrt((np.std(sv)**2 + np.std(uv)**2) / 2)
            d = (m_s - m_u) / sd if sd > 0 else 0
            out.append(f"{f:>20s} {m_s:>10.4f} {m_u:>10.4f} {m_s-m_u:>+10.4f} {d:>+9.2f}")
    out.append(f"(Stable n={len(S)}, Unstable n={len(U)})")
    out.append("```\n")

    # ---- simple confidence split ----
    out.append("### 3) Low vs High confidence flip rate\n```text")
    for pct in RATIOS:
        sub = [r for r in conf if int(r["ratio"]) == pct]
        # split by dist_to_boundary median (negative = deeper in prune zone = more confident)
        dvals = [r["dist_to_boundary"] for r in sub]
        med = np.median(dvals)
        hi = [r for r in sub if r["dist_to_boundary"] <= med]  # deeper = high conf
        lo = [r for r in sub if r["dist_to_boundary"] > med]   # near boundary = low conf
        flip_hi = sum(1 for r in hi if r["stable_at_all"] == 0.0)
        flip_lo = sum(1 for r in lo if r["stable_at_all"] == 0.0)
        out.append(f"  {pct}%: HiConf flip {flip_hi}/{len(hi)} ({flip_hi/len(hi)*100:.1f}%) | "
                   f"LoConf flip {flip_lo}/{len(lo)} ({flip_lo/len(lo)*100:.1f}%) | "
                   f"ratio {flip_lo/max(flip_hi,1):.1f}x")

        # also try total_evidence as confidence
        evals = [r["total_evidence"] for r in sub]
        ev_med = np.median(evals)
        hi_ev = [r for r in sub if r["total_evidence"] <= ev_med]  # low evi = confident prune
        lo_ev = [r for r in sub if r["total_evidence"] > ev_med]
        flip_he = sum(1 for r in hi_ev if r["stable_at_all"] == 0.0)
        flip_le = sum(1 for r in lo_ev if r["stable_at_all"] == 0.0)
        if hi_ev and lo_ev:
            out.append(f"        (by evidence): HiConf {flip_he}/{len(hi_ev)} "
                       f"({flip_he/len(hi_ev)*100:.1f}%) | LoConf {flip_le}/{len(lo_ev)} "
                       f"({flip_le/len(lo_ev)*100:.1f}%)")
        else:
            out.append("        (by evidence): one group empty")
    out.append("```\n")

    # ---- GO/NO-GO ----
    out.append("### GO/NO-GO\n```text")
    best_d = 0; best_f = ""
    for f in feats:
        sv = [r[f] for r in S if isinstance(r.get(f), (int, float))]
        uv = [r[f] for r in U if isinstance(r.get(f), (int, float))]
        if sv and uv:
            sd = np.sqrt((np.std(sv)**2 + np.std(uv)**2) / 2)
            d = abs((np.mean(sv) - np.mean(uv)) / sd) if sd > 0 else 0
            if d > best_d: best_d = d; best_f = f
    out.append(f"Best discriminator: {best_f} (effect d={best_d:.2f})")
    out.append("```")

    # ---- plots ----
    os.makedirs(f"{BASE}/plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (f, xl) in enumerate([("dist_to_boundary", "distance to boundary (percentile)"),
                                  ("total_evidence", "total 10-view evidence (log)")]):
        for grp, col, lbl in ((S, "#228833", "Stable"), (U, "#cc3311", "Unstable")):
            vals = [r[f] for r in grp if isinstance(r.get(f), (int, float))]
            if f == "total_evidence":
                vals = [max(v, 1e-8) for v in vals]
            axes[i].hist(vals, bins=50, alpha=0.5, density=True, color=col, label=lbl)
        axes[i].set_xlabel(xl); axes[i].set_ylabel("density")
        axes[i].set_title(f)
        axes[i].legend()
    fig.savefig(f"{BASE}/plots/stable_vs_unstable.png", dpi=130, bbox_inches="tight")
    plt.close()

    with open(f"{BASE}/data/b15c_stats.txt", "w") as f_:
        f_.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
