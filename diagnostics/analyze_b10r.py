#
# Paper B - B10-R analysis (host).  python3 diagnostics/analyze_b10r.py
#

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b10_r_extra_capacity_removability"


def main():
    meta = list(csv.DictReader(open(f"{BASE}/data/b10r_final_models.csv")))
    pru = list(csv.DictReader(open(f"{BASE}/data/b10r_pruning_results.csv")))
    for r in pru:
        for k, v in r.items():
            if k not in ("strategy", "seed", "target"):
                r[k] = float(v) if v not in ("", "NA") else None

    out = ["## B10-R Extra Capacity Removability\n"]

    # native baseline per seed
    nat = {(int(m["seed"])): float(m["psnr"]) for m in meta if m["strategy"] == "native"}
    nat_gs = {int(m["seed"]): int(m["n_gaussians"]) for m in meta if m["strategy"] == "native"}

    out.append("### Final models\n```text")
    out.append(f"{'strategy':>9s} {'seed':>4s} {'#GS':>8s} {'PSNR':>8s} {'SSIM':>7s} {'LPIPS':>7s}")
    for m in meta:
        out.append(f"{m['strategy']:>9s} {m['seed']:>4s} {int(m['n_gaussians']):>8d} "
                   f"{float(m['psnr']):>8.3f} {float(m['ssim']):>7.4f} {float(m['lpips']):>7.4f}")
    out.append("```\n")

    out.append("### Pruning results（immediate / +500-step recovery）\n```text")
    out.append(f"{'strategy':>9s} {'seed':>4s} {'target':>8s} {'#GS':>8s} {'PSNR_imm':>9s} "
               f"{'PSNR_rec':>9s} {'SSIM_rec':>8s} {'LPIPS_rec':>8s} {'ΔPSNR_nat':>10s}")
    for r in pru:
        d_nat = (r["psnr_rec500"] - nat[int(r["seed"])]) if r["psnr_rec500"] else None
        out.append(f"{r['strategy']:>9s} {int(r['seed']):>4d} {r['target']:>8s} "
                   f"{int(r['n_after_prune']):>8d} {r['psnr_imm']:>9.3f} "
                   f"{r['psnr_rec500']:>9.3f} {r['ssim_rec500']:>8.4f} {r['lpips_rec500']:>8.4f} "
                   f"{d_nat:>+10.3f}" if d_nat is not None else "")
    out.append("```\n")

    # aggregate
    out.append("### 汇总（跨 seed 均值）\n```text")
    agg = defaultdict(list)
    for r in pru:
        agg[(r["strategy"], r["target"])].append(r)
    out.append(f"{'strategy':>9s} {'target':>8s} {'#GS':>8s} {'PSNR_imm':>9s} {'PSNR_rec':>9s} "
               f"{'Δ_imm':>7s} {'Δ_rec':>7s} {'vs native #GS':>13s}")
    for (st, tg), sub in sorted(agg.items()):
        n = np.mean([r["n_after_prune"] for r in sub])
        pi = np.mean([r["psnr_imm"] for r in sub])
        pr = np.mean([r["psnr_rec500"] for r in sub])
        # native PSNR mean across seeds
        nat_mean = np.mean([nat[int(r["seed"])] for r in sub])
        nat_n = np.mean([nat_gs[int(r["seed"])] for r in sub])
        out.append(f"{st:>9s} {tg:>8s} {int(n):>8d} {pi:>9.3f} {pr:>9.3f} "
                   f"{pi - nat_mean:>+7.3f} {pr - nat_mean:>+7.3f} {n / nat_n - 1:>+12.1%}")
    out.append("```\n")

    # GO/NO-GO
    best = {}
    for (st, tg), sub in agg.items():
        pr = np.mean([r["psnr_rec500"] for r in sub])
        best[(st, tg)] = pr
    nat_mean = np.mean(list(nat.values()))
    out.append("### GO/NO-GO\n```text")
    for st in ("early3", "always3"):
        for tg in ("A=native", "B=90%", "C=80%"):
            if (st, tg) in best:
                d = best[(st, tg)] - nat_mean
                out.append(f"{st:>9s} {tg:>8s}: ΔPSNR(rec500) vs native = {d:+.3f} dB "
                           f"({'quality maintained' if d > -0.1 else 'quality drop'})")
    out.append("```\n")

    # plots
    os.makedirs(f"{BASE}/plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for st, col in (("native", "#777777"), ("early3", "#228833"), ("always3", "#4477aa")):
        ms = [m for m in meta if m["strategy"] == st]
        ax.scatter([int(m["n_gaussians"]) for m in ms], [float(m["psnr"]) for m in ms],
                   s=80, color=col, label=f"{st} (trained)", zorder=3)
    for r in pru:
        ax.scatter([r["n_after_prune"]], [r["psnr_rec500"]], s=30, marker="s",
                   color="#228833" if r["strategy"] == "early3" else "#4477aa", alpha=0.7,
                   zorder=2)
        ax.scatter([r["n_after_prune"]], [r["psnr_imm"]], s=20, marker="x",
                   color="#cc3311", alpha=0.5, zorder=2)
    ax.set_xlabel("#Gaussians"); ax.set_ylabel("test PSNR")
    ax.set_title("quality vs #GS (■=pruned+recovery, ×=immediate)")
    ax.legend(fontsize=8)
    fig.savefig(f"{BASE}/plots/quality_vs_gs.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for st, col in (("early3", "#228833"), ("always3", "#4477aa")):
        for tg, mk in (("A=native", "o"), ("B=90%", "s"), ("C=80%", "^")):
            sub = [r for r in pru if r["strategy"] == st and r["target"] == tg]
            if sub:
                comp = [1 - r["n_after_prune"] / nat_gs[int(r["seed"])] for r in sub]
                drop = [nat[int(r["seed"])] - r["psnr_rec500"] for r in sub]
                ax.scatter(comp, drop, color=col, marker=mk, s=50,
                           label=f"{st} {tg}" if True else None)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("compression vs native #GS (fraction removed)")
    ax.set_ylabel("PSNR drop vs native (dB)")
    ax.set_title("PSNR drop vs compression (after 500-step recovery)")
    ax.legend(fontsize=7)
    fig.savefig(f"{BASE}/plots/psnr_drop_vs_compression.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for st, col in (("early3", "#228833"), ("always3", "#4477aa")):
        sub = [r for r in pru if r["strategy"] == st]
        ax.scatter([r["psnr_imm"] for r in sub], [r["psnr_rec500"] for r in sub],
                   s=40, color=col, label=st)
    lim = [min(r["psnr_imm"] for r in pru) - 0.5, max(r["psnr_rec500"] for r in pru) + 0.5]
    ax.plot(lim, lim, "k--", lw=1, label="y=x")
    ax.set_xlabel("PSNR immediate"); ax.set_ylabel("PSNR after 500-step recovery")
    ax.set_title("immediate vs recovery")
    ax.legend(fontsize=8)
    fig.savefig(f"{BASE}/plots/immediate_vs_recovery.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(f"{BASE}/data/b10r_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
