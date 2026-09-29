#
# Paper B - B9-T analysis (host): temporal split demand from full-training
# curves. Time-to-quality (Q90/95/99 vs native final), final quality/cost,
# PSNR-vs-#GS, per-stage pattern.  python3 diagnostics/analyze_b9t.py
#

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b9_t_temporal_split_demand"
STRATS = ["native", "early3", "middle3", "late3", "always3"]
COL = {"native": "#777777", "early3": "#228833", "middle3": "#cc8800",
       "late3": "#aa3377", "always3": "#4477aa"}


def main():
    import csv as _csv
    curves = list(_csv.DictReader(open(f"{BASE}/data/training_curves.csv")))
    for r in curves:
        for k in r:
            if k not in ("strategy", "seed"):
                r[k] = float(r[k]) if r[k] not in ("", "NA") else None
    seeds = sorted({int(r["seed"]) for r in curves})
    by = defaultdict(dict)  # (strat, seed) -> {iter: row}
    for r in curves:
        by[(r["strategy"], int(r["seed"]))][int(r["iteration"])] = r

    out = ["## B9-T Temporal Split Demand（完整训练）\n"]

    # ---- final metrics ----
    fin_rows = []
    for st in STRATS:
        for sd in seeds:
            d = by[(st, sd)]
            last = max(d)
            r = d[last]
            fin_rows.append({"strategy": st, "seed": sd, "final_iter": last,
                             "test_psnr": r["test_psnr"], "test_ssim": r["test_ssim"],
                             "test_lpips": r["test_lpips"], "train_psnr": r["train_psnr"],
                             "n_gaussians": r["n_gaussians"], "wall_time_s": r["wall_time_s"]})
    with open(f"{BASE}/data/final_metrics.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(fin_rows[0].keys()))
        w.writeheader()
        w.writerows(fin_rows)

    out.append("### Final metrics（30k iters）\n```text")
    out.append(f"{'strategy':>9s} {'PSNR(mean±sd)':>18s} {'SSIM':>14s} {'LPIPS':>14s} "
               f"{'#GS':>10s} {'time(s)':>9s}")
    for st in STRATS:
        sub = [r for r in fin_rows if r["strategy"] == st]
        out.append(f"{st:>9s} {np.mean([r['test_psnr'] for r in sub]):>9.3f}±"
                   f"{np.std([r['test_psnr'] for r in sub]):.3f} "
                   f"{np.mean([r['test_ssim'] for r in sub]):>7.4f}±{np.std([r['test_ssim'] for r in sub]):.4f} "
                   f"{np.mean([r['test_lpips'] for r in sub]):>7.4f}±{np.std([r['test_lpips'] for r in sub]):.4f} "
                   f"{int(np.mean([r['n_gaussians'] for r in sub])):>10d} "
                   f"{np.mean([r['wall_time_s'] for r in sub]):>9.0f}")
    out.append("```\n")

    # ---- time-to-quality ----
    native_final = float(np.mean([r["test_psnr"] for r in fin_rows
                                  if r["strategy"] == "native"]))
    out.append(f"Native final PSNR (seed-mean) = {native_final:.4f}\n")
    out.append("### Time-to-quality（达到 native final 的 Q90/95/99 所需 wall-clock / iteration）\n```text")
    out.append(f"{'strategy':>9s} {'Q90 t(s)/it':>16s} {'Q95 t(s)/it':>16s} {'Q99 t(s)/it':>16s}")
    ttq = {}
    for st in STRATS:
        line = f"{st:>9s}"
        ttq[st] = {}
        for qn, q in (("Q90", 0.90), ("Q95", 0.95), ("Q99", 0.99)):
            tgt = native_final * 0 + (native_final - (1 - q) * (native_final - min(
                r["test_psnr"] for r in fin_rows)))  # q fraction of gap from min to native final
            # simpler & spec-consistent: fraction of native final PSNR
            tgt = q * native_final
            ts, its_ = [], []
            for sd in seeds:
                d = by[(st, sd)]
                hit_t = hit_i = None
                for it in sorted(d):
                    v = d[it]["test_psnr"]
                    if v is not None and v >= tgt:
                        hit_t, hit_i = d[it]["wall_time_s"], it
                        break
                if hit_t is not None:
                    ts.append(hit_t); its_.append(hit_i)
            ttq[st][qn] = (float(np.mean(ts)) if ts else None,
                           float(np.mean(its_)) if its_ else None)
            t_, i_ = ttq[st][qn]
            line += f" {('%.0f/%.0f' % (t_, i_)) if t_ else 'never':>16s}"
        out.append(line)
    out.append("```\n")

    # ---- PSNR at fixed early iterations (convergence speed) ----
    out.append("### 收敛速度：test PSNR @ 关键 iteration（seed-mean）\n```text")
    its_show = [1000, 2000, 3000, 5000, 8000, 12000, 20000, 30000]
    out.append(f"{'iter':>7s} " + " ".join(f"{st:>9s}" for st in STRATS))
    for it in its_show:
        line = f"{it:>7d}"
        for st in STRATS:
            vals = [by[(st, sd)][it]["test_psnr"] for sd in seeds
                    if it in by[(st, sd)] and by[(st, sd)][it]["test_psnr"] is not None]
            line += f" {np.mean(vals):>9.3f}" if vals else f" {'NA':>9s}"
        out.append(line)
    out.append("```\n")

    # ---- events per stage ----
    try:
        ev = list(_csv.DictReader(open(f"{BASE}/data/split_event_stats.csv")))
        out.append("### Stage 划分与事件统计（native 轨迹口径）\n```text")
        out.append("Early  = 1000 ~ 5500（10 events） · Middle = 5500 ~ 10000（9） · Late = 10000 ~ 14500（9）")
        native_ev = [e for e in ev if e["strategy"] == "native"]
        for lab, lo, hi in (("Early", 1000, 5500), ("Middle", 5500, 10000), ("Late", 10000, 14500)):
            sub = [e for e in native_ev if lo <= int(e["iteration"]) < hi]
            if sub:
                ch = sum(int(e["n_after"]) - int(e["n_before"]) for e in sub)
                out.append(f"{lab:>7s}: events {len(sub)} · split parents/事件 "
                           f"{int(np.mean([int(e['split_parents']) for e in sub])):.0f} avg · "
                           f"净增 GS {ch}")
        out.append("```\n")
    except FileNotFoundError:
        out.append("(event stats missing)\n")

    # ---- GO/NO-GO ----
    def g(st, key):
        return float(np.mean([r[key] for r in fin_rows if r["strategy"] == st]))

    e3_nat = g("early3", "test_psnr") - g("native", "test_psnr")
    m3_nat = g("middle3", "test_psnr") - g("native", "test_psnr")
    l3_nat = g("late3", "test_psnr") - g("native", "test_psnr")
    a3_nat = g("always3", "test_psnr") - g("native", "test_psnr")
    gs_cost = {st: g(st, "n_gaussians") / g("native", "n_gaussians") - 1 for st in STRATS}
    t90 = {st: ttq[st]["Q95"][0] for st in STRATS}
    out.append("### GO/NO-GO 检查\n```text")
    out.append(f"final ΔPSNR vs native: early3 {e3_nat:+.4f} · middle3 {m3_nat:+.4f} · "
               f"late3 {l3_nat:+.4f} · always3 {a3_nat:+.4f}")
    out.append(f"#GS 相对 native: " + " ".join(f"{st}:{gs_cost[st]:+.1%}" for st in STRATS))
    out.append(f"Q95 time-to-quality (s): " + " ".join(
        f"{st}:{t90[st]:.0f}" if t90[st] else f"{st}:never" for st in STRATS))
    out.append("```")

    # ---- plots ----
    os.makedirs(f"{BASE}/plots", exist_ok=True)

    def seedmean(st, key, xkey="iteration"):
        xs = sorted({it for sd in seeds for it in by[(st, sd)]})
        ys = []
        for it in xs:
            vals = [by[(st, sd)][it][key] for sd in seeds
                    if it in by[(st, sd)] and by[(st, sd)][it][key] is not None]
            ys.append(np.mean(vals) if vals else np.nan)
        return np.array(xs), np.array(ys)

    for key, name, ylab in (("test_psnr", "psnr_vs_iteration", "test PSNR"),
                            ("wall_time_s", "psnr_vs_time", None),
                            ("n_gaussians", "gs_vs_iteration", "#Gaussians")):
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        for st in STRATS:
            if name == "psnr_vs_time":
                # PSNR vs wall time: parametric
                ds = [by[(st, sd)] for sd in seeds]
                xs = sorted({it for d in ds for it in d})
                tm = np.array([np.mean([d[it]["wall_time_s"] for d in ds if it in d]) for it in xs])
                pm = np.array([np.mean([d[it]["test_psnr"] for d in ds if it in d]) for it in xs])
                ax.plot(tm / 60, pm, label=st, color=COL[st])
            else:
                xs, ys = seedmean(st, key)
                ax.plot(xs, ys, label=st, color=COL[st])
        ax.set_xlabel("wall-clock (min)" if name == "psnr_vs_time" else "iteration")
        ax.set_ylabel(ylab if ylab else "test PSNR")
        ax.set_title(name.replace("_", " "))
        if name != "gs_vs_iteration":
            ax.legend(fontsize=8)
        fig.savefig(f"{BASE}/plots/{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for st in STRATS:
        ds = [by[(st, sd)] for sd in seeds]
        xs = sorted({it for d in ds for it in d})
        gm = np.array([np.mean([d[it]["n_gaussians"] for d in ds if it in d]) for it in xs])
        pm = np.array([np.mean([d[it]["test_psnr"] for d in ds if it in d]) for it in xs])
        ax.plot(gm, pm, label=st, color=COL[st])
    ax.set_xlabel("#Gaussians"); ax.set_ylabel("test PSNR")
    ax.set_title("PSNR vs #GS")
    ax.legend(fontsize=8)
    fig.savefig(f"{BASE}/plots/psnr_vs_gs.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(len(STRATS))
    ps = [g(st, "test_psnr") for st in STRATS]
    gs = [g(st, "n_gaussians") / 1e3 for st in STRATS]
    ax.bar(x - 0.2, ps, 0.4, color="#4477aa", label="final PSNR")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, gs, 0.4, color="#cc8800", label="#GS (k)")
    ax.set_xticks(x); ax.set_xticklabels(STRATS, fontsize=9)
    ax.set_ylabel("PSNR"); ax2.set_ylabel("#GS (thousands)")
    ax.set_title("final quality vs cost")
    fig.savefig(f"{BASE}/plots/final_quality_cost.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for st in STRATS:
        vals = [ttq[st][q][0] / 60 for q in ("Q90", "Q95", "Q99") if ttq[st][q][0]]
        qs = [q for q in ("Q90", "Q95", "Q99") if ttq[st][q][0]]
        ax.plot(qs, vals, marker="o", label=st, color=COL[st])
    ax.set_xlabel("quality target"); ax.set_ylabel("time (min)")
    ax.set_title("time to quality")
    ax.legend(fontsize=8)
    fig.savefig(f"{BASE}/plots/time_to_quality.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(f"{BASE}/data/b9t_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
