#
# Paper B - B4 analysis: Who / How / Joint gains, rho-scaling, quality-#GS
# frontier, and rho=1 consistency vs B2-C. Host python3.
#
#   python3 diagnostics/analyze_b4.py [K100|K100,K300]
#

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b4_structural_capacity_oracle"
B2C = "paper_b/b2_c_oracle_scalability"
PCOL = {"keep_all": "#777777", "native_full": "#cc8800", "ow_nh": "#cc3311",
        "ow_oh": "#228833", "rw_nh": "#4477aa", "rw_oh": "#66ccee"}
PLAB = {"keep_all": "Keep-All", "native_full": "Native-Full", "ow_nh": "OW-NH",
        "ow_oh": "OW-OH", "rw_nh": "RW-NH", "rw_oh": "RW-OH"}


def sel(rs, K, s, p, rho=None):
    return [r for r in rs if r["K"] == K and r["group_seed"] == s
            and r["policy"] == p and (rho is None or r["rho"] == rho)]


def ms(v):
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)


def main():
    with open(f"{BASE}/data/b4_group_results.json") as f:
        G = json.load(f)
    rs = G["records"]
    Ks = sorted({r["K"] for r in rs})
    seeds = sorted({r["group_seed"] for r in rs})
    rhos = sorted({r["rho"] for r in rs if r["policy"] not in ("keep_all", "native_full")})
    out = ["## B4 统计输出（自动生成）\n"]
    out.append(f"records={len(rs)} Ks={Ks} seeds={seeds} rhos={rhos} N0={G['n_before']}\n")

    # ---------------- gains per (K, rho) ----------------
    out.append("### Who / How / Joint gains（L1 越低越好；gain>0 = 更好）\n")
    out.append("```text")
    out.append(f"{'K':>4s} {'rho':>5s} {'Who(RW-NH−OW-NH)':>20s} {'How(RW-NH−RW-OH)':>20s} "
               f"{'Joint(RW-NH−OW-OH)':>20s} {'RW-NH mean L1':>14s}")
    gain_data = {}
    for K in Ks:
        for rho in rhos:
            who, how, joint, rwnh_l1 = [], [], [], []
            for s in seeds:
                l_ow_nh = sel(rs, K, s, "ow_nh", rho)[0]["demand_l1_100"]
                l_ow_oh = sel(rs, K, s, "ow_oh", rho)[0]["demand_l1_100"]
                rw_nh = [r["demand_l1_100"] for r in sel(rs, K, s, "rw_nh", rho)]
                rw_oh = [r["demand_l1_100"] for r in sel(rs, K, s, "rw_oh", rho)]
                m_rw = float(np.mean(rw_nh))
                who.append(m_rw - l_ow_nh)
                joint.append(m_rw - l_ow_oh)
                how += [a - b for a, b in zip(rw_nh, rw_oh)]  # paired per repeat
                rwnh_l1.append(m_rw)
            g = {k: ms(v) for k, v in (("who", who), ("how", how), ("joint", joint))}
            gain_data[(K, rho)] = g
            out.append(f"{K:>4d} {rho:>5.2f} {g['who'][0]:>+12.6f}±{g['who'][1]:.6f} "
                       f"{g['how'][0]:>+12.6f}±{g['how'][1]:.6f} "
                       f"{g['joint'][0]:>+12.6f}±{g['joint'][1]:.6f} "
                       f"{np.mean(rwnh_l1):>14.6f}")
    out.append("```\n")

    # ---------------- policy table per rho ----------------
    out.append("### Policy 汇总（3 seeds 均值）\n")
    for K in Ks:
        out.append(f"#### K = {K}\n")
        out.append("| rho | M | RW-NH L1 | RW-OH L1 | OW-NH L1 | OW-OH L1 | OW-OH PSNR | OW-OH gPSNR | ΔTile@0 |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for rho in rhos:
            M = sel(rs, K, seeds[0], "ow_oh", rho)[0]["M"]
            vals = {}
            for p in ("rw_nh", "rw_oh", "ow_nh", "ow_oh"):
                per_seed = [np.mean([r["demand_l1_100"] for r in sel(rs, K, s, p, rho)])
                            for s in seeds]
                vals[p] = np.mean(per_seed)
            dpsnr = np.mean([sel(rs, K, s, "ow_oh", rho)[0]["demand_psnr_100"] for s in seeds])
            gpsnr = np.mean([sel(rs, K, s, "ow_oh", rho)[0]["global_psnr_100"] for s in seeds])
            dt = np.mean([sel(rs, K, s, "ow_oh", rho)[0]["dTile_0"] for s in seeds])
            out.append(f"| {rho:.2f} | {M} | {vals['rw_nh']:.6f} | {vals['rw_oh']:.6f} | "
                       f"{vals['ow_nh']:.6f} | {vals['ow_oh']:.6f} | {dpsnr:.3f} | "
                       f"{gpsnr:.4f} | {dt:+.1f} |")
        keep = np.mean([sel(rs, K, s, "keep_all")[0]["demand_l1_100"] for s in seeds])
        nfull = np.mean([sel(rs, K, s, "native_full")[0]["demand_l1_100"] for s in seeds])
        kg = np.mean([sel(rs, K, s, "keep_all")[0]["global_psnr_100"] for s in seeds])
        ng = np.mean([sel(rs, K, s, "native_full")[0]["global_psnr_100"] for s in seeds])
        out.append(f"\nKeep-All L1={keep:.6f} gPSNR={kg:.4f} · Native-Full L1={nfull:.6f} "
                   f"gPSNR={ng:.4f}\n")

    # ---------------- frontier ----------------
    out.append("### Frontier：OW-OH@rho vs Native-Full@1.0（同 K）\n```text")
    for K in Ks:
        nf = np.mean([sel(rs, K, s, "native_full")[0]["demand_l1_100"] for s in seeds])
        nfg = np.mean([sel(rs, K, s, "native_full")[0]["global_psnr_100"] for s in seeds])
        for rho in rhos:
            if rho >= 1.0:
                continue
            oo = np.mean([sel(rs, K, s, "ow_oh", rho)[0]["demand_l1_100"] for s in seeds])
            oog = np.mean([sel(rs, K, s, "ow_oh", rho)[0]["global_psnr_100"] for s in seeds])
            M = sel(rs, K, seeds[0], "ow_oh", rho)[0]["M"]
            out.append(f"K={K} rho={rho:.2f}: OW-OH L1 {oo:.6f} vs Native-Full {nf:.6f} "
                       f"(ΔL1 {oo-nf:+.6f}, 更优={oo < nf}) | gPSNR {oog:.4f} vs {nfg:.4f} "
                       f"(Δ{oog-nfg:+.4f}) | growth {M}/{K}")
    out.append("```\n")

    # ---------------- rho=1 consistency vs B2-C ----------------
    try:
        with open(f"{B2C}/data/b2c_group_results.json") as f:
            B = json.load(f)["records"]
        out.append("### rho=1 与 B2-C 一致性（CUDA 原子非确定性 → 允许 ~1e-4 级差异）\n```text")
        for K in [k for k in Ks if k in (30, 100, 300)]:
            for s in seeds:
                oo = sel(rs, K, s, "ow_oh", 1.0)[0]["demand_l1_100"]
                b2c_om = [r["group_demand_l1_100"] for r in B
                          if r["K"] == K and r["group_seed"] == s and r["policy"] == "oracle_mix"]
                nf = sel(rs, K, s, "native_full")[0]["demand_l1_100"]
                b2c_na = [r["group_demand_l1_100"] for r in B
                          if r["K"] == K and r["group_seed"] == s and r["policy"] == "native"]
                if b2c_om and b2c_na:
                    out.append(f"K={K} seed={s}: OW-OH@1 {oo:.6f} vs B2C oracle_mix {b2c_om[0]:.6f} "
                               f"(Δ{oo-b2c_om[0]:+.6f}) | Native-Full {nf:.6f} vs B2C native "
                               f"{b2c_na[0]:.6f} (Δ{nf-b2c_na[0]:+.6f})")
        out.append("```\n")
    except Exception as e:
        out.append(f"(B2C consistency check skipped: {e})\n")

    # ---------------- plots ----------------
    os.makedirs(f"{BASE}/plots", exist_ok=True)

    def save(fig, name):
        fig.savefig(f"{BASE}/plots/{name}", dpi=130, bbox_inches="tight")
        plt.close(fig)

    K = Ks[-1]  # main plots on the largest K
    # 1 quality vs rho
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for p, mk in (("rw_nh", "o"), ("rw_oh", "s"), ("ow_nh", "^"), ("ow_oh", "D")):
        xs, ys, es = [], [], []
        for rho in rhos:
            per_seed = [np.mean([r["demand_l1_100"] for r in sel(rs, K, s, p, rho)]) for s in seeds]
            xs.append(rho); ys.append(np.mean(per_seed)); es.append(np.std(per_seed))
        ax.errorbar(xs, ys, yerr=es, marker=mk, capsize=3, label=PLAB[p], color=PCOL[p])
    keep = np.mean([sel(rs, K, s, "keep_all")[0]["demand_l1_100"] for s in seeds])
    ax.axhline(keep, color=PCOL["keep_all"], ls="--", lw=1, label="Keep-All")
    nf = np.mean([sel(rs, K, s, "native_full")[0]["demand_l1_100"] for s in seeds])
    ax.axhline(nf, color=PCOL["native_full"], ls=":", lw=1.5, label="Native-Full")
    ax.set_xlabel("rho (= M/K added-capacity fraction)")
    ax.set_ylabel("group demand L1 @100 (lower better)")
    ax.set_title(f"quality vs growth ratio (K={K})")
    ax.legend(fontsize=8)
    save(fig, "quality_vs_growth_ratio.png")

    # 2 quality vs #GS (all K)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for p in ("rw_nh", "rw_oh", "ow_nh", "ow_oh", "native_full", "keep_all"):
        xs = [r["num_gaussians_after"] for r in rs if r["policy"] == p]
        ys = [r["demand_l1_100"] for r in rs if r["policy"] == p]
        ax.scatter(xs, ys, s=16, alpha=0.7, color=PCOL[p], label=PLAB[p])
    ax.set_xlabel("#GS after action (= N0 + M)")
    ax.set_ylabel("group demand L1 @100")
    ax.set_title("quality-#GS frontier (all K/rho/seeds)")
    ax.legend(fontsize=8)
    save(fig, "quality_vs_num_gaussians.png")

    # 3-5 gains vs rho
    for key, name, ttl in (("who", "quantity_gain_vs_rho", "Who Gain = L1(RW-NH)−L1(OW-NH)"),
                           ("how", "type_gain_vs_rho", "How Gain = L1(RW-NH)−L1(RW-OH) (paired)"),
                           ("joint", "joint_gain_vs_rho", "Joint Gain = L1(RW-NH)−L1(OW-OH)")):
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        for Kp in Ks:
            xs = [rho for rho in rhos]
            ys = [gain_data[(Kp, rho)][key][0] for rho in rhos]
            es = [gain_data[(Kp, rho)][key][1] for rho in rhos]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"K={Kp}")
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel("rho"); ax.set_ylabel(ttl)
        ax.set_title(name.replace("_", " "))
        ax.legend()
        save(fig, name + ".png")

    # 6 quality vs tile workload
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for p in ("rw_nh", "rw_oh", "ow_nh", "ow_oh", "native_full"):
        xs = [r["dTile_0"] for r in rs if r["policy"] == p and r.get("dTile_0") is not None]
        ys = [r["demand_l1_100"] for r in rs if r["policy"] == p and r.get("dTile_0") is not None]
        ax.scatter(xs, ys, s=16, alpha=0.7, color=PCOL[p], label=PLAB[p])
    ax.set_xlabel("ΔTile @0 vs Keep (structural workload)")
    ax.set_ylabel("group demand L1 @100")
    ax.set_title("quality vs tile workload")
    ax.legend(fontsize=8)
    save(fig, "quality_vs_tile_workload.png")

    with open(f"{BASE}/data/b4_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[b4a] stats -> {BASE}/data/b4_stats.txt ; plots -> {BASE}/plots/")


if __name__ == "__main__":
    main()
