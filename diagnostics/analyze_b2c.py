#
# Paper B - B2-C analysis: Oracle-Mix vs Native / All-Clone / All-Split /
# Shuffled-Mix, personalization gain vs K, native-vs-oracle binary confusion.
# Host python3 (numpy/matplotlib).  python3 diagnostics/analyze_b2c.py
#

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b2_c_oracle_scalability"
PCOL = {"keep": "#777777", "native": "#cc8800", "all_clone": "#cc3311",
        "all_split": "#4477aa", "oracle_mix": "#228833", "shuffled": "#aa3377"}
PLAB = {"keep": "Keep", "native": "Native", "all_clone": "All-Clone",
        "all_split": "All-Split", "oracle_mix": "Oracle-Mix", "shuffled": "Shuffled-Mix"}


def m(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else float("nan")


def main():
    with open(f"{BASE}/data/b2c_group_results.json") as f:
        G = json.load(f)
    with open(f"{BASE}/data/b2c_candidate_oracles.json") as f:
        O = json.load(f)
    recs, oracles = G["records"], O["oracles"]
    Ks = sorted({r["K"] for r in recs})
    seeds = sorted({r["group_seed"] for r in recs})
    out = ["## B2-C 统计输出（自动生成）\n"]

    def sel(K, s, p):
        return [r for r in recs if r["K"] == K and r["group_seed"] == s and r["policy"] == p]

    def agg(K, p, key):
        vals = []
        for s in seeds:
            sub = sel(K, s, p)
            if p == "shuffled":
                v = [r[key] for r in sub if r.get(key) is not None]
                if v:
                    vals.append(np.mean(v))
            else:
                for r in sub:
                    if r.get(key) is not None:
                        vals.append(r[key])
        return m(vals)

    # ---------------- section 29 table ----------------
    out.append("### 每 K Policy 汇总（3 seeds 均值；Shuffled 为 5 repeats 均值后跨 seed 平均）\n")
    for K in Ks:
        out.append(f"#### K = {K}\n")
        out.append("| Policy | GroupDemand L1@100 | GroupDemand PSNR@100 | Global PSNR@100 | ΔTile@0 | ΔTile@100 | Δ#GS |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for p in ("keep", "native", "all_clone", "all_split", "oracle_mix", "shuffled"):
            out.append(f"| {PLAB[p]}{' mean' if p=='shuffled' else ''} | "
                       f"{agg(K,p,'group_demand_l1_100'):.6f} | "
                       f"{agg(K,p,'group_demand_psnr_100'):.3f} | "
                       f"{agg(K,p,'global_psnr_100'):.4f} | "
                       f"{agg(K,p,'dTile_0'):+.2f} | {agg(K,p,'dTile_100'):+.2f} | "
                       f"{agg(K,p,'delta_num_gaussians'):+.0f} |")
        # within-seed shuffled std (mean over seeds of per-seed std of the 5 repeats)
        stds = []
        for s in seeds:
            v = [r["group_demand_l1_100"] for r in sel(K, s, "shuffled")
                 if r.get("group_demand_l1_100") is not None]
            if len(v) >= 2:
                stds.append(np.std(v))
        out.append(f"\nShuffled within-seed std (mean over seeds): {np.mean(stds):.6f}\n")

    # ---------------- oracle stability ----------------
    gaps = np.array([o["oracle_gap"] for o in oracles if o.get("oracle_gap") is not None], float)
    sems = np.array([o["split_sem"] for o in oracles if o.get("split_sem") is not None], float)
    n = len(oracles)
    n_clone = sum(1 for o in oracles if o.get("oracle_action") == "clone")
    n_split = sum(1 for o in oracles if o.get("oracle_action") == "split")
    n_none = n - n_clone - n_split
    n_hc = sum(1 for o in oracles if o.get("oracle_high_confident"))
    out.append("### Oracle stability\n```text")
    out.append(f"candidate count        : {n}")
    out.append(f"oracle clone %         : {n_clone/n*100:.1f}%   split %: {n_split/n*100:.1f}%   no-oracle(None): {n_none} ({n_none/n*100:.1f}%)")
    out.append(f"high-confidence %      : {n_hc/n*100:.1f}%")
    out.append(f"median |oracle gap|    : {np.median(np.abs(gaps)):.6f}")
    out.append(f"median split SEM       : {np.median(sems):.6f}")
    out.append(f"gap>0 (split better) % : {np.mean(gaps>0)*100:.1f}%   <0 (clone better) %: {np.mean(gaps<0)*100:.1f}%")
    out.append("```\n")

    # ---------------- native vs oracle binary ----------------
    conf = {a: {o: 0 for o in ("clone", "split")} for a in ("clone", "split")}
    for o in oracles:
        if o.get("oracle_action") in ("clone", "split"):
            conf[o["native_action"]][o["oracle_action"]] += 1
    tot = sum(conf[a][o] for a in conf for o in conf[a])
    agr = conf["clone"]["clone"] + conf["split"]["split"]
    out.append("### Native vs Oracle（binary，无 Keep）\n```text")
    out.append(f"binary agreement rate: {agr}/{tot} = {agr/tot*100:.1f}%")
    out.append("confusion (row=native, col=oracle):")
    out.append(f"{'':16s}{'oracle clone':>14s}{'oracle split':>14s}")
    for a in ("clone", "split"):
        out.append(f"native {a:9s}{conf[a]['clone']:14d}{conf[a]['split']:14d}")
    out.append("```\n")

    # ---------------- personalization gains ----------------
    out.append("### Personalization（Oracle-Mix vs 对照；跨 3 seeds mean±std）\n```text")
    out.append(f"{'K':>5s} {'pg_L1(shuf)':>16s} {'pg_PSNR(shuf)':>15s} {'dQ(native)':>14s} "
               f"{'dTile(native)':>15s} {'dQ(allclone)':>14s} {'dQ(allsplit)':>14s}")
    pg_stats = {}
    for K in Ks:
        pgL, pgP, dqn, dtn, dqc, dqs = [], [], [], [], [], []
        for s in seeds:
            om = sel(K, s, "oracle_mix")[0]
            sh = [r for r in sel(K, s, "shuffled") if r.get("group_demand_l1_100") is not None]
            nat = sel(K, s, "native")[0]
            ac = sel(K, s, "all_clone")[0]
            asp = sel(K, s, "all_split")[0]
            if sh and om.get("group_demand_l1_100") is not None:
                pgL.append(np.mean([r["group_demand_l1_100"] for r in sh]) - om["group_demand_l1_100"])
            if sh and om.get("group_demand_psnr_100") is not None:
                pgP.append(om["group_demand_psnr_100"] - np.mean([r["group_demand_psnr_100"] for r in sh]))
            if nat.get("group_demand_l1_100") is not None:
                dqn.append(nat["group_demand_l1_100"] - om["group_demand_l1_100"])
                dtn.append(om["dTile_0"] - nat["dTile_0"])
            dqc.append(ac["group_demand_l1_100"] - om["group_demand_l1_100"])
            dqs.append(asp["group_demand_l1_100"] - om["group_demand_l1_100"])
        def ms(v):
            return f"{np.mean(v):+.6f}±{np.std(v):.6f}" if v else "NA"
        pg_stats[K] = (pgL, pgP)
        out.append(f"{K:>5d} {ms(pgL):>16s} {ms(pgP):>15s} {ms(dqn):>14s} "
                   f"{ms(dtn):>15s} {ms(dqc):>14s} {ms(dqs):>14s}")
    out.append("（pg_L1 = shuffled_mean − oracle，>0 = oracle 更好；dQ(x) = x − oracle，>0 = oracle 更好）\n")

    # oracle-vs-shuffled per-seed detail
    out.append("### Oracle vs Shuffled per-seed（group_demand_L1@100）\n```text")
    for K in Ks:
        for s in seeds:
            om = sel(K, s, "oracle_mix")[0]["group_demand_l1_100"]
            sh = [r["group_demand_l1_100"] for r in sel(K, s, "shuffled")]
            out.append(f"K={K:>3d} seed={s}: oracle={om:.6f} shuffled={np.mean(sh):.6f} "
                       f"(min {np.min(sh):.6f} max {np.max(sh):.6f})")
    out.append("```\n")

    # ---------------- plots ----------------
    os.makedirs(f"{BASE}/plots", exist_ok=True)

    def save(fig, name):
        fig.savefig(f"{BASE}/plots/{name}", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 1 oracle vs shuffled quality vs K
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for p, mk in (("oracle_mix", "o"), ("shuffled", "s")):
        xs, ys, es = [], [], []
        for K in Ks:
            per_seed = []
            for s in seeds:
                sub = sel(K, s, p)
                v = [r["group_demand_l1_100"] for r in sub if r.get("group_demand_l1_100") is not None]
                if v:
                    per_seed.append(np.mean(v))
            xs.append(K); ys.append(np.mean(per_seed)); es.append(np.std(per_seed))
        ax.errorbar(xs, ys, yerr=es, marker=mk, capsize=4, label=PLAB[p], color=PCOL[p])
    ax.set_xlabel("K"); ax.set_ylabel("group demand L1 @100 (lower better)")
    ax.set_title("Oracle-Mix vs Shuffled-Mix")
    ax.legend()
    save(fig, "oracle_vs_shuffled_quality_vs_K.png")

    # 2 personalization gain vs K
    fig, ax = plt.subplots(figsize=(6, 4.2))
    xs = [K for K in Ks if pg_stats[K][0]]
    ys = [np.mean(pg_stats[K][0]) for K in xs]
    es = [np.std(pg_stats[K][0]) for K in xs]
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, color=PCOL["oracle_mix"])
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("K"); ax.set_ylabel("personalization_gain = shuffled_mean − oracle")
    ax.set_title("personalization gain vs K (>0 = oracle better)")
    save(fig, "personalization_gain_vs_K.png")

    # 3 policy quality vs K
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for p in ("keep", "native", "all_clone", "all_split", "oracle_mix", "shuffled"):
        xs, ys, es = [], [], []
        for K in Ks:
            per_seed = []
            for s in seeds:
                sub = sel(K, s, p)
                v = [r["group_demand_l1_100"] for r in sub if r.get("group_demand_l1_100") is not None]
                if v:
                    per_seed.append(np.mean(v))
            xs.append(K); ys.append(np.mean(per_seed)); es.append(np.std(per_seed))
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=PLAB[p], color=PCOL[p])
    ax.set_xlabel("K"); ax.set_ylabel("group demand L1 @100")
    ax.set_title("policy quality vs K")
    ax.legend(fontsize=8)
    save(fig, "policy_quality_vs_K.png")

    # 4 policy tile cost vs K
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for p in ("native", "all_clone", "all_split", "oracle_mix", "shuffled"):
        xs, ys, es = [], [], []
        for K in Ks:
            per_seed = []
            for s in seeds:
                sub = sel(K, s, p)
                v = [r["dTile_0"] for r in sub if r.get("dTile_0") is not None]
                if v:
                    per_seed.append(np.mean(v))
            xs.append(K); ys.append(np.mean(per_seed)); es.append(np.std(per_seed))
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=PLAB[p], color=PCOL[p])
    ax.set_xlabel("K"); ax.set_ylabel("ΔTile @0 vs Keep")
    ax.set_title("policy structural tile cost vs K")
    ax.legend(fontsize=8)
    save(fig, "policy_tile_cost_vs_K.png")

    # 5 quality vs tile cost
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for p in ("keep", "native", "all_clone", "all_split", "oracle_mix", "shuffled"):
        xs = [r["dTile_0"] for r in recs if r["policy"] == p and r.get("dTile_0") is not None]
        ys = [r["group_demand_l1_100"] for r in recs if r["policy"] == p
              and r.get("dTile_0") is not None]
        ax.scatter(xs, ys, s=30, alpha=0.8, color=PCOL[p], label=PLAB[p])
    ax.set_xlabel("ΔTile @0 vs Keep"); ax.set_ylabel("group demand L1 @100")
    ax.set_title("quality vs structural tile cost (all K/seeds)")
    ax.legend(fontsize=8)
    save(fig, "policy_quality_vs_tile_cost.png")

    # 6 confusion
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    mat = np.array([[conf["clone"]["clone"], conf["clone"]["split"]],
                    [conf["split"]["clone"], conf["split"]["split"]]])
    im = ax.imshow(mat, cmap="Greens")
    ax.set_xticks(range(2)); ax.set_xticklabels(["oracle clone", "oracle split"])
    ax.set_yticks(range(2)); ax.set_yticklabels(["native clone", "native split"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_title(f"binary agreement {agr/tot*100:.1f}% (n={tot})")
    fig.colorbar(im)
    save(fig, "native_vs_oracle_binary_confusion.png")

    # 7 oracle gap distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(gaps, bins=40, color=PCOL["oracle_mix"], edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("oracle_gap = split_quality − clone_quality (>0 → oracle=split)")
    ax.set_ylabel("candidates")
    ax.set_title(f"oracle gap distribution (n={len(gaps)})")
    save(fig, "oracle_gap_distribution.png")

    with open(f"{BASE}/data/b2c_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[analyze_b2c] stats -> {BASE}/data/b2c_stats.txt ; plots -> {BASE}/plots/")


if __name__ == "__main__":
    main()
