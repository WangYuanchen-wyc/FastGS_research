#
# Paper B - B5 analysis: cross-stage gains, per-iteration breakdown,
# gains vs snapshot state (#GS / candidates / split ratio), frontier per
# iteration. Host python3.  python3 diagnostics/analyze_b5.py
#

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b5_cross_stage_capacity_oracle"
PCOL = {"keep_all": "#777777", "native_full": "#cc8800", "ow_nh": "#cc3311",
        "ow_oh": "#228833", "rw_nh": "#4477aa", "rw_oh": "#66ccee"}


def main():
    import csv as _csv
    rs = json.load(open(f"{BASE}/data/b5_group_results.json"))["records"]
    stats = list(_csv.DictReader(open(f"{BASE}/data/b5_snapshot_stats.csv")))
    st = {int(s["iteration"]): {k: float(v) for k, v in s.items() if k != "iteration"}
          for s in stats}
    iters = sorted({r["iteration"] for r in rs})
    seeds = sorted({r["group_seed"] for r in rs})
    rhos = sorted({r["rho"] for r in rs if r["policy"] not in ("keep_all", "native_full")})
    out = ["## B5 统计输出（自动生成）\n"]

    # candidate oracle table from cache
    rows = []
    for f in glob.glob(f"{BASE}/cache/oracle_*.json"):
        o = json.load(open(f))
        if o.get("valid"):
            it, idx = f.split("oracle_")[1].removesuffix(".json").split("_")
            rows.append({"iteration": int(it), "parent_index": o["parent_index"],
                         "native_action": o["native_action"], "q_clone": o["q_clone"],
                         "q_split_mean": o["q_split_mean"], "q_split_std": o["q_split_std"],
                         "q_best": o["q_best"], "oracle_action": o["oracle_action"]})
    rows.sort(key=lambda r: (r["iteration"], r["parent_index"]))
    with open(f"{BASE}/data/b5_candidate_oracles.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def sel(it, s, p, rho=None):
        return [r for r in rs if r["iteration"] == it and r["group_seed"] == s
                and r["policy"] == p and (rho is None or r["rho"] == rho)]

    # ---------------- per-iteration gains ----------------
    # Statistics convention (B5-Fix #3): every gain is computed PER SEED;
    # HowR first averages the 5 paired repeats within each seed, then the two
    # seed means are aggregated (mean±std). Positive counts are reported at
    # BOTH levels, never mixed: seed-level = 20 comparisons (iter×rho×seed),
    # setting-level = 10 iteration×rho seed-means.
    out.append("### 每 iteration 增益（L1@100，>0 = 有收益；per-seed 值在 2 seeds 上 mean±std）\n")
    out.append("```text")
    out.append(f"{'iter':>6s} {'rho':>5s} {'WhoN':>18s} {'WhoO':>18s} {'HowR':>18s} {'HowO':>18s} {'Joint':>18s}")
    gains = {}
    seed_pos = {k: 0 for k in ("WhoN", "WhoO", "HowR", "HowO", "Joint")}
    set_pos = {k: 0 for k in ("WhoN", "WhoO", "HowR", "HowO", "Joint")}
    n_seed_events, n_settings = 0, 0
    for it in iters:
        for rho in rhos:
            per_seed = {k: [] for k in ("WhoN", "WhoO", "HowR", "HowO", "Joint")}
            for s in seeds:
                ow_nh = sel(it, s, "ow_nh", rho)[0]["demand_l1_100"]
                ow_oh = sel(it, s, "ow_oh", rho)[0]["demand_l1_100"]
                rwnh = [r["demand_l1_100"] for r in sel(it, s, "rw_nh", rho)]
                rwoh = [r["demand_l1_100"] for r in sel(it, s, "rw_oh", rho)]
                m = float(np.mean(rwnh))          # RW-NH seed mean
                mo = float(np.mean(rwoh))         # RW-OH seed mean (5 paired repeats)
                vals = {"WhoN": m - ow_nh, "WhoO": mo - ow_oh,
                        "HowR": m - mo,           # paired How within this seed (repeat-mean)
                        "HowO": ow_nh - ow_oh, "Joint": m - ow_oh}
                for k, v in vals.items():
                    per_seed[k].append(v)
                    seed_pos[k] += int(v > 0)
                n_seed_events += 1
            gains[(it, rho)] = per_seed
            for k in per_seed:
                if float(np.mean(per_seed[k])) > 0:
                    set_pos[k] += 1
            n_settings += 1
            out.append(f"{it:>6d} {rho:>5.2f} " + " ".join(
                f"{np.mean(per_seed[k]):>+11.6f}±{np.std(per_seed[k]):.6f}"
                for k in ("WhoN", "WhoO", "HowR", "HowO", "Joint")))
    out.append("```\n")
    out.append("```text")
    out.append(f"seed-level positive count（{n_seed_events} comparisons = 5 iter × 2 rho × 2 seeds）:")
    for k in ("WhoN", "WhoO", "HowR", "HowO", "Joint"):
        out.append(f"  {k}: {seed_pos[k]}/{n_seed_events} = {seed_pos[k]/n_seed_events*100:.0f}% positive")
    out.append(f"setting-level positive count（{n_settings} iteration×rho seed-means）:")
    for k in ("WhoN", "WhoO", "HowR", "HowO", "Joint"):
        out.append(f"  {k}: {set_pos[k]}/{n_settings} = {set_pos[k]/n_settings*100:.0f}% positive")
    out.append("```\n")

    # ---------------- OW-OH vs Native-Full per iteration ----------------
    out.append("### OW-OH vs Native-Full（每 iteration，rho 内最小/最优与各 rho）\n```text")
    for it in iters:
        nf_l = np.mean([sel(it, s, "native_full")[0]["demand_l1_100"] for s in seeds])
        nf_g = np.mean([sel(it, s, "native_full")[0]["global_psnr_100"] for s in seeds])
        for rho in rhos:
            oo = np.mean([sel(it, s, "ow_oh", rho)[0]["demand_l1_100"] for s in seeds])
            oog = np.mean([sel(it, s, "ow_oh", rho)[0]["global_psnr_100"] for s in seeds])
            M = sel(it, seeds[0], "ow_oh", rho)[0]["M"]
            out.append(f"it={it:>5d} rho={rho:.2f}: OW-OH {oo:.6f} vs NativeFull {nf_l:.6f} "
                       f"(Δ{oo-nf_l:+.6f}, 更优={oo<nf_l}) gPSNR {oog:.4f} vs {nf_g:.4f} "
                       f"({oog-nf_g:+.4f}) growth {M}/{sel(it,seeds[0],'native_full')[0]['M']}")
    out.append("```\n")

    # ---------------- gains vs state ----------------
    out.append("### Snapshot state 与增益（#GS / candidate / split-ratio）\n```text")
    out.append(f"{'iter':>6s} {'#GS':>8s} {'cand':>7s} {'split%':>7s} {'prune':>7s} "
               f"{'Joint(rho=.5)':>14s} {'Joint(rho=.75)':>15s} {'HowO(.5)':>12s}")
    for it in iters:
        s = st[it]
        j5 = np.mean(gains[(it, 0.5)]["Joint"]) if (it, 0.5) in gains else float("nan")
        j75 = np.mean(gains[(it, 0.75)]["Joint"]) if (it, 0.75) in gains else float("nan")
        h5 = np.mean(gains[(it, 0.5)]["HowO"]) if (it, 0.5) in gains else float("nan")
        out.append(f"{it:>6d} {int(s['num_gaussians_before']):>8d} {int(s['candidate_count']):>7d} "
                   f"{s['native_split_count']/s['candidate_count']*100:>6.1f}% "
                   f"{int(s['prune_count']):>7d} {j5:>+14.6f} {j75:>+15.6f} {h5:>+12.6f}")
    out.append("```\n")

    # ---------------- plots ----------------
    os.makedirs(f"{BASE}/plots", exist_ok=True)

    def save(fig, name):
        fig.savefig(f"{BASE}/plots/{name}", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 1 gain vs iteration
    fig, ax = plt.subplots(figsize=(7, 4.4))
    marks = {"WhoN": "v", "WhoO": "^", "HowR": "s", "HowO": "D", "Joint": "o"}
    for k, mk in marks.items():
        for rho, col in ((0.5, "#4477aa"), (0.75, "#228833")):
            xs = iters
            ys = [np.mean(gains[(it, rho)][k]) for it in iters]
            es = [np.std(gains[(it, rho)][k]) for it in iters]
            ax.errorbar(xs, ys, yerr=es, marker=mk, capsize=3, color=col,
                        label=f"{k} rho={rho}" if k == "Joint" or rho == 0.5 else None,
                        alpha=0.9)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("snapshot iteration")
    ax.set_ylabel("gain (L1 reduction)")
    ax.set_title("allocation gains vs training stage")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "gain_vs_iteration.png")

    # 2 gain vs #GS
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for k, mk in marks.items():
        xs = [st[it]["num_gaussians_before"] for it in iters]
        ys = [np.mean([np.mean(gains[(it, rho)][k]) for rho in rhos]) for it in iters]
        ax.plot(xs, ys, marker=mk, label=k)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("#GS before densify")
    ax.set_ylabel("mean gain (rho-averaged)")
    ax.set_title("gains vs representation size")
    ax.legend(fontsize=8)
    save(fig, "gain_vs_num_gaussians.png")

    # 3 quality vs growth by iteration
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for it in iters:
        xs, ys = [], []
        for p in ("keep_all", "native_full", "ow_nh", "ow_oh", "rw_nh", "rw_oh"):
            sub = [r for r in rs if r["iteration"] == it and r["policy"] == p]
            for r in sub:
                xs.append(r["M"])
                base = next(b for b in rs if b["iteration"] == it
                            and b["group_seed"] == r["group_seed"] and b["policy"] == "keep_all")
                ys.append(base["demand_l1_100"] - r["demand_l1_100"])
        ax.scatter(xs, ys, s=18, alpha=0.7, label=f"it{it}")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("M (added Gaussians)")
    ax.set_ylabel("L1 reduction vs Keep-All")
    ax.set_title("quality vs growth by snapshot")
    ax.legend(fontsize=8)
    save(fig, "quality_vs_growth_by_iteration.png")

    # 4 joint gain across snapshots (bar)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(iters))
    for i, rho in enumerate(rhos):
        ys = [np.mean(gains[(it, rho)]["Joint"]) for it in iters]
        es = [np.std(gains[(it, rho)]["Joint"]) for it in iters]
        ax.bar(x + (i - 0.5) * 0.38, ys, 0.38, yerr=es, capsize=3,
               label=f"rho={rho}", color=["#4477aa", "#228833"][i])
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"it{it}\nN={int(st[it]['num_gaussians_before'])}" for it in iters], fontsize=8)
    ax.set_ylabel("Joint gain")
    ax.set_title("joint gain across snapshots")
    ax.legend()
    save(fig, "joint_gain_across_snapshots.png")

    # 5 candidate count vs iteration
    fig, ax = plt.subplots(figsize=(5.8, 4))
    ax.plot(iters, [st[it]["candidate_count"] for it in iters], marker="o", color="#cc8800")
    for it in iters:
        ax.annotate(f"{int(st[it]['candidate_count'])}", (it, st[it]["candidate_count"]),
                    textcoords="offset points", xytext=(0, 6), fontsize=8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("candidate count (clone+split)")
    ax.set_title("densification demand vs training stage")
    save(fig, "candidate_count_vs_iteration.png")

    with open(f"{BASE}/data/b5_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[b5a] outputs -> {BASE}/data , {BASE}/plots")


if __name__ == "__main__":
    main()
