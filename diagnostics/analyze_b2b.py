#
# Paper B - B2-B analysis: compute additivity vs K, group quality vs K,
# tile cost vs K, quality-vs-cost trade-off. Host python3.
#
#   python3 diagnostics/analyze_b2b.py
#

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b2_b_scalability"
POL_COLORS = {"keep": "#777777", "native": "#cc8800", "all_clone": "#cc3311",
              "all_split": "#4477aa", "oracle_mix": "#228833"}
POL_LABEL = {"keep": "Keep", "native": "Native", "all_clone": "All-Clone",
             "all_split": "All-Split", "oracle_mix": "Oracle-Mix"}


def m(sub, key):
    """mean ignoring NA/None entries"""
    vals = [r[key] for r in sub if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    with open(f"{BASE}/data/b2b_results.json") as f:
        data = json.load(f)
    recs = data["records"]
    Ks = sorted({r["K"] for r in recs})
    policies = [p for p in ("keep", "native", "all_clone", "all_split", "oracle_mix")
                if any(r["policy"] == p for r in recs)]
    out = []
    out.append("## B2-B 统计输出（自动生成）\n")
    out.append(f"diag_iter={data['diag_iter']}, population(clone/split)="
               f"{data['population']['clone']}/{data['population']['split']}, "
               f"oracle_mix={data['oracle_mix']}, single-Q scaling={data['single_candidate_quality_scaling']}\n")

    # ---------------- additivity ----------------
    out.append("### Compute additivity（Actual − Predicted group ΔTile@0）\n")
    out.append("```text")
    out.append(f"{'K':>5s} {'policy':>10s} {'n':>3s} {'mean_rel_err':>14s} {'median_rel':>12s} "
               f"{'std_rel':>12s} {'max_abs_err':>12s}")
    for K in Ks:
        for p in ("native", "all_clone", "all_split"):
            sub = [r for r in recs if r["K"] == K and r["policy"] == p]
            rel = np.array([r["tile_additivity_relative_error"] for r in sub], float)
            ab = np.array([abs(r["tile_additivity_absolute_error"]) for r in sub], float)
            out.append(f"{K:>5d} {p:>10s} {len(sub):>3d} {rel.mean():>14.3e} "
                       f"{np.median(rel):>12.3e} {rel.std():>12.3e} {ab.max():>12.3f}")
    out.append("```\n")

    # ---------------- group quality / tile ----------------
    out.append("### Policy comparison（group-local L1@100 越低越好；ΔTile@0）\n")
    out.append("```text")
    out.append(f"{'K':>5s} {'policy':>10s} {'n':>3s} {'gL1@100':>10s} {'gPSNR@100':>10s} "
               f"{'dTile@0':>10s} {'dTile@100':>10s} {'dN':>5s} {'lat ms':>7s}")
    for K in Ks:
        for p in policies:
            sub = [r for r in recs if r["K"] == K and r["policy"] == p]
            if not sub:
                continue
            g1 = m(sub, "group_local_l1_100")
            gp = m(sub, "global_psnr_100")
            d0 = m(sub, "actual_delta_tile") if any(r["actual_delta_tile"] is not None for r in sub) else 0.0
            d100 = float(np.mean([r["tile_policy_100"] - r["tile_keep_0"] for r in sub]))
            dn = float(np.mean([r["delta_num_gaussians"] for r in sub]))
            lat = m(sub, "render_latency_ms")
            out.append(f"{K:>5d} {p:>10s} {len(sub):>3d} {g1:>10.6f} {gp:>10.4f} "
                       f"{d0:>10.2f} {d100:>10.2f} {dn:>5.0f} {lat:>7.3f}")
    out.append("```\n")

    # ---------------- deltas vs keep ----------------
    out.append("### 相对 Keep 的差值（mean over seeds）\n")
    out.append("```text")
    out.append(f"{'K':>5s} {'policy':>10s} {'ΔgL1@100':>12s} {'ΔgPSNR@100':>12s}")
    for K in Ks:
        keep = [r for r in recs if r["K"] == K and r["policy"] == "keep"]
        for p in policies:
            if p == "keep":
                continue
            sub = [r for r in recs if r["K"] == K and r["policy"] == p]
            if not sub or not keep:
                continue
            kg1 = m(keep, "group_local_l1_100")
            kp = m(keep, "global_psnr_100")
            dg1 = m(sub, "group_local_l1_100") - kg1
            dgp = m(sub, "global_psnr_100") - kp
            out.append(f"{K:>5d} {p:>10s} {dg1:>+12.6f} {dgp:>+12.4f}")
    out.append("```\n")

    # ---------------- plots ----------------
    os.makedirs(f"{BASE}/plots", exist_ok=True)

    def save(fig, name):
        fig.savefig(f"{BASE}/plots/{name}", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 1 additivity error vs K
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for p in ("native", "all_clone", "all_split"):
        xs, ys, es = [], [], []
        for K in Ks:
            sub = [r for r in recs if r["K"] == K and r["policy"] == p]
            if not sub:
                continue
            rel = np.array([r["tile_additivity_relative_error"] for r in sub], float)
            xs.append(K); ys.append(rel.mean()); es.append(rel.std())
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=POL_LABEL[p],
                    color=POL_COLORS[p])
    ax.set_xlabel("K (candidates per group)")
    ax.set_ylabel("relative additivity error |Actual−ΣdTile_i| / |Actual|")
    ax.set_title("tile-cost additivity vs group size (K=0 structural)")
    ax.legend()
    save(fig, "tile_additivity_error_vs_K.png")

    # 2 group quality vs K
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for p in policies:
        xs, ys, es = [], [], []
        for K in Ks:
            sub = [r for r in recs if r["K"] == K and r["policy"] == p]
            if not sub:
                continue
            vals = [r["group_local_l1_100"] for r in sub if r.get("group_local_l1_100") is not None]
            xs.append(K); ys.append(np.mean(vals)); es.append(np.std(vals))
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=POL_LABEL[p],
                    color=POL_COLORS[p])
    ax.set_xlabel("K")
    ax.set_ylabel("group-local L1 @100 (lower better)")
    ax.set_title("group quality vs K (Group ROI = union of candidate ROIs)")
    ax.legend()
    save(fig, "group_quality_vs_K.png")

    # 3 tile cost vs K
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for p in policies:
        xs, ys, es = [], [], []
        for K in Ks:
            sub = [r for r in recs if r["K"] == K and r["policy"] == p]
            if not sub:
                continue
            vals = [r["actual_delta_tile"] for r in sub if r.get("actual_delta_tile") is not None]
            xs.append(K); ys.append(np.mean(vals) if vals else 0.0); es.append(np.std(vals) if vals else 0.0)
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=POL_LABEL[p],
                    color=POL_COLORS[p])
    ax.set_xlabel("K")
    ax.set_ylabel("actual group ΔTile @0 (gaussian_tile_delta)")
    ax.set_title("structural tile cost vs K")
    ax.legend()
    save(fig, "group_tile_cost_vs_K.png")

    # 4 quality vs tile cost
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for p in policies:
        xs = [r["actual_delta_tile"] if r.get("actual_delta_tile") is not None else 0.0
              for r in recs if r["policy"] == p]
        ys = [r["group_local_l1_100"] for r in recs if r["policy"] == p]
        cs = [POL_COLORS[p] if r["policy"] != "oracle_mix" else POL_COLORS["oracle_mix"]
              for r in recs if r["policy"] == p]
        ax.scatter(xs, ys, s=42, alpha=0.8, color=POL_COLORS[p], label=POL_LABEL[p])
    ax.set_xlabel("actual ΔTile @0")
    ax.set_ylabel("group-local L1 @100 (lower better)")
    ax.set_title("quality vs structural tile cost (all K, all seeds)")
    ax.legend()
    save(fig, "group_quality_vs_tile_cost.png")

    with open(f"{BASE}/data/b2b_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[analyze_b2b] stats -> {BASE}/data/b2b_stats.txt ; plots -> {BASE}/plots/")


if __name__ == "__main__":
    main()
