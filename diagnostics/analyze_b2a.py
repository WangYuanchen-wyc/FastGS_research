#
# Paper B - B2-A analysis: split stochasticity + residual-split/parent alignment.
# Reads paper_b/b2_a_split_alignment/data/b2a_results.json (from diagnostic_b2a.py)
# and writes stats + 5 plots + a stats text block. Host python3 (numpy/matplotlib/scipy).
#
#   python3 diagnostics/analyze_b2a.py
#

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr, pearsonr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

BASE = "paper_b/b2_a_split_alignment"


def col(records, key):
    return np.array([np.nan if r.get(key) is None else float(r[key]) for r in records])


def spear(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return None, 0
    if HAVE_SCIPY:
        return float(spearmanr(x, y)[0]), len(x)
    rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


def pears(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None, 0
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def stats_str(a):
    a = a[~np.isnan(a)]
    if not len(a):
        return "n=0 (all NA)"
    return (f"n={len(a)} mean={np.mean(a):+.6f} median={np.median(a):+.6f} "
            f"std={np.std(a):.6f} min={np.min(a):+.6f} max={np.max(a):+.6f}")


def main():
    with open(f"{BASE}/data/b2a_results.json") as f:
        data = json.load(f)
    records = data["records"]
    n = len(records)
    rep = []
    rep.append("## B2-A 统计输出（自动生成）\n")

    # ---------------- split stochasticity ----------------
    std = col(records, "split_dQ_std")
    rng = col(records, "split_stochastic_range")
    gapabs = col(records, "clone_split_mean_gap_abs")
    ratio = std > gapabs
    rep.append("### Split stochasticity（ΔQ = demand-local L1 reduction @100）\n")
    rep.append("```text")
    rep.append(f"split_dQ_std          : {stats_str(std)}")
    rep.append(f"split_stochastic_range: {stats_str(rng)}")
    rep.append(f"|mean ΔQ_split − ΔQ_clone| : {stats_str(gapabs)}")
    m = ~(np.isnan(std) | np.isnan(gapabs))
    rep.append(f"candidate ratio  split_std > |Clone−Split mean gap| : "
               f"{np.sum(ratio[m])}/{int(np.sum(m))} = {np.nanmean(ratio[m])*100:.1f}%")
    rep.append(f"median split_std / median |gap| = "
               f"{np.nanmedian(std)/np.nanmedian(gapabs):.2f}")
    rep.append("```\n")

    # ---------------- residual-split alignment (pooled over repeats) --------
    px, py = [], []
    within = []
    for r in records:
        det = r.get("_split_detail") or []
        xs, ys = [], []
        for d in det:
            a = d.get("residual_split_alignment")
            q = (d.get("demand_l1") or {}).get("100")
            if a is None or q is None or r.get("keep_demand_error_100") is None:
                continue
            xs.append(a)
            ys.append(r["keep_demand_error_100"] - q)  # ΔQ_split@100 for this repeat
        if len(xs) >= 3:
            px += xs; py += ys
            s, _ = spear(np.array(xs), np.array(ys))
            if s is not None:
                within.append((s, r["candidate_id"]))
    px, py = np.array(px), np.array(py)
    pooled_p, npx = pears(px, py)
    pooled_s, nsx = spear(px, py)
    rep.append("### Residual–Split directional alignment（每个 split repeat 的 |cos| vs 该 repeat 的 ΔQ_split@100）\n")
    rep.append("```text")
    rep.append(f"pooled points n={len(px)}  Pearson={pooled_p:+.3f}  Spearman={pooled_s:+.3f}"
               if pooled_p is not None else "pooled: insufficient data")
    if within:
        ws = np.array([w[0] for w in within])
        rep.append(f"within-candidate Spearman (n={len(ws)} candidates, 5 pts each): "
                   f"mean={np.mean(ws):+.3f} median={np.median(ws):+.3f}")
        rep.append(f"positive-correlation candidate ratio: {np.mean(ws>0)*100:.1f}%  "
                   f"negative: {np.mean(ws<0)*100:.1f}%")
    rep.append("```\n")

    # ---------------- stable action gap correlations ----------------
    ykey = "quality_action_gap"
    xkeys = ["scale_max", "scale_anisotropy", "residual_parent_alignment_mean",
             "residual_extent_ratio_mean", "residual_centroid_offset_mean",
             "residual_anisotropy_mean", "proj_anisotropy_mean", "footprint_mean"]
    rep.append(f"### quality_action_gap = mean(ΔQ_split@100,5reps) − ΔQ_clone@100 相关性（n={n}）\n")
    rep.append("| descriptor | n | Pearson | Spearman |")
    rep.append("|---|---:|---:|---:|")
    for xk in xkeys:
        p_, np_ = pears(col(records, xk), col(records, ykey))
        s_, ns_ = spear(col(records, xk), col(records, ykey))
        rep.append(f"| {xk} | {np_} | {p_ if p_ is None else f'{p_:+.3f}'} | "
                   f"{s_ if s_ is None else f'{s_:+.3f}'} |")
    rep.append("")

    # ---------------- winners ----------------
    wk = sum(1 for r in records if r["oracle_winner"] == "keep")
    wc = sum(1 for r in records if r["oracle_winner"] == "clone")
    ws_ = sum(1 for r in records if r["oracle_winner"] == "split")
    conf = {a: {w: 0 for w in ("keep", "clone", "split")} for a in ("clone", "split")}
    for r in records:
        if r["oracle_winner"] in ("keep", "clone", "split"):
            conf[r["native_action"]][r["oracle_winner"]] += 1
    tot = sum(conf[a][w] for a in conf for w in conf[a])
    agree = conf["clone"]["clone"] + conf["split"]["split"]
    rep.append("### Action winner（split 用 5-repeat 平均质量）\n")
    rep.append("```text")
    rep.append(f"Keep best : {wk}/{n} = {wk/n*100:.1f}%")
    rep.append(f"Clone best: {wc}/{n} = {wc/n*100:.1f}%")
    rep.append(f"Split best: {ws_}/{n} = {ws_/n*100:.1f}%")
    rep.append(f"native vs oracle agreement: {agree}/{tot} = {agree/tot*100:.1f}%")
    rep.append("confusion (row=native, col=oracle):")
    rep.append(f"{'':14s}{'keep':>8s}{'clone':>8s}{'split':>8s}")
    for a in ("clone", "split"):
        rep.append(f"native {a:8s}{conf[a]['keep']:8d}{conf[a]['clone']:8d}{conf[a]['split']:8d}")
    cond_tot = sum(conf[a][w] for a in conf for w in ("clone", "split"))
    rep.append(f"conditional agreement (oracle picked an action): "
               f"{agree}/{cond_tot} = {agree/cond_tot*100:.1f}%")
    rep.append("```\n")

    # ---------------- tile ----------------
    dtc = col(records, "deltaTile_clone_0")
    dts = col(records, "deltaTile_split_0")
    rep.append("### Tile cost（gaussian_tile_delta @K=0；split 为 5-repeat 均值）\n")
    rep.append("```text")
    rep.append(f"ΔTile_clone@0: {stats_str(dtc)}")
    rep.append(f"ΔTile_split@0: {stats_str(dts)}")
    rep.append(f"split tile std@0 (repeat 间): {stats_str(col(records, 'split_tile_std_0'))}")
    rep.append("```\n")

    # ---------------- plots ----------------
    os.makedirs(f"{BASE}/plots", exist_ok=True)

    def save(fig, name):
        fig.savefig(f"{BASE}/plots/{name}", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 1
    fig, ax = plt.subplots(figsize=(6, 4))
    v = std[~np.isnan(std)]
    ax.hist(v, bins=25, color="#4477aa", edgecolor="white")
    ax.axvline(np.nanmedian(gapabs), color="#cc3311", ls="--",
               label=f"median |Clone−Split gap| = {np.nanmedian(gapabs):.5f}")
    ax.set_xlabel("std of ΔQ_split@100 over 5 repeats")
    ax.set_ylabel("candidates")
    ax.set_title(f"Split stochasticity (n={len(v)})")
    ax.legend()
    save(fig, "split_stochasticity_hist.png")

    # 2
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    x, y = gapabs[~np.isnan(gapabs)], std[~np.isnan(std)]
    ax.scatter(x, y, s=26, alpha=0.75, color="#4477aa")
    lim = max(np.max(x), np.max(y)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x (noise == signal)")
    ax.set_xlabel("|mean ΔQ_split − ΔQ_clone| (action gap magnitude)")
    ax.set_ylabel("std(ΔQ_split) (stochasticity)")
    ax.set_title("split noise vs clone/split preference")
    ax.legend()
    save(fig, "split_gain_variance_vs_action_gap.png")

    # 3 pooled scatter
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(px, py, s=14, alpha=0.55, color="#55a868")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("residual_split_alignment = |cos(residual dir, split separation dir)|")
    ax.set_ylabel("ΔQ_split@100 (per repeat)")
    ax.set_title(f"pooled n={len(px)}  r={pooled_p:+.2f} ρ={pooled_s:+.2f}"
                 if pooled_p is not None else "pooled")
    save(fig, "split_gain_vs_residual_split_alignment.png")

    # 4
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    x = col(records, "residual_parent_alignment_mean")
    y = col(records, ykey)
    m = ~(np.isnan(x) | np.isnan(y))
    ax.scatter(x[m], y[m], s=26, alpha=0.75,
               c=["#dd8452" if r["native_action"] == "clone" else "#4477aa"
                  for r, k in zip(records, m) if k])
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("residual_parent_alignment_mean (|cos|)")
    ax.set_ylabel("quality_action_gap (5-repeat split mean)")
    save(fig, "quality_gap_vs_residual_parent_alignment.png")

    # 5
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    x = col(records, "residual_extent_ratio_mean")
    m = ~(np.isnan(x) | np.isnan(y))
    ax.scatter(x[m], y[m], s=26, alpha=0.75,
               c=["#dd8452" if r["native_action"] == "clone" else "#4477aa"
                  for r, k in zip(records, m) if k])
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("residual_extent / parent footprint extent (mean)")
    ax.set_ylabel("quality_action_gap")
    save(fig, "quality_gap_vs_residual_extent_ratio.png")

    with open(f"{BASE}/data/b2a_stats.txt", "w") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print(f"\n[analyze_b2a] stats -> {BASE}/data/b2a_stats.txt ; plots -> {BASE}/plots/")


if __name__ == "__main__":
    main()
