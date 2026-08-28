#
# Paper B - B2 analysis: reads diagnostics/diagnostic_v2.py JSON output and
# produces statistics (A-F), Pearson/Spearman correlations, 6 plots and a
# markdown report. Pure post-processing: no FastGS dependency.
#
# Run on host python3 (numpy + matplotlib + scipy):
#   python3 diagnostics/analyze_v2.py project_md/action_diag_v2.json
#

import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def col(records, key):
    vals = []
    for r in records:
        v = r.get(key, None)
        vals.append(np.nan if v is None else float(v))
    return np.array(vals)


def pairs(records, xkey, ykey):
    xs, ys = col(records, xkey), col(records, ykey)
    m = ~(np.isnan(xs) | np.isnan(ys))
    return xs[m], ys[m]


def pearson(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    if len(x) < 3:
        return None
    if HAVE_SCIPY:
        return float(spearmanr(x, y)[0])
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return pearson(rx.astype(float), ry.astype(float))


def stats_str(a):
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return "n=0 (all NA)"
    return (f"n={len(a)} mean={np.mean(a):+.6f} median={np.median(a):+.6f} "
            f"std={np.std(a):.6f} min={np.min(a):+.6f} max={np.max(a):+.6f}")


def corr_block(records, xkeys, ykey):
    lines = []
    for xk in xkeys:
        x, y = pairs(records, xk, ykey)
        p, s = pearson(x, y), spearman(x, y)
        fp = "NA" if p is None else f"{p:+.3f}"
        fs = "NA" if s is None else f"{s:+.3f}"
        lines.append(f"| {xk} | {len(x)} | {fp} | {fs} |")
    return lines


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "project_md/action_diag_v2.json"
    with open(json_path) as f:
        data = json.load(f)
    records = data["records"]
    out_dir = os.path.join("project_md", "plots")
    os.makedirs(out_dir, exist_ok=True)
    n = len(records)
    rep = []
    rep.append(f"# Paper B — B2 Multi-Candidate Local Action Oracle 报告\n")
    rep.append(f"- 数据源: `{json_path}`（真实运行，无伪造）")
    rep.append(f"- scene: `{data['scene']}` , global probe: {data['global_probe_source']}")
    rep.append(f"- 总 candidate 数: **{n}** "
               f"(native clone: {sum(1 for r in records if r['native_action']=='clone')}, "
               f"native split: {sum(1 for r in records if r['native_action']=='split')})")
    rep.append(f"- 有效 local ROI candidate 数: {sum(1 for r in records if (r.get('num_valid_local_views') or 0) > 0)}")
    rep.append(f"- 有效 demand ROI candidate 数 (≥1 valid demand view): "
               f"{sum(1 for r in records if (r.get('num_valid_demand_views') or 0) > 0)}")
    rep.append("")

    if n == 0:
        print("no records"); return

    # ---------------- A. quality ----------------
    dqc = col(records, "deltaQ_demand_clone_100")
    dqs = col(records, "deltaQ_demand_split_100")
    gap = col(records, "quality_action_gap")
    dsc = col(records, "deltaQ_support_clone_100")
    dss = col(records, "deltaQ_support_split_100")
    rep.append("## A. Quality 分布（ΔQ = Err_keep − Err_action，正值 = 优于 Keep；主指标 demand-local L1@100）\n")
    rep.append("```text")
    rep.append(f"ΔQ_demand_clone  : {stats_str(dqc)}")
    rep.append(f"ΔQ_demand_split  : {stats_str(dqs)}")
    rep.append(f"ΔQ_support_clone : {stats_str(dsc)}")
    rep.append(f"ΔQ_support_split : {stats_str(dss)}")
    rep.append(f"quality_action_gap (split−clone, demand): {stats_str(gap)}")
    p_gap = gap[~np.isnan(gap)]
    if len(p_gap):
        rep.append(f"  gap>0 (split更好): {np.mean(p_gap>0)*100:.1f}%   "
                   f"gap<0 (clone更好): {np.mean(p_gap<0)*100:.1f}%   "
                   f"|gap|<1e-4 (无差): {np.mean(np.abs(p_gap)<1e-4)*100:.1f}%")
    rep.append("```\n")

    # ---------------- B. winners ----------------
    winners = [r.get("oracle_winner") for r in records]
    wk = winners.count("keep"); wc = winners.count("clone"); ws = winners.count("split")
    wn = winners.count(None)
    rep.append("## B. Action winner（oracle = demand-local L1@100 最小者；demand 无效时回退 support-local）\n")
    rep.append("```text")
    rep.append(f"Keep best  : {wk}/{n} = {wk/n*100:.1f}%")
    rep.append(f"Clone best : {wc}/{n} = {wc/n*100:.1f}%")
    rep.append(f"Split best : {ws}/{n} = {ws/n*100:.1f}%")
    if wn:
        rep.append(f"invalid    : {wn}/{n} (无法判定)")
    rep.append("```\n")

    # ---------------- C. native vs oracle ----------------
    conf = {a: {w: 0 for w in ("keep", "clone", "split")} for a in ("clone", "split")}
    for r in records:
        w = r.get("oracle_winner")
        if w in ("keep", "clone", "split"):
            conf[r["native_action"]][w] += 1
    tot = sum(conf[a][w] for a in conf for w in conf[a])
    agree = conf["clone"]["clone"] + conf["split"]["split"]
    rep.append("## C. FastGS native action vs short-horizon oracle\n")
    rep.append("```text")
    rep.append(f"agreement rate: {agree}/{tot} = {agree/tot*100:.1f}%" if tot else "no valid")
    rep.append("confusion (row=native, col=oracle winner):")
    rep.append(f"{'':14s}{'keep':>8s}{'clone':>8s}{'split':>8s}")
    for a in ("clone", "split"):
        rep.append(f"native {a:8s}{conf[a]['keep']:8d}{conf[a]['clone']:8d}{conf[a]['split']:8d}")
    rep.append("```\n")

    # ---------------- D/E. correlations ----------------
    ykey = "quality_action_gap"
    xkeys = ["scale_max", "scale_anisotropy", "residual_energy_mean",
             "residual_anisotropy_mean", "footprint_mean", "importance_score"]
    rep.append(f"## D/E. quality_action_gap 相关性（n={n}{' — 样本偏小，标记为初步' if n < 30 else ''}）\n")
    rep.append("| descriptor | n | Pearson | Spearman |")
    rep.append("|---|---:|---:|---:|")
    rep += corr_block(records, xkeys, ykey)
    rep.append("")

    # ---------------- F. compute ----------------
    dtc0 = col(records, "deltaTile_clone_0")
    dts0 = col(records, "deltaTile_split_0")
    tag0 = col(records, "tile_action_gap_0")
    dtc100 = col(records, "deltaTile_clone_100")
    dts100 = col(records, "deltaTile_split_100")
    net_c = col(records, "num_gaussians_clone") - col(records, "num_gaussians_keep")
    net_s = col(records, "num_gaussians_split") - col(records, "num_gaussians_keep")
    rep.append("## F. Compute（gaussian_tile_delta = ΔTile，K=0 结构性 / K=100 已实现）\n")
    rep.append("```text")
    rep.append(f"ΔTile_clone@0 : {stats_str(dtc0)}")
    rep.append(f"ΔTile_split@0 : {stats_str(dts0)}")
    rep.append(f"tile_action_gap@0 (split−clone): {stats_str(tag0)}")
    rep.append(f"ΔTile_clone@100: {stats_str(dtc100)}")
    rep.append(f"ΔTile_split@100: {stats_str(dts100)}")
    rep.append(f"Clone net Δ#GS: all=={int(np.nanmin(net_c))} ? {bool(np.all(net_c[~np.isnan(net_c)]==1))}"
               f"  (mean={np.nanmean(net_c):+.3f})")
    rep.append(f"Split net Δ#GS: all=={int(np.nanmin(net_s))} ? {bool(np.all(net_s[~np.isnan(net_s)]==1))}"
               f"  (mean={np.nanmean(net_s):+.3f})")
    rep.append("```\n")

    latk = col(records, "latency_keep_ms"); latc = col(records, "latency_clone_ms")
    lats = col(records, "latency_split_ms")
    rep.append("```text")
    rep.append(f"latency keep/clone/split ms (light protocol): "
               f"{np.nanmean(latk):.3f} / {np.nanmean(latc):.3f} / {np.nanmean(lats):.3f}")
    rep.append("```\n")

    gp = col(records, "global_PSNR_keep_100"); gc = col(records, "global_PSNR_clone_100")
    gs = col(records, "global_PSNR_split_100")
    rep.append("```text")
    rep.append(f"global PSNR@100 keep/clone/split: {np.nanmean(gp):.4f} / "
               f"{np.nanmean(gc):.4f} / {np.nanmean(gs):.4f}")
    rep.append("```\n")

    # ---------------- plots ----------------
    def save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        rep.append(f"- plots/{name}")

    # 1 quality gap histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    v = gap[~np.isnan(gap)]
    ax.hist(v, bins=30, color="#4477aa", edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("quality_action_gap = ΔQ_demand_split − ΔQ_demand_clone")
    ax.set_ylabel("count")
    ax.set_title(f"Split−Clone local quality gap (n={len(v)})")
    save(fig, "quality_gap_histogram.png")

    # 2 confusion
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    mat = np.array([[conf["clone"][w] for w in ("keep", "clone", "split")],
                    [conf["split"][w] for w in ("keep", "clone", "split")]])
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(["keep", "clone", "split"])
    ax.set_yticks(range(2)); ax.set_yticklabels(["native clone", "native split"])
    for i in range(2):
        for j in range(3):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_title(f"native vs oracle (agreement {agree/tot*100:.1f}%)" if tot else "confusion")
    fig.colorbar(im)
    save(fig, "native_vs_oracle_confusion.png")

    # 3 gap vs scale_max
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for act, c in (("clone", "#dd8452"), ("split", "#4477aa")):
        xs, ys = [], []
        for r in records:
            if r["native_action"] == act and r.get("quality_action_gap") is not None \
                    and r.get("scale_max") is not None:
                xs.append(r["scale_max"]); ys.append(r["quality_action_gap"])
        ax.scatter(xs, ys, s=22, alpha=0.75, label=f"native {act}", color=c)
    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("scale_max"); ax.set_ylabel("quality_action_gap")
    ax.legend(); ax.set_title("quality gap vs scale_max")
    save(fig, "quality_gap_vs_scale.png")

    # 4 gap vs residual anisotropy
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for act, c in (("clone", "#dd8452"), ("split", "#4477aa")):
        xs, ys = [], []
        for r in records:
            if r["native_action"] == act and r.get("quality_action_gap") is not None \
                    and r.get("residual_anisotropy_mean") is not None:
                xs.append(r["residual_anisotropy_mean"]); ys.append(r["quality_action_gap"])
        ax.scatter(xs, ys, s=22, alpha=0.75, label=f"native {act}", color=c)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("residual_anisotropy_mean"); ax.set_ylabel("quality_action_gap")
    ax.legend(); ax.set_title("quality gap vs residual anisotropy")
    save(fig, "quality_gap_vs_residual_anisotropy.png")

    # 5 delta tile clone vs split
    fig, ax = plt.subplots(figsize=(5, 4.4))
    x, y = pairs(records, "deltaTile_clone_0", "deltaTile_split_0")
    ax.scatter(x, y, s=24, alpha=0.75, color="#55a868")
    lim = max(abs(np.concatenate([x, y, [1]]))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y=x")
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("ΔTile_clone@0 (gaussian_tile_delta)")
    ax.set_ylabel("ΔTile_split@0")
    ax.legend(); ax.set_title(f"structural tile cost (n={len(x)})")
    save(fig, "delta_tile_clone_vs_split.png")

    # 6 quality vs tile cost
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for act, c in (("clone", "#dd8452"), ("split", "#4477aa")):
        xs, ys = [], []
        for r in records:
            if r["native_action"] == act and r.get("quality_action_gap") is not None \
                    and r.get("tile_action_gap_0") is not None:
                xs.append(r["tile_action_gap_0"]); ys.append(r["quality_action_gap"])
        ax.scatter(xs, ys, s=24, alpha=0.75, label=f"native {act}", color=c)
    ax.axhline(0, color="k", lw=1); ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("tile_action_gap@0 = ΔTile_split − ΔTile_clone")
    ax.set_ylabel("quality_action_gap")
    ax.legend(); ax.set_title("quality gain vs compute gap")
    save(fig, "quality_vs_tile_cost.png")

    # ---------------- write report ----------------
    rep_path = json_path.replace(".json", "_report.md").replace(".json", ".md") \
        if json_path.endswith(".json") else "project_md/action_diag_v2_report.md"
    rep_path = "project_md/action_diag_v2_report.md"
    with open(rep_path, "w") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print(f"\n[analyze] report -> {rep_path}, plots -> {out_dir}/")


if __name__ == "__main__":
    main()
