#
# Paper B - B8-L analysis (host): long-horizon cardinality persistence.
#   gain_t = L1_Split2(t) − L1_MSCC(t) at t = 100/500/1000 (C parents)
#            L1_Split2(t) − L1_Split3(t) at t (B parents, control)
#   persistence label (±1 SEM overlap, consistent with B8-A Fix):
#     persistent_higher_cardinality ⇔ mean_MSCC(1000) + SEM_MSCC < mean_Split2(1000)
#   python3 diagnostics/analyze_b8l.py
#

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b8_l_long_horizon_cardinality"
TS = [100, 500, 1000]


def sem_of(v):
    v = [x for x in v if x is not None]
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) >= 2 else 0.0


def main():
    d = json.load(open(f"{BASE}/data/b8l_repeat_results.json"))
    rows = d["branches"]
    life = d["lifecycle"]
    parents = defaultdict(dict)
    for r in rows:
        k = (r["iteration"], r["parent_index"], r["category"])
        if r["branch"] == "keep":
            parents[k]["keep"] = r
        else:
            parents[k].setdefault(r["branch"], []).append(r)

    out = ["## B8-L Long-Horizon Cardinality\n"]
    prow = []
    for (it, idx, cat), dd in sorted(parents.items()):
        mscc_branch = "split_mscc" if cat.startswith("C") else "split3"
        ctrl_branch = "split2"
        rec = {"iteration": it, "parent_index": idx, "category": cat,
               "keep_l1_1000": dd["keep"]["l1_1000"]}
        for t in TS:
            m2 = [b[f"l1_{t}"] for b in dd.get(ctrl_branch, []) if b.get(f"l1_{t}") is not None]
            mX = [b[f"l1_{t}"] for b in dd.get(mscc_branch, []) if b.get(f"l1_{t}") is not None]
            mean2 = float(np.mean(m2)) if m2 else None
            meanX = float(np.mean(mX)) if mX else None
            s2, sX = sem_of(m2), sem_of(mX)
            rec[f"mean_split2_l1_{t}"] = mean2
            rec[f"mean_mscc_l1_{t}"] = meanX
            rec[f"sem_split2_{t}"] = s2
            rec[f"sem_mscc_{t}"] = sX
            rec[f"gain_{t}"] = mean2 - meanX if (mean2 is not None and meanX is not None) else None
            rec[f"norm_gain_{t}"] = ((mean2 - meanX) / float(np.hypot(s2, sX))
                                     if (mean2 is not None and meanX is not None and s2 + sX > 0) else None)
        # persistence label (main, ±1 SEM conservative)
        # missing checkpoint -> "missing", excluded from every denominator
        mX, sX = rec.get("mean_mscc_l1_1000"), rec.get("sem_mscc_1000")
        m2 = rec.get("mean_split2_l1_1000")
        if mX is None or m2 is None:
            rec["persistent_label"] = "missing"
        else:
            rec["persistent_label"] = (
                "persistent_higher_cardinality" if (mX + sX < m2)
                else "not_persistent")
        prow.append(rec)

    import csv as _csv
    with open(f"{BASE}/data/b8l_parent_results.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(prow[0].keys()))
        w.writeheader()
        for r in prow:
            w.writerow({k: ("NA" if v is None else (f"{v:.8g}" if isinstance(v, float) else v))
                        for k, v in r.items()})

    cs = [r for r in prow if r["category"].startswith("C")]
    bs = [r for r in prow if r["category"] == "B_split2_sufficient"]
    out.append("### gain_t 定义：C 组 = L1_Split2(t)−L1_MSCC(t)；B 组（对照）= L1_Split2(t)−L1_Split3(t)"
               "（>0 = higher-N / Split-3 更好）\n")

    out.append("### C 组（higher-cardinality parents）\n```text")
    iters = sorted({r["iteration"] for r in prow})
    summary = []
    for scope, sub in [("ALL-C", cs)] + [(f"it{it}-C", [r for r in cs if r["iteration"] == it])
                                         for it in iters]:
        n = len(sub)
        if not n:
            continue
        miss = sum(1 for r in sub if r["persistent_label"] == "missing")
        nv = n - miss
        line = f"{scope:>9s} (valid {nv}/{n}, missing {miss}): "
        for t in TS:
            g = [r[f"gain_{t}"] for r in sub if r.get(f"gain_{t}") is not None]
            pos = sum(1 for x in g if x > 0)
            line += (f"gain{t} mean {np.mean(g):+.6f} (pos {pos}/{len(g)}) | ")
        pers = sum(1 for r in sub if r["persistent_label"] == "persistent_higher_cardinality")
        ng = [r["norm_gain_1000"] for r in sub if r.get("norm_gain_1000") is not None]
        line += (f"persistent@1000 {pers}/{nv} valid ({pers/nv*100:.0f}% of valid)" if nv else
                 "persistent@1000 NA (all missing)")
        line += f" | norm_gain1000>1 {sum(1 for x in ng if x > 1)}/{len(ng)}"
        out.append(line)
        summary.append({"scope": scope, "n": n, "valid": nv, "missing": miss,
                        **{f"mean_gain_{t}": float(np.mean([r[f"gain_{t}"] for r in sub
                                                            if r.get(f"gain_{t}") is not None]))
                           for t in TS},
                        "persistent_1000": pers,
                        "persistent_ratio_valid": (pers / nv) if nv else None,
                        "norm_gain1000_gt1": sum(1 for x in ng if x > 1)})
    out.append("```\n")

    out.append("### B 组对照（Split-2 sufficient：Split-3 是否长期仍无收益）\n```text")
    for scope, sub in [("ALL-B", bs)] + [(f"it{it}-B", [r for r in bs if r["iteration"] == it])
                                         for it in iters]:
        n = len(sub)
        if not n:
            continue
        miss = sum(1 for r in sub if r["persistent_label"] == "missing")
        line = f"{scope:>9s} (valid {n-miss}/{n}, missing {miss}): "
        for t in TS:
            g = [r[f"gain_{t}"] for r in sub if r.get(f"gain_{t}") is not None]
            pos = sum(1 for x in g if x > 0)
            line += (f"gain{t}(S3−S2 效果) mean {np.mean(g):+.6f} (pos {pos}/{len(g)}) | ")
        out.append(line + "（>0 = Split-3 仍更好；期望 B 组 ≈0 或负）")
    out.append("```\n")

    # trajectory: gain 100 -> 500 -> 1000
    out.append("### gain 轨迹（C 组，per parent）\n```text")

    def fg(r, t):
        v = r.get(f"gain_{t}")
        return "NA" if v is None else f"{v:+.6f}"

    for r in cs[:12]:
        out.append(f"it{r['iteration']} idx={r['parent_index']} {r['category'][:2]}: "
                   f"g100 {fg(r, 100)} -> g500 {fg(r, 500)} -> "
                   f"g1000 {fg(r, 1000)} ({r['persistent_label']})")
    if len(cs) > 12:
        out.append(f"…（共 {len(cs)} 个，全表见 b8l_parent_results.csv）")
    out.append("```\n")

    # lifecycle summary
    out.append("### Child lifecycle（C 组 MSCC branches，per-child 按亲属均值）\n```text")
    lf = [r for r in life if r["category"].startswith("C") and r["N"] > 2]
    for t in ("0", "100", "500", "1000"):
        sub = [r for r in lf if str(r["step"]) == t]
        if not sub:
            continue
        out.append(f"step {t:>4s}: opacity median {np.median([r['opacity'] for r in sub]):.4f} | "
                   f"visible_views median {np.median([r['visible_views'] for r in sub]):.1f} | "
                   f"xyz_disp median {np.median([r['xyz_disp_from_birth'] for r in sub]):.5f} | "
                   f"scale_change median {np.median([r['scale_change_from_birth'] for r in sub]):.5f}")
    op1000 = [r["opacity"] for r in lf if str(r["step"]) == "1000"]
    out.append(f"opacity@1000: p10 {np.percentile(op1000, 10):.4f} median {np.median(op1000):.4f} "
               f"| opacity<0.1 比例 {100*np.mean([o < 0.1 for o in op1000]):.0f}%")
    out.append("```\n")

    # GO/NO-GO
    n_c = len(cs)
    pers = sum(1 for r in cs if r["persistent_label"] == "persistent_higher_cardinality")
    g1000_pos = sum(1 for r in cs if (r.get("gain_1000") or 0) > 0)
    ng1 = sum(1 for r in cs if (r.get("norm_gain_1000") or 0) > 1)
    b_neg = sum(1 for r in bs if (r.get("gain_1000") or 0) <= 0)
    cross = all(any(r["persistent_label"] == "persistent_higher_cardinality"
                    for r in cs if r["iteration"] == it) for it in iters)
    shrink = (np.mean([r["gain_100"] for r in cs if r.get("gain_100") is not None]) -
              np.mean([r["gain_1000"] for r in cs if r.get("gain_1000") is not None]))
    n_c_valid = n_c - sum(1 for r in cs if r["persistent_label"] == "missing")
    n_b_valid = len(bs) - sum(1 for r in bs if r["persistent_label"] == "missing")
    out.append("### GO/NO-GO\n```text")
    out.append(f"C 组: persistent {pers}/{n_c_valid} valid (missing "
               f"{n_c-n_c_valid}) | gain1000>0 {g1000_pos}/{n_c_valid} | norm>1 {ng1}/{n_c_valid}")
    out.append(f"B 组: Split-3 长期无收益/更差 {b_neg}/{n_b_valid} valid (missing {len(bs)-n_b_valid})")
    out.append(f"跨 stage 持续: {'Y' if cross else 'N'}；gain 均值 100→1000 变化 {(-shrink):+.6f}（正=衰减）")
    out.append("```")
    ok = (n_c_valid > 0 and pers / n_c_valid >= 0.4) and \
         (n_b_valid == 0 or b_neg / n_b_valid >= 0.6) and cross
    out.append(f"\n判定: {'B8-L GO' if ok else 'B8-L NO-GO'}")

    # plots
    os.makedirs(f"{BASE}/plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for r in cs:
        ax.plot([100, 500, 1000], [r["gain_100"], r["gain_500"], r["gain_1000"]],
                alpha=0.45, lw=1, color="#4477aa")
    mc = [float(np.mean([r[f"gain_{t}"] for r in cs if r.get(f"gain_{t}") is not None]))
          for t in TS]
    ax.plot(TS, mc, lw=2.5, color="#cc3311", marker="o", label="C mean")
    mb = [float(np.mean([r[f"gain_{t}"] for r in bs if r.get(f"gain_{t}") is not None]))
          for t in TS]
    ax.plot(TS, mb, lw=2.5, color="#228833", marker="s", label="B control mean")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("step"); ax.set_ylabel("gain vs Split-2 (>0 = higher-N better)")
    ax.set_title("gain over time")
    ax.legend()
    fig.savefig(f"{BASE}/plots/gain_over_time.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for r, col in ((cs, "#cc3311"), (bs, "#228833")):
        for x_ in r:
            ax.scatter([1000], [x_["mean_mscc_l1_1000"]], s=22, alpha=0.7, color=col)
    ax.set_xlabel("step 1000"); ax.set_ylabel("mean L1 (MSCC branch)")
    ax.set_title("split2 vs mscc long horizon (per-parent MSCC value)")
    fig.savefig(f"{BASE}/plots/split2_vs_mscc_long_horizon.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(iters))
    for i, it in enumerate(iters):
        sub = [r for r in cs if r["iteration"] == it]
        pers = sum(1 for r in sub if r["persistent_label"] == "persistent_higher_cardinality")
        ax.bar(i, 100 * pers / len(sub), 0.5, color="#4477aa")
        ax.text(i, 100 * pers / len(sub) + 1, f"{pers}/{len(sub)}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"it{i}" for i in iters])
    ax.set_ylabel("% persistent")
    ax.set_title("per-stage persistence (C parents)")
    fig.savefig(f"{BASE}/plots/per_stage_persistence.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    steps_plot = [0, 100, 500, 1000]
    med_op, p10_op = [], []
    for t in steps_plot:
        sub = [r["opacity"] for r in lf if str(r["step"]) == str(t)]
        med_op.append(np.median(sub)); p10_op.append(np.percentile(sub, 10))
    ax.plot(steps_plot, med_op, marker="o", color="#4477aa", label="median opacity")
    ax.plot(steps_plot, p10_op, marker="x", color="#cc8800", label="p10 opacity")
    ax.set_xlabel("step"); ax.set_ylabel("child opacity")
    ax.set_title("child lifecycle (C-group higher-N children)")
    ax.legend()
    fig.savefig(f"{BASE}/plots/child_lifecycle.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(f"{BASE}/data/b8l_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
