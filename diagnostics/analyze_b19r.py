#
# Paper B - B19-R analysis: reliable densification + reliable pruning
#
# Inputs : b19_reliable_density_control/data/*_s{0,1,2}.csv
# Outputs: densify_view_evidence.csv, densify_child_fate.csv,
#          prune_multiview_support.csv, seed_summary.csv, b19r_stats.txt
#          (combined across seeds; per-seed inputs kept alongside)
#
# Grouping rules (fixed from the evidence distributions BEFORE reading any
# child-fate outcome; rule-setting used only Q1/Q3 distributions):
#   A. View-concentrated := top1_share > 0.5  (a single view contributes the
#      majority of the trigger's total evidence); View-distributed otherwise.
#      Rationale: semantic majority cut on a continuous heavy-tailed
#      distribution (median 0.687, p75 0.971 on seed-0 smoke).
#   B. Still-supported := nonzero VCD evidence in >= 5 of the 10 sampled views
#      (majority of sampled views flag the Gaussian as covering high-error
#      pixels). Corroborating, weaker signal reported separately:
#      frustum visibility (>=5/10 views).
#
# A-verdict: GO iff view-concentrated triggers are numerous and their children
#   show lower survival / higher prune / shorter lifetime consistently in 3/3
#   seeds, holding after birth-event & child-type stratification.
#   Thresholds: |dSurv30k| >= 0.05, 3/3 same sign, strata sign mostly holds.
# B-verdict: GO iff Still-supported fraction is non-negligible (>= 5%) and
#   stable in 3/3 seeds; NO-GO if candidates are essentially all low-support.
#

import os, csv, math
import numpy as np

BASE = "paper_b/b19_reliable_density_control"
DATA = f"{BASE}/data"
SEEDS = [0, 1, 2]
STRATA = [1, 2, 3, 4, 5]          # birth events in the early window
GO_D_SURV = 0.05
B_SUPPORTED_MIN_FRAC = 0.05
EQUAL_BAND = 0.02


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fi(x):
    return int(float(x))


def ff(x):
    v = float(x)
    return v


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ev_rows, fate_rows, cand_rows, rnd_rows = [], [], [], []
    for s in SEEDS:
        for r in read_csv(f"{DATA}/densify_view_evidence_s{seed_str(s)}.csv"):
            r["seed"] = s; ev_rows.append(r)
        for r in read_csv(f"{DATA}/densify_child_fate_s{seed_str(s)}.csv"):
            r["seed"] = s; fate_rows.append(r)
        for r in read_csv(f"{DATA}/prune_multiview_support_s{seed_str(s)}.csv"):
            r["seed"] = s; cand_rows.append(r)
        for r in read_csv(f"{DATA}/round_stats_s{seed_str(s)}.csv"):
            r["seed"] = s; rnd_rows.append(r)

    lines = []
    P = lines.append
    P("=" * 76)
    P("B19-R: Reliable Density Control (Room, native training, 3 seeds)")
    P("=" * 76)
    P(f"trigger-parent evidence rows: {len(ev_rows)};  children tracked: {len(fate_rows)};  "
      f"prune candidates: {len(cand_rows)}")
    P("")
    P("Grouping rules (fixed from distributions before reading fate):")
    P("  A: View-concentrated := top1_share > 0.5; distributed otherwise")
    P("  B: Still-supported := nonzero evidence in >= 5/10 sampled views")
    P("")

    # ---------- Q1 trigger view dominance ----------
    P("-" * 76)
    P("Q1  VCD trigger view concentration")
    P("-" * 76)
    P(f"{'seed':>4} {'n_trig':>7} {'conc%':>6} {'top1 med':>8} {'nz mean':>7} "
      f"{'ent mean':>8}  (early-window rows in parens)")
    group_of = {}
    for s in SEEDS:
        rows = [r for r in ev_rows if r["seed"] == s]
        early = [r for r in rows if fi(r["iteration"]) <= 3000]
        for r in rows:
            group_of[(s, r["parent_id"])] = (
                "concentrated" if ff(r["top1_share"]) > 0.5 else "distributed")
        for tag, sub in (("", rows), ("early", early)):
            t1 = np.array([ff(r["top1_share"]) for r in sub])
            nz = np.array([fi(r["nonzero_views"]) for r in sub])
            en = np.array([ff(r["entropy_norm"]) for r in sub])
            P(f"{s:>4} {len(sub):>7} {100 * (t1 > 0.5).mean():>5.1f}% "
              f"{np.median(t1):>8.3f} {nz.mean():>7.2f} {en.mean():>8.3f}"
              f"{'  (' + tag + ')' if tag else ''}")
    P("")

    # ---------- Q2 concentration vs child fate ----------
    P("-" * 76)
    P("Q2  Child fate by view-concentration group (early-window births)")
    P("-" * 76)
    P(f"{'seed':>4} {'group':>13} {'n_child':>8} {'@+500':>6} {'@+1000':>7} "
      f"{'@+3000':>7} {'@30k':>6} {'prune':>6} {'life':>6}")

    def fate_metrics(sub):
        sv = lambda k: mean([fi(r[k]) for r in sub])
        return (len(sub), sv("survived_500"), sv("survived_1000"),
                sv("survived_3000"), sv("survived_30k"),
                mean([fi(r["lifetime_iters"]) for r in sub]))

    seed_group = {s: {} for s in SEEDS}
    for s in SEEDS:
        for g in ("concentrated", "distributed"):
            sub = [r for r in fate_rows if r["seed"] == s and
                   group_of.get((s, r["parent_id"])) == g]
            n, v5, v10, v30, vk, lf = fate_metrics(sub)
            seed_group[s][g] = {"surv": vk, "life": lf, "n": n}
            P(f"{s:>4} {g:>13} {n:>8} {v5:>6.3f} {v10:>7.3f} {v30:>7.3f} "
              f"{vk:>6.3f} {1 - vk:>6.3f} {lf:>6.0f}")
    P("")

    # stratified by birth event AND child type (split/clone) — control
    P("Stratified by birth event (survival@30k), conc vs dist:")
    P(f"{'seed':>4} {'ev':>3} {'conc n/surv':>16} {'dist n/surv':>16}")
    strat_ok, strat_tot = 0, 0
    for s in SEEDS:
        for ev in STRATA:
            c = [r for r in fate_rows if r["seed"] == s and fi(r["birth_event"]) == ev
                 and group_of.get((s, r["parent_id"])) == "concentrated"]
            d = [r for r in fate_rows if r["seed"] == s and fi(r["birth_event"]) == ev
                 and group_of.get((s, r["parent_id"])) == "distributed"]
            vc = mean([fi(r["survived_30k"]) for r in c]) if len(c) >= 50 else float("nan")
            vd = mean([fi(r["survived_30k"]) for r in d]) if len(d) >= 50 else float("nan")
            P(f"{s:>4} {ev:>3} {len(c):>8}/{vc:>7.3f} {len(d):>8}/{vd:>7.3f}")
            if vc == vc and vd == vd:
                strat_tot += 1
                # A-go direction: concentrated children WORSE => vc < vd
                if vc < vd:
                    strat_ok += 1
    P("")

    # also split vs clone type control
    types = {}
    for r in fate_rows:
        g = group_of.get((r["seed"], r["parent_id"]))
        types.setdefault((r["seed"], r["child_type"], g), 0)
        types[(r["seed"], r["child_type"], g)] += 1
    P("child counts by (seed, type, group): " +
      ", ".join(f"{k}={v}" for k, v in sorted(types.items(), key=lambda x: str(x))))
    P("")

    # ---------- Q3/Q4 prune candidates ----------
    P("-" * 76)
    P("Q3/Q4  Low-opacity (<0.1) final-prune candidates: multi-view support")
    P("-" * 76)
    P(f"{'seed':>4} {'round':>5} {'it':>6} {'n_cand':>7} {'nz=0':>6} {'nz1-2':>6} "
      f"{'nz3-4':>6} {'nz>=5':>6} {'vis>=5':>7}")
    b_support_fracs = {s: [] for s in SEEDS}
    for r in sorted(rnd_rows, key=lambda x: (x["seed"], fi(x["round"]))):
        s, rd = r["seed"], fi(r["round"])
        sub = [c for c in cand_rows if c["seed"] == s and fi(c["round"]) == rd]
        if not sub:
            P(f"{s:>4} {rd:>5} {fi(r['it']):>6} {fi(r['n_candidates']):>7}  (no rows)")
            continue
        nz = np.array([fi(c["nonzero_views"]) for c in sub])
        vis = np.array([fi(c["visible_views"]) for c in sub])
        P(f"{s:>4} {rd:>5} {fi(r['it']):>6} {len(sub):>7} "
          f"{100 * (nz == 0).mean():>5.1f}% {100 * ((nz >= 1) & (nz <= 2)).mean():>5.1f}% "
          f"{100 * ((nz >= 3) & (nz <= 4)).mean():>5.1f}% "
          f"{100 * (nz >= 5).mean():>6.1f}% {100 * (vis >= 5).mean():>6.1f}%")
        b_support_fracs[s].append((nz >= 5).mean())
    P("")
    P("top1_share among candidates (pooled): "
      f"median={np.median([ff(c['top1_share']) for c in cand_rows]) if cand_rows else float('nan'):.3f}")

    # ---------- combined outputs ----------
    for r in ev_rows:
        r["group"] = group_of.get((r["seed"], r["parent_id"]), "")
    with open(f"{DATA}/densify_view_evidence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ev_rows[0].keys()))
        w.writeheader(); w.writerows(ev_rows)
    with open(f"{DATA}/densify_child_fate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fate_rows[0].keys()))
        w.writeheader(); w.writerows(fate_rows)
    with open(f"{DATA}/prune_multiview_support.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cand_rows[0].keys()))
        w.writeheader(); w.writerows(cand_rows)

    # seed_summary.csv
    summ = []
    for s in SEEDS:
        rows = [r for r in ev_rows if r["seed"] == s]
        n_conc = sum(1 for r in rows if r["group"] == "concentrated")
        cg = seed_group[s].get("concentrated", {"surv": float("nan"), "life": float("nan"), "n": 0})
        dg = seed_group[s].get("distributed", {"surv": float("nan"), "life": float("nan"), "n": 0})
        summ.append({
            "seed": s,
            "n_trigger_parents": len(rows),
            "pct_concentrated": round(100 * n_conc / max(len(rows), 1), 2),
            "A_n_children_conc": cg["n"], "A_n_children_dist": dg["n"],
            "A_surv30k_conc": cg["surv"], "A_surv30k_dist": dg["surv"],
            "A_lifetime_conc": cg["life"], "A_lifetime_dist": dg["life"],
            "B_stillsupported_frac_mean": round(mean(b_support_fracs[s]), 4),
            "B_n_candidates": sum(1 for c in cand_rows if c["seed"] == s),
        })
    with open(f"{DATA}/seed_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)

    # ---------- verdicts ----------
    P("-" * 76)
    P("VERDICT")
    P("-" * 76)
    # A: GO iff conc children consistently worse in 3/3 seeds by >= 0.05 surv
    d_surv, d_life, conc_big_enough = [], [], True
    for s in SEEDS:
        cg = seed_group[s].get("concentrated", {"n": 0})
        dg = seed_group[s].get("distributed", {"n": 0})
        if cg["n"] < 1000 or dg["n"] < 1000:
            conc_big_enough = False
        d_surv.append(dg["surv"] - cg["surv"])   # positive => conc worse
        d_life.append(dg["life"] - cg["life"])
    P(f"A: conc share per seed: "
      f"{[r_[1] for r_ in [(s, summ[i]['pct_concentrated']) for i, s in enumerate(SEEDS)]]}")
    P(f"A: dSurv30k (dist - conc) per seed: {['%+.3f' % d for d in d_surv]}; "
      f"dLifetime: {['%+.0f' % d for d in d_life]}")
    a_consistent = all(d >= GO_D_SURV for d in d_surv)
    if not conc_big_enough:
        a_verdict = "NO-GO (insufficient group sizes)"
    elif a_consistent and strat_ok >= 0.5 * strat_tot:
        a_verdict = "GO"
    else:
        a_verdict = "NO-GO"
    P(f"A: strata sign (conc worse) holds in {strat_ok}/{strat_tot} shared strata")
    P(f"A: Reliable Densification = {a_verdict}")

    # B: fraction of still-supported candidates, pooled per seed
    b_fracs = [mean(b_support_fracs[s]) for s in SEEDS]
    P(f"B: Still-supported fraction per seed (mean over rounds): "
      f"{['%.4f' % b for b in b_fracs]}")
    if all(b >= B_SUPPORTED_MIN_FRAC for b in b_fracs):
        b_verdict = "GO for further validation"
    elif all(b < EQUAL_BAND for b in b_fracs):
        b_verdict = "NO-GO"
    else:
        b_verdict = "NO-GO (supported fraction negligible/unstable)"
    P(f"B: Reliable Pruning = {b_verdict}")
    P("")
    P("NOTE: B measures FastGS's own VCD evidence (high-error-pixel coverage).")
    P("      Low evidence for a well-rendering Gaussian is NOT proof it is")
    P("      useless; a NO-GO only states that opacity<0.1 candidates are not")
    P("      flagrantly multi-view-supported by this metric.")

    stats = "\n".join(lines)
    print(stats)
    with open(f"{DATA}/b19r_stats.txt", "w") as f:
        f.write(stats + "\n")


def seed_str(s):
    return str(s)


if __name__ == "__main__":
    main()
