#
# Paper B - B18-D Final Fix analysis: One-shot vs Persistent child fate
#
# Inputs : final_fix/data/{fate_s*, lineage_s*, parents_s*}.csv  (3 seeds)
# Outputs: final_fix/data/{parent_persistence_groups, parent_child_lineage,
#          group_child_fate, seed_group_summary}.csv, b18d_final_fix_stats.txt
#
# Verdict rules (fixed before reading results):
#   INCONCLUSIVE if lineage checks fail or Persistent-3 children < 500 per seed.
#   GO  if Persistent-2/3 beat One-shot on >=1 primary metric (final survival,
#       lifetime, opacity, prune ratio) with same sign in 3/3 seeds, meaningful
#       magnitude (|dSurv| >= 0.05 or |dLifetime%| >= 10% or |dOpacity| >= 0.02
#       or |dPrune| >= 0.05), AND the sign still holds in most shared
#       birth-event strata (birth-timing control).
#   NO-GO if groups are basically equal (|dSurv| < 0.02), or signs are
#       inconsistent across seeds, or the strata control removes the effect.
#

import os, csv
import numpy as np

BASE = os.environ.get("B18DFF_ANALYSIS_BASE",
                      "paper_b/b18_densification_persistence/final_fix")
DATA = f"{BASE}/data"
GROUPS = ["One-shot", "Persistent-2", "Persistent-3"]
SEEDS = [0, 1, 2]
STRATA = [2, 3, 4, 5]  # birth events with potential group mix (ev1 = OS only)

MIN_P3_CHILDREN = 500

# GO magnitude thresholds
D_SURV = 0.05
D_LIFETIME_REL = 0.10
D_OPACITY = 0.02
D_PRUNE = 0.05
EQUAL_BAND = 0.02


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fi(x):
    return int(float(x))


def ff(x):
    v = float(x)
    return v if v == v else np.nan  # nan-safe


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def main():
    fates, lineages, parents = [], [], []
    for s in SEEDS:
        for r in read_csv(f"{DATA}/fate_s{s}.csv"):
            r["seed"] = s
            fates.append(r)
        for r in read_csv(f"{DATA}/lineage_s{s}.csv"):
            r["seed"] = s
            lineages.append(r)
        for r in read_csv(f"{DATA}/parents_s{s}.csv"):
            r["seed"] = s
            parents.append(r)

    lines = []
    P = lines.append
    P("=" * 76)
    P("B18-D FINAL FIX: One-shot vs Persistent Child Fate (Room, 3 seeds)")
    P("=" * 76)
    P(f"parent occurrences: {len(parents)}  children tracked: {len(fates)}  "
      f"lineage rows: {len(lineages)}")

    # ---------- lineage reliability ----------
    pruned_at_birth = sum(fi(r["n_children_born"]) - fi(r["n_children_kept"])
                          for r in parents)
    # every fate child must have a unique child_id (one parent each).
    # children pruned at birth are precisely identified and excluded from
    # fate; tolerate a negligible fraction (native multinomial prune can hit
    # newborn slots via its misaligned score->position mapping).
    cids = [r["child_id"] for r in fates]
    n_born_total = sum(fi(r["n_children_born"]) for r in parents)
    pab_frac = pruned_at_birth / max(n_born_total, 1)
    lineage_ok = (len(cids) == len(set(cids))) and pab_frac < 0.01
    P(f"lineage check: unique child_id = {len(cids) == len(set(cids))}, "
      f"pruned_at_birth = {pruned_at_birth}/{n_born_total} ({100 * pab_frac:.3f}%)")
    P("")

    # ---------- Q1 parent group composition ----------
    P("-" * 76)
    P("Q1  Parent persistence groups (parent occurrences, early births it1000-3000)")
    P("-" * 76)
    P(f"{'seed':>4} {'group':>13} {'n_parents':>9} {'pct':>7} "
      f"{'birth_it mean':>13} {'p25/p50/p75':>16}")
    pg_rows = []
    seed_p = {s: [r for r in parents if r["seed"] == s] for s in SEEDS}
    for s in SEEDS:
        ps = seed_p[s]
        for g in GROUPS:
            sub = [r for r in ps if r["group"] == g]
            its = sorted(fi(r["iteration"]) for r in sub)
            pct = 100.0 * len(sub) / max(len(ps), 1)
            q = (its[len(its) // 4], its[len(its) // 2], its[3 * len(its) // 4]) \
                if its else (0, 0, 0)
            P(f"{s:>4} {g:>13} {len(sub):>9} {pct:>6.1f}% "
              f"{mean(its):>13.0f} {q[0]:>5}/{q[1]}/{q[2]}")
            pg_rows.append({"seed": s, "group": g, "n_parents": len(sub),
                            "pct": round(pct, 2), "birth_it_mean": round(mean(its), 1),
                            "birth_it_p25": q[0], "birth_it_p50": q[1],
                            "birth_it_p75": q[2]})
    P("")
    with open(f"{DATA}/parent_persistence_groups.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pg_rows[0].keys()))
        w.writeheader(); w.writerows(pg_rows)

    # combined lineage + fate outputs
    with open(f"{DATA}/parent_child_lineage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lineages[0].keys()))
        w.writeheader(); w.writerows(lineages)
    with open(f"{DATA}/group_child_fate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fates[0].keys()))
        w.writeheader(); w.writerows(fates)

    # ---------- Q2/Q3 per-seed group summary ----------
    P("-" * 76)
    P("Q2/Q3  Child fate by group, per seed")
    P("-" * 76)
    P(f"{'seed':>4} {'group':>13} {'n_child':>8} {'@+500':>6} {'@+1000':>7} "
      f"{'@+3000':>7} {'@30k':>6} {'opacity':>8} {'visib':>6} {'prune':>6} "
      f"{'lifetime':>9}")
    sg_rows = []
    seed_f = {s: [r for r in fates if r["seed"] == s] for s in SEEDS}
    for s in SEEDS:
        fs = seed_f[s]
        for g in GROUPS:
            sub = [r for r in fs if r["parent_group"] == g]
            if not sub:
                continue
            sv = lambda k: mean([fi(r[k]) for r in sub])
            op = mean([ff(r["final_opacity"]) for r in sub])
            vis = mean([fi(r["visible_01"]) for r in sub])
            life = mean([fi(r["lifetime_iters"]) for r in sub])
            P(f"{s:>4} {g:>13} {len(sub):>8} {sv('survived_500'):>6.3f} "
              f"{sv('survived_1000'):>7.3f} {sv('survived_3000'):>7.3f} "
              f"{sv('survived_30k'):>6.3f} {op:>8.3f} {vis:>6.3f} "
              f"{1 - sv('survived_30k'):>6.3f} {life:>9.0f}")
            sg_rows.append({
                "seed": s, "group": g, "n_children": len(sub),
                "survival_500": round(sv("survived_500"), 4),
                "survival_1000": round(sv("survived_1000"), 4),
                "survival_3000": round(sv("survived_3000"), 4),
                "survival_30k": round(sv("survived_30k"), 4),
                "final_opacity": round(op, 4), "visible_frac": round(vis, 4),
                "prune_ratio": round(1 - sv("survived_30k"), 4),
                "mean_lifetime_iters": round(life, 1),
            })
    P("")
    with open(f"{DATA}/seed_group_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sg_rows[0].keys()))
        w.writeheader(); w.writerows(sg_rows)

    # ---------- Q5 birth-timing control (stratified by birth event) ----------
    P("-" * 76)
    P("Q5  Birth-timing control: survival@30k within each birth event")
    P("-" * 76)
    P(f"{'seed':>4} {'ev':>3} {'OS n/surv':>16} {'P2 n/surv':>16} {'P3 n/surv':>16}")

    def strat_val(seed, ev, g):
        sub = [r for r in seed_f[seed]
               if fi(r["birth_event"]) == ev and r["parent_group"] == g]
        return len(sub), mean([fi(r["survived_30k"]) for r in sub])

    strat_delta = {g: [] for g in ("Persistent-2", "Persistent-3")}  # per seed
    for s in SEEDS:
        for ev in STRATA:
            n1, v1 = strat_val(s, ev, "One-shot")
            n2, v2 = strat_val(s, ev, "Persistent-2")
            n3, v3 = strat_val(s, ev, "Persistent-3")
            P(f"{s:>4} {ev:>3} {n1:>8}/{v1:>7.3f} "
              f"{n2:>8}/{v2:>7.3f} "
              f"{n3:>8}/{v3:>7.3f}")
        P("")
        # per-seed stratified group means: unweighted mean over strata where the
        # group has >=50 children (equal birth-event weight)
        def strat_mean(g, key="survived_30k"):
            vals = []
            for ev in STRATA:
                sub = [r for r in seed_f[s]
                       if fi(r["birth_event"]) == ev and r["parent_group"] == g]
                if len(sub) >= 50:
                    vals.append(mean([fi(r[key]) for r in sub]))
            return mean(vals) if vals else float("nan")

        for g in ("Persistent-2", "Persistent-3"):
            strat_delta[g].append((strat_mean(g), strat_mean("One-shot")))

    P("Stratified (equal weight per birth event) survival@30k, per seed:")
    P(f"{'seed':>4} {'P2 vs OS':>16} {'P3 vs OS':>16}")
    for i, s in enumerate(SEEDS):
        P(f"{s:>4} "
          f"{strat_delta['Persistent-2'][i][0]:>7.3f}/{strat_delta['Persistent-2'][i][1]:.3f} "
          f"{strat_delta['Persistent-3'][i][0]:>7.3f}/{strat_delta['Persistent-3'][i][1]:.3f}")
    P("")

    # ---------- Q6 verdict ----------
    P("-" * 76)
    P("VERDICT")
    P("-" * 76)

    p3_counts = [sum(1 for r in seed_f[s] if r["parent_group"] == "Persistent-3")
                 for s in SEEDS]
    p2_counts = [sum(1 for r in seed_f[s] if r["parent_group"] == "Persistent-2")
                 for s in SEEDS]
    P(f"P2 children per seed: {p2_counts};  P3 children per seed: {p3_counts}")

    if not lineage_ok:
        verdict = "INCONCLUSIVE"
        P("* lineage reliability check FAILED -> INCONCLUSIVE")
    elif min(p3_counts) < MIN_P3_CHILDREN:
        verdict = "INCONCLUSIVE"
        P(f"* Persistent-3 children < {MIN_P3_CHILDREN} in at least one seed "
          f"-> sample too small -> INCONCLUSIVE")
    else:
        # gather per-seed, per-group metrics (whole early cohort + stratified)
        def seed_metrics(g):
            out = []
            for s in SEEDS:
                sub = [r for r in seed_f[s] if r["parent_group"] == g]
                out.append({
                    "surv": mean([fi(r["survived_30k"]) for r in sub]),
                    "life": mean([fi(r["lifetime_iters"]) for r in sub]),
                    "op": mean([ff(r["final_opacity"]) for r in sub]),
                })
            return out

        os_m = seed_metrics("One-shot")
        verdict_signals = []
        all_equal = True
        for g in ("Persistent-2", "Persistent-3"):
            gm = seed_metrics(g)
            dsurv = [gm[i]["surv"] - os_m[i]["surv"] for i in range(3)]
            dprune = [-d for d in dsurv]  # prune ratio = 1 - survival, by def
            dlife = [(gm[i]["life"] - os_m[i]["life"]) / max(os_m[i]["life"], 1)
                     for i in range(3)]
            dop = [gm[i]["op"] - os_m[i]["op"] for i in range(3)]
            P(f"{g} vs One-shot  dSurv30k={['%+.3f' % d for d in dsurv]}  "
              f"dPrune={['%+.3f' % d for d in dprune]}  "
              f"dLifetime%={['%+.1f%%' % (100 * d) for d in dlife]}  "
              f"dOpacity={['%+.3f' % d for d in dop]}")
            if not all(abs(d) < EQUAL_BAND for d in dsurv):
                all_equal = False
            for name, deltas, thr in (("survival", dsurv, D_SURV),
                                      ("lifetime", dlife, D_LIFETIME_REL),
                                      ("opacity", dop, D_OPACITY)):
                same_sign = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
                big = all(abs(d) >= thr for d in deltas)
                if same_sign and big:
                    verdict_signals.append((g, name))
        # strata control: does the sign of the pooled stratified delta hold in
        # most strata (per seed, >=2 of the strata where both groups have >=50)?
        strata_holds = {}
        for g in ("Persistent-2", "Persistent-3"):
            holds, common = 0, 0
            for s in SEEDS:
                for ev in STRATA:
                    n1, v1 = strat_val(s, ev, "One-shot")
                    n2, v2 = strat_val(s, ev, g)
                    if n1 >= 50 and n2 >= 50 and v1 == v1 and v2 == v2:
                        common += 1
                        pooled = strat_delta[g][SEEDS.index(s)]
                        sign = (pooled[0] - pooled[1]) > 0
                        if (v2 > v1) == sign:
                            holds += 1
            strata_holds[g] = (holds, common)
            P(f"strata control {g}: sign holds in {holds}/{common} shared strata")

        if verdict_signals:
            ok_strata = all(
                strata_holds[g][0] >= 0.5 * strata_holds[g][1]
                if strata_holds[g][1] > 0 else False
                for (g, name) in verdict_signals)
            if ok_strata:
                verdict = "GO"
                P(f"* GO-signals: {verdict_signals}; strata control passed "
                  f"-> Historical/Persistent Densification = GO")
            else:
                verdict = "NO-GO"
                P(f"* GO-signals present but birth-timing control NOT passed "
                  f"-> NO-GO (apparent advantage explained by birth timing)")
        elif all_equal:
            verdict = "NO-GO"
            P("* one-shot and persistent children are essentially equal "
              "(all |dSurv| < 0.02) -> NO-GO")
        else:
            verdict = "NO-GO"
            P("* differences exist but are not consistent + meaningful across "
              "3 seeds on any primary metric -> NO-GO (unstable differences)")

    P("")
    P(f"FINAL VERDICT: Historical / Persistent Densification = {verdict}")

    stats = "\n".join(lines)
    print(stats)
    with open(f"{DATA}/b18d_final_fix_stats.txt", "w") as f:
        f.write(stats + "\n")
    return verdict


if __name__ == "__main__":
    main()
