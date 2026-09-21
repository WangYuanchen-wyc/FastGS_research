#
# Paper B - B21-V1 analysis: densification benefit heterogeneity
#
# Primary: Benefit_ROI at horizon +400 (before the next event at +500).
# Verdict rules (fixed before running):
#   near-zero event  : |Benefit_ROI(+400)| <= 10% of the seed's median positive
#                      Benefit_ROI.
#   high-benefit     : Benefit_ROI(+400) >= 2x the seed's median positive.
#   GO  iff in >= 2/3 seeds: >= 2 of 6 events are near-zero AND those events'
#       Added_GS >= 30% of the seed's median Added_GS (low benefit not
#       explained by low capacity) AND >= 1 event is high-benefit (and the
#       third seed has >= 1 near-zero event).
#   NO-GO iff in >= 2/3 seeds: >= 5 of 6 events have benefit > 50% of the
#       seed's median positive AND CV across events < 0.5.
#   INVALID if implementation self-checks failed (diagnostic raises).
#

import os, csv
import numpy as np

BASE = "paper_b/b21_v1_densification_benefit"
DATA = f"{BASE}/data"
SEEDS = [0, 1, 2]
PRIMARY_HZ = 400
NEAR_ZERO_FRAC = 0.10
HIGH_MULT = 2.0
CAP_FRAC = 0.30


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ff(x):
    return float(x)


def fi(x):
    return int(float(x))


def main():
    roi, eff, evs, brs = [], [], [], []
    for s in SEEDS:
        for r in read_csv(f"{DATA}/roi_benefit_s{s}.csv"):
            r["seed"] = s; roi.append(r)
        for r in read_csv(f"{DATA}/capacity_efficiency_s{s}.csv"):
            r["seed"] = s; eff.append(r)
        for r in read_csv(f"{DATA}/densify_events_s{s}.csv"):
            r["seed"] = s; evs.append(r)
        for r in read_csv(f"{DATA}/paired_branch_metrics_s{s}.csv"):
            r["seed"] = s; brs.append(r)
    # combined copies (deliverables)
    for name, rows in (("roi_benefit", roi), ("capacity_efficiency", eff),
                       ("densify_events", evs), ("paired_branch_metrics", brs)):
        with open(f"{DATA}/{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    lines = []
    P = lines.append
    P("=" * 76)
    P("B21-V1: Densification Benefit (paired counterfactual, Room, 3 seeds)")
    P("=" * 76)
    P(f"native events recorded: {len(evs)}; selected-event pairs: "
      f"{len(set((r['seed'], r['event_it']) for r in roi))}")
    P("")

    # ---------- all native events ----------
    P("-" * 76)
    P("All native densification events (mainline)")
    P("-" * 76)
    P(f"{'seed':>4} {'it':>6} {'sel':>3} {'split':>6} {'clone':>5} "
      f"{'added':>6} {'#GS after':>9}")
    for r in evs:
        if fi(r["iteration"]) in (1500, 2500, 5000, 7000, 11000, 13000) or \
                fi(r["iteration"]) % 2000 == 0:
            P(f"{r['seed']:>4} {fi(r['iteration']):>6} {r['selected']:>3} "
              f"{fi(r['trigger_split']):>6} {fi(r['trigger_clone']):>5} "
              f"{fi(r['gs_added']):>6} {fi(r['gs_after']):>9}")
    P("")

    # ---------- paired results at +400 ----------
    P("-" * 76)
    P(f"Paired counterfactual results at horizon +{PRIMARY_HZ}")
    P("-" * 76)
    P(f"{'seed':>4} {'it':>6} {'phase':>6} {'added':>6} {'ROI err A':>10} "
      f"{'ROI err B':>10} {'Benefit_ROI':>11} {'dPSNR':>8} {'dLPIPS':>8}")
    per_seed = {s: {} for s in SEEDS}
    for r in roi:
        if fi(r["horizon"]) != PRIMARY_HZ:
            continue
        s = r["seed"]
        per_seed[s][fi(r["event_it"])] = r
        P(f"{s:>4} {fi(r['event_it']):>6} {r['phase']:>6} {fi(r['added_gs']):>6} "
          f"{ff(r['roi_l1_densify']):>10.5f} {ff(r['roi_l1_nodensify']):>10.5f} "
          f"{ff(r['benefit_roi']):>11.5f} {ff(r['dpsnr']):>+8.4f} "
          f"{ff(r['dlpips']):>+8.5f}")
    P("")

    # benefit sign stats
    ben = [ff(r["benefit_roi"]) for r in roi if fi(r["horizon"]) == PRIMARY_HZ]
    P(f"Benefit_ROI(+400) over {len(ben)} pairs: positive "
      f"{sum(1 for b in ben if b > 0)}/{len(ben)}, "
      f"median {np.median(ben):.5f}, mean {np.mean(ben):.5f}, "
      f"min {min(ben):.5f}, max {max(ben):.5f}")
    P("")

    # ---------- per-seed heterogeneity ----------
    P("-" * 76)
    P("Heterogeneity per seed (Benefit_ROI at +400)")
    P("-" * 76)
    n_low_seeds = 0
    n_similar_seeds = 0
    seed_stats = {}
    for s in SEEDS:
        d = per_seed[s]
        its = sorted(d)
        bvals = np.array([ff(d[i]["benefit_roi"]) for i in its])
        adds = np.array([fi(d[i]["added_gs"]) for i in its])
        pos = bvals[bvals > 0]
        med_pos = np.median(pos) if len(pos) else 0.0
        near_zero = [i for i, b in zip(its, bvals)
                     if abs(b) <= NEAR_ZERO_FRAC * max(med_pos, 1e-12)]
        high = [i for i, b in zip(its, bvals) if b >= HIGH_MULT * max(med_pos, 1e-12)]
        low_events_big = [i for i in near_zero
                          if fi(d[i]["added_gs"]) >= CAP_FRAC * np.median(adds)]
        cv = (bvals.std(ddof=1) / abs(np.mean(bvals))) if len(bvals) > 1 and \
            abs(np.mean(bvals)) > 0 else float("inf")
        similar = sum(1 for b in bvals if b > 0.5 * max(med_pos, 1e-12))
        seed_stats[s] = dict(near_zero=near_zero, high=high, low_big=low_events_big,
                             cv=cv, similar=similar, med_pos=med_pos)
        P(f"seed {s}: benefits " +
          ", ".join(f"{i}:{b:+.4f}" for i, b in zip(its, bvals)))
        P(f"  median_positive={med_pos:.5f}  near-zero events={near_zero} "
          f"(of which Added_GS>=30% median: {low_events_big})  "
          f"high-benefit events={high}  CV={cv:.2f}  "
          f"events>50% median: {similar}/6")
        P("")
        if len(near_zero) >= 2 and len(low_events_big) >= 1 and len(high) >= 1:
            n_low_seeds += 1
        elif len(near_zero) >= 1:
            n_low_seeds += 0  # partial credit handled in GO logic below
        if similar >= 5 and cv < 0.5:
            n_similar_seeds += 1

    # ---------- capacity efficiency ----------
    P("-" * 76)
    P("Capacity efficiency (Benefit_ROI / Added_GS at +400)")
    P("-" * 76)
    for s in SEEDS:
        d = per_seed[s]
        e = [(fi(d[i]["added_gs"]), ff(d[i]["benefit_roi"]) / max(fi(d[i]["added_gs"]), 1))
             for i in sorted(d)]
        P(f"seed {s}: " + ", ".join(f"it{i}:{v:.2e}" for i, v in e))
    P("")

    # ---------- verdict ----------
    P("-" * 76)
    P("VERDICT (pre-registered)")
    P("-" * 76)
    go_seeds = 0
    partial_seeds = 0
    for s in SEEDS:
        st = seed_stats[s]
        if len(st["near_zero"]) >= 2 and len(st["low_big"]) >= 1 and len(st["high"]) >= 1:
            go_seeds += 1
        elif len(st["near_zero"]) >= 1:
            partial_seeds += 1
    P(f"seeds meeting full GO pattern (2+ near-zero, low-not-low-capacity, "
      f"1+ high): {go_seeds}/3; seeds with >=1 near-zero: "
      f"{go_seeds + partial_seeds}/3")
    P(f"seeds with similar-and-positive benefits (>=5/6 events >50% median, "
      f"CV<0.5): {n_similar_seeds}/3")

    if go_seeds >= 2 and (go_seeds + partial_seeds) >= 3:
        verdict = "GO"
        P("* stable heterogeneity with low-benefit events not explained by "
          "low capacity -> V1 = GO (proceed to V2 in a later phase)")
    elif n_similar_seeds >= 2:
        verdict = "NO-GO"
        P("* most events gain similar and clear benefit -> NO-GO")
    elif go_seeds + partial_seeds == 0:
        verdict = "NO-GO"
        P("* no near-zero events anywhere -> NO-GO")
    else:
        verdict = "NO-GO"
        P("* pattern incomplete across seeds -> NO-GO")
    P("")
    P(f"FINAL VERDICT: V1 = {verdict}")

    stats = "\n".join(lines)
    print(stats)
    with open(f"{DATA}/b21v1_stats.txt", "w") as f:
        f.write(stats + "\n")

    # seed_event_summary.csv
    with open(f"{DATA}/seed_event_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "event_it", "phase", "added_gs",
                    "benefit_roi_400", "dpsnr_400", "near_zero", "high_benefit"])
        for s in SEEDS:
            d = per_seed[s]
            st = seed_stats[s]
            for i in sorted(d):
                r = d[i]
                w.writerow([s, i, r["phase"], r["added_gs"], r["benefit_roi"],
                            r["dpsnr"], int(i in st["near_zero"]),
                            int(i in st["high"])])


if __name__ == "__main__":
    main()
