#
# Paper B - B20-V analysis: rendering support validity
#
# Verdict rules (fixed BEFORE running the experiment):
#   V1 strength      : median total contribution of B_multi >= 10x that of
#                      A_low (per seed, round 1), and B's median
#                      mean-per-supported-view >= 100x TOL (>= 1e-4).
#   V2 temporal      : Persistent-supported fraction among candidates dying at
#                      rounds 2-4 >= 5% in all 3 seeds.
#   V3 fairness      : mean |delta opacity| of matched pairs <= 0.01 in every
#                      (seed, round); else report matching quality, and if
#                      > 0.05 anywhere -> INCONCLUSIVE.
#   V4 core          : dPSNR = drop(B) - drop(A) > 0 mean-per-seed in 3/3
#                      seeds, positive in >= 8/12 (seed,round) cells, and
#                      overall mean dPSNR >= 0.1 dB.
#   GO  iff V1 & V2 & V3 & V4.
#   INCONCLUSIVE if V3 fails hard, or K < 100 in every round of some seed.
#   Otherwise NO-GO.
#

import os, csv
import numpy as np

BASE = "paper_b/b20_rendering_support_validity"
DATA = f"{BASE}/data"
SEEDS = [0, 1, 2]
TOL = 1e-6
V1_RATIO = 10.0
V2_MIN_FRAC = 0.05
V3_CALIPER = 0.01
V3_HARD = 0.05
V4_MIN_MEAN_DPSNR = 0.1


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fi(x):
    return int(float(x))


def ff(x):
    return float(x)


def main():
    strength, temporal, dels = [], [], []
    for s in SEEDS:
        for r in read_csv(f"{DATA}/contribution_strength_s{s}.csv"):
            r["seed"] = s; strength.append(r)
        for r in read_csv(f"{DATA}/temporal_support_s{s}.csv"):
            r["seed"] = s; temporal.append(r)
        for r in read_csv(f"{DATA}/matched_deletion_results_s{s}.csv"):
            r["seed"] = s; dels.append(r)

    lines = []
    P = lines.append
    P("=" * 76)
    P("B20-V: Rendering Support Validity (Room, 3 seeds, matched deletion)")
    P("=" * 76)
    P(f"candidates: {len(strength)}   temporal rows: {len(temporal)}   "
      f"deletion branches: {len(dels)}")
    P("")

    # ---------- V1 contribution strength ----------
    P("-" * 76)
    P("V1  Contribution strength: A_low vs B_multi")
    P("-" * 76)
    P(f"{'seed':>4} {'round':>5} {'grp':>7} {'n':>7} {'total med':>11} "
      f"{'mean/sv med':>12} {'sv med':>7}")
    v1_ok = True
    for s in SEEDS:
        for grp, ratio_check in (("A_low", None), ("B_multi", None)):
            sub = [r for r in strength if r["seed"] == s and r["group"] == grp]
            tm = np.median([ff(r["total_contribution"]) for r in sub]) if sub else 0.0
            mm = np.median([ff(r["mean_per_supported_view"]) for r in sub]) if sub else 0.0
            sm = np.median([fi(r["support_view_count"]) for r in sub]) if sub else 0
            P(f"{s:>4} {'all':>5} {grp:>7} {len(sub):>7} {tm:>11.4f} "
              f"{mm:>12.5f} {sm:>7}")
        a1 = np.median([ff(r["total_contribution"]) for r in strength
                        if r["seed"] == s and r["group"] == "A_low" and fi(r["round"]) == 1])
        b1 = np.median([ff(r["total_contribution"]) for r in strength
                        if r["seed"] == s and r["group"] == "B_multi" and fi(r["round"]) == 1])
        bm = np.median([ff(r["mean_per_supported_view"]) for r in strength
                        if r["seed"] == s and r["group"] == "B_multi"])
        ok1 = (b1 >= V1_RATIO * max(a1, TOL))
        ok2 = (bm >= 100 * TOL)
        v1_ok &= (ok1 and ok2)
        P(f"     round1 median B/A total = {b1 / max(a1, TOL):.1f}x "
          f"(>= {V1_RATIO:.0f}x: {ok1}); B mean/sv median = {bm:.5f} (>=1e-4: {ok2})")
    P("")

    # ---------- V2 temporal stability ----------
    P("-" * 76)
    P("V2  Temporal support (candidates dying at rounds 2-4)")
    P("-" * 76)
    P(f"{'seed':>4} {'n_cand':>7} {'Low':>7} {'Transient':>10} {'Persistent':>11}")
    v2_ok = True
    for s in SEEDS:
        sub = [r for r in temporal if r["seed"] == s and fi(r["death_round"]) >= 2]
        n = len(sub)
        c = {k: sum(1 for r in sub if r["temporal_class"] == k)
             for k in ("Low-support", "Transient-supported", "Persistent-supported")}
        frac_p = c["Persistent-supported"] / max(n, 1)
        v2_ok &= (frac_p >= V2_MIN_FRAC)
        P(f"{s:>4} {n:>7} {c['Low-support']:>7} {c['Transient-supported']:>10} "
          f"{c['Persistent-supported']:>11}  ({100 * frac_p:.1f}%)")
    P("")

    # ---------- V3 + V4 matched deletion ----------
    P("-" * 76)
    P("V4  Matched deletion: quality drop by branch (dPSNR, dSSIM x1e3, dLPIPS x1e3)")
    P("-" * 76)
    P(f"{'seed':>4} {'round':>5} {'K':>6} {'|dOp|':>7} "
      f"{'A dPSNR':>8} {'B dPSNR':>8} {'R dPSNR':>8} {'B-A':>7}")
    cell_pos = 0
    n_cell = 0
    seed_mean = {s: [] for s in SEEDS}
    all_d = []
    v3_fair = True
    v3_hard_fail = False
    summary_rows = []
    by_sr = {}
    for d in dels:
        by_sr.setdefault((d["seed"], fi(d["round"])), {})[d["branch"]] = d
    for (s, rd), br in sorted(by_sr.items()):
        K = fi(next(iter(br.values()))["K"])
        gap = ff(next(iter(br.values()))["mean_abs_opacity_gap"])
        if gap > V3_CALIPER:
            v3_fair = False
        if gap > V3_HARD:
            v3_hard_fail = True
        da = ff(br["A_low"]["dpsnr"]); db = ff(br["B_multi"]["dpsnr"])
        dr = ff(br["R_random"]["dpsnr"])
        d = db - da
        n_cell += 1
        cell_pos += (d > 0)
        seed_mean[s].append((da, db, dr))
        all_d.append(d)
        P(f"{s:>4} {rd:>5} {K:>6} {gap:>7.4f} {da:>8.3f} {db:>8.3f} {dr:>8.3f} "
          f"{d:>+7.3f}")
        summary_rows.append({
            "seed": s, "round": rd, "K": K,
            "mean_abs_opacity_gap": gap,
            "dpsnr_A_low": da, "dpsnr_B_multi": db, "dpsnr_R_random": dr,
            "dpsnr_B_minus_A": round(d, 4),
            "dssim_A_low": ff(br["A_low"]["dssim"]),
            "dssim_B_multi": ff(br["B_multi"]["dssim"]),
            "dlpips_A_low": ff(br["A_low"]["dlpips"]),
            "dlpips_B_multi": ff(br["B_multi"]["dlpips"]),
        })
    P("")
    P("Per-seed mean dPSNR (B - A):")
    for s in SEEDS:
        if seed_mean[s]:
            m = np.mean([db - da for da, db, _ in seed_mean[s]])
            P(f"  seed {s}: {m:+.3f} dB over {len(seed_mean[s])} rounds")
    mean_all = float(np.mean(all_d)) if all_d else float("nan")
    P(f"overall: mean dPSNR(B-A) = {mean_all:+.3f} dB; positive cells "
      f"{cell_pos}/{n_cell}")

    # LPIPS direction (B should hurt more => dlpips_B > dlpips_A)
    lp_pos = sum(1 for row in summary_rows
                 if row["dlpips_B_multi"] > row["dlpips_A_low"])
    P(f"LPIPS direction (B hurts more): {lp_pos}/{len(summary_rows)} cells")
    P("")

    with open(f"{DATA}/contribution_strength.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(strength[0].keys()))
        w.writeheader(); w.writerows(strength)
    with open(f"{DATA}/temporal_support.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(temporal[0].keys()))
        w.writeheader(); w.writerows(temporal)
    with open(f"{DATA}/matched_deletion_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dels[0].keys()))
        w.writeheader(); w.writerows(dels)
    with open(f"{DATA}/seed_event_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    # ---------- verdict ----------
    P("-" * 76)
    P("VERDICT")
    P("-" * 76)
    P(f"V1 strength OK: {v1_ok}")
    P(f"V2 temporal OK: {v2_ok}")
    P(f"V3 matching fair (caliper {V3_CALIPER}): {v3_fair}; hard fail: {v3_hard_fail}")
    seed_pos = sum(1 for s in SEEDS if seed_mean[s] and
                   np.mean([db - da for da, db, _ in seed_mean[s]]) > 0)
    v4_ok = (seed_pos == 3 and cell_pos >= 0.67 * n_cell
             and mean_all >= V4_MIN_MEAN_DPSNR)
    P(f"V4 core: seeds 3/3 positive ({seed_pos}/3), cells {cell_pos}/{n_cell}, "
      f"mean dPSNR {mean_all:+.3f} (>= {V4_MIN_MEAN_DPSNR}): {v4_ok}")
    k_too_small = any(
        all(row["K"] < 100 for row in summary_rows if row["seed"] == s)
        for s in SEEDS)
    if v3_hard_fail or k_too_small:
        verdict = "INCONCLUSIVE"
    elif v1_ok and v2_ok and v3_fair and v4_ok:
        verdict = "GO"
    else:
        verdict = "NO-GO"
    P("")
    P(f"FINAL VERDICT: Reliable Pruning = {verdict}")

    stats = "\n".join(lines)
    print(stats)
    with open(f"{DATA}/b20v_stats.txt", "w") as f:
        f.write(stats + "\n")


if __name__ == "__main__":
    main()
