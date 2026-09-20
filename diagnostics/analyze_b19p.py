#
# Paper B - B19-P analysis: actual rendering support of opacity-prune candidates
#
# Inputs : b19_reliable_pruning/data/{prune_candidate_render_support,round_stats}_s*.csv
# Outputs: prune_candidate_render_support.csv, event_seed_summary.csv,
#          b19p_stats.txt
#
# Definitions (fixed a priori):
#   contribution  = per-Gaussian blend weight alpha*T summed over all pixels of
#                   a view (real rasterizer quantity, instrumented forward pass)
#   support_view_count = #views with contribution > TOL = 1e-6 (float-zero only)
#   no contribution    : total <= TOL (zero views supported)
#   few-view supported : support in 1-2 views
#   multi-view         : support in >= 3 views ("Still-supported" for verdict)
#   Verdict GO iff multi-view fraction >= 5% of candidates in ALL 3 seeds
#   (candidate-weighted) and present in every prune round with n>=500.
#

import os, csv
import numpy as np

BASE = "paper_b/b19_reliable_pruning"
DATA = f"{BASE}/data"
SEEDS = [0, 1, 2]
TOL = 1e-6
GO_MIN_FRAC = 0.05
MIN_N_ROUND = 500


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fi(x):
    return int(float(x))


def main():
    cand_rows, rnd_rows = [], []
    for s in SEEDS:
        for r in read_csv(f"{DATA}/prune_candidate_render_support_s{s}.csv"):
            r["seed"] = s
            cand_rows.append(r)
        for r in read_csv(f"{DATA}/round_stats_s{s}.csv"):
            r["seed"] = s
            rnd_rows.append(r)

    lines = []
    P = lines.append
    P("=" * 76)
    P("B19-P: Actual Rendering Support of Opacity-Prune Candidates (Room, 3 seeds)")
    P("=" * 76)
    P(f"candidates: {len(cand_rows)}   prune rounds: {len(rnd_rows)}")
    P("contribution = alpha*T blend weight summed over pixels (instrumented")
    P("rasterizer); support_view_count = #views with contribution > 1e-6")
    P("")

    # ---------- per (seed, round) table ----------
    P("-" * 76)
    P("Q1-Q4  Candidate support distribution per (seed, prune event)")
    P("-" * 76)
    P(f"{'seed':>4} {'round':>5} {'it':>6} {'n_cand':>7} {'none':>7} {'1-2 views':>9} "
      f"{'>=3 views':>9} {'>=5 views':>9} {'top1|multi':>10}")
    summary_rows = []
    frac_by_seed = {s: [] for s in SEEDS}
    for r in sorted(rnd_rows, key=lambda x: (x["seed"], fi(x["round"]))):
        s, rd = r["seed"], fi(r["round"])
        sub = [c for c in cand_rows if c["seed"] == s and fi(c["round"]) == rd]
        if not sub:
            continue
        sv = np.array([fi(c["support_view_count"]) for c in sub])
        f_none = (sv == 0).mean()
        f_12 = ((sv >= 1) & (sv <= 2)).mean()
        f_3 = (sv >= 3).mean()
        f_5 = (sv >= 5).mean()
        multi = [c for c, v in zip(sub, sv) if v >= 3]
        t1m = (np.median([float(c["top1_share"]) for c in multi])
               if multi else float("nan"))
        P(f"{s:>4} {rd:>5} {fi(r['it']):>6} {len(sub):>7} "
          f"{100 * f_none:>6.2f}% {100 * f_12:>8.2f}% "
          f"{100 * f_3:>8.2f}% {100 * f_5:>8.2f}% {t1m:>10.3f}")
        summary_rows.append({
            "seed": s, "round": rd, "it": fi(r["it"]), "n_candidates": len(sub),
            "frac_no_contribution": round(float(f_none), 5),
            "frac_few_view_1_2": round(float(f_12), 5),
            "frac_multi_view_ge3": round(float(f_3), 5),
            "frac_multi_view_ge5": round(float(f_5), 5),
            "median_top1_share_among_multi": (round(float(t1m), 4) if t1m == t1m else ""),
            "mean_total_contribution": round(float(np.mean(
                [float(c["total_contribution"]) for c in sub])), 6),
            "median_total_contribution": round(float(np.median(
                [float(c["total_contribution"]) for c in sub])), 6),
        })
        frac_by_seed[s].append((len(sub), f_3))
    P("")

    # ---------- combined + summary outputs ----------
    with open(f"{DATA}/prune_candidate_render_support.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cand_rows[0].keys()))
        w.writeheader(); w.writerows(cand_rows)
    with open(f"{DATA}/event_seed_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    # ---------- verdict ----------
    P("-" * 76)
    P("VERDICT")
    P("-" * 76)
    weighted = {}
    for s in SEEDS:
        n_tot = sum(n for n, _ in frac_by_seed[s])
        weighted[s] = (sum(n * f for n, f in frac_by_seed[s]) / max(n_tot, 1)
                       if n_tot else float("nan"))
        P(f"seed {s}: candidate-weighted multi-view(>=3) fraction = "
          f"{weighted[s]:.4f} over {n_tot} candidates")
    all_ge = all(weighted[s] >= GO_MIN_FRAC for s in SEEDS)
    small_rounds = [(s, rd["round"]) for s in SEEDS for rd in summary_rows
                    if rd["seed"] == s and rd["n_candidates"] < MIN_N_ROUND
                    and rd["frac_multi_view_ge3"] < GO_MIN_FRAC]
    if all_ge:
        verdict = "GO for method design"
        P("* multi-view-supported candidates are >= 5% in all 3 seeds "
          "-> Reliable Pruning = GO for method design")
    else:
        verdict = "NO-GO"
        P("* multi-view-supported fraction below 5% in at least one seed "
          "(opacity<0.1 candidates have essentially no real rendering "
          "contribution, or only scattered single-view support) "
          "-> Reliable Pruning = NO-GO")
    P("")
    P(f"FINAL VERDICT: Reliable Pruning = {verdict}")

    stats = "\n".join(lines)
    print(stats)
    with open(f"{DATA}/b19p_stats.txt", "w") as f:
        f.write(stats + "\n")


if __name__ == "__main__":
    main()
