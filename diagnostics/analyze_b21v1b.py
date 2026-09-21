#
# Paper B - B21-V1b analysis: long-horizon & event disentangling
#
# Verdict rules (fixed before running; primary horizon +2000, low-benefit
# events = {5000, 7000, 11000, 13000}; all comparisons per seed, then
# consistency across seeds; dB thresholds 0.1):
#   Q1 slow-benefit : Full - Skip at +400 < 0 but at +2000 >= -0.05 dB
#                     (Full catches up / overtakes Skip late).
#   Q2 capacity     : Add - Skip at +2000 <= +0.1 dB  -> adding Gaussians
#                     itself gives no clear benefit.
#   Q3 side effects : (Full - Add) at +2000 <= -0.1 dB AND
#                     (Side - Skip) at +2000 <= -0.1 dB -> side effects cancel
#                     the creation benefit.
#   Case A (slow benefit) : Q1 holds in >= 2/3 seeds and >= 2/4 low events.
#   Case C (side effects) : Q3 holds in >= 2/3 seeds and >= 2/4 low events.
#   Case B (capacity ineffective) : Q2 holds in >= 2/3 seeds on >= 3/4 low
#                     events.
#   Priority: A, then C, then B. it=1500 reported as sanity (Add-Only should
#   be positive there if early capacity is real).
#

import os, csv
import numpy as np

BASE = "paper_b/b21_v1b_event_disentangling"
DATA = f"{BASE}/data"
SEEDS = [0, 1, 2]
LOW_EVENTS = [5000, 7000, 11000, 13000]
SANITY_EVENT = 1500
BRANCHES = ["A_skip_all", "B_add_only", "C_side_only", "D_full"]
HORIZONS = [100, 400, 1000, 2000]
TH = 0.1
Q1_RECOVER = 0.05


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fi(x):
    return int(float(x))


def ff(x):
    return float(x)


def main():
    brs = []
    for s in SEEDS:
        for r in read_csv(f"{DATA}/branch_metrics_s{s}.csv"):
            r["seed"] = s; brs.append(r)
    roi = []
    for s in SEEDS:
        for r in read_csv(f"{DATA}/long_horizon_roi_s{s}.csv"):
            r["seed"] = s; roi.append(r)

    # lookup: (seed, event, branch, horizon) -> row
    L = {}
    for r in brs:
        L[(r["seed"], fi(r["event_it"]), r["branch"], fi(r["horizon"]))] = r

    lines = []
    P = lines.append
    P("=" * 76)
    P("B21-V1b: Long-Horizon & Event Disentangling (Room, 3 seeds)")
    P("=" * 76)
    P(f"rows: {len(brs)}; events: {sorted(set(fi(r['event_it']) for r in brs))}")
    P("")

    # ---------- full table ----------
    P("-" * 76)
    P("dPSNR vs Skip-All (positive = branch better than skipping)")
    P("-" * 76)
    P(f"{'seed':>4} {'it':>6} {'phase':>10} {'hz':>5} " +
      " ".join(f"{b.replace('_',''):>12}" for b in BRANCHES[1:]) +
      f" {'ROI_A':>9} {'ROI_D':>9}")
    for s in SEEDS:
        for ev in sorted(set(fi(r["event_it"]) for r in brs if r["seed"] == s)):
            for hz in HORIZONS:
                a = ff(L[(s, ev, "A_skip_all", hz)]["psnr"])
                cells = []
                for b in BRANCHES[1:]:
                    cells.append(ff(L[(s, ev, b, hz)]["psnr"]) - a)
                ra = ff(L[(s, ev, "A_skip_all", hz)]["roi_l1"])
                rd = ff(L[(s, ev, "D_full", hz)]["roi_l1"])
                P(f"{s:>4} {ev:>6} {L[(s, ev, 'D_full', hz)]['phase']:>10} "
                  f"{hz:>5} " + " ".join(f"{c:>+12.4f}" for c in cells) +
                  f" {ra:>9.5f} {rd:>9.5f}")
    P("")

    # ---------- Q1/Q2/Q3 per seed, low events, +2000 primary ----------
    P("-" * 76)
    P("Per-seed summary at +1000 / +2000 (low-benefit events, mean dPSNR vs Skip)")
    P("-" * 76)
    q1_seeds, q2_seeds, q3_seeds = 0, 0, 0
    seed_case = {}
    for s in SEEDS:
        rows_s = {}
        for ev in LOW_EVENTS:
            for hz in (400, 1000, 2000):
                a = ff(L[(s, ev, "A_skip_all", hz)]["psnr"])
                rows_s[(ev, hz)] = {
                    "B": ff(L[(s, ev, "B_add_only", hz)]["psnr"]) - a,
                    "C": ff(L[(s, ev, "C_side_only", hz)]["psnr"]) - a,
                    "D": ff(L[(s, ev, "D_full", hz)]["psnr"]) - a,
                }
        # Q1
        q1_cells = 0
        for ev in LOW_EVENTS:
            d400 = rows_s[(ev, 400)]["D"]
            d2000 = rows_s[(ev, 2000)]["D"]
            if d400 < 0 and d2000 >= -Q1_RECOVER:
                q1_cells += 1
        q1 = q1_cells >= 2
        # Q2
        q2_cells = sum(1 for ev in LOW_EVENTS
                       if rows_s[(ev, 2000)]["B"] <= TH)
        q2 = q2_cells >= 3
        # Q3
        q3_cells = sum(1 for ev in LOW_EVENTS
                       if rows_s[(ev, 2000)]["D"] - rows_s[(ev, 2000)]["B"] <= -TH
                       and rows_s[(ev, 2000)]["C"] <= -TH)
        q3 = q3_cells >= 2
        q1_seeds += q1; q2_seeds += q2; q3_seeds += q3
        seed_case[s] = {"q1": q1, "q2": q2, "q3": q3,
                        "q1_cells": q1_cells, "q2_cells": q2_cells,
                        "q3_cells": q3_cells}
        mb1000 = np.mean([rows_s[(ev, 1000)]["B"] for ev in LOW_EVENTS])
        mb2000 = np.mean([rows_s[(ev, 2000)]["B"] for ev in LOW_EVENTS])
        md1000 = np.mean([rows_s[(ev, 1000)]["D"] for ev in LOW_EVENTS])
        md2000 = np.mean([rows_s[(ev, 2000)]["D"] for ev in LOW_EVENTS])
        mc2000 = np.mean([rows_s[(ev, 2000)]["C"] for ev in LOW_EVENTS])
        P(f"seed {s}: Add-Skip mean {mb1000:+.4f}(+1000) {mb2000:+.4f}(+2000) | "
          f"Full-Skip mean {md1000:+.4f}(+1000) {md2000:+.4f}(+2000) | "
          f"Side-Skip mean {mc2000:+.4f}(+2000)")
        P(f"  Q1 slow-benefit cells {q1_cells}/4; Q2 capacity-ineffective cells "
          f"{q2_cells}/4; Q3 side-effect cells {q3_cells}/4")
        P("")

    # ---------- sanity event 1500 ----------
    P("-" * 76)
    P(f"Sanity (it={SANITY_EVENT}, expected real benefit): Add - Skip dPSNR")
    P("-" * 76)
    for s in SEEDS:
        a = ff(L[(s, SANITY_EVENT, "A_skip_all", 2000)]["psnr"])
        b = ff(L[(s, SANITY_EVENT, "B_add_only", 2000)]["psnr"])
        P(f"  seed {s}: {b - a:+.4f} dB at +2000")
    P("")

    # ---------- verdict ----------
    P("-" * 76)
    P("VERDICT (pre-registered; priority A > C > B)")
    P("-" * 76)
    P(f"Q1 slow-benefit seeds: {q1_seeds}/3   Q2 capacity-ineffective seeds: "
      f"{q2_seeds}/3   Q3 side-effect seeds: {q3_seeds}/3")
    if q1_seeds >= 2:
        verdict = "Case A - Slow Benefit (short-horizon artifact; do NOT enter V2 yet)"
    elif q3_seeds >= 2:
        verdict = "Case C - Event Side-Effect (pivot to event mechanism, not error-aware V2)"
    elif q2_seeds >= 2:
        verdict = "Case B - Capacity Ineffective (error-aware V2 hypothesis strengthened)"
    else:
        verdict = "MIXED / NO dominant case (report patterns; no single case across seeds)"
    P("")
    P(f"FINAL VERDICT: {verdict}")

    stats = "\n".join(lines)
    print(stats)
    with open(f"{DATA}/b21v1b_stats.txt", "w") as f:
        f.write(stats + "\n")

    # seed_event_summary.csv
    with open(f"{DATA}/seed_event_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "event_it", "phase", "horizon", "branch",
                    "added_gs", "psnr", "ssim", "lpips", "roi_l1", "gs_at_hz"])
        for r in brs:
            w.writerow([r["seed"], fi(r["event_it"]), r["phase"], fi(r["horizon"]),
                        r["branch"], fi(r["added_gs"]), ff(r["psnr"]), ff(r["ssim"]),
                        ff(r["lpips"]), ff(r["roi_l1"]), fi(r["gs_at_hz"])])


if __name__ == "__main__":
    main()
