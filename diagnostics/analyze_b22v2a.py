#
# Paper B - B22-V2A analysis: intra-event densification benefit
#
# Pre-registered rules:
#   heterogeneous cell (+2000 primary): among the 6 groups of a (seed,event),
#     (>=2 groups positive AND >=2 groups non-positive)  OR
#     (max |benefit| >= 3x median |benefit|).
#   V2-A = GO iff heterogeneous cells >= 60% (>=11/18) at +2000, every seed
#     contributes >=3/5 heterogeneous cells, and +1000 shows the same pattern
#     in >=50% of cells.
#   NO-GO otherwise.
#   Q5 (size explanation): Spearman-like rank correlation between Added_GS and
#     benefit within each cell; report share of cells with |rho| >= 0.7.
#

import os, csv
import numpy as np

BASE = "paper_b/b22_v2a_intra_event_benefit"
DATA = f"{BASE}/data"
SEEDS = [0, 1, 2]
PRIMARY_HZ = 2000


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fi(x):
    return int(float(x))


def ff(x):
    return float(x)


def rankdata(xs):
    order = np.argsort(xs)
    ranks = np.empty(len(xs))
    ranks[order] = np.arange(1, len(xs) + 1)
    return ranks


def main():
    groups, summ = [], []
    for s in SEEDS:
        for r in read_csv(f"{DATA}/groups_s{s}.csv"):
            r["seed"] = s; groups.append(r)
        for r in read_csv(f"{DATA}/seed_event_summary_s{s}.csv"):
            r["seed"] = s; summ.append(r)
    ben = read_csv(f"{DATA}/paired_capacity_benefit.csv")
    for r in ben:
        r["seed"] = int(r["seed"])

    lines = []
    P = lines.append
    P("=" * 76)
    P("B22-V2A: Intra-event Densification Benefit (Room, 3 seeds)")
    P("=" * 76)
    P(f"groups: {len(groups)}   benefit rows: {len(ben)}")

    # group cells: (seed, event) -> {group: {hz: benefit}}
    cells = {}
    adds = {}
    for r in summ:
        key = (r["seed"], fi(r["event_it"]))
        cells.setdefault(key, {})[fi(r["group"])] = {
            500: ff(r["benefit_500"]), 1000: ff(r["benefit_1000"]),
            2000: ff(r["benefit_2000"])}
        adds[key] = adds.get(key, {})
        adds[key][fi(r["group"])] = fi(r["added_gs"])

    P("")
    P("-" * 76)
    P("Capacity_Benefit by group, per (seed, event), horizons +500/+1000/+2000")
    P("-" * 76)
    het_2000 = 0
    het_1000 = 0
    n_cells = 0
    nan_groups = 0
    seed_het = {s: [0, 0] for s in SEEDS}
    size_explained = 0
    for (s, ev), gs in sorted(cells.items()):
        pairs = sorted(gs.items())
        vals2 = np.array([b[2000] for _, b in pairs])
        vals1 = np.array([b[1000] for _, b in pairs])
        ag_all = np.array([adds[(s, ev)][g] for g, _ in pairs])
        v2 = vals2[~np.isnan(vals2)]   # NaN = empty ROI (no valid pixel)
        v1 = vals1[~np.isnan(vals1)]
        nan_groups += int(np.isnan(vals2).sum())
        if len(v2) < 3:
            P(f"seed {s} ev {ev}: only {len(v2)}/{len(vals2)} groups with valid "
              f"ROI -> unclassifiable (excluded)")
            continue
        n_cells += 1
        ag = ag_all[~np.isnan(vals2)]
        n_pos2 = int((v2 > 0).sum())
        spread = float(v2.max() - v2.min())
        maxabs = float(np.max(np.abs(v2)))
        medabs = float(np.median(np.abs(v2))) or 1e-12
        het2 = (n_pos2 >= 2 and (len(v2) - n_pos2) >= 2) or (maxabs >= 3 * medabs)
        n_pos1 = int((v1 > 0).sum())
        het1 = (n_pos1 >= 2 and (len(v1) - n_pos1) >= 2) or \
               (float(np.max(np.abs(v1))) >= 3 * max(float(np.median(np.abs(v1))), 1e-12))
        het_2000 += het2
        het_1000 += het1
        seed_het[s][0] += het2
        seed_het[s][1] += 1
        rho = float(np.corrcoef(rankdata(ag), rankdata(v2))[0, 1]) \
            if len(ag) >= 3 and np.std(ag) > 0 else float("nan")
        if rho == rho and abs(rho) >= 0.7:
            size_explained += 1
        P(f"seed {s} ev {ev}: benefits(+2000) " +
          ", ".join(f"g{g}:{b[2000]:+.5f}" if b[2000] == b[2000] else f"g{g}:NaN"
                    for g, b in pairs) +
          f"  | spread {spread:.5f}  pos/neg {n_pos2}/{len(v2)-n_pos2}  "
          f"het={het2}  rho(size)={rho:+.2f}")
    P("")
    P(f"heterogeneous cells at +2000: {het_2000}/{n_cells}; at +1000: {het_1000}/{n_cells}")
    P(f"cells where |rank-corr(Added_GS, benefit)| >= 0.7: {size_explained}/{n_cells}")
    for s in SEEDS:
        P(f"  seed {s}: heterogeneous {seed_het[s][0]}/{seed_het[s][1]}")
    P("")

    # ---------- verdict ----------
    P("-" * 76)
    P("VERDICT (pre-registered)")
    P("-" * 76)
    per_seed_ok = all(seed_het[s][0] >= 3 for s in SEEDS)
    P(f"cells heterogeneous at +2000: {het_2000}/{n_cells} "
      f"({100 * het_2000 / max(n_cells, 1):.0f}%); per-seed >=3/5: {per_seed_ok}")
    if het_2000 >= 0.6 * n_cells and per_seed_ok and het_1000 >= 0.5 * n_cells:
        verdict = "GO"
        P("* stable intra-event heterogeneity -> V2-A = GO (enter V2-B in a "
          "later phase: which error features predict the difference)")
    else:
        verdict = "NO-GO"
        P("* intra-event group benefits are largely uniform -> V2-A = NO-GO "
          "(stop the error-aware error-type route)")
    P("")
    P(f"FINAL VERDICT: V2-A = {verdict}")

    stats = "\n".join(lines)
    print(stats)
    with open(f"{DATA}/b22v2a_stats.txt", "w") as f:
        f.write(stats + "\n")


if __name__ == "__main__":
    main()
