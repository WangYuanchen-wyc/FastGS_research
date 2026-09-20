#
# Paper B - B18-D Fix analysis: early densification persistence + child fate
#
# Questions:
#   Q1 (persistence): of event e's split triggers, what fraction were also
#       triggered at event e-1 (proximity-matched)? Early window vs rest.
#   Q2 (child fate): of the newborns born at event e, what fraction are alive at
#       the next event / at 30k? Do they re-trigger at the next event?
#   Verdict: persistent capacity demand vs one-shot transient demand.
#
# Reads: paper_b/b18_densification_persistence/fix/data/
# Writes: fix/data/b18dfix_stats.txt, fix/plots/*.png
#

import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("B18D_ANALYSIS_BASE",
                      "paper_b/b18_densification_persistence/fix")
DATA = f"{BASE}/data"
PLOTS = f"{BASE}/plots"
os.makedirs(PLOTS, exist_ok=True)

EARLY_MAX_IT = 3000


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return float("nan")


def main():
    trigger_rows = read_csv(f"{DATA}/trigger_events.csv")
    for r in trigger_rows:
        for k in r:
            if k != "seed":
                r[k] = fnum(r[k]) if k not in ("is_early",) else int(r[k])
        r["seed"] = int(r["seed"])

    child_rows = []
    seen_child = set()
    for s in (0, 1, 2):
        p = f"{DATA}/child_fate_s{s}.csv"
        if os.path.exists(p):
            for r in read_csv(p):
                key = (int(r["seed"]), int(float(r["birth_event"])))
                if key in seen_child:   # older runs wrote cumulative per-seed files
                    continue
                seen_child.add(key)
                for k in r:
                    r[k] = fnum(r[k]) if k not in ("is_early",) else int(r[k])
                r["seed"] = int(r["seed"])
                child_rows.append(r)

    lines = []
    P = lines.append

    P("=" * 72)
    P("B18-D FIX: Early Densification Persistence & Child Fate Validation")
    P("=" * 72)
    P(f"events total: {len(trigger_rows)}  (seeds: {sorted(set(r['seed'] for r in trigger_rows))})")
    P(f"proximity tol: 0.05 (same as B18-D); early window: it in [1000, {EARLY_MAX_IT}]")
    P("")

    # ---------- Q1 persistence ----------
    P("-" * 72)
    P("Q1  Trigger persistence (fraction of triggers matched to previous event)")
    P("-" * 72)
    early, late = [], []
    per_seed = {}
    for r in trigger_rows:
        if r["n_trigger"] <= 0 or not np.isfinite(r.get("n_persist_from_prev", np.nan)):
            continue
        row_seed = r["seed"]
        # skip the first event of each seed (no previous event to match)
        if r["event"] == 1:
            continue
        ratio = r["n_persist_from_prev"] / r["n_trigger"]
        d = per_seed.setdefault(row_seed, {"early": [], "late": []})
        if r["is_early"]:
            d["early"].append(ratio); early.append(ratio)
        else:
            d["late"].append(ratio); late.append(ratio)

    P(f"{'seed':>5} {'early mean±std (n_ev)':>28} {'late mean±std (n_ev)':>28}")
    for s in sorted(per_seed):
        e = np.array(per_seed[s]["early"]); l = np.array(per_seed[s]["late"])
        P(f"{s:>5} {e.mean():>12.3f}±{e.std(ddof=1):.3f} (n={len(e):>2})    "
          f"{l.mean():>12.3f}±{l.std(ddof=1):.3f} (n={len(l):>2})")
    e_all = np.array(early); l_all = np.array(late)
    sem_e = e_all.std(ddof=1) / np.sqrt(len(e_all)) if len(e_all) > 1 else float("nan")
    sem_l = l_all.std(ddof=1) / np.sqrt(len(l_all)) if len(l_all) > 1 else float("nan")
    P(f"ALL   early persist_ratio = {e_all.mean():.3f} ± {sem_e:.3f} (SEM, n={len(e_all)} events)")
    P(f"      late  persist_ratio = {l_all.mean():.3f} ± {sem_l:.3f} (SEM, n={len(l_all)} events)")
    P("")

    # ---------- Q2 child fate ----------
    P("-" * 72)
    P("Q2  Child fate (newborns = clones + 2x split children, per event)")
    P("-" * 72)
    P(f"{'seed':>5} {'ev':>3} {'it':>6} {'early':>5} {'n_born':>8} "
      f"{'alive_next':>10} {'retrig_next':>11} {'alive@30k':>9} {'op@30k':>7}")
    early_alive_next, early_retrig, early_alive_final = [], [], []
    late_alive_next, late_alive_final = [], []
    for r in sorted(child_rows, key=lambda x: (x["seed"], x["birth_event"])):
        # alive_next / retrig come from the NEXT event's row in trigger_events
        nxt = [t for t in trigger_rows
               if t["seed"] == r["seed"] and t["event"] == r["birth_event"] + 1]
        alive_next = retrig = float("nan")
        if nxt and r["n_born"] > 0:
            alive_next = nxt[0]["n_prev_newborn_alive_next"] / r["n_born"]
            retrig = nxt[0]["n_child_retrigger_next"] / max(r["n_born"], 1)
        af = r["alive_frac_final"]
        P(f"{r['seed']:>5} {int(r['birth_event']):>3} {int(r['birth_it']):>6} "
          f"{int(r['is_early']):>5} {int(r['n_born']):>8} "
          f"{alive_next:>10.3f} {retrig:>11.4f} {af:>9.3f} {r['mean_opacity_final_alive']:>7.3f}")
        tgt_a, tgt_r, tgt_f = ((early_alive_next, early_retrig, early_alive_final)
                               if r["is_early"] else
                               (late_alive_next, [], late_alive_final))
        if np.isfinite(alive_next):
            tgt_a.append(alive_next)
            if r["is_early"]:
                early_retrig.append(retrig)
        if np.isfinite(af):
            tgt_f.append(af)

    P("")
    for name, arr in [("early child alive@next-event", early_alive_next),
                      ("early child re-trigger@next-event", early_retrig),
                      ("early child alive@30k", early_alive_final),
                      ("late  child alive@next-event", late_alive_next),
                      ("late  child alive@30k", late_alive_final)]:
        a = np.array([x for x in arr if np.isfinite(x)])
        if len(a):
            sem = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
            P(f"{name:>38}: {a.mean():.3f} ± {sem:.3f} (SEM, n={len(a)} events)")
    P("")

    # ---------- Verdict ----------
    P("-" * 72)
    P("VERDICT (criteria fixed before reading results)")
    P("-" * 72)
    ep = e_all.mean() if len(e_all) else float("nan")
    verdict_lines = []
    if np.isfinite(ep):
        if ep >= 0.5:
            verdict_lines.append(
                f"early persist_ratio = {ep:.3f} >= 0.5 -> trigger demand is PERSISTENT "
                f"(same regions re-trigger); a persistence-aware densification signal has headroom.")
        elif ep <= 0.25:
            verdict_lines.append(
                f"early persist_ratio = {ep:.3f} <= 0.25 -> trigger demand is mostly ONE-SHOT "
                f"transient; persistence-aware re-densification would chase noise. NO-GO.")
        else:
            verdict_lines.append(
                f"early persist_ratio = {ep:.3f} in (0.25, 0.5) -> MIXED; no dominant regime.")
    ef = np.array([x for x in early_alive_final if np.isfinite(x)])
    if len(ef):
        if ef.mean() <= 0.3:
            verdict_lines.append(
                f"early-born children alive@30k = {ef.mean():.3f} <= 0.3 -> early capacity "
                f"largely overshoots (children do not persist); supports one-shot demand.")
        elif ef.mean() >= 0.6:
            verdict_lines.append(
                f"early-born children alive@30k = {ef.mean():.3f} >= 0.6 -> early capacity "
                f"persists; demand is structural.")
        else:
            verdict_lines.append(
                f"early-born children alive@30k = {ef.mean():.3f} -> intermediate survival.")
    for v in verdict_lines:
        P("* " + v)
    P("")
    P("NOTE: alive@next-event for birth event E is measured at event E+1 (500 iters later);")
    P("      alive@30k is proximity-matched against the final model. Both use tol=0.05.")

    stats = "\n".join(lines)
    print(stats)
    with open(f"{DATA}/b18dfix_stats.txt", "w") as f:
        f.write(stats + "\n")

    # ---------- plots ----------
    evs = sorted(set((r["iteration"]) for r in trigger_rows))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    # persist ratio by iteration
    for s in sorted(per_seed):
        xs = [r["iteration"] for r in trigger_rows
              if r["seed"] == s and r["event"] != 1 and r["n_trigger"] > 0]
        ys = [r["n_persist_from_prev"] / r["n_trigger"] for r in trigger_rows
              if r["seed"] == s and r["event"] != 1 and r["n_trigger"] > 0]
        ax[0].plot(xs, ys, "o-", ms=3, label=f"seed{s}")
    ax[0].axvspan(1000, EARLY_MAX_IT, alpha=0.15, color="red", label="early window")
    ax[0].set_xscale("log"); ax[0].set_xlabel("iteration"); ax[0].set_ylabel("persist ratio")
    ax[0].set_title("Trigger persistence (vs prev event)"); ax[0].legend(fontsize=7)
    # child alive next
    for s in (0, 1, 2):
        pts = []
        for r in child_rows:
            if r["seed"] != s or r["n_born"] <= 0:
                continue
            nxt = [t for t in trigger_rows
                   if t["seed"] == s and t["event"] == r["birth_event"] + 1]
            if not nxt:
                continue
            v = nxt[0]["n_prev_newborn_alive_next"] / r["n_born"]
            if np.isfinite(v):
                pts.append((r["birth_it"], v))
        if pts:
            xs, ys = zip(*pts)
            ax[1].plot(xs, ys, "o-", ms=3, label=f"seed{s}")
    ax[1].axvspan(1000, EARLY_MAX_IT, alpha=0.15, color="red")
    ax[1].set_xscale("log"); ax[1].set_xlabel("birth iteration")
    ax[1].set_ylabel("alive@next-event frac"); ax[1].set_title("Newborn survival to next event")
    ax[1].legend(fontsize=7)
    # child alive 30k
    for s in (0, 1, 2):
        xs = [r["birth_it"] for r in child_rows if r["seed"] == s]
        ys = [r["alive_frac_final"] for r in child_rows if r["seed"] == s]
        ax[2].plot(xs, ys, "o-", ms=3, label=f"seed{s}")
    ax[2].axvspan(1000, EARLY_MAX_IT, alpha=0.15, color="red")
    ax[2].set_xscale("log"); ax[2].set_xlabel("birth iteration")
    ax[2].set_ylabel("alive@30k frac"); ax[2].set_title("Newborn survival to 30k")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/persistence_and_child_fate.png", dpi=130)
    print(f"[analyze] plots -> {PLOTS}/persistence_and_child_fate.png")


if __name__ == "__main__":
    main()
