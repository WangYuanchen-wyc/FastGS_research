#
# Paper B - B16-E analysis (host).  python3 diagnostics/analyze_b16e.py
#
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "paper_b/b16_e_evidence_formulation"
ETYPES = ["binary", "continuous", "contribution"]
COL = {"binary": "#777777", "continuous": "#228833", "contribution": "#4477aa"}
LAB = {"binary": "Binary (FastGS)", "continuous": "Continuous", "contribution": "Contribution"}


def main():
    rem = list(csv.DictReader(open(f"{BASE}/data/evidence_pruning_results.csv")))
    for r in rem:
        for k, v in r.items():
            if k not in ("evidence",):
                try: r[k] = float(v)
                except: pass

    out = ["## B16-E Evidence Formulation Diagnostic\n"]

    # ---- All-view reference ----
    out.append("### All-view pruning quality（oracle reference，各 evidence type）\n```text")
    out.append(f"{'evidence':>13s} {'5%':>8s} {'10%':>8s} {'20%':>8s}")
    for et in ETYPES:
        line = f"{et:>13s}"
        for pct in (5, 10, 20):
            a = next((r["psnr"] for r in rem if r["evidence"] == et
                      and r["source"] == "allview" and int(r["ratio"]) == pct), None)
            line += f" {a:>8.4f}" if a else f" {'NA':>8s}"
        out.append(line)
    out.append("```\n")

    # ---- 10-view quality ----
    out.append("### 10-view pruning quality（20 perms mean±std）\n```text")
    out.append(f"{'evidence':>13s} {'5%':>16s} {'10%':>16s} {'20%':>16s}")
    for et in ETYPES:
        line = f"{et:>13s}"
        for pct in (5, 10, 20):
            sub = [r["psnr"] for r in rem if r["evidence"] == et
                   and r["source"] == "10view" and int(r["ratio"]) == pct]
            if sub:
                line += f" {np.mean(sub):>8.4f}±{np.std(sub):.4f}"
            else:
                line += f" {'NA':>16s}"
        out.append(line)
    out.append("```\n")

    # ---- key comparison ----
    out.append("### Core comparison: Continuous vs Binary vs Contribution（10-view, seed-mean）\n```text")
    out.append(f"{'ratio':>6s} {'Binary':>10s} {'Continuous':>11s} {'Contribution':>13s} "
               f"{'Cont−Bin':>9s} {'Contr−Bin':>10s}")
    for pct in (5, 10, 20):
        vals = {}
        for et in ETYPES:
            sub = [r for r in rem if r["evidence"] == et
                   and r["source"] == "10view" and int(r["ratio"]) == pct]
            vals[et] = np.mean([r["psnr"] for r in sub]) if sub else None
        d1 = vals["continuous"] - vals["binary"] if vals["continuous"] and vals["binary"] else None
        d2 = vals["contribution"] - vals["binary"] if vals["contribution"] and vals["binary"] else None
        out.append(f"{pct:>6d} {vals['binary']:>10.4f} {vals['continuous']:>11.4f} "
                   f"{vals['contribution']:>13.4f} {d1 if d1 is None else format(d1,'>+9.4f')} "
                   f"{d2 if d2 is None else format(d2,'>+10.4f')}")
    out.append("```\n")

    # global PSNR
    out.append("### Global PSNR (same 10-view, same ratio)\n```text")
    for pct in (5, 10, 20):
        vals = {}
        for et in ETYPES:
            sub = [r["gpsnr"] for r in rem if r.get("gpsnr") and r["evidence"] == et
                   and r["source"] == "10view" and int(r["ratio"]) == pct]
            vals[et] = np.mean([float(x) for x in sub]) if sub else None
        out.append(f"  {pct}%: binary={vals.get('binary', 'NA')} "
                   f"continuous={vals.get('continuous', 'NA')} "
                   f"contribution={vals.get('contribution', 'NA')}")

    out.append("```\n")

    # ---- GO/NO-GO summary ----
    out.append("### GO/NO-GO（10-view pruning quality，vs Binary）\n```text")
    for pct in (5, 10, 20):
        for et in ("continuous", "contribution"):
            b = next((r["psnr"] for r in rem if r["evidence"] == "binary"
                      and r["source"] == "10view" and int(r["ratio"]) == pct), None)
            c = next((r["psnr"] for r in rem if r["evidence"] == et
                      and r["source"] == "10view" and int(r["ratio"]) == pct), None)
            if b and c:
                out.append(f"  {pct}% {et:>12s} vs binary: ΔPSNR = {c - b:+.4f} "
                           f"({'better' if c > b else 'worse'})")
    out.append("```")

    with open(f"{BASE}/data/b16e_stats.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
