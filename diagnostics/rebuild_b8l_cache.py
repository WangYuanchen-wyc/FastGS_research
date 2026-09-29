#
# Paper B - B8-L cache fix: rebuild b8l_repeat_results.{csv,json} from the
# existing branches.json WITHOUT any GPU rerun. Restores the 12 smoke-cache
# parents whose ev checkpoint keys were JSON-serialized as strings.
#

import json
import csv

BASE = "paper_b/b8_l_long_horizon_cardinality"

done = json.load(open(f"{BASE}/cache/branches.json"))
plan = json.load(open(f"{BASE}/cache/parent_plan.json"))

branch_rows = []
for p in plan:
    it, idx = p["iteration"], p["parent_index"]
    # keep
    rec = done[f"{it}|{idx}|{0}|keep"]
    ev = {int(k): v for k, v in rec["ev"].items()}
    row = {"iteration": it, "parent_index": idx, "category": p["category"],
           "branch": "keep", "N": 0, "repeat": 0, "seed": None, "d_n": rec["d_n"]}
    for k in (0, 100, 500, 1000):
        e = ev.get(k, {})
        row[f"l1_{k}"] = e.get("l1")
        row[f"psnr_{k}"] = e.get("psnr")
        row[f"gpsnr_{k}"] = e.get("gpsnr")
        row[f"gssim_{k}"] = e.get("gssim")
        row[f"glpips_{k}"] = e.get("glpips")
        row[f"tile_{k}"] = e.get("tile")
    branch_rows.append(row)
    for n, act in p["actions"]:
        for r in range(5):
            sd = 950000 + idx * 100 + n * 10 + r
            rec = done[f"{it}|{idx}|{n}|{sd}"]
            ev = {int(k): v for k, v in rec["ev"].items()}
            row = {"iteration": it, "parent_index": idx, "category": p["category"],
                   "branch": act, "N": n, "repeat": r, "seed": sd, "d_n": rec["d_n"]}
            for k in (0, 100, 500, 1000):
                e = ev.get(k, {})
                row[f"l1_{k}"] = e.get("l1")
                row[f"psnr_{k}"] = e.get("psnr")
                row[f"gpsnr_{k}"] = e.get("gpsnr")
                row[f"gssim_{k}"] = e.get("gssim")
                row[f"glpips_{k}"] = e.get("glpips")
                row[f"tile_{k}"] = e.get("tile")
            branch_rows.append(row)

with open(f"{BASE}/data/b8l_repeat_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(branch_rows[0].keys()))
    w.writeheader()
    for r in branch_rows:
        w.writerow({k: ("NA" if v is None else (f"{v:.8g}" if isinstance(v, float) else v))
                    for k, v in r.items()})
life = json.load(open(f"{BASE}/data/b8l_repeat_results.json")).get("lifecycle", [])
json.dump({"branches": branch_rows, "lifecycle": life},
          open(f"{BASE}/data/b8l_repeat_results.json", "w"))

na = sum(1 for r in branch_rows if r["l1_1000"] is None)
print(f"rebuilt {len(branch_rows)} rows; l1_1000 NA: {na}")
parents = {(r['iteration'], r['parent_index']) for r in branch_rows}
b = sum(1 for p in plan if p['category'] == 'B_split2_sufficient')
c = sum(1 for p in plan if p['category'].startswith('C'))
print(f"parents: {len(parents)} (B={b}, C={c})")
