"""Paired analysis of the leakage control (results_leak/): same-patient minus other-patient support
on identical test slices."""
import json, glob, collections, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon

rows = collections.defaultdict(dict)
for f in glob.glob("results_leak/*.jsonl"):
    d, mode = Path(f).stem.rsplit("_", 1)
    for r in map(json.loads, open(f)):
        rows[(d, r["k"], r["set"])][mode] = r
keys = [("image memory", "image_max_memory"), ("image ROAM", "image_max_memory_two"), ("image fusion", "image_max_testminmax"),
        ("pixel memory", "pixel_memory"), ("pixel ROAM", "pixel_memory_two"), ("pixel fusion", "pixel_testminmax")]
pairs = {k: v for k, v in rows.items() if len(v) == 2}
for d, sh in sorted({(k[0], k[1]) for k in pairs}):
    ps = [v for k, v in pairs.items() if k[0] == d and k[1] == sh]
    print(f"{d} k={sh}: {len(ps)} pairs")
    for name, key in keys:
        same = [100 * p["same"]["variants"][key] for p in ps]
        other = [100 * p["other"]["variants"][key] for p in ps]
        diff = [a - b for a, b in zip(same, other)]
        pv = wilcoxon(diff).pvalue if len(diff) >= 5 and any(diff) else float("nan")
        print(f"  {name:13s} same {st.mean(same):5.1f}  other {st.mean(other):5.1f}  diff {st.mean(diff):+5.1f}  "
              f"same>other {sum(x > 0 for x in diff)}/{len(diff)}  p={pv:.3f}")
