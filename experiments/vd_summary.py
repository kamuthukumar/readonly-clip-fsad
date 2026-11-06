"""Volume-disjoint vs all-test results of the ROAM runs (results_vd/), per dataset and shot count."""
import json, glob, collections, statistics as st

rows = [json.loads(l) for f in glob.glob("results_vd/*.jsonl") for l in open(f)]
A = collections.defaultdict(list)
for r in rows:
    v = r["variants"]
    if "vd_image_memory" not in v:
        continue
    A[(r["backbone"], r["dataset"], r["k"])].append(r)
keys = [("all mem", "image_max_memory"), ("all two", "image_max_memory_two"), ("all tmm", "image_max_testminmax"),
        ("vd mem", "vd_image_memory"), ("vd two", "vd_image_memory_two"), ("vd tmm", "vd_image_testminmax"),
        ("vd pxmem", "vd_pixel_memory"), ("vd pxtwo", "vd_pixel_memory_two"), ("vd pxtmm", "vd_pixel_testminmax")]
print(f"{'':24s}" + "".join(f"{n:>10s}" for n, _ in keys) + "   n+ min  runs  wins")
for key in sorted(A):
    rs = A[key]
    m = [100 * st.mean(r["variants"][k] for r in rs) for _, k in keys]
    wins = sum(r["variants"]["vd_image_memory_two"] > r["variants"]["vd_image_memory"] for r in rs)
    print(f"{key[0][:6]} {key[1][:11]:11s} k{key[2]:<3d} " + "".join(f"{x:10.1f}" for x in m) +
          f"  {min(r['vd_n_pos'] for r in rs):5d}  {len(rs):4d}  {wins}")
