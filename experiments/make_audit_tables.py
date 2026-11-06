"""Tables for the patient audit: tables/leak_control.tex (results_leak/) and
tables/patient_disjoint.tex (results_vd/). Numbers are also printed for the text."""
import collections, glob, json, statistics as st
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "paper_v3" / "tables" if (ROOT / "paper_v3").is_dir() else ROOT / "tables"
TAB.mkdir(exist_ok=True)
NAMES = {"Liver": "Liver CT", "Retina_RESC": "Retinal OCT (RESC)", "Brain": "Brain MRI"}

pairs = collections.defaultdict(dict)
for f in (ROOT / "results_leak").glob("*.jsonl"):
    d, mode = f.stem.rsplit("_", 1)
    for r in map(json.loads, f.open()):
        pairs[(d, r["k"], r["set"])][mode] = r
lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"\textbf{Dataset} & $k$ & \textbf{Same patients} & \textbf{Other patients} & \textbf{Difference} & \textbf{Same $>$ other} \\", r"\midrule"]
for d in ("Liver", "Retina_RESC"):
    for k in (4, 8):
        ps = [v for key, v in pairs.items() if key[:2] == (d, k) and len(v) == 2]
        for label, key in (("image", "image_max_testminmax"), ("pixel", "pixel_testminmax")):
            s = [100 * p["same"]["variants"][key] for p in ps]
            o = [100 * p["other"]["variants"][key] for p in ps]
            df = [a - b for a, b in zip(s, o)]
            pv = wilcoxon(df).pvalue
            first = NAMES[d] if (k == 4 and label == "image") else ""
            lines.append(f"{first} & {k} ({label}) & {st.mean(s):.1f} & {st.mean(o):.1f} & {st.mean(df):+.1f} "
                         f"($p={pv:.2f}$) & {sum(x > 0 for x in df)}/{len(df)} \\\\")
            print(d, k, label, f"{st.mean(s):.1f} {st.mean(o):.1f} {st.mean(df):+.1f} p={pv:.3f}")
    if d == "Liver":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "leak_control.tex").write_text("\n".join(lines) + "\n")

A = collections.defaultdict(list)
for f in glob.glob(str(ROOT / "results_vd" / "*.jsonl")):
    for r in map(json.loads, open(f)):
        if "vd_image_testminmax" in r["variants"]:
            A[(r["backbone"], r["dataset"], r["k"])].append(r)
lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"\textbf{Dataset} & $k$ & \textbf{Runs} & \textbf{All test} & \textbf{Patient-disjoint} & \textbf{Abn.\ images left (min)} \\", r"\midrule"]
for d in ("Liver", "Retina_RESC", "Brain"):
    for k in (2, 4, 8, 16):
        rs = A[("biomedclip", d, k)]
        a = 100 * st.mean(r["variants"]["image_max_testminmax"] for r in rs)
        v = 100 * st.mean(r["variants"]["vd_image_testminmax"] for r in rs)
        lines.append(f"{NAMES[d] if k == 2 else ''} & {k} & {len(rs)} & {a:.1f} & {v:.1f} & {min(r['vd_n_pos'] for r in rs)} \\\\")
    if d != "Brain":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "patient_disjoint.tex").write_text("\n".join(lines) + "\n")
