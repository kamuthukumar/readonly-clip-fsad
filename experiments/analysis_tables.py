"""Protocol and ablation tables for the paper (epoch selection, support-set effects, ablations).

All numbers are computed from results_fsad/*.jsonl.
  python experiments/analysis_tables.py --out paper_v3/tables
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT.parent / "results_fsad"
SEG = ["Brain", "Liver", "Retina_RESC"]
DS = SEG + ["Retina_OCT2017", "Chest", "Histopathology"]
SHORT = {"Brain": "Brain", "Liver": "Liver", "Retina_RESC": "RESC", "Retina_OCT2017": "OCT17",
         "Chest": "Chest", "Histopathology": "HIS"}
RO = "openai__dual_L4-8-12_fuse-testminmax_p-mvfa_e50.jsonl"


def load(name):
    p = RES / name
    return [json.loads(line) for line in open(p)] if p.exists() else []


def img(r, var="testminmax", variants=None):
    v = variants or r["variants"]
    return 100 * v[("image_max_" if r["dataset"] in SEG else "image_det_") + var]


def epoch_selection():
    rows = load("openai__dual_L4-8-12_fuse-testminmax_p-mvfa_e50_curve10.jsonl")
    out = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
           r"\textbf{Dataset} & \textbf{Fixed (epoch 50)} & \textbf{Best epoch on test} & \textbf{Inflation} \\", r"\midrule"]
    gaps, gaps_px = [], []
    for d in DS:
        rs = [r for r in rows if r["dataset"] == d]
        if not rs:
            continue
        fin, best, pf, pb = [], [], [], []
        for r in rs:
            eps = sorted(r["curve"], key=int)
            vals = [img(r, variants=r["curve"][e]) for e in eps]
            fin.append(vals[-1]); best.append(max(vals))
            if d in SEG:
                pv = [100 * r["curve"][e]["pixel_testminmax"] for e in eps]
                pf.append(pv[-1]); pb.append(max(pv))
        g = st.mean(best) - st.mean(fin); gaps.append(g)
        cellf = f"{st.mean(fin):.1f}" + (f" / {st.mean(pf):.1f}" if pf else "")
        cellb = f"{st.mean(best):.1f}" + (f" / {st.mean(pb):.1f}" if pb else "")
        if pf:
            gaps_px.append(st.mean(pb) - st.mean(pf))
        out.append(f"{SHORT[d]} & {cellf} & {cellb} & $+${g:.1f} \\\\")
    out += [r"\midrule", f"Mean & & & $+${st.mean(gaps):.1f} (image), $+${st.mean(gaps_px):.2f} (pixel) \\\\",
            r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out), st.mean(gaps), st.mean(gaps_px)


def ablation():
    cfgs = [("RO-MVFA (two adapter sets, blocks 4/8/12)", RO, "testminmax"),
            ("one shared adapter set", "openai__shared_L4-8-12_fuse-testminmax_p-mvfa_e50.jsonl", "testminmax"),
            ("block 12 only", "openai__dual_L12_fuse-testminmax_p-mvfa_e50.jsonl", "testminmax"),
            ("blocks 3/6/9/12", "openai__dual_L3-6-9-12_fuse-testminmax_p-mvfa_e50.jsonl", "testminmax"),
            ("memory score only (no test statistics)", RO, "memory"),
            ("text score only", RO, "text"),
            ("+ memory-consistency loss, dual memory", "openai__dual_L4-8-12_fuse-testminmax_p-mvfa_e50_mem1.0_dualmem.jsonl", "testminmax")]
    ds = [d for d in DS if d != "Chest"]  # Chest features were cached for blocks 4/8/12 only
    out = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
           r"\textbf{Variant} ($k\in\{4,16\}$, mean of 5 datasets) & \textbf{Image AUC} & \textbf{Pixel AUC} \\", r"\midrule"]
    res = {}
    for label, f, var in cfgs:
        rows = [r for r in load(f) if r["k"] in (4, 16) and r["dataset"] in ds]
        cells = [(d, k) for d in ds for k in (4, 16)]
        if not all(any(r["dataset"] == d and r["k"] == k for r in rows) for d, k in cells):
            out.append(f"{label} & \\NUM{{running}} & \\NUM{{running}} \\\\")
            continue
        i = st.mean(st.mean(img(r, var) for r in rows if (r["dataset"], r["k"]) == c) for c in cells)
        p = st.mean(st.mean(100 * r["variants"]["pixel_" + var] for r in rows if (r["dataset"], r["k"]) == c)
                    for c in cells if c[0] in SEG)
        res[label] = (i, p)
        out.append(f"{label} & {i:.1f} & {p:.1f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out), res


def support_effects():
    rows = load(RO)
    s0 = [img(r) for r in rows if r["set"] == 0]
    s12 = [img(r) for r in rows if r["set"] > 0]
    spread = []
    for d in DS:
        for k in (2, 4, 8, 16):
            v = [img(r) for r in rows if r["dataset"] == d and r["k"] == k]
            if len(v) == 3:
                spread.append((max(v) - min(v), d, k))
    spread.sort(reverse=True)
    return {"set0": st.mean(s0), "random": st.mean(s12), "max_range": spread[0], "median_range": st.median(x[0] for x in spread)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "paper_v3" / "tables"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t, gi, gp = epoch_selection(); (out / "epoch_selection.tex").write_text(t)
    t2, res = ablation(); (out / "ablation.tex").write_text(t2)
    se = support_effects()
    (out / "numbers.json").write_text(json.dumps({"epoch_inflation_img": gi, "epoch_inflation_px": gp,
                                                  "ablation": res, "support": se}, indent=2, default=str))
    print(t); print(t2); print(json.dumps(se, default=str))


if __name__ == "__main__":
    main()
