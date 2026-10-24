"""Build the paper's result tables from results_fsad/*.jsonl and published_results.json.

Every number in the LaTeX output comes from a stored run; nothing is typed by hand.
  python experiments/make_tables.py --out paper_v3/tables
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ["Brain", "Liver", "Retina_RESC", "Retina_OCT2017", "Chest", "Histopathology"]
SEG = ["Brain", "Liver", "Retina_RESC"]
SHORT = {"Brain": "Brain", "Liver": "Liver", "Retina_RESC": "RESC", "Retina_OCT2017": "OCT17",
         "Chest": "Chest", "Histopathology": "HIS"}
SHOTS = [2, 4, 8, 16]

# (row label, backbone, config name, scoring variant)
MAIN_ROWS = [
    ("Frozen text head", "openai", "none_L4-8-12_fuse-support_p-mvfa_e50", "text"),
    ("Frozen memory head", "openai", "none_L4-8-12_fuse-support_p-mvfa_e50", "memory"),
    ("Frozen memory head (BiomedCLIP)", "biomedclip", "none_L4-8-12_fuse-support_p-mvfa_e50", "memory"),
    ("RO-MVFA", "openai", "dual_L4-8-12_fuse-testminmax_p-mvfa_e50", "testminmax"),
    ("RO-MVFA, memory score only", "openai", "dual_L4-8-12_fuse-testminmax_p-mvfa_e50", "memory"),
    ("RO-MVFA (BiomedCLIP)", "biomedclip", "dual_L4-8-12_fuse-testminmax_p-mvfa_e50", "testminmax"),
    ("+ memory-consistency variant", "openai", "dual_L4-8-12_fuse-testminmax_p-mvfa_e50_mem1.0_dualmem", "testminmax"),
]


def load_runs(res_dir):
    runs = []
    for p in glob.glob(str(Path(res_dir) / "*.jsonl")):
        if "__dev" in p:
            continue
        runs += [json.loads(line) for line in open(p)]
    return runs


def metric(run, variant, pixel):
    v = run.get("variants", {})
    if pixel:
        return v.get(f"pixel_{variant}")
    key = f"image_max_{variant}" if run["dataset"] in SEG else f"image_det_{variant}"
    return v.get(key)


def collect(runs, backbone, config, variant):
    """-> {(dataset, k, 'img'|'px'): [values over support sets]}"""
    out = defaultdict(list)
    for r in runs:
        if r["backbone"] != backbone or r["config"] != config:
            continue
        for pixel, tag in ((False, "img"), (True, "px")):
            if pixel and r["dataset"] not in SEG:
                continue
            m = metric(r, variant, pixel)
            if m is not None:
                out[(r["dataset"], r["k"], tag)].append(100 * m)
    return out


def cell(vals, show_std=True):
    if not vals:
        return "--"
    m = np.mean(vals)
    if show_std and len(vals) > 1:
        return f"{m:.1f}\\textsubscript{{$\\pm${np.std(vals, ddof=1):.1f}}}"
    return f"{m:.1f}"


def main_table(runs, published, k):
    cols = [(d, "img") for d in DATASETS] + [(d, "px") for d in SEG]
    head = " & ".join([SHORT[d] for d in DATASETS] + [SHORT[d] for d in SEG])
    lines = [r"\begin{tabular}{@{}l" + "c" * len(cols) + "cc@{}}", r"\toprule",
             r" & \multicolumn{6}{c}{\textbf{Image AUC}} & \multicolumn{3}{c}{\textbf{Pixel AUC}} & \multicolumn{2}{c}{\textbf{Mean}} \\",
             r"\cmidrule(lr){2-7}\cmidrule(lr){8-10}\cmidrule(l){11-12}",
             "\\textbf{Method} & " + head + r" & img & px \\", r"\midrule",
             r"\multicolumn{12}{@{}l}{\emph{Published results (one support set; MVFA and MadCLIP use ViT-L/14 at 240 px)}} \\"]
    for name in ("APRIL-GAN", "MediCLIP", "MVFA", "MadCLIP"):
        p = published[name][str(k)]
        vals = [p.get(d) for d in DATASETS] + [p.get(f"{d}_px") for d in SEG]
        img = np.mean([p[d] for d in DATASETS]); px = np.mean([p[f"{d}_px"] for d in SEG])
        lines.append(name + " & " + " & ".join(f"{v:.1f}" for v in vals) + f" & {img:.1f} & {px:.1f} \\\\")
    lines += [r"\midrule", r"\multicolumn{12}{@{}l}{\emph{ViT-B/16 on identical features, mean$\pm$std over 3 support sets}} \\"]
    for label, bb, cfg, var in MAIN_ROWS:
        c = collect(runs, bb, cfg, var)
        vals = [c.get((d, k, "img"), []) for d in DATASETS] + [c.get((d, k, "px"), []) for d in SEG]
        if not any(vals):
            continue
        img = np.mean([np.mean(v) for v in vals[:6]]) if all(vals[:6]) else float("nan")
        px = np.mean([np.mean(v) for v in vals[6:]]) if all(vals[6:]) else float("nan")
        lines.append(label + " & " + " & ".join(cell(v) for v in vals) + f" & {img:.1f} & {px:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def shots_table(runs, published):
    """Mean image / pixel AUC over datasets for each k."""
    lines = [r"\begin{tabular}{@{}l" + "cc" * len(SHOTS) + "@{}}", r"\toprule",
             r"\textbf{Method} & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$k={k}$}}" for k in SHOTS) + r" \\",
             " & " + " & ".join(["img & px"] * len(SHOTS)) + r" \\", r"\midrule"]
    for name in ("MVFA", "MadCLIP"):
        row = []
        for k in SHOTS:
            p = published[name][str(k)]
            row += [f"{np.mean([p[d] for d in DATASETS]):.1f}", f"{np.mean([p[d + '_px'] for d in SEG]):.1f}"]
        lines.append(name + " (published) & " + " & ".join(row) + r" \\")
    for label, bb, cfg, var in MAIN_ROWS:
        c = collect(runs, bb, cfg, var)
        row = []
        for k in SHOTS:
            img = [np.mean(c[(d, k, "img")]) for d in DATASETS if c.get((d, k, "img"))]
            px = [np.mean(c[(d, k, "px")]) for d in SEG if c.get((d, k, "px"))]
            row += [f"{np.mean(img):.1f}" if len(img) == 6 else "--", f"{np.mean(px):.1f}" if len(px) == 3 else "--"]
        if any(x != "--" for x in row):
            lines.append(label + " & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT.parent / "results_fsad"))
    ap.add_argument("--out", default=str(ROOT.parent / "paper_v3" / "tables"))
    a = ap.parse_args()
    runs = load_runs(a.results)
    published = json.loads((ROOT / "published_results.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in SHOTS:
        (out / f"main_k{k}.tex").write_text(main_table(runs, published, k))
    (out / "shots.tex").write_text(shots_table(runs, published))
    print(f"{len(runs)} runs -> tables in {out}")
    print((out / "main_k4.tex").read_text())


if __name__ == "__main__":
    main()
