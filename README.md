# Read-only CLIP adapters for few-shot medical anomaly detection

Code, support lists and per-run results for:

> Revisiting few-shot medical anomaly detection with CLIP: read-only adapters and a leak-free
> evaluation on BMAD. Submitted to *Computerized Medical Imaging and Graphics*.

The adapters read a frozen CLIP ViT-B/16 encoder after blocks 4, 8 and 12 and never write back
into it, so encoder tokens are computed once per image and cached. Every support set, shot count
and ablation reuses the same cache.

## Setup

Python 3.11. Versions used for the paper:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Experiments ran on an Apple M1 GPU (MPS). The code also runs on CUDA or CPU.

## Data

BMAD (Bao et al., 2023). Only the valid and test splits are needed, plus the OCT2017 train
normals that MVFA's 16-shot list uses (`--datasets Retina_OCT2017 --splits train`).
`fetch_bmad.py` streams the requested folders from the Hugging Face mirror of the official
release without downloading the full 21.8 GB archive:

```bash
python experiments/fetch_bmad.py --out ~/data/BMAD --splits valid test
```

Set `BMAD_ROOT` and `FSAD_CACHE` to use other locations (defaults `~/data/BMAD`, `~/data/fsad_cache`).

## Reproducing the paper

```bash
# 1. cache frozen tokens (once per backbone: openai or biomedclip)
python experiments/fsad.py cache --backbone openai

# 2. all test-split runs, in the order used for the paper (resumable)
bash experiments/run_test_grid.sh

# 3. in-line (MVFA placement) baseline at k=4, and the qualitative figure
python experiments/inline_baseline.py --shots 4 --sets 0 1 2 \
    --datasets Brain Liver Retina_RESC Retina_OCT2017 Histopathology
python experiments/qualitative.py --backbone biomedclip --k 4 --set 1

# 4. tables and every number quoted in the text
python experiments/make_tables.py --out tables
python experiments/analysis_tables.py --out tables
```

Step 4 needs only the JSONL files in `results_fsad/`, which are included, so the tables can
be checked without a GPU.

## Main options of `fsad.py run`

| Option | Paper setting |
|---|---|
| `--adapter dual` | two adapter sets (detection, localization); `shared`, `none` for ablations |
| `--fusion testminmax` | MVFA's min-max fusion over the test set; the memory-only score is always recorded too |
| `--blocks 4 8 12` | tapped blocks; `12` and `3 6 9 12` for ablations |
| `--w_mem_loss 1 --dual_memory` | memory-consistency variant (negative result) |
| `--eval_every 10` | epoch-selection study; the reported runs never use it for selection |
| `--dev` | evaluate on valid minus support (development split) |

## Files

- `experiments/fewshot_seed/`: MVFA's published support lists (support set 0), MIT licence,
  see `LICENSE_MVFA`. Sets 1 and 2 are drawn by `support_set()` with fixed seeds and exclude
  images whose 64x64 hash matches a test image.
- `experiments/published_results.json`: published numbers used in the comparison tables, with sources.
- `results_fsad/*.jsonl`: one line per test run (dataset, k, support set, all score variants,
  timings, number of support images duplicated in the test split).
- `results_dev/`: development-split runs and `DECISION.md`, the design decisions made before
  the test split was used.

## Known issue in the benchmark

10 of the 11 abnormal brain volumes in BMAD's valid split also appear in its test split, and
10 valid slices are pixel-identical to test slices. Our random support sets exclude the identical
slices but not the shared volumes; see Section 4.1 of the paper.

## Licence

Code: MIT (see `LICENSE`). MVFA's support lists in `experiments/fewshot_seed/` keep their own MIT
licence (`LICENSE_MVFA`). BMAD images are not included; BMAD is distributed under CC BY-NC-SA.
