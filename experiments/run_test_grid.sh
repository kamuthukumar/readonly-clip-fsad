#!/bin/bash
# BMAD test-split experiments for the paper, in priority order. Settings were fixed on the dev
# split (results_dev/DECISION.md) before the test split was used. One process per
# (config, dataset) so GPU memory is released between runs; finished runs are skipped, so the
# script can be stopped and restarted at any point.
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
OUT=results_fsad
DATASETS="Brain Liver Retina_RESC Retina_OCT2017 Histopathology Chest"
NOCHEST="Brain Liver Retina_RESC Retina_OCT2017 Histopathology"
MVFA="--adapter dual --fusion testminmax"            # MVFA objective + fusion on read-only adapters
OURS="$MVFA --w_mem_loss 1 --dual_memory"            # + memory-consistency loss + dual memory

run() {  # run BACKBONE "SHOTS" "SETS" "DATASETS" extra args...
  local bb=$1 shots=$2 sets=$3 ds=$4; shift 4
  for d in $ds; do
    .venv/bin/python experiments/fsad.py run --backbone $bb --datasets $d --shots $shots --sets $sets --out $OUT "$@" 2>&1 \
      | grep -E --line-buffered "k=|GPU fault|FAILED|Traceback|Error" | sed -u "s|^|[$bb $*] |"
  done
}

# 1. main comparison, OpenAI CLIP ViT-B/16, k = 2/4/8/16, support sets 0 (MVFA's list), 1, 2
run openai "2 4 8 16" "0 1 2" "$DATASETS" --adapter none        # frozen: text / memory / both
run openai "2 4 8 16" "0 1 2" "$DATASETS" $MVFA                 # MVFA-B/16
run openai "2 4 8 16" "0 1 2" "$DATASETS" $OURS                 # memory-consistency variant (ablation)

# 2. BiomedCLIP backbone
.venv/bin/python experiments/fsad.py cache --backbone biomedclip --datasets $NOCHEST --blocks 4 8 12
.venv/bin/python experiments/fsad.py cache --backbone biomedclip --datasets Chest --blocks 4 8 12
run biomedclip "2 4 8 16" "0 1 2" "$DATASETS" --adapter none
run biomedclip "2 4 8 16" "0 1 2" "$DATASETS" $MVFA

# 3. epoch-selection study: score the test split every 10 epochs (k = 4)
run openai "4" "0 1 2" "$DATASETS" $MVFA --eval_every 10

# 4. design ablations of MVFA-B/16 at k = 4 and 16
run openai "4 16" "0 1 2" "$DATASETS" --adapter shared --fusion testminmax
run openai "4 16" "0 1 2" "$NOCHEST" $MVFA --blocks 12
run openai "4 16" "0 1 2" "$NOCHEST" $MVFA --blocks 3 6 9 12

# 5. memory-consistency variant on BiomedCLIP (lowest priority)
run biomedclip "2 4 8 16" "0 1 2" "$DATASETS" $OURS
echo TEST_GRID_DONE
