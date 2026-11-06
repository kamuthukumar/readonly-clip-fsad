#!/bin/bash
# ROAM test-split runs on the datasets with masks. Finished runs are skipped, so this can be
# stopped and restarted. Datasets without masks reuse the memory score of the existing runs.
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p results_vd
LOG=results_vd/run.log
for bb in biomedclip openai; do
  for d in Liver Retina_RESC Brain; do
    .venv/bin/python experiments/fsad.py run --backbone $bb --datasets $d --shots 2 4 8 16 --sets 0 1 2 \
      --adapter dual --fusion testminmax --extra_scores --xs_lite --out results_vd 2>&1 \
      | grep -E --line-buffered "k=|GPU fault|FAILED|Traceback|Error" | sed -u "s|^|[$bb] |" >> $LOG
  done
done
echo "ALL DONE" >> $LOG
