# Pre-registered configurations (fixed before any test result was seen)

2026-10-31: adapter ensemble, M=5 members (seeds set*100+k+1000*m), BiomedCLIP, dual adapters,
blocks 4/8/12, 50 epochs, MVFA losses. Reported scores: memory-only (no test statistics) and
test min-max fusion, both recorded for every run. M was chosen a priori and is not tuned on test.

2026-10-31 (later): the adapter ensemble was stopped after one test run (Liver k=2 set 0: 71.8 vs 72.0
single model); it is not part of the method.

Dev-split selection (BiomedCLIP, k=2/4, sets 0-2, all six datasets; results_dev/*_xs__dev.jsonl and
results_dev/xs2/): neighbourhood averaging (agg), adapted+frozen memory (dual) and pseudo-lesion
detection memory (pseudo) did not help and are dropped. The two-sided lesion memory beat the
normal-only memory in 17 of 18 dev runs on the mask datasets.

FIXED METHOD for the test split (no further changes):
- datasets with masks (Brain, Liver, RESC): score map = sum over levels of
  d(x, normal support patches) - d(x, lesion patches of the abnormal support images), adapted
  segmentation tokens; image score = max of the map  -> variants pixel_memory_two / image_max_memory_two
- datasets without masks (OCT2017, Chest, Histopathology): memory-only detection score -> image_det_memory
- no test-set statistics anywhere. Test min-max fusion is reported only for comparison.

2026-11-05: volume-disjoint evaluation added BEFORE any such result was seen.
experiments/patient_overlap.py and a slice-correlation check show that all Liver and RESC volumes
(normal and abnormal) occur in both valid and test, as Brain does for 10 of 11 abnormal valid volumes.
PRIMARY evaluation for Brain, Liver and RESC from now on: test images whose <volume> id is not the
volume of any support image (variants vd_*, counts vd_n_pos/vd_n_neg). All-test numbers stay recorded
only for comparison with published results. The fixed method itself is unchanged.

2026-11-05: leakage control, design fixed before any result (experiments/leak_control.py).
Liver and RESC, BiomedCLIP, k=4 and 8, same model and scores as ROAM. Patients are split in half,
3 random partitions, both halves in turn. Evaluation images are always the test slices of one half T.
Condition "same": support drawn from valid slices of T. Condition "other": support drawn from valid
slices of the other half. Same training seed in a pair, so any paired AUC difference on identical
test images measures the effect of support patients appearing in the evaluation set.
Reported: paired mean difference and Wilcoxon signed-rank test for memory, memory_two and testminmax.

2026-11-06: outcome. The two-sided lesion memory is NOT adopted as the proposed method. On test, volume
disjoint (results_vd/): clear image-AUC gain only on Liver with BiomedCLIP; no real gain on Brain or
RESC; loses on Liver with OpenAI weights at k=8 and 16; pixel AUC usually lower. Pixel AUC fell below 60
in 3 of 48 leakage-control runs. It is reported as a negative result. The dev split overestimated it,
which also shows dev selection on BMAD inherits the same patient overlap.
