# Settings fixed on the dev split before any final test-split run (2026-10-22)

Dev split = BMAD valid images not in the support set; k=4, support set 1; OpenAI CLIP ViT-B/16.

1. Adapters: two branches (detection / segmentation). One shared branch lost up to 26 pixel-AUC
   points (Brain 70.6 vs 96.6).
2. Training losses: MVFA text-alignment losses + memory-consistency loss (weight 1).
3. Memory: dual (adapted + frozen CLIP distances). Adapted-only memory lost to frozen features
   on Brain, Chest and Histopathology; dual memory removed those losses.
4. Primary score at test time: memory distance only (`variants.*_memory`). It matched or beat
   the fused score on the dev runs and needs no test statistics. MVFA's test-set min-max fusion
   (`*_testminmax`) is reported alongside for comparability with published numbers.
5. Fixed 50 epochs, Adam lr 1e-3, batch 8; no epoch selection.
