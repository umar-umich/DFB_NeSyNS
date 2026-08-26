# Step 1 — like-for-like anchor comparison

Does the Celeb-DF-v2 anchor gap belong to the CLIP branch, or to DiCoME's decomposition and internal DS fusion? Five readouts, one exporter, the same six datasets.

## 0. Correctness check — does the new exporter reproduce DiCoME's own eval path?

`p_fused` here must match `DiCoME/eval_adaptation/RESULTS.md`, which came from DiCoME's untouched `test_step`. If it does not, the exporter is reading the model wrong and nothing below means anything.

| dataset | published | this exporter | delta |
|---|---:|---:|---:|
| CDFv2 | 0.9729 | 0.9731 | +0.0002 |
| DFD | 0.9392 | 0.9396 | +0.0004 |
| DFDC | 0.8822 | 0.8825 | +0.0003 |
| DFDCP | 0.8799 | 0.8799 | -0.0000 |
| FFpp | 0.9905 | 0.9905 | -0.0000 |

## 1. Video AUROC

| readout | FFpp | CDFv2 | DFD | DFDC | DFDCP | DFEval24 |
|---|---|---|---|---|---|---|
| DiCoME-released fused | 0.9905 | 0.9731 | 0.9396 | 0.8825 | 0.8799 | 0.6735 |
| DiCoME-released semantic | 0.9903 | 0.9675 | 0.9384 | 0.8794 | 0.8782 | 0.6807 |
| DiCoME-released artifact | 0.9907 | 0.9778 | 0.9462 | 0.8853 | 0.8826 | 0.6645 |
| DiCoME-P0DS fused | 0.9929 | 0.9646 | 0.9421 | 0.8828 | 0.8573 | 0.6922 |
| DiCoME-P0DS semantic | 0.9929 | 0.9564 | 0.9399 | 0.8812 | 0.8547 | 0.6909 |
| CLIP port | 0.9909 | 0.9225 | 0.9222 | 0.8477 | 0.8912 | 0.6357 |

## 2. The operational column — FPR on REAL videos at a frozen tau

One tau per readout, the EER on that readout's OWN FF++ scores, frozen across every other dataset. Per-readout because the five are differently calibrated; a shared tau would measure calibration offset rather than operating quality.

| readout | tau | FFpp | CDFv2 | DFD | DFDC | DFDCP | DFEval24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DiCoME-released fused | 0.5639 | 0.036 | 0.079 | 0.019 | 0.150 | 0.187 | 0.147 |
| DiCoME-released semantic | 0.6635 | 0.036 | 0.090 | 0.030 | 0.158 | 0.196 | 0.192 |
| DiCoME-released artifact | 0.3567 | 0.029 | 0.079 | 0.041 | 0.152 | 0.204 | 0.157 |
| DiCoME-P0DS fused | 0.4279 | 0.036 | 0.169 | 0.209 | 0.366 | 0.452 | 0.388 |
| DiCoME-P0DS semantic | 0.6099 | 0.029 | 0.219 | 0.220 | 0.367 | 0.448 | 0.439 |
| CLIP port | 0.3789 | 0.036 | 0.101 | 0.088 | 0.201 | 0.217 | 0.285 |

## 3. Probability separation `d_RF`

| readout | FFpp | CDFv2 | DFD | DFDC | DFDCP | DFEval24 |
|---|---|---|---|---|---|---|
| DiCoME-released fused | +0.849 | +0.625 | +0.624 | +0.495 | +0.459 | +0.169 |
| DiCoME-released semantic | +0.715 | +0.503 | +0.517 | +0.407 | +0.366 | +0.157 |
| DiCoME-released artifact | +0.739 | +0.520 | +0.545 | +0.423 | +0.399 | +0.112 |
| DiCoME-P0DS fused | +0.812 | +0.531 | +0.543 | +0.424 | +0.368 | +0.162 |
| DiCoME-P0DS semantic | +0.659 | +0.386 | +0.381 | +0.304 | +0.248 | +0.127 |
| CLIP port | +0.864 | +0.531 | +0.570 | +0.459 | +0.491 | +0.140 |

## 4. Verdict

On Celeb-DF-v2: DiCoME-fused **0.9731**, DiCoME-semantic-only **0.9675**, our CLIP port **0.9225**.

- semantic-only minus port: **+0.0451** — the CLIP *recipe* gap.
- fused minus semantic-only: **+0.0055** — what the artifact view and internal DS fusion add.

**A genuine recipe gap exists.** DiCoME's own CLIP branch beats our port by more than the noise band on identical LoRA settings, so the difference is in the training recipe — batch size (128 vs 32), precision (bf16-mixed vs fp32), or the VAE/alignment loss terms shaping the shared encoder. Investigate those before choosing a chassis.

## 5. The operational finding — and it changes Step 2

Mean FPR on REAL videos across the five OOD sets: DiCoME-released **0.116**, our CLIP port **0.179**, DiCoME-P0DS **0.317**.

**Our retrain of DiCoME has worse real-side health than our own CLIP port**, on five of six datasets, despite matching or beating it on AUROC. On DFDC the port scores 0.8477 AUROC against P0-DS's 0.8828, yet calls 20.1% of reals fake against P0-DS's 36.6%. On DFDCP: port 0.8912 / 0.217, P0-DS 0.8573 / 0.452.

The likely cause is checkpoint selection, and it is the failure mode this brief was written around. P0-DS was picked at **epoch 1** on the highest `val_auroc_video` (0.9960) — an in-domain metric that is saturated, where every candidate epoch scores above 0.995 and the ranking among them is noise. The released checkpoint is epoch 4. Two other P0-DS checkpoints exist (epochs 2 and 5) and were never evaluated on anything but that saturated number.

So Step 2 must not simply adopt P0-DS as the strongest retrainable anchor on the strength of its AUROC. It must re-select among the available P0-DS checkpoints on the health dashboard, with real-side FPR overriding AUROC, exactly as the brief specifies. That is cheap — the checkpoints are on disk.
