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

One tau per readout, the EER on that readout's own **FF++ VAL** scores, frozen across every dataset below. Per-readout because the five are differently calibrated; a shared tau would measure calibration offset rather than operating quality. VAL and not test: an earlier version of this table took tau from FF++ test, which handed the released checkpoint a higher threshold and inflated the real-side gap from 0.048 to 0.201.

| readout | tau | FFpp | CDFv2 | DFD | DFDC | DFDCP | DFEval24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DiCoME-released fused | 0.4069 | 0.036 | 0.180 | 0.091 | 0.239 | 0.309 | 0.292 |
| DiCoME-released semantic | 0.5164 | 0.043 | 0.242 | 0.099 | 0.257 | 0.339 | 0.346 |
| DiCoME-released artifact | 0.3455 | 0.029 | 0.084 | 0.050 | 0.160 | 0.217 | 0.164 |
| DiCoME-P0DS fused | 0.4784 | 0.007 | 0.129 | 0.174 | 0.317 | 0.391 | 0.341 |
| DiCoME-P0DS semantic | 0.6086 | 0.029 | 0.225 | 0.223 | 0.368 | 0.448 | 0.439 |
| CLIP port | 0.3260 | 0.050 | 0.124 | 0.124 | 0.229 | 0.239 | 0.336 |

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

## 5. The operational column

Mean FPR on REAL videos across the five OOD sets, tau frozen on FF++ VAL: our CLIP port **0.210**, DiCoME-released **0.222**, DiCoME-P0DS **0.271**.

**Correction to an earlier version of this table.** It took tau from FF++ TEST, which handed the released checkpoint a threshold of 0.5639 against P0-DS's 0.4279 — and a higher threshold mechanically calls fewer reals fake. That produced 'released 0.116 vs P0-DS 0.317' and a conclusion that P0-DS had badly broken real-side health. On the permitted development source the gap is +0.048, not 0.201. The threshold must come from development data for the same reason it must be frozen: otherwise it is fitted to the thing it is being used to judge.

What survives the correction: P0-DS is still the weakest of the three on real-side health, and our CLIP port is now level with the released checkpoint rather than far behind it. So the port's deficit is a RANKING deficit (-0.045 AUROC on Celeb-DF-v2), not an operational one — which sharpens the recipe question rather than answering it, and is consistent with the gap living inside the learned representation.
