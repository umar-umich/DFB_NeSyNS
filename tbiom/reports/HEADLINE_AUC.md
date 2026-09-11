# Headline cross-dataset AUC

Video-level AUROC. **The OOD mean is computed only over the columns every row shares** — averaging over different column sets is not a comparison, and the released checkpoint was initially missing Celeb-DF-v3, the hardest column, which would have flattered it.

Shared OOD columns: Celeb-DF-v2, Celeb-DF-v3, DFD, DFDC, DFDCP, Deepfake-Eval-2024.

| model · readout | FF++ | Celeb-DF-v2 | Celeb-DF-v3 | DFD | DFDC | DFDCP | Deepfake-Eval-2024 | **OOD mean** |
|---|---|---|---|---|---|---|---|---|
| DiCoME released (frozen) · artifact | 0.9907 | 0.9778 | 0.8886 | 0.9462 | 0.8853 | 0.8826 | 0.6645 | **0.8742** |
| DiCoME released (frozen) · fused | 0.9905 | 0.9731 | 0.8777 | 0.9396 | 0.8825 | 0.8799 | 0.6735 | **0.8710** |
| DiCoME released (frozen) · semantic | 0.9903 | 0.9675 | 0.8697 | 0.9384 | 0.8794 | 0.8782 | 0.6807 | **0.8690** |
| P0-DS (retrainable) · artifact | 0.9912 | 0.9685 | 0.8490 | 0.9487 | 0.8764 | 0.8570 | 0.6712 | **0.8618** |
| P0-DS (retrainable) · fused | 0.9929 | 0.9646 | 0.8409 | 0.9421 | 0.8828 | 0.8573 | 0.6922 | **0.8633** |
| P0-DS (retrainable) · semantic | 0.9929 | 0.9564 | 0.8199 | 0.9399 | 0.8812 | 0.8547 | 0.6909 | **0.8572** |
| _CLIP port (where we started)_ | 0.9909 | 0.9225 | — | 0.9222 | 0.8477 | 0.8912 | 0.6357 | _0.8439_ |

## Headline

**Best OOD mean: DiCoME released (frozen) · artifact at 0.8742** over 6 columns (Celeb-DF-v2, Celeb-DF-v3, DFD, DFDC, DFDCP, Deepfake-Eval-2024).

Against the CLIP port we started from, restricted to the 5 columns the port has: **0.8713** vs **0.8439** — **+0.0274**. The port was never scored on Celeb-DF-v3, so comparing its mean against a 6-column mean would repeat the very error this table exists to avoid.

On Celeb-DF-v2 specifically: **0.9778** against the port's 0.9225 — **+0.0553**.

## Frozen vs retrainable — the choice that matters later

| | best readout | OOD mean |
|---|---|---:|
| frozen (released) | artifact | 0.8742 |
| retrainable (P0-DS) | fused | 0.8633 |

Gap **+0.0109** on shared columns. The frozen checkpoint cannot be retrained, which forecloses any later corpus or recipe work; the retrainable one keeps that open for that cost.

## Caveats that belong with these numbers

- These are TEST splits. Nothing here was used to select a checkpoint or fit a threshold — selection ran on FF++ VAL_select and VALmix throughout.
- Single seed. The spec's §22 rule wants a confirming seed for |ΔAUROC| < 0.01, which covers most of the gaps between readouts in this table.
- AUROC alone hid an operational collapse once in this project. The real-side health for these same models is in `tbiom/STEP3_ANCHOR_HEALTH.md`; the artifact readout has markedly better zero-shot FPR_real (0.169) than fused (0.247) at similar AUROC.
