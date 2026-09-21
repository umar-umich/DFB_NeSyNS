# Stage 1A — reliability controller ladder: frozen before any controller is fitted

Written 2026-09-20. Governs `tbiom/reports/EXP_RELIABILITY_CONTROLLER.md`. Cites `tbiom/reports/REFERENCE_MANIFEST.md`.

## What is held fixed

* **Branches:** the frozen reference — V-JEPA 2 LoRA (B-augmented, `vjepa2_lora_B_seed42`), and Appearance +
  Off-manifold from the warm+gate epoch-19 checkpoint. Never outputs from Stage 1B's changing checkpoints.
* **Discount rule:** evidence scaling exactly as implemented, α' = 1 + w(α − 1) (manifest §2). Only the predictor of
  w changes between controllers; the equation never does.
* **Segments:** 4 contiguous segments per video over the stored crops (manifest §3–4), from
  `logs/tbiom/acquisition/v2/`. Empty segments have no row and are masked, never filled. Spatial segment opinions are
  means of the cached per-frame opinions over the segment's crops.
* **Aggregation:** within each view, the discounted segment evidence is **averaged** over the video's available
  segments, α_v = 1 + mean_s(w_{v,s} · e_{v,s}); then Dempster's rule across the three views. Segments are disjoint
  crops, and averaging (not summing) keeps them from counting as independent evidence of the same video.
* **Data and access:** the development pool (VALmix, VALmix-v2, FF++ val), clean + JPEG q40 + resize 0.60 versions of
  each video in the **same** source group; the 5 source-grouped splits used everywhere; fitting fold trains, selection
  fold scores. The fusion therefore learns from labelled external development data — disclosed as such, and every
  competing controller has exactly the same access.

## The ladder — identical inputs, splits, discounting, aggregation and budget

| controller | predictor of w_{v,s} |
|---|---|
| constant DS-R (reference) | one weight per view, grid {0, .25, .5, .75, 1} with max = 1, fitted on the fitting fold (the existing procedure) |
| segment MLP | 2 × 32 hidden, current-segment features only |
| GRU | 1 layer, 32 hidden, over segments in time order |
| Mamba | one small selective state-space block (d = 32, state 8, causal depthwise conv of width 2), pure PyTorch |

Learned controllers output a per-segment, per-view offset Δ and use w = σ(logit(clip(w_const, .02, .98)) + Δ), with
the output layer initialised to zero, so each starts exactly at the constant weights and must earn any departure.
Inputs per segment: each view's p and u, cross-view disagreement (range and s.d. of p, largest pairwise conflict),
the five quality descriptors (fitting-fold standardised), and availability. The condition label is never an input.
Training: fused log-loss over clean + processed fitting examples, Adam, lr 1e-3, weight decay 1e-4, 60 full-batch
epochs, seed 0 — the same for all three learned controllers, no per-controller tuning.

## Controls

1. No quality descriptors (each learned controller refitted without them).
2. Shuffled chronology (segment order permuted per video, fixed permutation, labels untouched).
3. Quality trajectories reassigned across videos without regard to labels.
4. The clean + processed logistic stacker on the same segment-averaged opinions and the same access.
5. **Stress condition:** each segment's condition is drawn independently from {clean, JPEG q40, resize 0.60}, by a
   draw that ignores the label, so quality changes between contiguous segments while the distribution is identical
   across classes. Uniform degradation throughout a video does not test history; this does.

## Acceptance, development selection folds, DS-R-style fused output, frozen now

Scored separately on clean, JPEG q40, resize 0.60 and the stress condition. Δ against constant DS-R, paired
source-group bootstrap, 1000 draws per split.

1. **Adaptive reliability succeeds** if the best of MLP / GRU / Mamba beats constant DS-R by **≥ +0.005** v3.1 averaged
   over clean, JPEG and resize, with the CI lower bound > 0 in **≥ 3 of 5** splits, and none of these regressions:
   clean Δ < −0.005; reenactment Δ with CI upper bound < 0 in ≥ 3 of 5 splits; FPR at the frozen τ up by > +0.02.
2. **History helps** only if GRU or Mamba beats the MLP on the stress condition by ≥ +0.005 **and** that advantage
   disappears under shuffled chronology.
3. **Mamba earns inclusion** only if it beats both the MLP and the GRU by ≥ +0.002 on the criterion-1 average, or
   matches the better of them within 0.002 at lower measured compute. The reliability mechanism does not fail because
   the MLP wins.
4. τ is reported both at the constant controller's numeric value (drop-in) and refitted per controller by the same
   EER-on-FF++-val procedure (comparative); neither is refitted per test condition. Decisions use the refitted τ.
5. Also reported: error-detection AUROC and risk-coverage of the new fused output (fused vacuity and margin), within
   each dataset and condition as well as pooled, and compute (parameters, wall-clock per 1,000 videos).

No test set enters any of the above.
