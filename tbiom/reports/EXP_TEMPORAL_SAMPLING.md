# Stage 2 — bounded temporal experiment

Pre-registration: `tbiom/docs/stage2-temporal-preregistration.md`. Each configuration replaces the temporal view of the reference triple; Appearance and Off-manifold are the frozen reference views. Fusion refitted on the fitting part; candidate chosen on the selection part; the stopping rule assessed on the confirmation part only. Partition sizes: fit 803, select 628, confirm 628 videos.

## 1. The 2 × 2 (plus the same-window diagnostic), fused DS-R subset-v3.1

| configuration | branch alone (select) | fused, select clean | select JPEG | select resize | fused, confirm clean | confirm reenactment | export ms/video |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse × pooling (reference) | 0.8012 | 0.8463 | 0.7915 | 0.8347 | 0.8728 | 0.9407 | 49.1 |
| sparse × GRU head | 0.7924 | 0.8370 | 0.7701 | 0.8232 | 0.8765 | 0.9349 | 34.3 |
| dense × pooling **(candidate)** | 0.7927 | 0.8576 | 0.8101 | 0.8455 | 0.8784 | 0.9307 | 34.7 |
| dense × GRU head | 0.7910 | 0.8571 | 0.8015 | 0.8409 | 0.8798 | 0.9349 | 34.4 |
| dense-subsampled × pooling (diagnostic) | 0.7198 | 0.7834 | 0.7451 | 0.7493 | 0.8405 | 0.8328 | 77.0 |
| sparse × Mamba head (only if triggered) | TODO(run) | TODO(run) | TODO(run) | TODO(run) | TODO(run) | TODO(run) | TODO(run) |
| dense × Mamba head (only if triggered) | TODO(run) | TODO(run) | TODO(run) | TODO(run) | TODO(run) | TODO(run) | TODO(run) |

## 2. The stopping rule, on the confirmation part

Candidate (best fused DS-R on the selection part): **dense × pooling**.

1. Margin: fused Δ **+0.0056**, 95% CI [-0.0352, +0.0467] (need ≥ +0.005 and CI excluding zero) → NOT met.
2. Robustness: mean Δ over JPEG q40 and resize 0.60 **-0.0098** (need ≥ -0.005) → NOT met.
3. Reenactment: Δ **-0.0100** (need ≥ -0.005) → NOT met.
4. Compute: 0.71× the reference's export time per video (need ≤ 2.0×); the backbone sees 16 frames in every configuration, so model FLOPs are equal up to the small GRU; dense windows additionally need decoding and face cropping of 32 consecutive source frames, reported separately.

**Verdict: STOP temporal-architecture development for this paper.** Reported as insufficient demonstrated benefit under the tested configurations within the allocated budget — not as evidence that temporal information is absent.
FPR / TPR at the fitting part's τ, confirmation clean: candidate 0.066 / 0.800, reference 0.049 / 0.748.

## Mamba trigger (pre-registered 2026-09-21, before any fused number was computed)

A small temporal head is useful if a GRU configuration beats pooling at the same sampling by ≥ +0.005 fused DS-R subset-v3.1 on the selection part; only then is a Mamba head trained, for that sampling.

- sparse: GRU − pooling = -0.0093 → trigger **does not fire**; no Mamba head trained.
- dense: GRU − pooling = -0.0005 → trigger **does not fire**; no Mamba head trained.

## 3. Order diagnostics (branch alone, whole subset) — diagnostics, never gates

| configuration | ordered AUROC | reversed | locally permuted | shuffled | mean |Δp| reversed | mean |Δp| shuffled |
|---|---:|---:|---:|---:|---:|---:|
| sparse × pooling (reference) | 0.8934 | 0.8908 | 0.8927 | 0.8873 | 0.0638 | 0.0826 |
| sparse × GRU head | 0.8940 | 0.8832 | 0.8869 | 0.8838 | 0.0649 | 0.0806 |
| dense × pooling | 0.9045 | 0.9053 | 0.8796 | 0.8606 | 0.0756 | 0.3545 |
| dense × GRU head | 0.9049 | 0.9089 | 0.8427 | 0.6890 | 0.0733 | 0.3290 |
| dense-subsampled × pooling (diagnostic) | 0.8587 | 0.8611 | 0.8587 | 0.7972 | 0.0750 | 0.1671 |

Frames for every configuration are cut by the same dlib / 81-landmark aligned-crop function; the dense window is 32 consecutive source frames at the video's midpoint. One seed per configuration.
