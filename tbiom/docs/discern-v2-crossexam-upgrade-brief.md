# CROSS-EXAM upgrade brief (four-day plan)

This brief supersedes `discern-v2-npa-brief.md` and the unfinished Parts 3 to 6 of `discern-v2-independent-views-brief.md`. The joint IBDC brief stays in force.

## What this plan must deliver by the end of day 4

Whatever the individual experiments show, the final report contains seven things.
1. A final view configuration, chosen by the pre-set rules below, with its full decision trail.
2. The DF40 family table and the view-complementarity table in slide format, for that configuration.
3. The ten-set table, the processed tests, and threshold and uncertainty metrics, for that configuration.
4. An ablation table for the processing-suppression study.
5. Joint IBDC results on the final views.
6. A table of negative results.
7. A short statement of whether the architecture shown on the talk slides (nuisance-purified appearance on I-JEPA) matches the selected system, and what would have to change if it does not.

If every new idea fails, items 2 to 6 are still produced for the current system.

## Rules

- Never fabricate a number. Unrun cells are `TODO(run)`.
- **Data roles.**
  - Fitting (subspaces, mappings, normalization statistics) uses FF++ c23 train only.
  - Checkpoint selection uses VALmix-v2.
  - Trust weights use the 5 source-grouped development folds, and the threshold uses FF++ c23 validation EER, exactly as deployed.
  - Test sets are evaluated once per final row and never used to choose anything.
- **Pre-set rules.** Every rule is fixed before its run. A failed rule is reported as a finding.
- **Fusion.** PCR6+-R as deployed. Refit trust weights and threshold per seed and per configuration.
- **"Incremental contribution of view X"** means the fused out-of-fold development AUROC with X minus the same without X, with a source-grouped bootstrap 95% interval (1,000 resamples).
- **Seeds.** One seed during screening, three seeds for any row that reaches a final decision.
- **Budget.** Estimate on 100 videos before any extraction or training run longer than an hour. If one adaptation run would exceed 10 GPU hours, stop and report the estimate.
- **Terminology.** In the diagnosis, the fake-minus-real mean contrast is the "class-separation direction". It is not the decision direction of the nonlinear evidential head.
- Put code in `upgrade/` and reports in `analysis/upgrade/`.

## GPU lanes and schedule

GPUs 1, 2 and 3 are free. If measured throughput suggests a better split, change it and report why.

Deviation recorded 2026-10-01: this run is restricted to GPUs 1 and 3 (UMAR), both shared with
another user, so the three lanes below are serialised across two cards. Part 3d was added on day 1.

| Day | Lane A (GPU 1) | Lane B (GPU 2) | Lane C (GPU 3) |
|---|---|---|---|
| 1 | Part 1 (IBDC on current views), Part 2 (diagnostics) | Part 3a extraction, DINOv3 and PE-Spatial | Part 6a extraction, V-JEPA 2.1 |
| 2 | Part 3b (frozen screen and fused evaluation) | Part 3c adaptation, candidate 1 | Part 6b and 6c (video heads), then Part 3c adaptation, candidate 2 |
| 3 | Part 4 (suppression study) and Part 5 (consistency study) | Part 4a paired extraction on the adapted winner | Part 7 (fillers) |
| 4 | Part 8 (final assembly, three seeds, tests) | Part 8 IBDC on final views | Part 8 tables and report |

---

## Part 1. Finish IBDC on the current views (day 1, lane A)

Complete the remaining runs in `discern-v2-joint-ibdc-brief.md` and its Option B addition, including J1s, the λ sweeps and the J2b verdict. Heads-only runs take minutes each. This guarantees a complete IBDC result even if the views change later.

## Part 2. Cheap diagnostics (day 1, lane A, cached features)

### 2a. Threshold transfer versus ranking

High FPR on DFDCP may reflect a shifted threshold rather than dataset shortcuts. For every test set and the current system, report:
- AUROC
- FPR and TPR at the FF++ threshold
- FPR at the test set's own EER threshold
- ECE

A test set where AUROC is good but FPR at the FF++ threshold is high has a threshold-transfer problem, not a ranking problem. This is analysis only and drives no decision.

### 2b. Feature-space factor profile (adapted from ESP, Xiao et al., 2026)

Describe this as an adaptation of evidence subspace projection, not a reproduction of its neuron-level method.

**Views.** APP (CLIP with LoRA), frozen CLIP, LFS head input, TD pooled feature and frozen I-JEPA. Add DINOv3, PE-Spatial and V-JEPA 2.1 when their features exist.

**Reference populations.** Each group has its own.

| Group | Population | Labels |
|---|---|---|
| Class separation | FF++ c23 val, real and fake | real, fake |
| Processing | FF++ c23 val, paired renders of each video | clean, JPEG q70, q50, q30, resize 0.75, 0.50, 0.35 |
| Source | Development-pool real videos only | dataset |
| Manipulation family | FF++ c23 val fakes only | swap, reenactment |
| Face size | FF++ c23 val | 3 quantile bins |

**Method.** Standardize with FF++ train statistics. Build the contrast vectors r_g and the SVD basis Q, take the class-separation direction d, and report E∆ = ‖Qᵀd‖² and E∆ divided by effective rank.

**Null.** 1,000 permutations.
- For processing, permute setting labels within each video.
- For the other groups, permute labels across videos, grouped by source video.

Report the 95th percentile of the null.

**Pre-registered predictions,** each reported either way.
- P1. APP and LFS profiles are more similar to each other than either is to TD.
- P2. APP shows more processing and source alignment than frozen CLIP.

These profiles describe representations. They do not by themselves show that a nuisance causes the detector's errors, and the report must say so.

Neuron-level ESP is deferred. It may run in Part 7 on the selected system only.

## Part 3. Spatial backbone screen (days 1 and 2)

**Candidates.**
- I-JEPA ViT-H/14 (reference, reuse cached features).
- DINOv3 ViT-L/16.
- PE-Spatial, the smallest released checkpoint that fits the budget.

Frozen CLIP, the deployed APP's base without LoRA, is the floor. Record the exact checkpoint names and licenses.

**3a. Extraction.**
- Use face crops at each model's native resolution, cut from the original frames with the deployed boxes. Report the fraction of faces that are upsampled.
- Use the same frames as APP.
- Features are the standard global embedding of each model, and for PE-Spatial also the mean of patch tokens from one intermediate layer at two-thirds depth. Record the choice.
- Cache FF++ train, val, VALmix-v2, the development pool and the test sets.

**3b. Frozen screen** (one seed).
- Head: linear to 256, GELU, dropout 0.1, linear to 2, softplus evidence, with the deployed EDL settings and VALmix-v2 selection.
- Report standalone FF++ val AUROC and dev v3.1, plus the incremental contribution over LFS and TD.
- Report APP's own incremental contribution over LFS and TD as the bar.

**Shortlist rule.** The top two candidates by incremental contribution over LFS and TD. Frozen scores alone do not eliminate a candidate whose standalone dev v3.1 is within 0.03 of the best.

**3c. Adaptation of the shortlist.** Two recipes, one seed each.
- **L.** LoRA rank 8, alpha 16, dropout 0.1 on query and value.
- **N.** Layer Normalization only, with L2-normalized features, following GenD (credit it).

FF++ c23 train, source-paired as deployed, VALmix-v2 selection.

**Winner rule (pre-set).**
- The winner is the adapted candidate with the largest incremental contribution over LFS and TD.
- It replaces APP only if that contribution beats APP's, with the interval of the difference above zero.
- Otherwise APP stays, and the report says that no independent backbone improved on it.

## Part 3d. Layer readout: last, intermediate, and combinations (added 2026-10-01, UMAR)

**Premise.** The last block of a pretrained encoder is the most task-abstract and tends to be the
most robust to processing, while an intermediate block keeps detail that a forgery detector may need
on clean data. The deployed appearance view reads the last layer only, and that choice has never been
tested. If the two carry different information, a combined readout should beat either alone on clean
data without giving up the last layer's robustness.

**Candidates.** For each shortlisted backbone from Part 3b, and for frozen CLIP as the floor:

| Readout | Definition |
|---|---|
| `last` | the deployed readout: global embedding of the final block (the Part 3b bar) |
| `mid` | mean of patch tokens at two-thirds depth |
| `half` | mean of patch tokens at one-half depth |
| `last+mid` | concatenation, each L2-normalised before concatenation so neither dominates by scale |
| `last+mid+half` | three-way concatenation, same normalisation |

Record the exact block index used for every depth, since "two-thirds" is ambiguous for a 24-block
encoder and the index is what another lab would need.

**Measurements,** one seed for screening, three for any row that reaches a decision.
- Standalone FF++ val AUROC and dev v3.1.
- Incremental contribution over LFS and TD, with the source-grouped bootstrap interval.
- **Robustness, which is the point of the premise.** FF++ val AUROC at JPEG q30 and resize 0.35,
  reported for every readout. The clean and processed columns are reported side by side.

**Keep rule (pre-set).** A multi-layer readout replaces `last` only if BOTH hold.
1. Its incremental contribution over LFS and TD beats `last`'s, with the interval of the difference
   above zero.
2. Its FF++ val AUROC at JPEG q30 and at resize 0.35 is each at most 0.01 below `last`'s.

Condition 2 exists because the premise predicts the trade: if an intermediate readout buys clean
accuracy by giving up robustness, that is a finding and not an upgrade, and the rule must be able to
say so. Report both halves either way, including the case where `mid` wins clean and loses processed.

**Interaction with Part 4.** If a readout is kept, Parts 4 and 5 run on it. If the premise fails and
`last` survives, say so and note that the deployed readout was tested rather than assumed.

## Part 4. Processing suppression on the winner (day 3)

If APP stays, run this part on APP instead, and report it as suppression applied to the CLIP appearance view.

**4a. Paired features.** Render 1,000 FF++ c23 training videos (balanced, source-paired) at all 7 processing settings. Extract adapted-winner features, and compute paired differences Δ_{i,a} = z(T_a(x_i)) − z(x_i).

**4b. Subspaces,** each fitted on FF++ train only.
- **MC.** Mean-contrast basis across processing labels.
- **PD.** Top-k principal directions of the paired differences, with k chosen by 90% explained variance and capped at 32.
- **LEACE.** LEACE for the processing labels (Belrose et al., 2023), implemented and not just cited.
- **RND.** Random subspaces with the same rank as PD, 5 draws.

**4c. Rows,** each a new head on frozen adapted features, three seeds once shortlisted.

| Row | Features |
|---|---|
| A0 | Adapted winner, no suppression |
| A-MC | Full projection on the MC basis |
| A-PD | Full projection on the PD basis |
| A-PD-λ | Partial suppression z′ = z − λ UUᵀ z on the PD basis, with one shared λ chosen on VALmix-v2 from {0.25, 0.5, 0.75} |
| A-LEACE | LEACE erasure |
| A-RND | Rank-matched random control for PD |

**Keep rule (pre-set).** A suppression row is kept over A0 only if all four hold.
1. Its dev v3.1 gain over A-RND has an interval above zero.
2. FF++ val AUROC drops by at most 0.01 against A0.
3. FF++ val AUROC at JPEG q30 and resize 0.35 is at least A0's.
4. In the fused system, development real-video FPR at the threshold is at most 0.01 above A0's.

If more than one row is kept, choose by incremental contribution over LFS and TD. If LEACE is the best kept row, use it and report that the contribution is the application, not the projection.

## Part 5. Label-preserving consistency (day 3, cached paired features)

Principle: moderate processing should keep the authenticity label while evidence strength may change.

Train heads on the 4a paired features.
- **C-AUG.** EDL on clean and processed versions, no consistency term. This is the augmentation control.
- **C-CONS.** C-AUG plus a symmetric KL term between the projected class probabilities of a clean and processed pair, with weight 0.1. The loss never matches evidence magnitude or vacuity.
- **C-CONS + best suppression from Part 4,** if any row was kept.

**Keep rule.** C-CONS is kept only if it beats C-AUG on dev v3.1 with an interval above zero and passes the Part 4 conditions 2 to 4.

Backbone-level consistency training is deferred.

## Part 6. Video view upgrade (days 1 and 2, lane C)

**6a. Extraction.** V-JEPA 2.1 ViT-L/16. Confirm the released checkpoint and its resolution, and record it. Use the same 16-frame clips as TD. Cache:
- global pooled features
- dense token features pooled into the five facial regions from the I-JEPA brief, per frame group

**6b. Global swap.** The TD head architecture and recipe on V-JEPA 2.1 global features, with LoRA as in TD if the budget allows, otherwise frozen. Compare against TD on V-JEPA 2.

**6c. Regional evidence.**
- **R-REG.** The five regional features per temporal token group feed a two-layer transformer of width 128 over region-by-time tokens with no positional shortcut beyond learned region and time embeddings, then mean pooling and the evidential head.
- **Controls.**
  - **R-GLOB.** Global pooling with a parameter-matched MLP head.
  - **R-POOL.** Regional features mean-pooled, with the same head capacity.
- **Diagnostics.** Shuffle frame order at evaluation (5 permutations). Ablate one region at a time.

**Keep rules (pre-set).**
- V-JEPA 2.1 replaces V-JEPA 2 only if its incremental contribution over the spatial views beats TD's, with an interval above zero, and its VALmix-v2 reenactment AUROC is not lower.
- R-REG is kept only if it beats both R-GLOB and R-POOL on dev v3.1, with intervals above zero.
- Gains that survive frame shuffling are described as regional aggregation, not temporal modeling.

InternVideo-Next is deferred.

## Part 7. Fillers (lane C, day 3, only when the lane is idle)

These run in this order and never delay Parts 1 to 6 or 8.
1. **Perturbation stability** (training-free, from Part 6 of the independent views brief) on the screen backbones.
2. **Cross-backbone feature mapping** (Part 5 of the independent views brief) between the Part 3 winner and frozen CLIP.
3. **Neuron-level ESP** on the selected spatial backbone and frozen CLIP.

Report whatever completes. Unfinished fillers are listed as not run.

## Part 8. Final assembly (day 4)

**Configuration.** The spatial appearance view (Part 3 winner, with Part 4 and 5 rows if kept), LFS, and the video view (Part 6 if kept). Three seeds for every changed view.

**Development report.**
- Out-of-fold AUROC, real-video FPR, vacuity error AUROC.
- Trust weights per fold and zeroing counts.
- Every view's incremental contribution.
- The Part 2b profile of the final views.

**Success criterion (pre-set).**
- The final system's development AUROC is within −0.005 of the deployed system's.
- Every view has an incremental contribution with an interval above zero.
- Report the outcome either way. If it fails, keep the deployed system as the reference and report the changes as negatives.

**Tests,** evaluated once per final configuration and seed, and once for the deployed system for comparison.
- Ten-set table, processed tests and the Part 2a threshold metrics.
- DF40 family table in slide format (four families, mean ± standard deviation across generators, macro).
- View-complementarity table by family: single views, all pairs, all three.

**IBDC on the final views.** Rerun J1, J2a and J2b from the joint IBDC brief with fusion fixed, if any view changed.

## Final report `analysis/upgrade/UPGRADE-REPORT.md`

1. The final configuration and the decision trail, with every pre-set rule's verdict and evidence file.
2. Part 2 diagnostics, including the P1 and P2 verdicts.
3. Screen, adaptation, suppression, consistency and video tables, with seed standard deviations.
4. Final development and test tables, and both slide-format tables.
5. IBDC results on the current and the final views.
6. Negative results, one line each, with the number that decided them.
7. Slide consistency: does the selected system match "nuisance-purified appearance on I-JEPA"? If not, state the accurate name and backbone for the architecture slide.
8. Checkpoints, licenses, GPU hours per lane and cache paths.