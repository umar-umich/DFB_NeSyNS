# DISCERN-v2 — Final Finish-Line Brief (corrected reference, FreqBlender, acquisition pilot)

For the coding agent in the live tree (DFB_NeSyNS, DISCERN_Ext). This supersedes the previous finish-line brief and fixes the reference-configuration inconsistency in it.

## Reference architecture, frozen for every experiment below

Three branches, no Face-prior:

- **Temporal**: V-JEPA-2 with LoRA, B-augmented checkpoint.
- **Appearance**: CLIP ViT-L LoRA view.
- **Off-manifold**: β-VAE residual view.

Primary proposed fusion is DS-R. Logistic stacking is the mandatory matched comparator, run beside DS-R everywhere. Do not describe DS-R as demonstrated superior, and do not call logistic the deployed reference, without the corresponding results for this exact triple. Face-prior is removed from the reference, the confirmation runs, and the main ablation matrix, and is retained only in explicitly labelled historical comparisons.

## Stage 0 — Resolve the reference manifest first (do this before any experiment)

Read the live configuration and reports and resolve, do not guess: the three checkpoint paths and their hashes, each branch's preprocessing, the clip and frame aggregation procedure, the exact DS-R definition and its fitted parameters, and the development-fitting and development-selection splits. Write one `tbiom/reports/REFERENCE_MANIFEST.md` recording all of it, and use that manifest identically in every experiment. If any checkpoint identity cannot be resolved from the live tree, stop and report it rather than proceeding on a guess.

Pass condition. The manifest names three resolved checkpoints, the DS-R definition and its fitted parameters, and the fold splits. Every later experiment cites this manifest.

## Order of work

Experiments 1 to 3 are cached-output CPU work, start now, no GPU, no dependence on running exports. Experiment 4 is the confirmation seed, queue behind current exports. Experiment 5 is the FreqBlender study, after 1 to 4. Experiment 6 is the optional joint-training ablation. Experiment 7 is the optional acquisition-policy pilot, after the reference is fixed. Audit and seal FaceShifter before any test-set reading.

B augmentation mixture, referenced by every training arm. 40 percent unchanged, JPEG q70/q50/q30 at 10 percent each, downscale-then-upscale by 0.75/0.50/0.35 at 10 percent each, one draw per clip independent of class. The q40 and resize 0.60 severities are test-only.

---

## Experiment 1 — Uncertainty safeguard (cached, CPU, start now)

The deployed fusion is DS-R with logistic as comparator, so report both. Do not attach a fusion's uncertainty to a different fusion's prediction without validating the relationship.

On cached opinions of the reference triple, compare as error-ranking signals: the fused decision margin, per-branch vacuities, branch disagreement, and the fused DS-R vacuity where DS-R is the fusion. Report error-detection AUROC and a risk-coverage curve for each, on clean and processed development conditions separately, and count confident false negatives. The objective is useful error identification, not making vacuity beat margin. If margin ranks errors best, report that plainly. Write `tbiom/reports/EXP_UNCERTAINTY.md`.

---

## Experiment 2 — Three-view ablation (cached, CPU, start now)

Under matched fusion-fitting procedures (DS-R primary, logistic comparator), compare the full triple against its three pairs: V-JEPA + Appearance, V-JEPA + Off-manifold, and Appearance + Off-manifold. Report which pair each triple beats and by how much, so it is clear whether all three branches earn their place. These are ablations of the three-view claim, not proposals to drop a view. Face-prior does not appear here. Write `tbiom/reports/EXP_THREEVIEW_ABLATION.md`.

---

## Experiment 3 — DFDCP diagnosis (cached, CPU, start now)

The reference's known weakness is processed DFDCP. B augmentation improved general JPEG on development but did not fix DFDCP, suggesting a fusion-level cause. Diagnose which of two applies.

- If branches still discriminate under processing but fusion deteriorates, refit the fusion on clean-plus-processed development examples and evaluate on separate source-grouped selection examples, no encoder retraining. Check whether DFDCP recovers, for both DS-R and logistic.
- If V-JEPA's own discrimination collapses under compression, report whether the B-augmented branch recovers DFDCP relative to the unaugmented one.

Both can coexist, report each separately. Note DFDCP false positives run high (roughly 0.24 to 0.46) at frozen tau for every model, which ties to Experiment 1. Write `tbiom/reports/EXP_DFDCP_DIAGNOSIS.md`.

Note on FreqBlender. If the fusion refit recovers DFDCP, the processing story is complete and the FreqBlender study becomes an additional study rather than a needed fix. Record that, and still run Experiment 5 as requested.

---

## Experiment 4 — Confirmation seed (GPU, queue after current exports)

Train the reference recipe with seed 1337, changing only the seed. Same V-JEPA checkpoint init, clips, sampling, preprocessing, LoRA config, losses, and budget, including the same B augmentation. Same fixed Appearance and Off-manifold checkpoints. Same development-only selection, refitting DS-R and logistic on the same source-grouped folds. Label this as temporal-branch training variability conditional on the fixed branches, not full three-seed detector replication. The only variable is the seed. Do not interrupt running exports. Write `tbiom/reports/EXP_CONFIRM_SEED.md`.

---

## Experiment 5 — FreqBlender spatial study (GPU, after 1 to 4)

A bounded matched study of whether frequency-domain synthetic fakes help the spatial branches, temporal branch frozen. FreqBlender is an experimental augmentation, not an assumed novelty and not part of the reference unless it earns inclusion on development.

Setup discipline.
- Verify the upstream FreqBlender implementation first: generator, required weights, output labels, preprocessing. Distinguish any local reproduction from the official pretrained configuration and record which is used.
- Two matched arms. Spatial-B: Appearance + Off-manifold trained with B processing. Spatial-B-Freq: identical recipe plus FreqBlender replacing 10 percent of fake training slots. The 10 percent is a fixed pilot setting, no fraction sweep.
- Match total samples, class balance, initialization, optimizer, losses, and budget. Preserve ordinary fakes, apply the same class-independent B processing after sample construction, preserve the existing β-VAE configuration and losses. Appearance and Off-manifold share the CLIP feature path, so treat them as one coupled spatial experiment. Leave the reference checkpoints untouched.
- Reuse an existing Spatial-B run only if it genuinely matches these conditions, otherwise train the matched Spatial-B control, since comparing against unaugmented spatial checkpoints would confound processing exposure with FreqBlender.
- Keep the temporal branch frozen at the reference B-augmented V-JEPA checkpoint for both arms. Do not feed frame-level pseudo-fakes into temporal training. Never combine a synthetic spatial input's opinion with a temporal opinion from its original genuine clip in a fused loss, they are different samples, so confine FreqBlender examples to the spatial losses.

Evaluation. After spatial training, export all three views on the same evaluation clips, both real and fake videos. Refit DS-R separately per arm on identical development folds, keep logistic as the matched comparator. Decide inclusion on predefined development criteria before reading any test set.

Acceptance criterion, frozen before launch, on development not test. Report clean swaps, reenactment, processed performance, and FPR/TPR together. Swaps improving while reenactment drops does not qualify. The disqualifier, given the arm-C precedent where synthetic pseudo-fakes collapsed reenactment to 0.527: any fused-level development reenactment regression beyond the bootstrap noise band disqualifies the FreqBlender arm regardless of swap gains. Verify at the fused three-view level, because freezing the temporal branch preserves its parameters but does not guarantee the fused detector's reenactment performance, a swap-tuned spatial branch can still shift the fused reenactment decision through DS-R. Report CDFv3 test reenactment only afterward, never as the selection criterion. Do not claim FreqBlender improves compression robustness before it is demonstrated on the frozen development criteria. Write `tbiom/reports/EXP_FREQBLENDER.md` with arm C recorded as the precedent.

---

## Experiment 6 — Joint-training ablation (GPU, optional, after 5)

Initialize from the selected staged checkpoints, train all branch adapters and heads together with branch-level supervision plus the fusion loss, and compare against the staged system. Keep FreqBlender examples confined to spatial losses. Do not assume joint training reduces interpretability, measure its effect on per-branch contributions and on performance, and report both. The staged system remains the reference, joint training is the ablation. Write `tbiom/reports/EXP_JOINT_ABLATION.md`.

---

## Experiment 7 — Acquisition-policy pilot (optional, after the reference is fixed)

One bounded experiment that makes a decision the current detector cannot: which additional clip window to inspect under a budget. Frozen reference detector, no backbone training. This is not the paper's headline, it is an optional pilot with a strict go/no-go before any RL.

State: observed branch opinions, disagreement, quality measurements, inspected timestamps, remaining budget. Actions: inspect one of a small predefined set of unvisited clip windows, or stop. Budget: at most two additional inspections. Output: one updated video score using the fixed reference aggregation and fusion. Training labels available only during controller training, none at inference, and no access to uninspected clips' detector outputs at inference.

Step 1, cache and controls before any PPO. Precompute candidate-clip outputs on policy-training and development videos. Then compare, at equal acquisition budget: fixed uniform additional clips, random additional clips, quality-based selection, a supervised one-step action-value selector, and a bounded two-step assessment so that a lack of one-step benefit does not automatically rule out sequential value. All methods use the same action pool and comparable budgets. Count decoding, quality-estimation, and controller overhead, not only detector calls. During simulation, hide unselected cached outputs from the policy, or it effectively sees the answers before choosing. Aggregate acquired observations within each branch before fusion, do not fuse overlapping clips as independent sources.

Step 2, PPO only if Step 1 shows useful variation in which action helps which video. A small PPO policy is sufficient, a newer optimiser's name does not establish contribution. Reward `r_t = loss(p_t, y) - loss(p_{t+1}, y) - lambda * cost(a_t)`, which rewards improved prediction not increased confidence. Compare PPO against an equal-budget multi-step greedy policy. Note that log-loss improvement does not guarantee AUROC improvement, so select and assess policies on a predefined development objective and report AUROC alongside log loss, FPR/TPR, and cost.

Reporting and scope. Report full-population detection performance and computational cost separately from any abstention results, an improvement measured only after rejecting hard videos is a selective-prediction result, not an AUROC improvement. Novelty is narrow and needs comparison against existing frame-selection methods (AdaFrame, AdaFocus) before any claim. Do not add RL augmentation, fusion-weight RL, or human-feedback training to this experiment. Write `tbiom/reports/EXP_ACQUISITION_PILOT.md`.

---

## Standing safeguards and labeling corrections

- **FaceShifter.** Audit its exact source split before assigning any confirmation role. The official FF++ FaceShifter release manipulates the original FF++ videos, so it can test an unseen manipulation method while sharing source content with training. Describe it as unseen-manipulation, not zero-leakage, and check source overlap before sealing. Seal it before any test-set reading in Experiments 4 to 7.
- **Predictive temporal residual.** Deferred and unrun, not a reported negative. Do not describe it as a negative result.
- **Residual operators on frozen pooled temporal features.** Those were run and are reported negatives, keep them as such.
- Numbers from result files, not memory. Unrun cells `TODO(run)`. Test sets read only after development selection.
- Do not, in this queue: build the predictive temporal residual, run further residual-operator variants, adapt additional temporal encoders beyond V-JEPA 2, add human-in-the-loop training, sweep the FreqBlender fraction, or run RL augmentation or RL fusion-weight tuning.

Start Experiments 1, 2, 3 now on cached outputs. Queue Experiment 4 behind current exports. Run Experiment 5 after 4. Experiments 6 and 7 are optional and come last, and Experiment 7 proceeds to PPO only if its Step-1 controls justify it.