# DISCERN-v2 — Fusion Audit and Conditional Follow-ups (current-paper scope)

For the coding agent in the live tree (DFB_NeSyNS, DISCERN_Ext). Scope is the current T-BIOM paper. The only experiment to launch now is Experiment 1, the cached fusion audit, which trains nothing and runs on saved prediction tensors. Experiments 2 through 5 are specified but conditional, and none of them starts until the audit reports and a human decides. Do not queue training runs ahead of the audit.

The central question the whole sequence answers: does the learned participation fusion exploit the third branch better than ordinary stacking, especially under processing and on reenactment. Everything else waits on that.

What is explicitly out of scope for this paper: FreqBlender integration, a learned hard-augmentation policy, full T1 to T5 test-time controller, human-feedback RL, and expansion past three branches. These are legitimate follow-up work, not this manuscript.

Framing corrections that must hold in the writeup, stated once here and enforced everywhere below:

- The audit does not establish "whether you have a paper." It answers one narrower question, whether learned participation beats simpler fusion on these branches. A tie is unresolved, not proof that fusion research is exhausted, and a small numerical win is not a fusion contribution.
- Replay measures how fusion behaves with these particular learned branches. Because the prior head was trained with the gate present, replay alone cannot establish which fusion-training method is best. State this.
- There is one warm+gate seed. Reinitializing an MLP on its cached predictions does not create independent detector seeds. Do not present MLP reinitializations as seed replication.
- Error overlap is a diagnostic, not a rejection rule. A branch earns or loses its place by whether it improves the selected objective on development data, not by its correlation.
- Numbers from files, not memory. No fabricated cells.

---

## Experiment 1 — Cached fusion audit (launch now, zero GPU training)

On the existing warm+gate checkpoint's saved per-branch opinions, compare five fusion arms on identical inputs. Nothing about the detector is retrained.

Arms:
1. The existing learned participation gate.
2. Full prior participation (g = 1).
3. Development-fitted constant participation (a single g fit on development data).
4. Logistic and small-MLP stacking.
5. Frozen two-view DiCoME base, as the branch-removal control.

Input parity. Give stacking the same deployment-available inputs the gate sees, including the uncertainty quantities (per-branch vacuity, evidence, disagreement), not only class probabilities. A stacking baseline handed only probabilities is a strawman.

Fitting and selection protocol. The constant, logistic, and MLP arms are fitted, so they need clean fit-and-select separation. Split the development predictions into a fusion-fitting fold and a fusion-selection fold, grouped by source video so no source appears in both. Fit each alternative on the fitting fold, compare on the selection fold. Do not select the winning arm by its ten-test-set mean, and do not touch the test sets to choose the arm.

Report, per arm: OOD mean, and separately the reenactment subset (CDFv3 reenactment, KoDF) and the processed conditions (CDFv2 and DFDCP under JPEG and resize), plus real-side FPR at frozen tau. The reenactment and processed numbers matter more than the mean, since those are where the third branch is supposed to earn its place.

Decision rule. If learned participation beats MLP stacking by a meaningful margin across the grouped selection fold, the fusion is a candidate contribution and the paper can build on it. If stacking matches or beats it, the honest framing is that gains come from three-view representation complementarity, not the gating machinery, and the paper is written that way. If the result is close, report it as unresolved, not a win and not a proof that fusion is exhausted.

Write `tbiom/EXP1_FUSION_AUDIT.md` with the five-arm table, the grouped-fold protocol, the decision, and the two limitations (replay cannot establish the best fusion-training method, and one warm+gate seed is one seed).

Pass condition. The audit completes with the grouped fit-and-select protocol and input parity, and returns one of the three verdicts above. Stop and report. Do not start Experiment 2 in the same unattended run.

---

## Experiment 2 — One processing-augmentation pilot (conditional, after the audit)

Only after Experiment 1 selects a fusion configuration. Train that configuration with the existing B recipe. Do not introduce a new hard-example generator here.

**Correction (2026-09-16).** The B recipe is NOT "JPEG q40, spatial resize 0.60" — those are the held-out *evaluation* conditions. B's training mix (`DISCERN_Ext/src/dataset/training_arms.py`, `SharedProcessing`): one setting drawn per sample, independent of the label — none 40%, JPEG q70 / q50 / q30 10% each, down-then-up bilinear resize 0.75 / 0.50 / 0.35 10% each; applied to the uint8 frame after de-normalisation, then re-normalised; per-worker RNG from commit `29e7f18`. Decision (UMAR, 2026-09-16): Experiment 1 was unresolved; run ONE matched pilot, warm+gate + B augmentation (`warm_dicome_gate_B_seed42`), identical to `warm_dicome_gate_seed42` except `training_arm: B`. It is an augmentation ablation, not a declaration that the learned gate won Experiment 1.

Setup discipline. Start from the same base checkpoint and match the unaugmented comparison's initialization, budget, and trainable modules, so the only change is processing exposure. Evaluate development performance during training, select the checkpoint by a rule frozen before launch, and do not run every checkpoint on the ten test sets.

Mandatory caveat in the report. With the DiCoME backbone frozen, this trains only the prior head and fusion components, not the full detector. It is a narrower experiment than B's original full-detector adaptation and must be labeled as such, or the gain is misattributed.

Pass condition. The pilot completes under the frozen-selection rule with the frozen-backbone caveat recorded.

---

## Experiment 3 — X-CLIP temporal probe (conditional, after Experiment 2)

Only after the augmentation pilot. Evaluate before any full training, which for a representation means train a small probe, not attach an unfitted evidential head.

Steps:
1. Extract frozen X-CLIP features from tracked face clips.
2. Train a small supervised classifier on them.
3. Compare against spatial pooling over the same frames, to separate temporal information from spatial signal.
4. Fit lightweight fusion with the existing views on the grouped development fold.
5. Measure incremental development performance on reenactment (CDFv3, KoDF), processed conditions, and clean swaps.

Report error overlap with the existing views as a diagnostic, not a veto. A frozen probe that fails is evidence against that configuration, not proof that bounded adaptation cannot help, so a failing frozen probe leads to Experiment 4 only if the failure analysis points at temporal dynamics.

Pass condition. The probe reports incremental value on the reenactment subsets with the spatial-pooling control and the error-overlap diagnostic.

---

## Experiment 4 — Bounded X-CLIP adaptation (conditional, only if Experiment 3 justifies it)

Only if the frozen probe was promising-but-insufficient and the failure analysis points at under-adaptation. Adapt temporal modules or selected adapters plus the head, with an explicit compute budget fixed before launch, preserving the complete β-VAE path unchanged. AV-HuBERT stays in reserve and is built only if the error analysis shows visible mouth dynamics are the dominant failure, and it is scoped as mouth-appearance features, not lip-sync, since without audio it cannot measure audio-visual synchronization.

---

## Experiment 5 — Confirmation (conditional, on the final selected recipe)

Repeat the selected final recipe across seeds to establish reproducibility. Distinguish independently trained detector seeds from fusion-head reinitializations, which are not the same thing.

---

## Two safeguards that apply throughout

- **Trigger and uncertainty validation.** Before claiming vacuity is useful for error referral or triggering inspection, plot the empirical vacuity distribution for misclassified versus correct samples and confirm u actually spikes on errors, and compare u against decision margin for error ranking. Do not assume EDL uncertainty is useful, your own diagnostics questioned it. If margin ranks errors better than vacuity, say so and use margin.
- **Sealed benchmark.** Keep at least one major test set (DFEval24 or FFIW) completely uninspected during selection and threshold tuning, reserved as a zero-leakage confirmation benchmark for the final manuscript. Previously inspected benchmarks remain reportable but cannot be made untouched again, so name the sealed one now and do not read it until final confirmation.

The immediate instruction is Experiment 1 only. Run the cached fusion audit, report the five-arm table and the verdict, and stop. Experiments 2 through 5 are conditional follow-ups that a human authorizes after the audit, so nothing trains before the audit informs the choice.