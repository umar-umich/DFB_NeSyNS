# DISCERN-v2 — Freeze, then Symbolic Explanation Overlay

For the coding agent in the live tree (DFB_NeSyNS, DiCoME, DISCERN_Ext). Two phases, in order. Phase F freezes the detector on the arm E and seed-1337 results that finished but were not consolidated into a decision. Phase S builds a symbolic explanation overlay on the frozen model and validates it. Phase S does not start until Phase F names a frozen configuration, because the overlay reads out that configuration's evidence and cannot be built on an undecided architecture.

Hard rules for both phases.

- Freeze before building on top. Nothing downstream, overlay included, is finalized until Phase F declares the canonical detector.
- Numbers from result files, never memory. Unrun cells are `TODO(run)`.
- The overlay only explains, it does not decide. The Real/Fake output stays the fused evidential decision. The symbolic layer narrates it. It is a symbolic explanation overlay, not a neuro-symbolic detector, and the writeup names it that way unless the symbolic layer later earns its way into inference.
- Do not touch training or the AUC path in Phase S. The overlay is downstream and non-invasive.

---

## Phase F — Freeze the detector

### F1 — Report the two finished runs

From the export files on disk, report per dataset and mean OOD video AUROC, with real-side FPR at frozen τ, against P0-DS.

- Arm E (semantic view removed from fusion, trained aware of the exclusion) against the post-hoc replay result (0.8784), against arm C, against P0-DS. The question is whether training-aware exclusion beat the replay, matched it, or fell short.
- Seed-1337 (arm B, second seed) against seed-42, both against P0-DS, with per-branch vacuity for both seeds. The question is whether arm B's headline holds under a second initialization and whether all three branches stay alive.

### F2 — Freeze decision

Recommend one canonical configuration, arm B three-view, arm C three-view, or arm E two-view, and state its mean OOD AUROC and real-side FPR against P0-DS. Apply these rules.

- If arm E only matched the replay, prefer dropping the semantic view at inference on a normally trained model, since that is simpler and applies to any checkpoint. If arm E genuinely beat the replay beyond noise, freeze the two-view trained config.
- State what the seed evidence supports honestly. Two seeds is a sign check, not a spread. "Arm B is +X vs P0-DS across two seeds" is allowed. A full arm ordering (B > C > D) is not, unless every arm has matched seeds. Do not overstate.
- Name DFEval24 explicitly as the holdout if it remains the set nothing beats P0-DS on. That is a stated limitation, not something to average away.

### F3 — Consolidate

Update `tbiom/BRANCH3_REPORT.md` with the arm E and seed-1337 numbers and the freeze decision, so the record is one document rather than scattered runs. Record the frozen configuration's exact checkpoint path and its exported opinions, since Phase S reads from them.

Pass condition. One configuration is frozen, its checkpoint and exported per-branch opinions are recorded, and the honest AUC, real-side FPR, and DFEval24 limitation are stated. Phase S starts only now.

---

## Phase S — Symbolic explanation overlay (post-hoc, on the frozen model)

Build the overlay on the frozen configuration's already-exported quantities. No retraining, no GPU beyond reading exports.

### S1 — Predicates from measured evidence, with principled thresholds

Derive predicates from the quantities the frozen model already computes per branch: branch probability and evidence, branch vacuity, cross-branch agreement/disagreement, fused probability. Candidate predicates: HighArtifactEvidence, StrongFacePriorViolation (FS-VFM branch), BranchesAgree, BranchesConflict, EvidenceInsufficient.

Thresholds are the point of failure, so set them on a principled basis, not tuned to make traces read well. Use the operating points the model already uses, the frozen EER threshold and dev-set quantiles of each quantity computed on FF++/VALmix development data only, never on OOD test data. Record every threshold and its basis. A threshold chosen to make the explanation look convincing reintroduces the arbitrariness the overlay is supposed to avoid.

### S2 — Rules and the decision trace

Small, readable rule set over the predicates, for example: HighArtifactEvidence ∧ StrongFacePriorViolation ⇒ StrongFakeSupport; LowArtifactEvidence ∧ LowFacePriorViolation ⇒ RealSupport; BranchesConflict ∧ EvidenceInsufficient ⇒ Inconclusive. Emit a per-sample decision trace listing each branch's evidence direction and strength, the cross-branch agreement, and the overall assessment.

### S3 — Validation, two tests, and intervention is the one that decides

Agreement alone is not sufficient. Rules written to match the fused output will always match it, so a high agreement score can be an artifact of construction. Run both.

- **Agreement (necessary).** Fraction of samples where the symbolic assessment matches the model's Real/Fake decision. Report it, but do not treat it as proof of faithfulness on its own.
- **Intervention (decisive).** For samples the overlay explains by naming a branch as the driver, suppress that branch's evidence (set its opinion vacuous) and re-fuse. A faithful overlay's decision must change or weaken correspondingly. If suppressing the branch the overlay named as the reason does not change the decision, the overlay named a reason that was not the reason, and it is decorating, not explaining. This is the same test that showed the earlier FPAD localization map decorated rather than explained. Report the fraction of explanations that survive intervention.

Also report **rule coverage**, the fraction of samples explained by a specific rule versus falling to the Inconclusive fallback, so the writeup is honest about how often the rules actually fire.

### S4 — Verdict and naming

- If agreement is high and a strong majority of explanations survive intervention, the overlay is faithful and is a legitimate secondary contribution. Describe it as a symbolic evidence overlay translating heterogeneous branch evidence into human-readable forensic reasoning traces.
- If explanations do not survive intervention, it is a UI, not a contribution. Do not claim it as novelty. Report the negative and stop.
- Do not call the detector neuro-symbolic. The symbolic layer is post-hoc and does not participate in inference, so it is an explanation overlay. Only a symbolic layer that enters the Real/Fake decision and is shown to hold or improve performance earns the neuro-symbolic label, and that is a separate, later, measured claim, not this phase.

Write `tbiom/SYMBOLIC_OVERLAY.md` with the predicates and their thresholds and bases, the rules, the agreement and intervention numbers, rule coverage, and the verdict.

Pass condition. The overlay is validated on both agreement and intervention with principled thresholds, and named according to what the validation supports.

---

## Sequencing and what not to do

Phase F before Phase S, always. Within Phase S, thresholds on development data only, both validations run, intervention decides. Do not start Stage 8 applicability or any new detector arm, the detector question is settled at freeze. Do not blur the overlay into a neuro-symbolic claim in the writing. The strongest presentation of this work is the honest reliability and fusion-redundancy findings made clear, with a faithful explanation overlay on top, not a symbolic layer added to sound sophisticated. Keep it faithful and it strengthens the story. Skip the intervention test or overstate the naming and it becomes the one addition that invites the scrutiny the rest of the work does not need.