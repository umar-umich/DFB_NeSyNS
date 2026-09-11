# DISCERN-v2 — Nuisance-Aware Evidence Diagnostic (pre-framework)

For the coding agent in the live tree (DFB_NeSyNS, DiCoME, DISCERN_Ext). This is a diagnostic phase, not a framework build. Its job is to decide, cheaply, whether a nuisance-aware abductive evidence layer deserves the larger investment, before any of it is built at scale. Everything runs on the frozen detector's measurements. No new backbone, no unfreezing.

The claim under test is narrow and must stay narrow: a forensic system that separates manipulation evidence from nuisance-induced anomalies and from unavailable measurements, and shows that distinction improves decisions under shift. These experiments establish whether that claim is real. If they fail, the honest outcome is a smaller reliability paper, not a forced framework.

Non-negotiable controls, stated once, enforced everywhere below.

- Same-input neural control. Every symbolic or probabilistic mechanism is compared against a plain neural head that receives the identical inputs, quality indicators, availability masks, and comparable tuning effort. A mechanism is a contribution only if it beats the same-input neural head. If it merely matches on AUC, it may still contribute through robustness or calibration, but that must be shown, not asserted.
- Evaluate more than AUC. Report FPR at matched TPR across conditions, and threshold transfer to held-out genuine sources, with the threshold fixed before touching untouched target datasets. Lowering false positives by missing more fakes is not an improvement, so report forced-binary performance separately from any deferral, with coverage and missed-fake counts.
- Freeze protects the experiment, it does not guarantee the experiment has enough information. If the frozen features contain nothing that separates processing from manipulation, no reasoning layer recovers it, and that itself is a finding to report rather than engineer around.
- Numbers from files, held-out test sources never in model selection.

---

## Stage 0 — Confirm the frozen detector and inventory measurements

Confirm the canonical frozen configuration (arm B, arm C, or arm E per the freeze decision) and its exported per-branch quantities. Inventory, per branch and per sample, what is actually available: branch probability, branch evidence, branch vacuity, cross-branch agreement, fused probability, and any reconstruction or manifold diagnostics. This is the raw material every later stage reasons over.

Pass condition. The frozen configuration is named and its per-sample measurement exports are located.

---

## Stage 1 — Build the measurement record with three separate axes

The core design correction: observability, reliability, and evidence direction are distinct and must be represented separately, not collapsed into one three-state label. For each cue, retain four things.

- **Value.** The measurement itself.
- **Availability.** Was the cue measurable at all (region visible, tracking succeeded). Never infer availability from a low fake score. It must come from the measurement process, not the decision.
- **Reliability.** A quality estimate for the measurement under this sample's operating conditions (compression, camera, resolution), separate from whether the cue points to fake. Available does not mean reliable.
- **Source.** Which region, segment, and detector produced it, so duplicate and correlated evidence can be identified later.

Evidence direction (what the value supports) stays a separate downstream quantity, not folded into availability or reliability. A cue can be available, reliable, and point to real; or available, unreliable, and point to fake; these are different states and the record must distinguish them.

Pass condition. A per-sample measurement record exists carrying value, availability, reliability, and source as separate fields, with availability derived from the measurement process rather than the label or the score.

---

## Stage 2 — Learn how ordinary processing changes measurements

Create controlled processing variants of the same source videos, real and fake, holding authenticity fixed: compression levels, resizing, frame-rate changes. Apply the same processing distribution to both classes, so the system cannot learn "compressed means fake." Measure, per cue, which stay useful and which become unreliable under each processing severity.

Two limits to record explicitly, because they bound the claim. Re-encoding already-compressed video does not remove its earlier compression history, so treat processing as additive, not a clean intervention. And processing augmentation cannot supply missing genuine identities, cameras, or scene types, so if the real-class support is too narrow this stage cannot fix it, and that limit must be stated rather than hidden.

Pass condition. Per-cue reliability-versus-processing curves exist, fit on both classes under matched processing.

---

## Stage 3 — Three decision mechanisms on identical inputs (the central comparison)

This is the experiment that decides whether the direction is worth the larger build. All three receive exactly the same Stage 1 measurement record and Stage 2 reliability estimates.

- **M1, neural fusion head.** A plain neural head on the measurements. Tests whether the added measurements themselves are sufficient, with no probabilistic model and no symbolic structure.
- **M2, probabilistic benign-vs-manipulation comparison.** Two competing hypotheses, H0 authentic content with processing, H1 manipulated content with processing, scored as a likelihood ratio that aggregates over plausible processing histories with a shared processing prior. The shared prior is what stops "heavily compressed" from being treated as intrinsically fake. Tests whether explicitly accounting for processing helps beyond M1.
- **M3, M2 plus explicit reasoning constraints.** The same comparison with symbolic constraints on which measurements are applicable and which share a source. Tests whether the symbolic structure adds measurable benefit beyond M2.

The three-way split isolates the value of the inputs (M1), the probabilistic model (M2 − M1), and the symbolic component (M3 − M2). Report all three on AUC, FPR at matched TPR, and calibration. A mechanism earns its complexity only by beating the simpler one beside it beyond the noise band.

Critical safety valve on M2 and M3, from the interpretation-uncertainty analysis. Allow both explanations to be inadequate. If H0 explains a sample poorly and H1 explains it slightly-less-poorly, the ratio still favors manipulation on a sample neither model understands, which is exactly the unseen-real failure mode. So add a separate adequacy check: does either hypothesis adequately explain these measurements under the evaluated conditions. If neither does, flag unsupported-interpretation, which is distinct from assigning the sample to real. Route that state to inconclusive, and in the forced-binary report show what it would have decided so deferral cannot hide error.

Pass condition. The three mechanisms are compared on identical inputs, with the adequacy check active, and the differences M2 − M1 and M3 − M2 are reported with the same-input discipline intact.

---

## Stage 4 — Three property tests

Run these on the frozen measurements. The first two are design-correctness requirements, not novelty. The third is the central empirical claim.

- **Missing-measurement handling.** Deliberately mask a cue independently of the label. The system must not fabricate a measurement or a rule violation, and must not convert unavailable into contradicted. Note that real-world missingness can itself be informative, so keep this deliberate controlled masking separate from naturally occurring missingness.
- **Duplicate-evidence handling.** Supply the identical observation twice. Confidence must not rise merely from double-counting. This is the correlated-evidence check, and it connects to the Stage 1 source field and to your earlier fusion-redundancy finding.
- **Processing discrimination (the central test).** Under held-out processing conditions, does the system reduce genuine-video false positives while retaining sensitivity to manipulation. This is the property that would justify the whole direction, and it is measured, not assumed.

Pass condition. Missing and duplicate handling behave correctly by construction, and processing discrimination is measured on held-out conditions with FPR-at-matched-TPR reported.

---

## Stage 5 — Validate the uncertainty itself (cheap, and it guards the whole story)

Two cheap checks that decide whether the reliability narrative rests on something real.

- **Interpretation uncertainty via a small ensemble.** Train a small ensemble of interpretation heads on the frozen features. A single fused probability hides model disagreement: every head near 50% and half-at-1%-half-at-99% both average 50% but mean different things. Report whether ensemble disagreement predicts errors better than the single fused probability or the vacuity, on the held-out sets. This needs no new backbone, and it is the concrete comparator for the next check.
- **Does vacuity mean what we claim.** Validate u = K/S against the ensemble's epistemic uncertainty rather than assuming its meaning from the formula, per the ICML 2024 result that EDL uncertainty may not faithfully represent epistemic uncertainty. If vacuity does not track ensemble disagreement, the reliability story cannot lean on vacuity-as-epistemic-uncertainty, and that must be stated.

Pass condition. Ensemble disagreement is compared against fused probability and vacuity on error prediction, and vacuity's interpretation is validated or qualified against it.

---

## Reading the outcome

- If M2 or M3 beats M1 beyond noise on FPR-at-matched-TPR under held-out processing, and the property tests pass, and vacuity survives Stage 5, the direction is worth the full framework build, and the paper claim is the narrow one at the top. Its comparison set is not only DiCoME but the 2025 trusted-multi-view and evidence-filtering work, so position against those explicitly.
- If M2 and M3 only match M1, the symbolic and probabilistic structure is not earning its place on discrimination, and the honest paper is smaller, reliability and calibration on the frozen detector, with the negative reported.
- If processing discrimination fails because the frozen features do not separate processing from manipulation, that is the finding, and it says improve the measurements before any reasoning layer, not build the reasoning layer anyway.

Do not build the full abductive framework, and do not start active evidence acquisition, until Stage 3 and Stage 4's processing test return. Active acquisition is a separate later paper on this same substrate, preserved, not cut, but not in scope here.