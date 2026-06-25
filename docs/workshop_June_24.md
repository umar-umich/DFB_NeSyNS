# Workshop Paper — Faithful Deepfake Explanations Verified by Causal Intervention

**Target** ECCV 2026 Workshop (AI for Multimedia Forensics and Disinformation Detection, ~10 days, or Foundation and Generative Models in Biometrics, ~17 days).
**One-line claim** deepfake explanations can be verified by intervention. A cited region is real evidence only if neutralizing it moves the verdict, and we validate this verifier against ground-truth manipulation masks.
**What it is NOT** not a better detector. Detection accuracy is flat by design. The contribution is verified explanation on a bad baseline (current explainers hallucinate).

---

## 1. The gap it fills

Current explainer detectors, including MLLM judges, cite reasons that are often decorative. TriDF shows hallucination disrupts the verdict. Prior work measures faithfulness after the fact, rewards it during training, or corroborates by retrieval. None makes faithfulness a structural property enforced by independent measurement with a ground-truth check. That is the opening.

## 2. Method, kept deliberately small

1. A frozen VLM proposes localized predicates over face regions. Each predicate is `type(region_bbox, confidence)` from a small fixed vocabulary, not free text.
2. Each predicate goes through the intervention test. Neutralize the cited region, re-run the frozen detector, measure the change in p(fake).
3. A predicate is admitted only if its removal moves the verdict beyond a calibrated margin. Otherwise dropped. If nothing survives, abstain rather than explain.

No multi-stream, no causal graph, no symbolic engine. That is the main-track paper.

## 3. Components and exact choices

| Component | Choice | Note |
|---|---|---|
| Detector | strong pretrained FROZEN detector with public weights (GenD, ForAda, or Effort) | frozen for independence, which is the whole asset. Not your contribution, just the instrument. |
| Predicate proposer | frozen Qwen2.5-VL (7B, scale up if memory allows), InternVL as cross-check | proposer not authority. Expect noisy proposals, the gate filters them. |
| Predicate vocabulary | ~6-10 typed, localized predicates | blending seam, over-smoothed skin, asymmetric eye reflections, irregular teeth, warped ear, hair-edge inconsistency, face-surround lighting mismatch |
| Forensic probes | your existing freq/noise/blending code | SUPPORTING evidence only, not the primary verifier |
| Intervention (paper) | inpainting with a real-face prior (LaMa or similar) | validated against paired-real reversion, see pilot |
| Validation (route 2) | ground-truth manipulation masks (FF++ etc) | the credibility anchor |

## 4. Why the detector is frozen and pretrained, not trained by us

GenD, ForAda, and Effort are all frozen foundation encoder plus a tiny adapter, and reach SOTA cross-dataset detection. Using one off the shelf, frozen, means the intervention test runs on a detector we did not tune, so the faithfulness result cannot be dismissed as self-rigged. Independence is the point. The pilot also serves as detector selection, since a detector that does not respond to region reversion is too global, so we pick another.

## 5. The three results, each on a bad baseline

1. **Necessity rate.** Fraction of explanations that survive the erase test, ours vs free-text VLMs and MLLM judges. Theirs is low (TriDF), so this is a large gap.
2. **Mask alignment.** On mask-bearing datasets, intervention-confirmed predicates overlap the true manipulation region far more than rejected ones. This is the unimpeachable result, it proves our causal verifier agrees with ground truth.
3. **Control.** The same neutralization on a genuinely real region does not move the verdict. Reported as rigor, not headline.

Detection accuracy reported as flat, stated plainly.

## 6. The honest framing to write into the paper

The intervention test proves a cited cue is causally NECESSARY for the verdict, and mask overlap proves it is correctly LOCALIZED. State the claim as "necessary and correctly localized," not "the explanation is true." Name this gap before a reviewer does. The pilot uses paired-real reversion as the cleanest validator, the paper uses inpainting for general use, justified by the pilot's correlation check. Be explicit about that.

## 7. What must not leak in from the main track

- Do NOT train the detector on predicate captions. That makes the intervention test circular, since you would teach the model to attend to the regions you then test sensitivity on. Frozen detector only.
- No symbolic reasoner, no attribution, no rule editing. Those are the main-track contributions.

## 8. Data and effort

Existing datasets only (FF++ c23 with masks for the validation, plus CDFv2, DFDC, DFD for breadth). Frozen perception, no detector training, conformal-free pilot. Finishable in the workshop window because there is no training loop on the detector side.

## 9. Go/no-go before writing

Run the pilot spec (intervention_verifier_pilot.md). The single number is the gap between mean drop_true and mean drop_ctrl on 100 masked FF++ frames. Large positive gap, paper is alive. Near-zero gap, stop.

## 10. Dissertation role

Operationalizes verified symbol grounding. Discharges survey RQ on faithfulness metrics and grounding. The abstention and ground-truth-validation story is the part Prof Mohammad responds to, since it is testability and verification vocabulary.