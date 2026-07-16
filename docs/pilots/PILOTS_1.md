# Claude Code instructions, pilot phase only, revision 3

Purpose. Run the two gating pilots for the Verified Evidence Graph project. Do not build the full pipeline yet. Both pilots are pass or fail gates, Pilot 1 additionally has a partial outcome. Report numbers and stop at each gate for a human decision.

Read PLANNING.md and CODEBOOK.md first. Do not deviate from the codebook. Do not add components beyond what each pilot specifies. Complexity that is not in the pilot spec is out of scope.

Revision 3 changes. The dose response sweep is off the Pilot 1 gate, the gate is decided at a single LPIPS matched corruption point, embedding distance is logged as a secondary check only. A corruption only magnitude sweep may be logged since it costs only detector forward passes. Pilot 2 is split, part one with the unverified baselines is the gate, part two with the verified refit runs only on GO. Run Pilot 1 first, Pilot 2 part two depends on it.

Hard rules.
- Do not train or fine tune any perception model. Encoder, VLM, and diffusion model stay frozen. Temperature scaling of the detector output is the single permitted fit, one scalar on a held out calibration split.
- Do not put a text to pixel grounding model in the loop. Regions come only from deterministic facial landmarks bound to the codebook region ids.
- Report real measured numbers only. Never fill a table with placeholder or invented values. If a run did not complete, say so.
- Set and log all random seeds. Report results averaged over at least three seeds where sampling is involved.
- Keep every predicate probabilistic per the codebook. The only Boolean fields are the structural ones.
- Pre register the DF40 generator to family mapping and the held out generators in a file committed before any Pilot 2 results are produced.

Environment. Two GPUs, roughly 148 GB total. Batch heavily. Prefer 8 bit or 16 bit inference for large models.

---

## Pilot 1, verification test on pixels

Goal. Decide whether the counterfactual test measures the cited cue rather than detector fragility, and whether diffusion repair approaches ground truth repair. This gates the entire framework.

Data. Roughly one hundred FF++ frames that have ground truth manipulation masks, plus their paired real source frames. Balance across the FF++ manipulation types.

Steps.
0. Calibration. Temperature scale the frozen detector on a held out calibration split disjoint from the pilot frames. Log the temperature. All p(fake) values below are calibrated.
1. Load the frozen detector backbone. Log which model.
2. Run deterministic landmarking, map to the codebook region ids, produce region boxes. Apply the landmarking failure policy from the codebook, log abstained frames.
3. For each masked frame, identify the region that overlaps the true manipulation mask, and select one non overlapping region as the wrong region control target.
4. Ground truth repair. Paste the paired real region back with Poisson blending, masked to the face side of any seam. Re run the detector, record the drop in p(fake). This is the upper bound intervention.
5. Diffusion repair. Repair the same true region with the frozen diffusion inpainter, re run the detector, record the drop. Report as a fraction of the ground truth repair effect per frame.
6. Inert corruption control. On the same true region, apply a magnitude matched non semantic distortion, a small patch shift and separately a light localized blur. Matched to the repair in LPIPS, this is the gate criterion. Log detector embedding distance as a secondary check, it is not a second requirement. The gate is decided at this single matched point. Optionally log a corruption only magnitude sweep, it costs only detector forward passes, but it is not a gate criterion. The full dose response figure including repair is deferred to the paper build.
7. Wrong region specificity control. Apply diffusion repair to the selected non manipulated region of the same fake frame. Record the drop.
8. Real repair offset. Apply the same repair operations to the corresponding region of the paired real frame. Record the change in p(fake). Do not require this to be zero. Record its distribution, it defines the normalization for verified_confidence.
9. Spectral interventions. For whole_face frequency predicates proposed on the fake frames, notch the cited band, suppress the checkerboard peak, renormalize the residual, each separately, re run the detector, record the drops. Apply the same operations to paired real frames for the offset distribution.
10. VLM proposal recall. Run the frozen VLM proposer on all fake frames against the fixed codebook. Measure the fraction of true manipulated regions on which the VLM proposes at least one predicate, and precision of proposals against the masks. Log which VLM.
11. Localization check. For the set of proposed predicates, compare ground truth mask overlap of predicates that pass the necessity condition against those that fail. Log co verification, all predicates admitted by the same spatial intervention on the same region.

Report.
- p(fake) drops at the LPIPS matched point for ground truth repair, diffusion repair, patch shift, and blur, on the true region, plus the corruption only sweep if logged.
- Diffusion repair effect as a fraction of ground truth repair effect, distribution over frames.
- Wrong region repair drop distribution.
- Real repair offset distribution per intervention type.
- Spectral intervention drops on fakes versus real offsets.
- VLM proposal recall and precision against masks.
- Mask overlap, confirmed versus rejected predicates, with co verification rates.

Gate.
- GO if ground truth repair drops p(fake) clearly more than matched corruption at the matched point, diffusion repair recovers a substantial fraction of that effect, wrong region repair is near inert, the real repair offset is small relative to the fake repair effect, and confirmed predicates have clearly higher mask overlap than rejected.
- PARTIAL if ground truth repair passes but diffusion repair does not. The test is sound and the inpainter is the weak link. Report and stop, the human decides the replacement repair operator.
- STOP if ground truth repair and matched corruption produce similar drops, or if wrong region repair drops the score comparably to true region repair. Report this plainly. The foundation fails and the human decides the pivot.

Stop here and wait for a human decision before Pilot 2 unless told to run both. If told to run both, still run Pilot 1 first.

---

## Pilot 2, attribution signal on DF40

Goal. Decide whether frozen features plus predicate signatures separate generator families, whether the predicates carry that signal, and whether the verification gate preserves it. This gates the attribution story.

Data. DF40 with attribution labels, grouped into the codebook families per the pre registered mapping. Hold out at least one generator per family for the transfer test. The mapping file must exist before any results are produced.

Part one, the gate. Cheap, runs first, decides GO or STOP on its own.
1. Extract frozen encoder features for all samples. Log which encoder.
2. Compute the codebook predicate attributes per sample, using deterministic probes for the frequency, noise, and geometry attributes, and the frozen VLM for the semantic predicates over landmark regions. Whole_face frequency predicates included.
3. Baseline A. Linear probe on frozen encoder features, attribution head over seen families. Report seen and held out accuracy.
4. Baseline B. Raw DCT or wavelet features, attribution head. Report seen and held out accuracy.
5. Baseline C. Predicate count vector, a simple model over how many of each predicate type fire per sample. Report seen and held out accuracy.
6. Baseline D. MLP over the full unverified predicate attribute vector. Report seen and held out accuracy.

Part two, the interaction check. Runs only after a GO on part one. Confirmatory, not gating.
7. On a smaller subsample, sized to fit within a day of GPU time and logged, run the Pilot 1 verification gate, ground truth repair is unavailable here so use diffusion repair and spectral interventions with the controls, and refit Baselines C and D on verified only predicates. Report seen and held out accuracy of the verified variants next to the unverified ones, same subsample for both so the comparison is clean.

Report.
- Part one. Attribution accuracy for all four baselines, seen and held out, chance level stated, three seeds, mean and spread. The seen to held out gap for each. Whether the predicate based models, C and D, retain the signal that raw feature models, A and B, have.
- Part two, if run. The verified versus unverified comparison, including per predicate family verification rates by generator family.

Gate, decided on part one alone.
- GO if a predicate based model separates seen families well above chance and the predicate count baseline beats raw DCT. The signal exists and the predicates carry it. Run part two, then proceed toward the symbolic layer whose job is to match D verified on seen and beat it on held out.
- STOP if predicates collapse the signal that raw features retain. Report plainly. The paper pivots to certification of evidence only.
- If part two shows the verified refit collapses the signal the unverified predicates carry, stop and report. The human decides between fixing the gate and the certification only pivot.

---

## After both gates

Do not proceed to the full build without an explicit human go. When both pilots pass, the next instruction set will cover the Verified Evidence Graph construction, the probabilistic Scallop or DeepProbLog reasoner, hand written rules first, the full attribution evaluation including the unverified predicate MLP baseline, the calibrated attribution confidence with reported ECE, and the quantified rule editing study against few shot MLP fine tuning.
