# Verified Evidence Graph for Deepfake Reasoning
## Master planning document, revision 3

This is the reference of record for the merged single paper. It supersedes the earlier two paper split and revisions 1 and 2. Read the design rules first. Everything after the pilots is conditional on the pilots passing.

Revision 3 changes, from the meta review. The dose response sweep and dual metric matching are moved off the pilot gate's critical path, the gate uses a single LPIPS matched corruption point. Pilot 2 is split, the cheap unverified baselines are the gate and the verified refit interaction check is a confirmatory follow on that runs only on GO. Verified confidence normalization constants are refit on a calibration split disjoint from evaluation before any headline numbers. The region level versus predicate level necessity distinction is stated as one unmissable sentence in the claim scope.

Revision 2 changes, from the external critique. Ground truth repair added as the gold standard intervention. Wrong region specificity control added. Real inertness reframed from a pass or fail requirement to a measured baseline offset. A global region and spectral interventions added so global fingerprints are not systematically rejected by a spatially local gate. Pilot 2 now includes a verified subset run so the interaction of the gate and the attribution signal is tested before the full build. Calibration is obtained, not assumed. Magnitude matching is specified and dose response curves replace single point comparisons. The rule editing demonstration becomes a measured comparison. The generator shift claim is scoped to unseen generators within known families.

Writing style for the paper follows the standing preference. Concise declarative prose. No colons, no semicolons, no em dashes. No fabricated numbers in any results table. No AI writing signatures.

---

## 1. One line thesis

Binary real versus fake detection is saturated. The two things detectors still cannot do are explain themselves with evidence that is provably load bearing and attribute a fake to its generator family on generators never seen in training. We freeze perception, certify each piece of forensic evidence with a physical counterfactual test before any reasoning uses it, assemble the survivors into a Verified Evidence Graph, and run a symbolic program over that graph to attribute the generator family with an editable, checkable proof.

The contribution is not the depth of neural symbolic coupling. It is that only causally verified evidence is allowed to enter reasoning, and that the resulting rules transfer to unseen generators within known families and can be edited by a human without retraining perception.

---

## 2. Design rules carried from the DISCERN post mortem

These are hard constraints. Every component is checked against them.

1. Confirm the baseline is actually bad before building anything. A mechanism cannot look good against a saturated target. Both pilots exist to enforce this.
2. Do not aim at saturated targets. Aim at explanation load bearingness and cross generator attribution, both of which have headroom.
3. Decouple. The neural detector owns the binary verdict. The symbolic layer never sits on the detection critical path, so interpretability can never lower detection accuracy.
4. Do not train perception. Frozen foundation models avoid the cross generator collapse that comes from training on generator specific artifacts.
5. Every component earns its place through an ablation or a falsification test. No decoration.
6. Prefer the smallest mechanism that moves the bad number over an elegant architecture that matches the baseline.

Failure patterns to watch, learned from prior reviews and the revision 1 critique.
- Region necessity is not predicate necessity. Removing a region can move the verdict because you damaged local statistics, not because you removed the cited cue. The magnitude matched corruption control exists to catch this. Even after that control, region level repair verifies every predicate proposed on that region jointly. Co verification is therefore logged and reported, and frequency and noise predicates get predicate targeted spectral interventions instead of spatial repair.
- The inpainter is itself a generator. Diffusion repair may replace one artifact with another the detector is blind to. Ground truth repair from the paired real frame is the gold standard against which diffusion repair is measured, and the real face repair delta is a measured offset that verified confidence is normalized against, not a control that is required to be zero.
- Verification is spatially local by default but attribution fingerprints are often global. Without a global region and spectral interventions, the gate would systematically reject exactly the predicates that carry attribution signal.
- Do not put a grounding model on the validation critical path. Predicates are bound to deterministic facial landmarks so a failed check means the evidence failed, not that a grounder missed.

---

## 3. What the framework is, precisely

Type 3 neuro symbolic on the Kautz taxonomy. Neural perception feeds a symbolic solver through one discrete interface. No gradient crosses the boundary in the safe version. Reasoning happens entirely in the symbolic program. The neural parts perceive and measure, they do not reason. There is no inference time loop and no symbolic to neural feedback. This is a deliberate choice, not a limitation, because a loop would let the current family hypothesis steer what evidence perception looks for, which would contaminate the independence of the evidence that is the core claim.

Framing sentence for the paper. A neuro symbolic forensic reasoning framework in which neural perception certifies forensic evidence and symbolic reasoning performs executable attribution over that verified evidence.

### 3.1 Components, all frozen except the rules

- Perception. A frozen grounding capable VLM proposes localized predicates from a fixed codebook and emits the region for each. No free text. The fixed vocabulary is the inductive bias. The VLM covers only the semantic predicates. Deterministic probes own the frequency, noise, and geometry predicates. VLM proposal recall against ground truth masks is measured in Pilot 1, and a with and without VLM ablation is planned from the start. The framework survives a droppable VLM. The framing does not depend on it.
- Landmarking. Deterministic facial landmark detection produces the region boxes for intervention. This keeps a fallible grounding model off the critical path. Failure policy fixed in advance. If landmarking confidence falls below threshold on a frame, the frame emits an abstention node and no predicates are proposed on it.
- Detector backbone. A frozen contrastive encoder with a light head, in the style of GenD, ForAda, or Effort, produces p(fake). The raw score is temperature scaled on a held out calibration split before any verification uses it. This backbone also owns the binary verdict.
- Counterfactual generator. A frozen diffusion inpainter reconstructs a plausible clean version of a cited region for the verification test. Its quality is bounded above by ground truth repair, defined in section 4.
- Symbolic reasoner. Scallop or DeepProbLog runs rules over the Verified Evidence Graph to output the generator family, a proof trace, and an abstention when nothing is verified. Evidence enters as probabilistic facts, not Boolean.

### 3.2 Communication channel

The only thing that crosses from neural to symbolic is the Verified Evidence Graph, a set of structured nodes with probabilistic confidences and typed relations. The symbolic layer never sees pixels, latents, or VLM text. This strict bottleneck is what keeps the design honest and interpretable.

---

## 4. The counterfactual verification test

This is the heart of the paper. A predicate proposed over a region is admitted only if it passes the necessity gate under the controls below.

Interventions, by predicate class.
- Spatially localized predicates. Repair the cited region.
  - Ground truth repair, gold standard where a paired real frame exists. Paste the true real region back with Poisson blending, masked strictly to the face side of any seam so background pixels are untouched. This is the upper bound intervention.
  - Diffusion repair, the deployable intervention. A frozen diffusion inpainter reconstructs the region. Reported against the ground truth repair upper bound wherever pairs exist.
- Global frequency and noise predicates, bound to the whole_face region. Spatial inpainting cannot remove a global fingerprint, so these get predicate targeted spectral interventions. Notch the cited band, suppress the checkerboard peak, renormalize the residual statistics, each followed by re running the detector. This tests the cited attribute, not the region.

Controls, all required.
- Inert corruption control. Apply a magnitude matched non semantic distortion to the same region, a small patch shift and separately a light localized blur. Magnitude matched means matched in LPIPS as the single gate criterion, with detector embedding distance logged as a secondary check, not a second requirement. The pilot gate is decided at this one well matched point. A corruption only magnitude sweep may be logged since it costs only detector forward passes, but it is not a gate criterion. The full dose response figure including repair belongs to the paper build after the gate passes, never on the gate's critical path.
- Wrong region specificity control. Apply the same repair to a non manipulated region of the same fake frame. p(fake) must not drop comparably. Without this, nothing rules out that any inpainting anywhere on a fake drops the score.
- Real repair offset. Apply the same repair operation to the paired real frame. This delta is expected to be nonzero because inpainting synthesizes pixels. It is measured, its distribution is recorded, and verified confidence is defined as the normalized drop on the fake relative to this real repair delta distribution, not as a raw drop required to clear an arbitrary margin.

Optional for the stronger version, gated on time and pilot outcome.
- Sufficiency. Isolate the cited cue against a neutral baseline and show p(fake) rises. Necessity plus sufficiency is much harder to dismiss. Add this for the journal version, not the first conference pass.

Claim scope discipline. Never call this faithful without qualification. The property proven is that the cited region or attribute is necessary for the detector verdict, correctly localized, and specific, in the sense that matched inert corruptions and wrong region repairs do not reproduce the effect. When multiple predicates share a region they are co verified jointly, and co verification rates are reported so the confound is quantified rather than hidden. The paper states region level necessity for spatially repaired predicates and predicate level necessity for spectrally intervened predicates, and never blurs the two. This one sentence is the claim, everywhere it appears. Verified means necessary for this detector's verdict. Evidence the detector is insensitive to will not verify even if forensically real, and per predicate family verification rates are reported by generator family so this filter is visible.

---

## 5. The Verified Evidence Graph

Nodes are structured records, not flat scalars and not opaque embeddings. Each node carries its landmark region, a verified confidence, the intervention type that verified it, and a short list of named human readable forensic attributes. The rule for an attribute is simple. A human must be able to read it and could write a rule over it. If not, it belongs upstream inside frozen perception, not on the node.

Attribute lists are kept deliberately generous at first so ablation can prune, since dropping an attribute is free and adding one means re running verification.

Verified confidence is a normalized effect size mapped into zero to one, defined against the real repair offset distribution and the matched corruption reference point. Normalization constants measured during the pilot are pilot scoped only. Before any headline attribution numbers, they are refit on a calibration split disjoint from all evaluation data, so the confidence scale never sees the evaluation set. It enters the logic program as a fact probability. The reasoner is probabilistic so confidence propagates to the attribution output. The attribution confidence out the other end is then explicitly calibrated on a held out split and its ECE is reported. Calibration is measured, never claimed by construction.

See CODEBOOK.md for the full node schema, relation types, and per attribute Boolean versus probabilistic tags.

---

## 6. Reasoning and attribution

The symbolic program holds rules that map patterns of verified evidence to generator families, for example heavy verified boundary artifact together with verified identity drift supports the face swap family and contradicts the diffusion family. Predicate interactions are expressed by the rules, which is the reason to use a relational program rather than an MLP over predicate scores. The MLP would model interactions implicitly and opaquely. The program models them explicitly and readably.

Rules are either hand written or synthesized once by inductive logic programming over the verified predicate matrix, then fixed. Do not add ILP unless a hand written rule set fails to clear the baseline. Adding it before that is a small DISCERN.

Adaptation to a new generator happens in the rules, not the weights. This is evaluated, not demonstrated. The rule editing study is a measured comparison. For k pre registered confounds, predict the behavior change before the edit, apply one rule edit per confound, and measure attribution accuracy change, side effect rate on all other families, and human time. Compare against few shot fine tuning of the same input MLP baseline on the identical handful of samples from the new generator. The moat claim becomes three numbers, accuracy recovered, side effects induced, and cost.

Honest framing of the symbolic advantage. Hand written rules inject human forensic priors the MLP does not receive. The contribution is stated as such. The symbolic representation is the thing that admits human knowledge injection and human correction, and the claim is that this representation degrades less under generator shift at matched input information. Capacity fairness against the MLP baseline is reported.

---

## 7. The headline result and the baselines

The headline number is attribution accuracy under generator shift, measured on held out generators within known families, symbolic program versus a same input neural baseline over the identical verified predicates. Matching on seen generators while losing less on unseen generators is the win. A genuinely new family is unattributable by construction, and abstention is the designed behavior there. Abstention rate on a truly novel family is reported as a supplementary result. Do not sell interpretability as the primary contribution, it is a weak 2026 selling point on its own.

Baselines to beat, at minimum.
- Frozen encoder features with a linear probe, attribution head.
- A simple frequency baseline, DCT or wavelet features, attribution head.
- An MLP over the exact same verified predicate vector. This is the critical baseline. If the symbolic program does not match it on seen generators and beat it on unseen generators, the symbolic layer is decorative.
- An MLP over the unverified predicate vector, to isolate what the gate itself contributes or costs.
- A free text VLM asked to attribute, for the faithfulness and checkability contrast.

Report attribution accuracy seen and unseen, the seen to unseen gap, mask overlap of verified versus rejected predicates, co verification rates, per family verification rates by generator family, VLM proposal recall, abstention rate and its correlation with difficulty, ECE of the attribution confidence, and the quantified rule editing study of section 6. Variance across at least three seeds everywhere sampling is involved.

---

## 8. The two pilots, both this week, both pass or fail gates

Nothing after this section is committed until both pilots pass. This is the single most important discipline in the whole plan.

### 8.1 Verification pilot, decides whether the core test works on pixels

Question. Does the counterfactual test measure the cue rather than detector fragility, and does diffusion repair approach ground truth repair.

Procedure. On roughly one hundred FF++ masked frames with paired real sources, run ground truth repair, diffusion repair, magnitude matched corruption on the same region, wrong region repair on the same fake, and the same repair on the paired real frame. Spectral interventions run for the global frequency predicates. Predicates drawn from the fixed codebook so no grounding model is in the loop.
- GO if ground truth repair drops calibrated p(fake) clearly more than matched corruption, diffusion repair recovers a substantial fraction of the ground truth repair effect, wrong region repair is near inert, the real repair offset is small relative to the fake repair effect, and intervention confirmed predicates beat rejected ones on ground truth mask overlap.
- STOP if repair and matched corruption produce similar drops even for ground truth repair, or if wrong region repair drops the score comparably. Then the test measures fragility and the framework foundation evaporates. Better to know now.
- PARTIAL outcome now recognized. If ground truth repair passes but diffusion repair does not, the test is sound and the inpainter is the weak link. The pivot is a better repair operator, not abandoning the framework.

### 8.2 Attribution pilot, decides whether Paper B lives

Question. Do frozen encoder features plus predicate signatures separate generator families above chance, do the predicates carry that signal, and does the verification gate preserve it.

Procedure, in two parts so the expensive check never blocks the cheap answer.
- Part one, the gate. On DF40 with attribution labels, family grouping and held out generators pre registered before any results exist. Frozen encoder features, raw DCT features, predicate count vectors, and the full unverified predicate attribute vector, each with an attribution head, seen and held out.
- Part two, the interaction check, runs only on GO. On a smaller subsample, run the verification gate from the pilot above and refit the predicate models on verified only predicates. Confirmatory, not gating, but a collapse here is a late STOP and goes to the human.

Gate, decided on part one alone.
- GO if a predicate based model separates seen families well above chance and the predicate count baseline beats raw DCT. The signal exists and the predicates carry it. Then run part two.
- STOP if predicates collapse the signal that raw features retain. Then no symbolic layer rescues attribution and the paper pivots to certification of evidence only.
- If part two later shows verification destroys the signal the unverified predicates carry, stop and report, the human decides between fixing the gate and the certification only pivot.

---

## 9. What is cut and why

- Inference time loop where the reasoner requests re analysis. Reintroduces an agent, adds latency and failure modes, and contaminates evidence independence. Cut.
- Detector ensemble for verification, five detectors voting. Multiplies cost, moves no target number, gives reviewers more to attack. Cut.
- Conformal certification of the proof. Elegant, moves no attribution number, weeks of debugging in a probabilistic logic stack. One future work sentence only.
- RL trained grounding, causal drop as reward. Trains perception, breaks rule 4, brittle reward hacking risk. Future work sentence only.
- Trained adapter on the encoder. Only if frozen predicates underperform, and only with the three entity independence guard, predicates from model A, verification by detector B, masks as C. Not in the first build.

---

## 10. Compute, where it is allowed to go

Two GPUs, roughly 148 GB. Spend it on the large grounding VLM at inference, the diffusion counterfactuals including the dose response sweeps, and full DF40 scale so numbers carry statistical weight. Do not spend it on parallel ensembles of the same idea. Having compute is not a reason to add machinery. Machinery still has to move a target number.

---

## 11. Dissertation placement and research questions

This one paper carries two dissertation research questions. Numbering follows the committee preference for chapter level tags distinct from dissertation level tags.

- DRQ. Can forensic evidence be certified as load bearing before it is used, and does reasoning over certified evidence attribute generators better under distribution shift than reasoning over unverified evidence or than a same capacity neural baseline.

Chapter level.
- RQ x.1. Does counterfactual verification admit evidence that is necessary for the detector verdict, correctly localized against ground truth masks, and specific under matched corruption and wrong region controls. Falsified if admitting every predicate with no gate yields equal mask overlap, or if the controls reproduce the repair effect.
- RQ x.2. Does a symbolic program over verified evidence match a neural baseline on seen generators and beat it on unseen generators within known families for family attribution. Falsified if it does not beat the same input MLP on unseen generators.
- RQ x.3. Can a human edit rules to correct pre registered generator confounds and change behavior as predicted, at lower cost and side effect rate than few shot fine tuning of the neural baseline, without retraining perception. Falsified if edits do not change behavior as predicted or if fine tuning dominates on all three measures.

Connection to the rest of the dissertation. The published DBaGNet established generalization through explicit multi signal features. DISCERN established calibrated uncertainty and produced the lesson that mechanisms must target a metric with headroom. The survey defines the epistemic ladder and names symbol grounding and faithfulness measurement as the open problems. This paper discharges both, grounding through verification and attribution through reasoning over verified evidence.

---

## 12. Execution order

1. This week. Preprocess DF40, pre register the family grouping and the held out generators before any results exist.
2. This week. Calibrate the detector by temperature scaling on a held out split. Run both pilots against the fixed codebook, verification pilot first since the attribution pilot's interaction check depends on it.
3. Gate. If the verification pilot fails on ground truth repair, stop and reconsider the whole approach. If it fails only on diffusion repair, swap the repair operator. If the attribution pilot fails, pivot to the certification of evidence paper.
4. If both pass, build the full pipeline, frozen perception, codebook predicates, counterfactual gate with all controls, Verified Evidence Graph, probabilistic symbolic reasoner, hand written rules first.
5. Run the full attribution evaluation with all baselines, including the unverified predicate MLP.
6. Add ILP only if hand written rules stall. Add sufficiency only for the journal version. Decide WACV versus TIFS by how strong attribution looks.

---

## 13. Open decisions to confirm before freezing the codebook

- Venue and therefore scope, WACV lean version first versus TIFS full version.
- DF40 family grouping and which generators are held out. Pre registered, written down before pilot results exist. Boundary cases such as diffusion based face swaps assigned by manipulation scope, a swap is face_swap regardless of the generative backbone, and the assignment rationale logged.
- Which frozen detector backbone is the verdict owner and the verification instrument, and whether they are the same model or two different models for independence.
- Which grounding capable VLM is the proposer.
- Which diffusion inpainter is the counterfactual generator.
- Landmarking confidence threshold for the abstention policy.
