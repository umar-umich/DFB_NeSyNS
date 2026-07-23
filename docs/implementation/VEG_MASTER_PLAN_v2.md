# VEG — Master Implementation Plan (Parked State, v2)
**VEG: Causal Certification of Deepfake Explanations via Counterfactual Grounding**
Muhammad Umar Farooq · WACV second-round (~40-day clock) → IEEE TIFS (full) · Dissertation Ch. 4 (verification)
Consolidated July 9, 2026. This is the single source of truth for the paper's *current* design. It supersedes scattered decisions across chats. Companion files: `VEG_Protocol_PreRegistration_v1.md` (freeze-ready protocol), `CLAUDE_CODE_VEG_PHASE0.md` (Phase-0 agent instructions), `Umar_Phase0_Checklist.md` (operator checklist), `VEG_Framework_Complete_and_SOTA_Comparison.md` (positioning).

> **Read order for a fresh chat:** this file → protocol → framework/SOTA doc. This file records *what changed and why* since those were written; where they conflict, this file wins and those get updated.

---

## 0. One-paragraph state of the paper

The instrument (CGT) is validated and frozen (Phase-1: NM 0.45–0.84 across four detectors, GAP 0.67, controls inert). The paper has pivoted from a pure *audit* ("MLLM explanations are ungrounded") to a **deployable causal arbitration framework** that keeps a strong detector's accuracy as a floor, adds a causal false-positive filter that targets precision, emits certified explanations with tiered abstention that never discards a detection, and — as a fenced bonus arm — trains an explainer on CGT-derived labels to improve grounding at the root. Training is now **in this paper** (not deferred), via offline DPO. Formal backbone is the **Pearl Causal Hierarchy** (single backbone, no citation soup). Two new foundation models (VL-JEPA, SAM3) are **conditional, gated Phase-1 pilots**, not core commitments.

---

## 1. What changed since the protocol/framework docs were written (the deltas)

These are decisions made in discussion that the older docs don't yet reflect. Apply these when reconciling.

| # | Change | Rationale | Affects |
|---|---|---|---|
| D1 | **Spine = Config B** (detector holds verdict authority; MLLM is *region proposer*, not decider). Config A (MLLM grounded against its own verdict) survives as a *reported audit sub-study* only. | Filtering a weak MLLM's verdicts down is a regression, not a system. Detector decides; MLLM localizes. | Arms 2 & 3, deployment |
| D2 | **The bridge is NOT GradCAM.** Detector and MLLM share the *image coordinate frame*; the CGT (pixel-space repair) is the joint space. Detector attribution overlap is *reported*, never fed to the MLLM. | GradCAM injection contaminates the audit (leaks the answer, kills P3) and reintroduces the correlational signal the paper argues against. | Arm 2 mechanism |
| D3 | **Confident-FP mechanism replaces the uncertain-band mechanism.** Per DISCERN, most FPs are *confidently wrong*, not uncertain. The discriminator is the **shape of the causal-necessity profile**: genuine partial-manipulation fakes have a *peaked* profile (one dominant region, high GAP); confident false positives have a *flat* profile (no dominant region → diffuse/global cue). Flag flat-profile confident-fakes as likely FPs. | Uncertainty-gating catches the wrong error population. Uses the validated Phase-1 GAP quantity + the CLIP-spectral-blindness finding. | Accuracy story, Arm 3, pilots |
| D4 | **Training is in-paper** via **offline DPO** on CGA-labeled grounded/ungrounded claims (not online GRPO, not deferred to TBIOM). No CGT in the training loop → no reward-degeneracy (ASG's STE problem vanishes; labels are external). Trainable on 2 GPUs. | User needs real numbers; offline DPO fits the clock and the compute. ASG-style online loop stays as the *future* extension. | New Arm (training), schedule |
| D5 | **Formal backbone = Pearl Causal Hierarchy (PCH).** CGT = physical realization of an L3 counterfactual; Necessity Margin = **probability of necessity (PN)**; dual-process (System 1/2) = L1 (MLLM instinct) vs L3 (counterfactual verification). NSF-CoT's **two-property decomposition** (Perceptual Grounding × Causal Necessity) is the scoring structure *inside* PCH. **Reject** RSCM, CFQL, SMT/Z3, LLM-judge fusion, LASSO/attribution "Validity" term, game-theory equilibria. | One coherent formal story that *describes what the CGT already does*, vs. decorative citation soup. Attribution terms reintroduce the correlational signal we reject. | Intro, Arms 2/3, related work |
| D6 | **Two-property claim scoring + Claim Cards.** Each claim: Perceptual Grounding (does the artifact exist? — from the paired-real diff you already compute) × Causal Necessity (does repair flip the verdict? — the CGT). Separates Case-B "real but causally inert" from Case-C "hallucinated." Present as Claim Cards, not a single Grounding Rate scalar. | Fixes the coarse grounded/ungrounded binary; the "real-but-inert" finding is sharper than pure hallucination. | Arms 2/3, metrics |
| D7 | **Evaluation is protocol-scoped by pairing availability** (see §4). Cross-manipulation = full pipeline (paired). Cross-dataset pairing-recoverable (CDF/DFDC/DFDCP) = partial + unique diagnostic. Unpaired/in-the-wild/fully-synthetic = detector+trained-explainer only (out of certification scope). **VEG does not enter the cross-dataset AUC race.** | The CGT is a paired-data instrument; honest protocol scoping turns this from a liability into a novel axis. | Evaluation section |
| D8 | **VL-JEPA & SAM3 = conditional gated Phase-1 pilots.** SAM3 for open-vocabulary grounding of cited regions → masks (alongside, not replacing, the fixed face-parser); gated on mask-IoU. VL-JEPA as architecturally-distinct 2nd detector for the FP-signature filter; gated on (a) competent frame-level face verdict and (b) CLIP-vs-JEPA disagreement. Either can be struck without disturbing the freeze. | "New model shiny" is the trap; each must earn a number. SAM3 is the higher-probability win (plugs the untestable-claim hole); VL-JEPA is higher-risk (tuned for video/motion, not appearance). | Phase-1 pilots |

---

## 2. The framework (current, post-deltas)

### Dissertation arc
DBaGNet (prediction) → DISCERN (calibration) → **VEG (verification)**. VEG discharges the survey's two open problems: symbol grounding (CGT operationalizes it causally) and explanation-faithfulness measurement (CGA measures it). DISCERN's role in VEG: calibrated uncertainty cleanly handles the *genuinely ambiguous* slice; the confident-FP causal-signature filter (D3) handles the confident-FP slice DISCERN's calibration can't gate.

### Deployable arbitration (Config B) — the deployed object
Detector screens everything (fast, strong verdict). MLLM proposes candidate artifact regions (its one real strength: localization + language). CGT tests proposed regions against the **detector** in pixel space. Tiered output:

| Detector | CGT on MLLM-proposed region(s) | Output |
|---|---|---|
| Confident FAKE | ≥1 region causally grounded (peaked) | **Certified Fake** + grounded rationale |
| Confident FAKE | flat profile, none grounded | **Fake, `rationale_ungrounded`** — verdict kept, explanation flagged, route to spectral/human. *Also the confident-FP flag (D3): flat profile ⇒ suspect spurious FP.* |
| Ambiguous (DISCERN) | ≥1 grounded | **Resolve toward FAKE** on causal evidence |
| Ambiguous (DISCERN) | none grounded | **Abstain / escalate** |

Floor guarantee (provable, no experiment): detector verdict is never overridden by the weaker MLLM → detection accuracy never *drops*, explanation precision *rises*. Upside bet (must be measured): confident-FP filter reduces false positives → precision gain, magnitude unknown, registered as hypothesis.

### The three arms + training arm
- **Arm 1 — CGT instrument validation.** ✅ FROZEN. Phase-1 anchor + Pilot 1L (localization) + Studies 1A/1B carryover + inpainting/spectral negatives.
- **Arm 2 — Causal Grounding Audit (CGA), Config B.** Per fake, per model: Type-B elicitation → parse to predicates → repair MLLM-cited region with paired-real pixels → re-query **detector** → two-property score (Perceptual Grounding × Causal Necessity) → Claim Card. Per-MLLM control re-validation. Metrics: GR / control-adjusted GR / PN (Necessity Margin) / untestable fraction / GR↔CHAIR / judge-correlation.
- **Arm 3 — Symbolic Adjudicator + tiered arbitration.** Frozen small ruleset over two-valued (then three-valued) claim truth; certified rationale = grounded claims only; abstention; contradiction flags. Evaluated by CHAIR-by-construction + external DeepfakeJudge-7B score + abstention/FP-filter numbers. **No user study** (IRB cut).
- **Arm 4 (bonus, fenced) — CGT-supervised training.** Offline DPO on CGA-labeled grounded≻ungrounded claims → higher-grounding explainer + better region proposals. **Hard fence:** runs only AFTER CGA freeze (Gate G3); **hard fallback:** if not clean by ~Day 34, moves to TIFS and the paper stands on Arms 1–3.

### What the paper does NOT claim
Not a cross-dataset AUC winner. Not a new detector architecture. Not fully-synthetic-capable (no paired real). No invented numbers — every accuracy/precision figure is measured or registered as a hypothesis.

---

## 3. Novelty & positioning (ranked by defensibility)
1. **CGT** — first causal-necessity instrument for forensic evidence (validated). Hardest to attack.
2. **CGA** — first causal audit of MLLM deepfake explanations; reframes the field's annotation-agreement standard.
3. **P3 result (if it holds)** — annotation-supervised "grounded" training buys annotation consistency, not causal grounding. The finding that stings SOTA.
4. **Causal arbitration + confident-FP filter** — deployable, precision-targeting, never loses detection accuracy.
5. **Symbolic Adjudicator** — minimal NeSy certifier; defensible only while tiny and frozen pre-CGA.

**Positioning one-liner:** the 13+ recent VLM methods compete on generalization AUC and measure explanation quality by annotation agreement; VEG contributes the causal faithfulness axis they cannot measure, shows it generalizes across manipulations, and uses it to reduce confident false positives — orthogonal to and composable with any of their detectors. **Closest related work to position against: VIGIL** (part-grounded, plan-then-examine, stage-gated injection — independently arrives at "keep region-selection driven by the model's own perception," conceptually adjacent to our GradCAM-rejection; but its grounding is annotation-structured, ours is causal). Primary face-swap baseline/SUT: **DD-VQA** (FF++, paired, region-annotated, ECCV 2024). Fusion methods that inject a detector score into the MLLM prompt = the related-work family our causal-intervention bridge distinguishes from.

---

## 4. Evaluation plan (protocol-scoped by pairing — D7)

**Field-standard protocols:** cross-dataset (train FF++, test CDF/DFDCP/DFDC/DFD) and cross-manipulation (train one FF++ type, test all four). Metric: AUC, sometimes pAUC@10%FPR, EER.

**Core insight — the CGT is a *training-time* signal, not a *test-time* requirement.** Trained explainer emits grounded explanations at inference without running the CGT. So VEG operates in unpaired test settings the same way every method does; the CGT ran during training on paired FF++.

**Three inference cases (increasing difficulty):**
- **Case A — Cross-manipulation (FF++ held-out type): FULL pipeline.** All FF++ manipulations share source actors ⇒ test set still paired ⇒ live CGT + certification + FP-filter all run. **This is VEG's best protocol** — the one place the complete pipeline including test-time verification is demonstrable. Centerpiece.
- **Case B — Cross-dataset, pairing-recoverable (CDF/DFDC/DFDCP): partial + unique diagnostic.** Detector verdict (always). Trained explainer emits grounded explanation (no CGT needed). Live CGT/FP-filter run only on the pairing-recovered subset. **Novel:** CGT as *generalization-failure diagnostic* — localize *why* AUC drops FF++→CDF causally (are the load-bearing regions the same off-distribution?), something no competitor can produce.
- **Case C — Unpaired/in-the-wild (FFIW) / fully-synthetic: detector + trained explainer only.** No paired real ⇒ CGT can't run at test time ⇒ certification out of scope (stated honestly). Same position as competitors (unverified explanation), but explainer trained on a stronger signal.

**Five protocol-level claims (axes competitors can't report):**
1. Causal explanation faithfulness under cross-manipulation (Case A, live verification).
2. Generalization of grounding (Case B: does FF++-trained grounded explainer stay grounded on CDF/DFDC? — GR-vs-dataset curve).
3. Causal generalization-failure diagnostic (Case B, pairing-recovered subset).
4. Confident-FP precision gain (Cases A+B) — report as pAUC@10%FPR; honest because no TP is ever dropped.
5. Detection AUC — reported for completeness, detector positioned competitive-not-SOTA, explicitly disclaimed as *not* the contribution.

**Load-bearing feasibility check (add to Phase-0):** confirm pairing-recovery for a meaningful CDF/DFDC subset (recover target-real frames + usable masks). If too lossy, Case B degrades toward Case C and the paper leans on Case A (airtight on pairing).

---

## 5. Formal backbone (PCH) — how to write it (D5/D6)

- **CGT = physical realization of an L3 counterfactual.** Repairing the cited region with paired-real pixels and re-querying *realizes* "what would the verdict be had this region been real?" Cite counterfactual-realizability: we can *estimate* the necessity quantity directly (not merely bound it) *because* the counterfactual is physically realizable via the paired real.
- **Necessity Margin = probability of necessity (PN).** Pearl's probability of causation. Gives the informal metric a rigorous, citable grounding.
- **Untestable (non-spatial) claims = non-realizable counterfactuals → non-identifiable → bound, don't discard.** (Partial-identification framing; TIFS-grade upgrade to the untestable-fraction finding.)
- **Dual-process = PCH layers.** MLLM claim = L1 (instinctive/associative); CGT = L3 (counterfactual verification); arbitration = the layer-selection decision. Free within the PCH backbone; no separate citation.
- **Scoring inside PCH = Perceptual Grounding × Causal Necessity** (NSF-CoT parallel, cited as the NLP analogue). **No attention/attribution term** — it's the weak correlational signal we reject.
- **Explicitly rejected:** Relational SCMs (wrong problem — identification without intervention; we *can* intervene), Causal Flow Q-Learning (sequential RL, not our offline-DPO structure), SMT/Z3 (no FOL for "mouth blending"; CGT is a stronger verifier than symbolic entailment), LLM-judge fusion (reintroduces the sycophancy we indict), game-theory equilibria (no competing agents).

---

## 6. Gated execution schedule (Day 0 = freeze day; WACV ~Day 40)

| Days | Track | Deliverables & gates |
|---|---|---|
| 0–5 | **Phase 0 — freeze + launch** | Pre-registration committed; FF++ + regenerated paired set (DiffSwap/BlendFace/CSCS) on H200; model weights verified; **pairing-recovery feasibility check (CDF/DFDC)**; throughput. **Gate G0 → FREEZE.** (See `CLAUDE_CODE_VEG_PHASE0.md`.) |
| 5–12 | **Phase 1 — instrument + new-model pilots** | Pilot 1L (localization); Pilot 1.5-M (per-MLLM control validation — *over-invest, top reviewer attack*); Pilot 2P (parser reliability). **New gated pilots:** **SAM3 mask-IoU gate** (open-vocab region grounding → §A.2/A.3, lowers untestable fraction); **VL-JEPA gate** (competent frame-level face verdict? CLIP-vs-JEPA FP-disagreement?). **Confident-FP signature pilot (D3): do TP profiles peak and FP profiles stay flat?** — de-risks the entire detection-accuracy claim; run before CGA. |
| 12–24 | **Phase 2 — CGA (Arm 2), Config B** | Full audit, two-property Claim Cards, per-MLLM controls, judge-correlation, DD-VQA bonus. **Gate G3 — CGA FREEZE (hard, Day 24).** |
| 24–32 | **Phase 3 — Adjudicator (Arm 3) + arbitration** | Frozen-CGA ruleset; tiered output; CHAIR-by-construction; external judge; abstention/FP-filter numbers. |
| 24–34 | **Phase 4 — training (Arm 4, fenced)** | Offline DPO on CGA labels, *after* G3. **Fallback: not clean by ~Day 34 → TIFS.** |
| 26–38 | **Writing** (parallel) | Draft as results land; PCH backbone; positioning table (incl. VIGIL/DD-VQA); red-team vs known attacks. |
| 38–40 | **Buffer + submit** | |

**Discipline invariants (never violate):** freeze before you measure · strike before you freeze · never tune the audit to flatter the remedy (Arm 4 strictly post-G3) · every component traces to a confound it kills or a number it earns · no invented numbers.

---

## 7. Open decisions parked for the next working session
1. **Claim-selection rule under K=4 cap:** order-of-listing (recommended, neutral, biases toward SUTs) vs. random-among-claims. *Confirm at freeze.*
2. **Tier-C API models:** set dollar cap or drop. *Confirm at freeze.*
3. **Incompatibility list** (≤10 mutually-exclusive artifact pairs for Adjudicator flags): draft + approve *before* G3.
4. **Parser LLM:** Gemini 2.5 Flash-Lite (TriDF-matching) vs. pinned open LLM. *Confirm at freeze.*
5. **VL-JEPA go/no-go:** depends on Phase-1 disagreement gate — likely drop if it's too video/motion-tuned for the appearance task.
6. **Three-valued vs two-valued Adjudicator truth:** start two-valued (grounded/ungrounded); promote to three-valued (add UNKNOWN for untestable) if SAM3 meaningfully changes the testable fraction.

---

## 8. Phase-1 pilot additions to encode in Claude Code (next agent session)
When you return to implementation, these are the new gated sub-pilots to add to a `CLAUDE_CODE_VEG_PHASE1.md` (not yet written):
- **Confident-FP signature pilot** (D3) — run region-wise necessity profiles on confirmed TPs vs. confident FPs; gate: FP profiles measurably flatter than TP profiles. *Precedes CGA; gates the accuracy claim.*
- **SAM3 grounding pilot** — phrase-grounded masks vs. hand-drawn region masks (IoU gate); on pass, SAM3 grounds open-vocab cited claims alongside the fixed face-parser, lowering untestable fraction. *Higher-probability win.*
- **VL-JEPA second-detector pilot** — (a) frame-level face-verdict competence check, (b) CLIP-vs-JEPA causal-region disagreement on FPs. Both gates must pass or strike. *Higher risk; verify openness/interface first.*
- **Pairing-recovery pilot** — recover target-real + masks for a CDF/DFDC subset (feasibility for Case-B claims).
- Each pilot: struck cleanly without disturbing the freeze; each earns a specific number or is dropped.

---

## 9. Files in this project
- `VEG_MASTER_PLAN_v2.md` — **this file** (parked source of truth).
- `VEG_Protocol_PreRegistration_v1.md` — freeze-ready protocol (update with D1–D8 before freeze).
- `CLAUDE_CODE_VEG_PHASE0.md` — Phase-0 agent instructions (add pairing-recovery check).
- `Umar_Phase0_Checklist.md` — operator checklist.
- `VEG_Framework_Complete_and_SOTA_Comparison.md` — positioning/critical-eval (reconcile with D1–D8).
- *(to write next session)* `CLAUDE_CODE_VEG_PHASE1.md` — Phase-1 pilots incl. §8 additions.

---

## 10. Note for the survey chapter (the task you're moving to)
The survey — *"Explainable Deepfake Detection: A Survey of Neural, LLMs, and Neuro-Symbolic Approaches for Forensic Reasoning"* — is where VEG's two discharged open problems live: **symbol grounding** and **explanation-faithfulness measurement**. Frame both as open problems in the survey, then the dissertation cites VEG as discharging them. Keep the survey's faithfulness-measurement section annotation-agreement-centric (that's the field's actual state) so VEG's causal-necessity contribution reads as the gap-filler. VIGIL, TriDF, DeepfakeJudge, DD-VQA, SIDA, FakeShield, Skyra, and the fusion-score methods are the MLLM/NeSy-adjacent works to taxonomize.
