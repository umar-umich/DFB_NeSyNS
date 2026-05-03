# NeSy-DeFake — Paper-Facing Companion
### Framework state and presentation guide as of 2026-05-01

This document complements the engineering READMEs (April-22: interpretability pipeline; April-28: sub-graph internals). Those describe **what the system produces**. This one describes **what the paper claims and how the framing must hang together** for NeurIPS submission. Read it alongside `PROJECT_INSTRUCTIONS.md`.

If a claim in the paper is not backed by something in this document or in the two prior READMEs, the claim is unsupported and must either be supported or removed.

---

## 1. The pitch in one paragraph

DeFakeNet is a CLIP-based deepfake detector that matches the strongest CLIP-adapter detectors on cross-dataset accuracy while uniquely emitting (i) per-prediction symbolic violation traces grounded in FACS, (ii) inspectable class-conditioned structural-graph divergences, and (iii) calibrated deferral signals — all at no additional inference compute. The contribution is the *integration mechanism*, not any single component: a confidence-modulated evidence fusion (CMEF) layer that reweights symbolic streams by their per-sample Dirichlet strength, a branch-level auxiliary supervision (PBAS) that prevents gradient starvation, and an inter-branch disagreement calibration (IBDC) that binds fused epistemic uncertainty to detached pairwise stream disagreement rather than evidence magnitude.

The strongest single technical novelty is **IBDC**. It is currently underplayed in the abstract and introduction.

---

## 2. What the paper actually claims

### 2.1 Architecture claims

The paper claims a three-stream architecture fused under an extended evidential reasoning module:

| Stream | Substrate | Output |
|--------|-----------|--------|
| Visual | Frozen CLIP ViT-L/14, LayerNorm-only tuning [Yermakov 2026] | 2-d softplus evidence |
| Symbolic | 58-dim semantic vector (InsightFace + DeepFace + LibreFace + MediaPipe) + ~18 differentiable FACS-grounded predicates | 2-d softplus evidence |
| Structural | 4 sub-domain SCM pairs (identity, structural, noise, spectral); per-class reconstruction residuals | 2-d softplus evidence |

The fused output is a Dirichlet distribution `α = e_total + 1` over two classes, from which the prediction `p̂ = α/S` and epistemic uncertainty `u = K/S` follow in closed form.

### 2.2 What is novel and defensible

1. **IBDC** — fused epistemic uncertainty bound to detached pairwise cosine disagreement between stream expected probabilities. Standard EDL [Sensoy 2018] ties uncertainty to evidence magnitude. TMC [Han 2021] fuses homogeneous views. Neither expresses uncertainty as a function of inter-source reasoning conflict. The detached-disagreement-as-calibration-target formulation is the single sharpest novelty in the paper.

2. **Class-conditioned SCM-pair structural divergence** — four sub-domain SCM pairs `{SCM_real, SCM_fake}` per sub-domain, with detection signal `Δr_g = r_g^(fake) − r_g^(real)`. Adjacency matrices retained as inspectable artifacts. The architectural commitment to *pairs* (not a single SCM) is what makes structural divergence available as a detection signal.

3. **CMEF + PBAS** — per-sample symbolic stream reweighting via Dirichlet strength prevents the visual stream from drowning out symbolic streams when the visual is uncertain; per-stream auxiliary EDL loss prevents gradient starvation. Both mechanisms address a known failure mode of multi-stream fusion that prior EDL/TMC literature does not handle.

4. **FACS-grounded predicate selection methodology** — candidate set defined a priori from FACS literature, retained via discriminative-gap validation on training-fold data only. This converts "where's the evidence each predicate helps?" from an ablation demand into a methodology, while foreclosing post-hoc-selection suspicion.

### 2.3 What we are NOT claiming

The paper must not claim:

- Identifiability of causal mechanisms in the Pearlian sense.
- Interventional or counterfactual queries on observed faces.
- A full first-order-logic symbolic reasoning system.
- That detection AUC is the headline win (it is parity, not a margin).
- That the symbolic stream is sufficient on its own (it is complementary, not standalone).

The §3.2 disclaimer paragraph is load-bearing for the first two. The §6 limitations paragraph must explicitly acknowledge the third.

---

## 3. The symbolic substrate — what the paper says about it

### 3.1 Substrate sourcing (§3.1)

The 58-dim semantic vector is sourced from four production-grade models, not hand-coded. The paper must state this:

| Source | What it provides | Indices | Citation anchor |
|--------|------------------|---------|-----------------|
| InsightFace (buffalo_l) | Gender, age, head pose (yaw/pitch/roll), detection confidence | 0–1, 34–36, 51 | InsightFace |
| DeepFace | Emotion (7 classes), ethnicity entropy | 2–10 | DeepFace |
| LibreFace (DISFA + BP4D) | 17 FACS Action Units (12 intensity from DISFA, 5 detection-only from BP4D) | 11–33 (mostly) | LibreFace, FACS [Ekman & Friesen 1978] |
| MediaPipe FaceLandmarker | 478 3D landmarks, gaze, geometric ratios, symmetry, blendshape AU fallback | 37–57 | MediaPipe |

The paper currently underclaims this by saying "computed from 3D facial landmarks." That is technically true but represents <40% of the substrate. The pedigree must be stated explicitly in §3.1, with citations to LibreFace and FACS at minimum. DISFA and BP4D are peer-reviewed FACS-validated datasets and citing them strengthens the substrate claim materially.

### 3.2 Predicate selection — the protocol the paper presents

Predicates are FACS-grounded **inductive priors**. The paper does not call them "causal priors" or "intervention priors" — those terms are reserved for Pearlian commitments not made here.

**Candidate set (a priori, ~25–30 predicates):**

The current `consistency_rules_v7.py` defines 12 predicates. The candidate set extends this with the four faceswap-failure-mode predicates from April-28 README §5 plus additional candidates spanning the same coherence categories:

| Category | Examples | Approx. count |
|----------|----------|---------------|
| Expression–AU coherence | `cr_happy_au6`, `cr_happy_au12`, `cr_surprise_au1au2`, `cr_sad_au15`, `cr_angry_au4`, `cr_fear_au1au5`, `cr_disgust_au9`, `cr_contempt_au14`, `cr_neutral_any_au` | 9 (current) + ~3 candidates |
| Pose–gaze coherence | `cr_gaze_lr_divergence`, `cr_yaw_gaze_misalign`, `cr_pose_facewidth` | 1 (current) + 2 (April-28) + ~2 candidates |
| Geometric / structural | `cr_lip_geometry_au12`, `cr_brow_eye_couple` | 0 (current) + 2 (April-28) + ~3 candidates |
| Bilateral symmetry | `cr_eye_asymmetry`, `cr_jaw_asymmetry` | 2 (current) + ~2 candidates |
| **Total candidate** | | **~25–30** |

**Validation protocol:**

1. Compute `gap_p = |mean(score_p | fake) − mean(score_p | real)|` for each candidate predicate `p` on a held-out validation slice of FF++ training data.
2. Rank predicates by gap.
3. Retain top-18 (default; final number resolved by inspecting actual gap distribution — possibly elbow-point).
4. **Freeze retained set before any test or cross-dataset evaluation.**

**Reporting in paper:**

- Main paper §3.1 (one paragraph): protocol description with explicit leakage-prevention sentence.
- Supplementary: full candidate set, validation-slice gaps, retained set, selection rule.
- Optional main-paper figure: top-k discriminative-gap bar plot if page budget allows.

The leakage-prevention sentence is non-optional. Required wording (or equivalent): *"Predicate selection is performed once on a held-out slice of the FaceForensics++ training fold; the retained set is frozen before any test-set or cross-dataset evaluation, and no predicate is added or removed based on results from these benchmarks."*

### 3.3 Defending the symbolic stream against "decorative neuro-symbolic"

When a reviewer writes "12 hand-coded predicates is not neuro-symbolic," the defense chain is:

1. The substrate is FACS-validated AU recognition (LibreFace DISFA + BP4D), not hand-coded features.
2. Predicates are FACS-grounded — every predicate references named FACS Action Units or pose/gaze/symmetry features with literature anchors.
3. The retained predicate set is empirically validated, not chosen for narrative convenience.
4. Differentiable-predicate + learned-fusion is a recognized neuro-symbolic family (DeepProbLog, NS-CL, Semantic Loss). Cite Garcez 2023 as the definitional anchor.
5. Limitations honest: §6 explicitly concedes this is a lightweight symbolic layer rather than full first-order logic.

---

## 4. The structural stream — what the paper says about it

### 4.1 Why "structural" and not "causal"

The Hybrid framing decision: keep "structural causal model" terminology because it accurately describes the NOTEARS/DAGMA formulation, but explicitly scope the claim to structural divergence between class-conditioned mechanisms — no identifiability, no interventions, no counterfactuals. The §3.2 disclaimer paragraph is load-bearing.

Outside §3.2, prefer "structural" over "causal" wherever feasible. "Class-conditioned structural mechanisms" is the most accurate phrase.

### 4.2 The four sub-domains

| Sub-domain | Input dim | What it captures |
|------------|-----------|------------------|
| Identity | 70 (32 latent + 26 curated + 12 predicates) | Expression–AU couplings, pose–gaze, geometric ratios, attribute coherence |
| Structural | 62 (32 latent + 30 macro forensic) | Boundary gradients, blur, symmetry, color, low-frequency DCT |
| Noise | 75 (32 latent + 43 micro-noise) | Patch noise consistency, cross-channel noise, SRM residuals, multi-scale wavelets |
| Spectral | 42 (32 latent + 10 FFT) | Radial FFT bins, spectral fall-off slope, HF/LF ratio |

Each sub-domain owns a pair of nonlinear MLPs, `SCM_real(x) ≈ x` and `SCM_fake(x) ≈ x`, trained on real-only and fake-only samples respectively. At inference, both are run on every input; the difference between their reconstruction errors `Δr_g = r_g^(fake) − r_g^(real)` is the per-sub-domain detection signal.

### 4.3 Cross-dataset dominance — the load-bearing experimental result

The key empirical claim that must be in the main paper:

> *Sub-domain dominance is dataset-conditional. On FaceForensics++ (heavy compression, classical face-swap pipelines), the noise sub-domain dominates ~84% of fake-class predictions, reflecting the dominance of compression-induced micro-noise artifacts. On Celeb-DF-v2 (cleaner post-processing, broader identity gap), identity rises to ~35% on real-class predictions and noise drops to ~60% on fakes. Different forensic regimes engage different mechanisms; a single-stream detector cannot adapt this way.*

This converts the FF++ identity-weakness number (1.7% fake-class dominance per April-28 README §4) from a liability into evidence that the architecture *correctly* allocates trust to the dominant signal regime per dataset. The figure must compare FF++ vs CDF dominance side-by-side.

If DFDC and other benchmarks produce additional dominance flips, those reinforce the claim. If they don't (e.g., DFDC dominance pattern resembles FF++), the claim narrows to "FF++ vs cleaner post-processing benchmarks" — still defensible but less sweeping.

### 4.4 The auxiliary objectives — what each loss term does

The paper has four structural auxiliary objectives. Each must be tied to a specific failure mode it prevents:

| Loss term | Failure mode it prevents | Required in paper |
|-----------|--------------------------|-------------------|
| DAGMA acyclicity | Cyclic adjacencies that aren't structurally interpretable | Yes |
| L1 sparsity | Dense uninterpretable adjacencies | Yes |
| Graph divergence (`−L1(A_real, A_fake)`) | The two SCMs in each pair learning identical mechanisms (signal collapse) | Yes |
| Label-conditioned reconstruction | Cross-contamination — `SCM_real` learning fake patterns or vice versa | Yes |

Without all four, structural divergence collapses or becomes uninformative. The paper must state each term and its purpose in §3.2 (or the loss subsection), not just list them in Eq. (8).

---

## 5. The fusion mechanism — IBDC as the headline novelty

### 5.1 Why IBDC matters

Standard EDL ties epistemic uncertainty to evidence magnitude — a confident-but-wrong prediction can have low uncertainty if the dominant stream's evidence is high. TMC [Han 2021] fuses homogeneous views; it does not address heterogeneous-stream reasoning conflict.

IBDC inverts the relationship:

- Compute pairwise cosine disagreement `d` between expected class probabilities `p_b = α_b / S_b` of each stream pair.
- **Detach `d` from the computation graph** (this is critical — `d` is a calibration target, not a learned quantity).
- Calibrate fused epistemic uncertainty `u = K/S` against `d` via binary cross-entropy: `L_ibdc = −d log u − (1−d) log(1−u)`.

Effect: when streams disagree, the model is trained to produce high uncertainty; when streams agree, the model is trained to produce low uncertainty. The disagreement signal acts as a fixed calibration target, and gradient flows only through `u`.

### 5.2 How to defend IBDC novelty

A reviewer who knows EDL/TMC will ask: "Is this just adding a regularizer to the EDL loss?" The defense:

1. **Different supervision target.** Standard EDL supervises `α` against class labels. TMC fuses `α` across views via Dempster's rule. IBDC supervises *uncertainty* against an inter-stream consistency signal — a different gradient target that no prior EDL or multi-view fusion paper has formulated.
2. **Detached disagreement.** The disagreement signal is a fixed target, not a learnable quantity. This is a deliberate choice — making `d` learnable would let the model game the calibration. The detach-and-calibrate-against pattern is what makes IBDC a calibration term rather than another evidence source.
3. **Heterogeneous streams.** TMC assumes homogeneous views. IBDC is designed for streams with sharply different inductive biases (high-capacity neural vs lightweight symbolic vs structural). The asymmetry is what makes inter-stream conflict a meaningful signal.

These three points together establish IBDC as a novel calibration mechanism, not a regularizer variant.

### 5.3 Surfacing IBDC in the paper

Currently underplayed. The abstract should reference it explicitly. The introduction's contribution-bullet for IBDC should be the first or second bullet, not third. The method section already describes it well (§3.3 IBDC subsection); the issue is upstream, in the framing.

---

## 6. The interpretability pipeline — what to surface in the main paper

The April-22 README documents seven levels of interpretability. The main paper has space for a subset. Priority for surfacing:

| Level | What it shows | Main paper? |
|-------|---------------|-------------|
| 1. EDL uncertainty + reliability diagram | Calibration | Yes (small) |
| 2. Branch evidence decomposition | Which stream drove the prediction | Yes |
| 3. FACS predicate violations | Named failures the model detected | Yes (case studies) |
| 4. SCM per-sample residuals (radar) | Sub-domain decomposition per sample | Yes (per-method radar promoted to main) |
| 5. SCM adjacency matrices | Class-conditioned structural difference | Yes (one sub-domain, side-by-side) |
| 6. Evidence gates (CMEF) | Trust allocation across streams | Supplementary |
| 7. Forensic feature attribution | Per-method dominance | Yes (cross-dataset comparison) |

Plus selective prediction (risk–coverage curve) and case-study gallery — both load-bearing if Headline Result (A) or (C) is selected.

---

## 7. Headline result — three options to draft and decide

The paper has not yet committed to a single headline. Three defensible options:

### (A) AUC parity + selective-prediction win

- **Lead figure:** Risk–coverage curve cross-dataset (FF++ training → CDF, DFDC, etc. testing). Three lines per benchmark: EDL uncertainty, softmax margin, oracle.
- **Lead number:** E-AURC vs softmax-margin gap, averaged across cross-dataset benchmarks.
- **Strength:** Cleanest measurable win. Detection AUC is saturated; selective prediction is where the paper can actually claim a margin.
- **Defensible against:** "Why not just use temperature scaling?" — because temperature scaling doesn't decompose evidence and doesn't surface inter-stream conflict.

### (B) Interpretability + accuracy parity

- **Lead figure:** Per-method causal fingerprint radar (FF-DF, FF-F2F, FF-FS, FF-NT) plus FF++ vs CDF dominance comparison.
- **Lead number:** AUC parity table with strongest CLIP-adapter baselines.
- **Strength:** Strongest narrative. The "interpretable detection at no accuracy cost" framing is rhetorically powerful.
- **Risk:** "Interpretable" is hard to falsify without a faithfulness experiment. Rejection vector unless the intervention test is included.

### (C) Calibration + deferral story

- **Lead figure:** Reliability diagram + selective-prediction case-study gallery (uncertain / confident-correct / confident-wrong groups from April-22 README §5.9).
- **Lead number:** ECE + AURC + confidence-vs-accuracy at multiple coverage levels.
- **Strength:** Most deployment-credible. Speaks directly to the forensic/regulatory deployment claim in the introduction.
- **Risk:** Most hybrid — pulls focus across calibration, selective prediction, and case studies. Hardest to fit in 9 pages.

The first working session should produce drafts of all three (abstract + Fig. 1 caption + contribution bullets) for direct comparison.

---

## 8. Mapping paper sections to evidence

For every paper section, the corresponding evidence source:

| Section | Evidence source | Key artifacts |
|---------|----------------|---------------|
| §1 Introduction | Cross-dataset baselines + selective prediction | `fig1_mosaic.png`, `risk_coverage_curve.png` |
| §2 Related Work | Literature only | — |
| §3.1 Substrate + predicates | `precompute_fast_semantic.py`, `consistency_rules_v7.py`, predicate selection protocol | Substrate table (this doc §3.1), candidate-set table (supplementary), `rule_discriminative_gap.png` |
| §3.2 SCM streams | `improved_scm_branch.py` (per April-28 README), CDF dominance numbers | `scm/r_diff_radar.png`, `scm/identity_side_by_side.png`, FF++ vs CDF dominance comparison |
| §3.3 NeSy-EDL fusion | `nesy_edl_loss.py`, `EvidenceFusion` | Branch evidence figure, gate distributions, IBDC math |
| §3.4 Training | Training recipe + UA loss + source-paired batches | — |
| §4 Experiments | Cross-dataset table | Main results table, ablation table |
| §5 Analysis | Interpretability pipeline | Per-method radar, reliability diagram, case study gallery, intervention test |
| §6 Limitations | Honest concessions | — |

---

## 9. Known liabilities and how the paper handles them

| Liability | Mitigation in paper |
|-----------|---------------------|
| FF++ identity sub-graph contributes only 1.7% on fakes | Cross-dataset dominance figure (§4.3) reframes as correct allocation |
| Symbolic stream looks "thin" if presented as 12 rules | §3.1 surfaces FACS-validated AU substrate; predicate selection protocol presents predicates as empirically validated |
| "Causal" terminology will trigger Pearlian objections | §3.2 disclaimer paragraph; "structural" preferred outside §3.2 |
| Detection AUC is parity, not a margin | Recast headline around selective prediction (Option A) |
| Predicate selection vulnerable to p-hacking suspicion | Leakage-prevention sentence; selection on training-fold validation slice only |
| Compute cost vs simpler CLIP-adapter baselines | Justified by selective prediction win + interpretability outputs at zero inference overhead |

---

## 10. What is still open

These are not blockers for starting paper work, but should be resolved during the first working sessions:

- **Headline result selection** — A, B, or C. Resolve in Session 1 by drafting all three.
- **Final retained predicate count** — top-18 default; final number resolved by inspecting actual gap distribution. Could be 16, 18, or 20.
- **Per-method radar placement** — main paper Fig. 2 vs Fig. 3 vs supplementary depending on page budget after experimental table is finalized.
- **Intervention test design** — leverage `INTERVENTION_RULE_DEFS_V7` hook in `consistency_rules_v7.py`. Minimum: synthetically violate one predicate per category, verify symbolic-stream evidence drops, report magnitude.
- **DFDC sub-domain dominance** — required to know whether the dominance-flip claim generalizes beyond FF++/CDF or must be narrowed.
- **Cross-dataset baselines** — confirm all five recent CLIP-adapter wave baselines (Forensics Adapter, CLIPping the Deception, Token Shuffling, LayerNorm-only, Reprogramming) are runnable. If any is not, document why and note explicitly in supplementary.

---

## 11. First-session recommendation

The first conversation in the project should produce:

1. Three abstract drafts (~200 words each) for headline options A, B, C.
2. Three Fig. 1 caption drafts matching each abstract.
3. Three contribution-bullet sets (3–4 bullets each) matching each abstract.
4. A rewritten §3.1 paragraph that surfaces the FACS-validated substrate and references the predicate selection protocol.
5. A side-by-side comparison so the user can pick the headline.

This single session resolves the headline-result question, the substrate-visibility question, and provides working language for the rest of the paper.

The second session should be `/review` on the current intro + related work + method, using the locked project instructions. The output becomes the working punch-list for everything that follows.

# Critique: Abstract, Introduction, Literature Review
I'll be direct about what's broken before we move to Method. These are the issues that will sink the paper at NeurIPS, ranked by severity.

Abstract — Issues
Severity: High. Needs rewrite once headline result is locked, but structural problems are fixable now.

Buries the lede. Opens with a generic threat-model paragraph that any deepfake paper opens with. By sentence three a reviewer is still waiting for the contribution. The differentiator (IBDC, structural divergence, calibrated deferral) appears only after 100+ words of throat-clearing.
"Outperforms on several, matches on the remainder" is the weakest possible accuracy claim. It signals you don't have a clean accuracy win and are hedging. If the headline is going to be (A) selective-prediction or (C) calibration, the abstract should commit to that as the win and treat AUC as parity, not as a soft win.
IBDC — the strongest novelty — is described as "inter-branch disagreement directly inflates uncertainty." That's mechanistic but strips the technical claim. The novel binding (uncertainty bound to detached pairwise disagreement, not evidence magnitude) is what differentiates from EDL/TMC. Currently invisible.
"Causal" appears unscoped. "Label-conditioned structural causal models … detects manipulations from the structural divergence between the real and fake-conditioned mechanisms" — better than most uses, but the abstract still leans on "causal" as a virtue. Per the Hybrid framing decision, this needs to read as structural divergence with the SCM term used precisely (NOTEARS/DAGMA).
Symbolic stream underclaimed. "Differentiable predicates derived from FACS" undersells the LibreFace-validated 58-dim substrate and the discriminative-gap predicate selection protocol. These are the two things that defang the "decorative symbolic component" objection.
No headline number. Abstract needs one quantitative claim that lands. Currently zero numbers.
"Opacity undermines both trust and generalization." The trust half is fine; the generalization half is unsupported and reviewers will challenge it. Cut.

Hold on rewriting the abstract until headline result is decided. I'll draft three abstracts (one per option) when we get to that decision point — per project instructions.

Introduction — Issues
Severity: Medium-High. Workable but bloated; central insight doesn't land hard enough or early enough.

Paragraph 1 is generic. Four sentences establishing "deepfakes are bad and generalization matters" — every reviewer has read this paragraph 200 times. Cut to two sentences. The page-budget argument: this paragraph is ~⅛ of the intro real estate spent on consensus material.
Paragraph 2 is well-structured but the gap-claim is the weak version. The current framing — "every method in this family compresses its decision into a single scalar fakeness score" — is the aesthetic claim flagged in the project instructions. Every classifier ever made does this. The locked sharpened version needs to land here:

Cross-dataset CLIP-adapter detectors fail silently and indistinguishably from confident-correct predictions because their uncertainty is uncalibrated and their evidence is undecomposable; an analyst has no surface on which to audit, contest, or defer.

Currently the intro hints at this in paragraph 2 final sentences but doesn't commit. It must commit, and the commitment must precede the contribution list.
Paragraph 3 — "common cause" framing is good but mistakes the punchline. It correctly identifies that detectors don't reason over semantic regularities, then immediately introduces three streams. The technical insight that should land here — the binding mechanism that ties uncertainty to disagreement — is buried as a clause at the end. IBDC is the strongest single novelty claim per the project framing and should be named in the intro.
Contribution (1) is overclaimed and over-hedged simultaneously. "First interpretable neuro-symbolic deepfake detector" is the kind of "first" claim that invites a reviewer to find a prior work and reject. Drop "first" framing; lead with what the contribution does — emits per-prediction predicate violations and inspectable class-conditioned structural-graph divergences as the explanation, not as post-hoc visualization. "To our knowledge" is on the project's banned-phrases list.
Contribution (2) is the strongest but currently second. IBDC (uncertainty bound to detached disagreement, not evidence magnitude) should be contribution (1) once headline is locked, especially under option (A).
Contribution (3) is the experimental claim, not a contribution. Convert to a results-preview sentence at the end of the intro and recover the bullet for IBDC or the predicate-validation methodology.
Three contributions, not four. Project instructions specify 3–4 bullets each tied to an experiment. Currently three but the third is results, not contribution. Net: two real contributions in the bullet list. Need to re-distribute.
No FACS-validated substrate surfacing. The intro says "predicates derived from FACS" but never names LibreFace, DISFA, BP4D, or the 58-dim substrate. Defense against the "decorative symbolic" objection requires this pedigree visible in the intro.
No causal scoping. "Causal reasoning stream … learns label-conditioned structural causal models" appears without the §3.2 disclaimer existing yet. Per the Hybrid framing lock, every appearance of "causal" before §3.2 must either be scoped or be replaced with "structural" / "class-conditioned structural." Currently the intro uses "causal" four times, all unscoped.
Page budget. Current intro is ~1.0 page — at budget — but if we're adding the IBDC technical claim and predicate-validation sentence, something has to come out. Candidates: (a) cut paragraph 1 to two sentences (~3 lines saved), (b) collapse the two-strategy taxonomy in paragraph 2 into one sentence with citations (the related work covers it). Net: room for the sharper insight without overrun.


Literature Review — Issues
Severity: Medium. Functional but underweights the positioning that protects the contribution.

CLIP-adapter wave under-positioned. Paragraph 2 lists six adapter methods in one breath and treats them as undifferentiated. The project instructions name Forensics Adapter [Cui 2025] and Yermakov 2026 as primary positioning anchors. These two need a sentence each that names the specific design (bottleneck adapters; LayerNorm-only) and exactly what they leave on the table (no uncertainty, no decomposition, no deferral). One-sentence treatment for the rest is fine.
The closing sentence of paragraph 2 is the right gap claim but undersells. "DeFakeNet retains the generalization benefit … restructuring the prediction itself as a composition of inspectable evidence streams." Add a clause naming the calibrated deferral and inter-branch disagreement binding — these are what no adapter method can offer regardless of post-hoc add-ons.
Neuro-symbolic paragraph cites Garcez 2023 and DeepProbLog but doesn't position DeFakeNet within that family. Reviewer-2 will write "what makes this neuro-symbolic beyond a few hand-coded rules?" The differentiable-predicate + learned-fusion family (NS-CL, Semantic Loss, DeepProbLog) needs a one-sentence acknowledgment that DeFakeNet sits in this family with predicates empirically validated via discriminative-gap selection on training-fold data. This pre-empts the "decorative symbolic" objection in the lit-review rather than deferring it to method.
Causal paragraph conflates causal representation learning with NOTEARS/DAGMA. Schölkopf 2021 is about identifying invariant mechanisms — a Pearlian commitment we explicitly don't make. NOTEARS/DAGMA is structural-equation discovery without identifiability claims. Per the Hybrid framing, the lit-review should cite NOTEARS/DAGMA as the technical lineage and cite Schölkopf 2021 only as motivation that distribution shift is real. As currently written, a careful reviewer will flag the same conflation the §3.2 disclaimer tries to fix.
Missing citations from the locked positioning anchors. Yermakov 2026 (LayerNorm-only) is cited in the deepfake-detection paragraph but should be cited again in the foundation-model paragraph as the closest baseline. Forensics Adapter [Cui 2025] is correctly cited.
EDL paragraph is commented out but should arguably be in. The IBDC contribution differentiates against Sensoy 2018 and Han 2021 specifically. Without an EDL paragraph in the lit-review, the differentiation lives only in §3.2 of method, which is too late. Recommend uncommenting a tightened version (~3 sentences) and finding the space by collapsing the deepfake-detection paragraph. This trade is the right one — the EDL positioning protects the strongest novelty claim.
Page budget. Currently ~0.5–0.75 page. If we add a tight EDL paragraph (3 sentences) and tighten the adapter and neuro-symbolic positioning, we stay within budget by also collapsing the pseudo-fake list (Face X-ray, SBI, FreqBlender, ProDet) into one citation cluster — the lit-review doesn't need to enumerate them.


Summary: What to fix before iteration 2
Abstract: hold pending headline-result decision.
Introduction:

Compress paragraph 1.
Replace "scalar score" gap claim with the locked "fail silently and indistinguishably" framing.
Surface IBDC in paragraph 3 as the technical insight.
Rebuild contribution bullets: (1) IBDC binding, (2) class-conditioned SCM-pair structural divergence, (3) FACS-validated substrate + predicate validation methodology + interpretability-by-design — with "first" and "to our knowledge" cut.
Convert contribution (3) → results-preview sentence.
Scope every "causal" mention or replace with "structural / class-conditioned structural."
Surface LibreFace + FACS substrate pedigree.

Literature Review:

Differentiate Forensics Adapter and Yermakov 2026 individually; one-sentence-each for the rest of the wave.
Position DeFakeNet within the differentiable-predicate neuro-symbolic family; mention discriminative-gap validation.
Separate Schölkopf-style causal representation learning (motivation) from NOTEARS/DAGMA (technical lineage); cite them for what they each are.
Uncomment a tight EDL paragraph (3 sentences) for IBDC differentiation; recover budget by collapsing the pseudo-fake enumeration.

---

## Implementation status update — 2026-05-02

Two pieces of supporting infrastructure landed for the NeurIPS submission. Both are described in the same imperative voice the paper will use; treat the bullet text as draft caption / methods-section material.

### A. Discriminative-gap predicate selection — *what §3.2 of the paper now references*

The hand-picked-12 problem is gone. The symbolic stream is now backed by a **selection protocol with explicit leakage prevention**, exactly as described in §3.2 of the README.

- **Candidate set (28 predicates)** in `training/networks/nesy_defake/semantic/consistency_rules_v8.py`. Composition: 12 inherited from v7, 4 faceswap-specific predicates from the April-28 README (`cr_yaw_gaze_misalign`, `cr_pose_facewidth`, `cr_lip_geometry_au12`, `cr_brow_eye_couple`), and 12 FACS-grounded additions covering negative-emotion AU coherence, pitch / roll-gaze, geometric ratios, mouth symmetry, and Duchenne / genuine-expression AU pairs. All predicates are differentiable, output ∈ [0, 1], operate on the 58-d fast-semantic substrate, and have one-line FACS or geometric justifications in their docstrings.
- **Validation script** `scripts/validate_predicates.py`. Loads only FF++ training-fold features; deterministic 90 / 10 *video-level, per-class* split with `seed=42`; computes per-predicate gap, single-predicate AUC, mean / std / sample counts, and sanity flags (NaN, constant, all-zero) on the 10 % slice. Writes `results/predicate_gaps.csv`, `predicate_gaps_sorted.csv`, `predicate_gap_barplot.png`, `predicate_gap_distribution.png`, and `run_log.txt` (CLI invocation, git hash, manifest sha256, slice statistics, top-/bottom-5).
- **Retained set** `configs/retained_predicates.yaml`. **Top-18 by gap**, justified by an elbow at gap ≈ 0.005 visible in `predicate_gap_distribution.png`; equivalently `gap > 0.005`. The decision is documented in `results/predicate_selection_decision.md`. Of the four README §5 candidates, three were retained (`cr_brow_eye_couple` rank 2, `cr_pose_facewidth` rank 12, `cr_lip_geometry_au12` rank 14); `cr_yaw_gaze_misalign` did not survive on FF++ (front-facing pose distribution under-stresses the constraint) and is parked in the candidate module for future cross-pose evaluation.
- **Runtime enforcement.** A `RetainedConsistencyRules` wrapper reads the YAML, indexes the `(B, 28)` candidate output to the 18 retained dims, and registers the indices as a buffer so they ship in `state_dict`. On training start, `train.py` calls `verify_against_yaml()` whenever `concept_branch.consistency_rules_version: v8_retained` is set; tampering with either the YAML or the candidate-name table raises before optimisation begins.
- **Paper assets.** `scripts/render_paper_assets.py` emits `results/tab_predicate_gaps.tex` (booktabs, single-column-friendly, with publication-style predicate labels and category names) and a restyled bar plot (3.4 in width, sans-serif, retained = blue, dropped = hatched grey, gap-threshold dashed line at 0.005, grayscale-readable). Display labels live in the renderer; the YAML and v8 module use the canonical `cr_*` identifiers as ground truth.
- **Leakage statement.** The 10 % validation slice is never seen by the detector during training (training uses only the complementary 90 %); no predicate gap is recomputed on test or cross-dataset data; the retained set is fixed before the first test evaluation runs. State this in §3.2 of the methods and in any reviewer rebuttal.

### B. Ablation evaluation pipeline — *what feeds Tables 4 / 5 / 6*

A minimal runner suite reproduces every number the paper's ablation tables need from existing checkpoints, without re-implementing model loading or re-sampling frames. All metric definitions come verbatim from `scripts/calibration_metrics.py`.

- `scripts/run_ablation_eval.py` — per-ablation wrapper. Subprocess calls `training/test.py`, normalises `test_predictions.csv` to the canonical `(sample_id, video_id, frame_idx, label, prediction, confidence, prob_fake)` schema via `calibration_metrics.predictions_to_per_sample`, computes frame *and* video-level AUC / ECE / E-AURC / CW@0.9 (the four metrics the paper's Table 4 uses). Writes `results/ablations/<name>/per_sample.csv` and `metrics.json` incrementally. Default config: `nesy_defake_ablation4_causal.yaml`; default test set: `Celeb-DF-v2`.
- `scripts/run_faithfulness.py` — predicate-substrate intervention. Loads the full DeFakeNet checkpoint, iterates CDFv2, restricts to correctly-classified fakes (`label == 1` and `prob_base ≥ 0.5`). For each k ∈ {1, 3, 5}, runs two interventions per sample: zeroing the **top-k firing predicates** and zeroing **k random retained predicates** (deterministic per `(seed, batch, k)`). Records the relative drop in symbolic-stream evidence `Ev_sym = Σ concept_evidence` and the rate of hard-prediction flips. Writes `results/faithfulness/CDFv2.json` after every batch. The intervention point is the new `predicate_mask` kwarg on `ConceptBranch.forward` (see §C below).
- `scripts/run_selective.py` — full DeFakeNet vs. GenD-CLIP at video level on CDFv2. Reads the two per-sample CSVs, aggregates by `video_id` (mean `prob_fake`, mean confidence), reports full-coverage AUC and AUC at 90 % coverage (sort descending by confidence, keep top 90 %, recompute). Writes `results/selective/CDFv2.json`. Reports `UNAVAILABLE` with a clear reason if either CSV is missing rather than fabricating numbers.
- `scripts/aggregate.py` — assembles `results/ABLATION_SUMMARY.md` in the paper's three-table format (Component Ablation / Faithfulness / Selective Prediction) plus a per-config run-log section. Rows for missing artefacts render `—` and are flagged in the run log; no silent fabrication.
- `scripts/run_all.sh` — single-command driver. Invokes the per-ablation evaluator for every entry in an `ABLATIONS=(...)` array (`full_defakenet`, `no_ibdc`, `no_cmef`, `no_pbas`, `no_causal`, `no_symbolic`, `visual_edl_only`), then faithfulness, then selective, then aggregator. Header points to the checkpoint paths the user must update for their layout. Failures are tolerated — each one prints `FAILED: <name>` and the pipeline continues.
- **One command:** `bash scripts/run_all.sh`.

### C. Single supporting code change

Faithfulness needs to zero specific predicates between the rule module and the concept MLP. To stay non-invasive:

- `ConceptBranch.forward(combined_features, predicate_mask=None)` — when `None` the forward is bit-for-bit unchanged from before (default for every existing checkpoint and every existing call site). When a `(B, K)` or `(1, K)` `{0, 1}` tensor is supplied, violations are element-wise multiplied by it before the concat into `concept_input`.
- `NeSyDeFakeHybridDetector.forward` looks up `data_dict.get('predicate_mask')` and threads it into the concept branch.
- The change is strictly additive; no checkpoint key shapes change, no defaults change, and no existing run path takes the new code path.

### D. What the paper now legitimately claims

- The symbolic substrate is **selected, not curated**, with a documented protocol on a held-out FF++ slice (§3.2). The 18 retained predicates are reproducible from the candidate module and the frozen YAML.
- Faithfulness is reported as a **causal intervention on the predicate substrate** (zero specific predicates → measure Ev_sym drop and prediction flips), not as a post-hoc attribution. The intervention is mathematically exact under the existing forward graph.
- The component ablation, faithfulness, and selective-prediction tables are produced from existing checkpoints by a single command, with every metric computed by the verbatim functions in `scripts/calibration_metrics.py`. No bespoke metric reimplementation is hidden in the runner.