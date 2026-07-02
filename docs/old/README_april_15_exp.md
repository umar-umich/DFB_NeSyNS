---

## Explainability & Interpretability for NeurIPS

The NeSyDeFake framework provides **seven distinct levels of interpretability**, spanning from per-prediction uncertainty to fine-grained forensic decomposition. Unlike black-box detectors, every prediction can be accompanied by a human-readable explanation of *why* the model flagged a face as fake and *how confident* it is.

The causal branch comes in two variants — **CCV** (Causal Constraint Verification) and **SCM** (Structural Causal Models) — each offering a different explainability trade-off. This document covers both, with a comparative analysis for publication novelty at the end.

### Level 1: EDL Uncertainty Quantification (Per-Prediction)

Every prediction produces a full Dirichlet distribution, not just a class probability:
```
evidence  = softplus(logits)        → (B, 2) non-negative evidence
alpha     = evidence + 1            → (B, 2) Dirichlet concentration params
S         = sum(alpha)              → (B,)   Dirichlet strength
prob      = alpha / S               → (B, 2) expected class probability
uncertainty = K / S                 → (B,)   epistemic uncertainty ∈ (0, 1]
```
**NeurIPS angle:** Unlike softmax (which can output high confidence for OOD inputs), EDL uncertainty monotonically decreases with evidence. A prediction with uncertainty > 0.5 means the model has *less evidence than a uniform prior* — a principled "I don't know." This is critical for deployment: flag uncertain predictions for human review.

**Visualization:** Uncertainty heatmaps over test sets, uncertainty vs accuracy calibration plots, uncertainty distribution for in-domain (FF++) vs OOD (CelebDF, DFDC) data.

### Level 2: Branch-Level Evidence Decomposition

The three-branch architecture produces independent evidence streams that can be inspected:
```
spatial_evidence  = softplus(classifier(proj(CLIP_features)))     → (B, 2)
concept_evidence  = softplus(concept_mlp(features + rules))       → (B, 2)
causal_evidence   = softplus(ccv_mlp(rules + learned + anomaly + cf)) → (B, 2)
```
**Per-prediction explanation:**
- "Spatial branch contributed 3.2 evidence for fake (CLIP visual features)"
- "Concept branch contributed 1.8 evidence for fake (semantic inconsistencies)"
- "Causal branch contributed 0.9 evidence for fake (forensic anomalies)"

**NeurIPS angle:** This is a *decomposed* prediction — the user can see which reasoning pathway drove the decision. If spatial evidence is high but concept evidence is near zero, the detection is based on visual artifacts, not semantic inconsistencies. If concept evidence is high, the face has detectable attribute contradictions.

### Level 3: 23 Named Consistency Rule Violations (Symbolic Interpretability)

Each of the 23 hand-coded rules produces a named, interpretable violation score in [0, 1]:
```
Expression-AU Coherence (9 rules):
  cr_happy_au6:      |happy_score - AU6_intensity|       "Happy but no cheek raiser"
  cr_happy_au12:     |happy_score - AU12_intensity|      "Happy but no lip corner puller"
  cr_surprise_au1au2:|surprise - mean(AU1, AU2)|         "Surprised but no brow raise"
  cr_sad_au15:       |sad - AU15|                        "Sad but no lip corner depressor"
  cr_angry_au4:      |angry - AU4|                       "Angry but no brow lowerer"
  cr_fear_au1au5:    |fear - mean(AU1, AU5)|             "Fearful but no upper lid raise"
  cr_disgust_au9:    |disgust - AU9|                     "Disgusted but no nose wrinkle"
  cr_contempt_au14:  |contempt - AU14|                   "Contemptuous but no dimpler"
  cr_neutral_any_au: neutral × max(key_AUs)              "Neutral face with active AUs"

Cross-Region Identity (5 rules):
  cr_gender_beard:      female_score × beard             "Female face with beard"
  cr_gender_makeup:     male_score × heavy_makeup        "Male face with heavy makeup"
  cr_hair_age:          gray_hair × (1 - age)            "Gray hair but young face"
  cr_bald_longhair:     bald × long_hair                 "Bald with long hair"
  cr_skin_tone_conflict:fair_skin × dark_skin            "Fair and dark skin simultaneously"

Skin/Age (2), Pose-Gaze (1), Landmark Symmetry (2), Image Quality (1):
  cr_smooth_age:        smooth_skin × high_age           "Smooth skin but elderly"
  cr_wrinkle_age:       wrinkled_skin × low_age          "Wrinkled but young"
  cr_gaze_lr_divergence:|left_gaze_x - right_gaze_x|    "Eyes looking in different directions"
  cr_eye_asymmetry:     eye_lr_symmetry                  "Left-right eye shape mismatch"
  cr_jaw_asymmetry:     jaw_symmetry                     "Jaw contour asymmetry"
  cr_image_quality:     blurry × sharp                   "Both blurry and sharp"
```
**NeurIPS angle:** These are *domain-grounded* symbolic rules that encode FACS (Facial Action Coding System) expert knowledge. Each rule has a clear human interpretation. A deepfake with mismatched expression-AU patterns can be explained: "This face shows a happy expression (score 0.82) but AU12 (lip corner puller) activation is only 0.11 — real happy faces always have strong AU12." This is the kind of explanation a forensic analyst can verify.

### Level 4: Causal Branch — Two Variant Architectures

The framework supports two causal branch variants, each with distinct interpretability strengths:

---

#### Level 4-CCV: Causal Constraint Verification (type: ccv, ~15K params)

The CCV branch decomposes its evidence into three independently meaningful signals:

##### 4-CCV-a. Learned Constraint Violations (K=16 features)
```python
class LearnedConstraintFunctions:
    # Shared trunk + K sigmoid heads
    # combined_features (B, 122) → LayerNorm → Linear(122→48) → GELU → Linear(48→16) → sigmoid
    # Output: (B, 16) violation scores in [0, 1]
```
These complement the 23 hand-coded rules by discovering *data-driven* attribute interactions that domain experts missed. Post-hoc analysis: identify which input features each learned constraint is most sensitive to (gradient-based feature importance per constraint head).

##### 4-CCV-b. Forensic Anomaly Scores (5 semantically grouped scores)
```
Per-group autoencoder reconstruction error:
  boundary_texture (12 features → bottleneck 8 → reconstruction error)
  symmetry_color   (18 features → bottleneck 8 → reconstruction error)
  patch_noise      (16 features → bottleneck 8 → reconstruction error)
  srm_noise        (27 features → bottleneck 8 → reconstruction error)
  fft_spectral     (10 features → bottleneck 8 → reconstruction error)
```
**Per-prediction explanation:**
- "Forensic anomaly highest in `patch_noise` group (score 2.3) — patch-level noise consistency is disrupted"
- "Anomaly lowest in `fft_spectral` (score 0.1) — frequency domain looks authentic"

**NeurIPS angle:** Real faces have consistent forensic patterns within each group; fakes break this consistency. The per-group decomposition tells you *where* the forensic artifacts are: boundary blending? noise injection? frequency domain? This maps directly to known deepfake generation artifacts (e.g., face swapping creates boundary anomalies, GANs create spectral artifacts).

##### 4-CCV-c. Counterfactual Mismatch (1 score)
```python
class CounterfactualPredictor:
    # "If a face truly had these attributes, what would CLIP see?"
    # semantic_summary (B, 228) → MLP → predicted_z (B, 32)
    # actual_z = compress(spatial_raw.detach()) (B, 32)
    # mismatch = ||predicted_z - actual_z||^2 → (B, 1)
```
**Per-prediction explanation:**
- "Counterfactual mismatch = 4.7 — the face's semantic attributes predict very different CLIP features than what was actually observed. The appearance doesn't match the attributes."

**NeurIPS angle:** This is *counterfactual reasoning*: "if this face were real and truly had these detected attributes (age, expression, pose), its visual representation would look like X — but it actually looks like Y." High mismatch = the visual appearance is inconsistent with the detected attributes, a hallmark of manipulation.

---

#### Level 4-SCM: Structural Causal Models (type: improved_scm, ~180K params)

The SCM branch learns *explicit causal graph structure* — adjacency matrices that encode directional dependencies between facial attributes, forensic signals, and latent CLIP features. This is fundamentally different from CCV: instead of checking constraint violations, the SCM learns *how variables causally influence each other* and detects fakes via *structural graph divergence* between real and fake causal mechanisms.

##### 4-SCM-a. Four Learned Sub-Graphs (8 SCMs total: 4 sub-graphs x real/fake)
```
Each sub-graph pair learns separate nonlinear SCMs for real vs fake data:

Identity sub-graph (106 nodes):
  z_causal(32) + curated_attrs(51) + rules(23)
  Real SCM: learns how attributes causally relate in authentic faces
  Fake SCM: learns how these relationships break in manipulated faces

Forensic-structural sub-graph (62 nodes):
  z_causal(32) + forensic_structural(30) [boundary, blur, symmetry, color, DCT, quality]

Forensic-noise sub-graph (75 nodes):
  z_causal(32) + forensic_noise(43) [PPNC, CCNC, SRM, multi-scale noise]

Forensic-spectral sub-graph (42 nodes):
  z_causal(32) + forensic_spectral(10) [FFT features]
```
Each `NonlinearSCM` is a 1-hidden-layer MLP (d → 64 → d, SiLU) with NOTEARS-MLP adjacency approximation: A ≈ |W₂| @ |W₁|. The adjacency matrix A is a (d x d) matrix where A[i,j] > 0 means "node i causally influences node j."

**Per-prediction explanation:**
- "Identity sub-graph: real SCM residual = 0.3, fake SCM residual = 2.1 — the face's attribute relationships match real causal structure, not fake"
- "Forensic-noise sub-graph: differential residual is high — noise patterns follow fake causal mechanisms"

##### 4-SCM-b. Differential Causal Residuals (Per Sub-Graph)
```python
class SubGraphPair:
    # For each sub-graph g:
    #   r_real = ||x - SCM_real(x)||^2    # reconstruction under real causal model
    #   r_fake = ||x - SCM_fake(x)||^2    # reconstruction under fake causal model
    #   r_diff = r_fake - r_real           # differential residual
    #   summary_g = Linear(d, 8)(r_diff)   # projected to 8-d summary
    # Evidence input = concat([summary_1, ..., summary_4]) → 32-d
```
The differential residual is the core interpretability signal: a sample that fits the real SCM better than the fake SCM will have r_diff > 0 (evidence for real). The per-sub-graph decomposition tells you *which causal domain* is discriminative for each sample.

##### 4-SCM-c. Learned Adjacency Matrices (Visualizable Causal Graphs)
```
Each of the 8 SCMs produces a weighted adjacency matrix:
  A_identity_real:  (106 x 106) — causal structure of real identity attributes
  A_identity_fake:  (106 x 106) — causal structure of fake identity attributes
  A_forensic_struct_real: (62 x 62) — real forensic structural patterns
  A_forensic_struct_fake: (62 x 62) — fake forensic structural patterns
  ... (4 more for noise and spectral sub-graphs)
```
**Per-prediction explanation:**
- "In real faces, AU12 (lip corner puller) → happy_score has weight 0.82. In fakes, this edge weight drops to 0.21 — the generator does not maintain FACS-consistent causal pathways."
- "In real faces, boundary_gradient → blur_score has a strong causal link. In fakes, this link is weak because GAN blending creates boundary artifacts that break the natural causal relationship."

**NeurIPS angle (strongest novelty):** These are *learned, differentiable causal graphs* with named nodes. No prior deepfake detection work learns separate real/fake SCMs and uses their *structural divergence* as a detection signal. The adjacency matrices are directly visualizable as causal DAGs — a reviewer can inspect the graph and verify that the learned causal relationships are forensically meaningful. This goes beyond feature attribution: it provides a *causal mechanism explanation* of what distinguishes real from fake.

##### 4-SCM-d. DAG Acyclicity Constraint (Principled Causal Structure)
```
DAGMA constraint: h(A) = -log det(sI - A∘A) + d·log(s)
h(A) = 0  ⟺  A is a DAG (directed acyclic graph)
```
The SCM enforces that learned graphs are valid DAGs — variables cannot cause themselves through cycles. This is not just a regularizer; it's a *structural guarantee* that the learned relationships are causally interpretable (not just correlational).

**Additional SCM training losses:**
- **Graph divergence**: −L1(A_real, A_fake) — *maximizes* structural difference between real/fake graphs, forcing the model to discover where real and fake causal mechanisms truly differ
- **Label-conditioned reconstruction**: real SCM is trained only on real samples, fake SCM only on fake — each SCM learns class-specific causal structure
- **Sparsity**: L1 on adjacency matrices to keep graphs human-readable

##### 4-SCM-e. Evidence Production
```python
# 4 sub-graph summaries (each 8-d) → 32-d vector
evidence_input = concat([summary_identity, summary_struct, summary_noise, summary_spectral])
# LayerNorm → Linear(32, 64) → GELU → Linear(64, 2) → softplus
causal_evidence = softplus(evidence_mlp(evidence_input))
```

### Level 5: Evidence Gate Analysis (Per-Image Trust Allocation)

#### Static Gates (nesy_fusion: false)
```
concept_gate = σ(learnable_scalar)  → single value for all images
causal_gate  = σ(learnable_scalar)  → single value for all images
```
Shows the model's learned global trust: "concept branch contributes ~38% weight, causal branch ~31%"

#### Feature-Conditioned Gates (evidence_gate.conditioned: true)
```python
class FeatureConditionedGate:
    # Linear(1024 → 2) → Sigmoid
    # Input: spatial_raw (CLIP features)
    # Output: per-image [concept_weight, causal_weight] ∈ (0, 1)
```
**Per-prediction explanation:**
- "For this image: concept_gate=0.72, causal_gate=0.15 — model trusts semantic rules heavily for this clear frontal face, less trust in causal forensics"
- "For this image: concept_gate=0.18, causal_gate=0.68 — model trusts forensic analysis for this compressed/degraded image where semantic extraction is unreliable"

**NeurIPS angle:** The gates reveal the model's *reasoning strategy per image*. This can be aggregated across datasets: e.g., "on CelebDF (OOD), the model shifts trust toward symbolic branches" — demonstrating that the neuro-symbolic architecture enables principled domain adaptation.

#### CMEF Confidence Modulation (nesy_fusion: true)
```
phi(S_branch) = sigmoid((S_branch - K) / tau)
modulated_weight = sigma(gate) × phi(S_branch)
```
Each symbolic branch's contribution is further modulated by its own Dirichlet confidence. A branch that is uncertain about a specific sample automatically contributes less evidence.

### Level 6: Inter-Branch Disagreement as Calibrated Uncertainty (IBDC)

```
disagreement = 1 - cosine_similarity(p_neural, p_symbolic)
L_bdc = -d × log(u) - (1-d) × log(1-u)
```
**Visualization:** Plot branch agreement vs prediction uncertainty. When neural (CLIP) says "real" but symbolic (concept+causal) says "fake," the framework produces high uncertainty — flagging the prediction as unreliable.

**NeurIPS angle:** This is *calibrated* uncertainty that reflects *reasoning conflict*, not just low evidence. Standard EDL can be confidently wrong when evidence is high but homogeneous. IBDC ensures that conflicting evidence sources produce appropriately high uncertainty. This is a novel contribution.

### Level 7: Forensic Feature Attribution

The 83-d forensic feature vector has named, semantically grouped features:
```
Indices 0-11:  Boundary/texture gradients (ff_grad_*, ff_blur_*)
Indices 12-29: Symmetry, color, DCT, quality (ff_sym_*, ff_color_*, ff_dct_*, ff_quality_*)
Indices 30-37: Patch-level noise consistency (ff_ppnc_*)
Indices 38-45: Cross-channel noise consistency (ff_ccnc_*)
Indices 46-60: Steganalysis rich model filters (ff_srm_*)
Indices 61-72: Multi-scale noise analysis (ff_noise_*)
Indices 73-82: FFT spectral features (ff_fft_*)
```
Each feature has a known forensic interpretation. Combined with the ForensicAnomalyDetector's per-group scores, the model can explain *which specific forensic signals* are anomalous.

---

### CCV vs SCM: Comparative Analysis for Publication Novelty

#### Head-to-Head Comparison

| Dimension | CCV (Constraint Verification) | SCM (Structural Causal Models) |
|-----------|-------------------------------|--------------------------------|
| **Core idea** | Check if facial attributes violate known/learned constraints | Learn directed causal graphs, detect via structural divergence |
| **Parameters** | ~15K (lightweight) | ~180K (12x heavier) |
| **Graph structure** | None — flat constraint checks | 8 explicit adjacency matrices (DAGs) |
| **Interpretability type** | Feature-level: "which constraints fired" | Mechanism-level: "how causal relationships differ" |
| **Counterfactual** | Semantic→CLIP mismatch (single scalar) | Full differential residuals per sub-graph |
| **Training complexity** | Standard backprop, no auxiliary losses | DAG acyclicity + sparsity + divergence + label-conditioned reconstruction |
| **Visualization** | Bar charts of violations/anomalies | Graph diagrams with named edges and weights |
| **Novelty risk** | Constraint checking is well-studied (logic-based AI) | Learned per-class SCMs for forensics is novel but harder to stabilize |
| **Failure mode** | Constraints may not capture all manipulation types | DAG penalty can explode; graph divergence can cause adversarial dynamics |

#### Publication Novelty Assessment

**CCV strengths for a paper:**
- Clean, easy-to-explain story: "hand-coded rules + learned constraints + forensic anomalies + counterfactual reasoning"
- Each sub-component has a clear, independent interpretation
- Lightweight — easy to ablate, fast to train, reproducible
- Counterfactual predictor ("if this face were real, CLIP would see X") is a compelling narrative
- Maps well to the NeSy (neuro-symbolic) framing: symbolic rules + neural constraints
- **Risk**: Reviewers may see learned constraints as "just another MLP" and counterfactual as "just a reconstruction error." The novelty is in the composition, not the individual components.

**SCM strengths for a paper:**
- **Strongest novelty**: No prior deepfake work learns separate real/fake SCMs with DAG-enforced structure. This is a genuinely new contribution.
- Adjacency matrices are *inspectable artifacts* — a reviewer can look at the graph and see "AU12 → happy has weight 0.82 in real, 0.21 in fake"
- Graph divergence loss is a *principled objective*: the model is explicitly trained to discover where real and fake causal mechanisms differ
- Maps to causal inference literature (Pearl, DAGMA, NOTEARS) — positions the work at the intersection of causal ML and forensics
- Label-conditioned reconstruction is a form of *causal disentanglement*: real SCM sees only real data, fake SCM sees only fake data
- Four semantically meaningful sub-graphs (identity, structural forensics, noise forensics, spectral forensics) provide domain-grounded decomposition
- **Risk**: Training instability (DAG penalty explosion, graph divergence adversarial dynamics). Heavier model, harder to reproduce. Reviewers may question whether the learned graphs truly capture causal structure vs. just fitting correlations under a DAG constraint.

#### Recommended Strategy

**For maximum novelty (NeurIPS):** Lead with SCM. The learned causal graphs are the differentiating contribution — no one else does this for deepfake detection. Frame CCV as an ablation baseline ("we also evaluate a lightweight constraint-verification variant that trades causal graph interpretability for training efficiency").

**Ablation table structure:**
| Ablation | Causal Branch | Key Interpretability |
|----------|---------------|---------------------|
| Abl 3 | None (spatial + concept only) | Rules + EDL uncertainty |
| Abl 4-CCV | Constraint Verification | + learned constraints, anomaly decomposition, counterfactual |
| Abl 4-SCM | Structural Causal Models | + learned DAGs, causal divergence, per-class mechanisms |

**Key experiments unique to SCM:**
1. **Graph visualization**: Show A_identity_real vs A_identity_fake as heatmaps/DAG diagrams. Highlight edges that exist in one but not the other (causal mechanisms specific to manipulation).
2. **Edge-level forensic analysis**: For correctly detected fakes, which edges have the largest real-vs-fake weight difference? Do these correspond to known manipulation artifacts?
3. **Graph divergence correlation**: Does L1(A_real, A_fake) correlate with detection confidence? Higher divergence should mean easier detection.
4. **Sub-graph specificity**: Do different deepfake methods (F2F, DF, FS, NT) produce different graph divergence profiles across the 4 sub-graphs? Expected: face swaps break forensic-structural, reenactment breaks identity, GANs break spectral.
5. **DAG sparsity analysis**: How sparse are the learned graphs? Sparser = more interpretable. Track sparsity over training — does the model discover a compact set of key causal edges?

**Key figures unique to SCM:**
1. **Causal DAG comparison**: Side-by-side DAG visualization of identity sub-graph for real vs fake, with node names (AU6, happy, age, beard...) and edge weights
2. **Edge divergence heatmap**: |A_real - A_fake| for each sub-graph, sorted by magnitude, with forensic interpretation of top divergent edges
3. **Per-method graph fingerprint**: Radar chart of sub-graph divergence magnitudes for each deepfake method — showing that each method has a distinct "causal fingerprint"
4. **Training dynamics**: Plot DAG acyclicity h(A), graph sparsity, and graph divergence over epochs — showing that the model progressively discovers structured, sparse, divergent causal graphs

### Suggested Presentation for NeurIPS Paper

#### Quantitative Explainability Experiments (Shared)
1. **Uncertainty calibration:** Expected Calibration Error (ECE) on in-domain vs OOD test sets; show that NeSy-EDL with IBDC produces better-calibrated uncertainty than standard EDL or softmax
2. **Branch ablation as explanation validation:** Disable concept/causal branches → which predictions change? If a prediction was "explained by" concept evidence, removing the concept branch should flip it
3. **Rule violation statistics:** For correctly detected fakes, which consistency rules fire most? Expected: expression-AU rules for expression-based fakes (F2F), boundary rules for face swaps (FS, DF)
4. **Gate adaptation on OOD data:** Show that feature-conditioned gates shift trust toward symbolic branches on unseen datasets (cross-dataset generalization through interpretable trust allocation)

#### Quantitative Explainability Experiments (CCV-specific)
5. **Forensic group specificity:** Per-manipulation-method forensic anomaly profiles (radar charts) showing that different deepfake methods break different forensic groups
6. **Learned constraint sensitivity:** Gradient-based feature importance per learned constraint head — which input features does each of the K=16 constraints rely on?
7. **Counterfactual mismatch distribution:** Compare mismatch scores for real vs fake; show separation and correlation with detection confidence

#### Quantitative Explainability Experiments (SCM-specific)
8. **Causal graph divergence:** Compute L1(A_real, A_fake) per sub-graph per method; show that different methods produce distinct divergence profiles
9. **Edge-level analysis:** Rank edges by |A_real[i,j] - A_fake[i,j]|; report top-k edges with forensic interpretation. Do these correspond to known manipulation signatures?
10. **DAG quality metrics:** Track h(A) (acyclicity), L1 sparsity, and number of edges > threshold over training. Show convergence to sparse, valid DAGs.
11. **Sub-graph ablation:** Remove individual sub-graphs (identity, structural, noise, spectral) and measure AUC drop. Which causal domain contributes most per method?
12. **Label-conditioned fit:** Compare r_real on real samples vs r_real on fake samples (and vice versa). The real SCM should reconstruct real samples better — measure this gap as a function of training epoch.

#### Qualitative Examples
1. **Case study: correctly detected + well-explained** — decomposed evidence, top violated rules, forensic anomaly profile, counterfactual mismatch
2. **Case study: high uncertainty → abstain** — branches disagree, IBDC produces high uncertainty, flagged for human review
3. **Case study: failure mode** — model confident but wrong, analysis of which branch(es) failed and why

#### Figures (Shared)
1. **Architecture diagram** with evidence flow arrows and named intermediate outputs (two variants: CCV path and SCM path)
2. **Evidence decomposition bar chart** per prediction (spatial/concept/causal stacked)
3. **Consistency rule heatmap** across test set (rules x samples, color = violation magnitude)
4. **Uncertainty reliability diagram** (ECE plot)
5. **Gate distribution violin plots** across in-domain vs OOD datasets

#### Figures (CCV-specific)
6. **Forensic anomaly radar chart** per deepfake method (5 axes = 5 forensic groups)
7. **Counterfactual mismatch scatter** — mismatch score vs prediction confidence, colored by real/fake

#### Figures (SCM-specific)
8. **Causal DAG comparison** — side-by-side graph visualization of identity sub-graph: A_real vs A_fake with node labels (AU6, happy, age, ...) and edge widths proportional to weight
9. **Edge divergence heatmap** — |A_real - A_fake| for each of the 4 sub-graphs, sorted by magnitude, top-k labeled with forensic interpretation
10. **Per-method causal fingerprint** — radar chart with 4 axes (identity, structural, noise, spectral divergence) per deepfake method
11. **Training dynamics plot** — DAG acyclicity h(A), graph sparsity (% edges > 0.01), and mean divergence L1(A_real, A_fake) over epochs
12. **CCV vs SCM ablation table** — AUC, ECE, and interpretability metrics side by side across FF++ and CelebDF

---

### Implementation: `training/interpretability/` Package

The interpretability levels above are implemented as a modular Python package
that integrates into the eval loop of the trainer. It collects intermediate
outputs per batch, produces aggregate analysis, generates visualizations, and
writes per-sample human-readable explanations.

#### Package Layout

```
training/interpretability/
  __init__.py                     — exports InterpretabilityEngine
  engine.py                       — orchestrator (collect batches, finalize)
  visualization.py                — shared matplotlib plotting (Agg backend)
  analyzers/
    __init__.py                   — re-exports all analyzer classes
    base.py                       — BaseAnalyzer ABC (collect / analyze / visualize / explain_sample)
    edl_uncertainty.py            — Level 1: ECE, uncertainty histograms, reliability diagram
    branch_evidence.py            — Level 2: per-branch contribution ratios, stacked bars
    consistency_rules.py          — Level 3: top violated rules with FACS names, firing rates
    ccv_analysis.py               — Level 4-CCV: learned constraints, forensic radar, CF mismatch
    scm_analysis.py               — Level 4-SCM: adjacency divergence, top edges, DAG sparsity
    gate_analysis.py              — Level 5: gate distributions (static or conditioned)
    disagreement.py               — Level 6/7: inter-branch disagreement vs uncertainty
```

#### Integration Points

**Detector** (`training/detectors/nesy_defake_detector.py`):
CCV-specific keys (`violation_scores`, `anomaly_scores`, `counterfactual_residual`)
are now propagated from `causal_out` to the top-level prediction dict alongside
the existing `A_*` adjacency matrices.

**Trainer** (`training/trainer/trainer.py`):
- `test_one_dataset()` accepts an optional `interp_engine` parameter. When
  provided, each batch's prediction dict is fed to the engine via
  `collect_batch()`.
- `test_epoch()` creates an `InterpretabilityEngine` per test dataset when
  `interpretability.enabled: true` in config. After testing, it calls
  `collect_model_params(model)` (for SCM adjacency matrices) and
  `finalize(save_dir)` to produce all outputs.
- Interpretability runs every `graph_viz_every_n_epochs` epochs (default 5).

#### Config

The config YAML controls which levels are active:

```yaml
interpretability:
  enabled: true                    # master switch
  levels:
    edl_uncertainty: true          # Level 1
    branch_evidence: true          # Level 2
    consistency_rules: true        # Level 3
    ccv_analysis: true             # Level 4-CCV (auto-skipped if type != ccv)
    scm_analysis: true             # Level 4-SCM (auto-skipped if type != improved_scm)
    gate_analysis: true            # Level 5
    disagreement: true             # Level 6/7
  num_explain_samples: 10          # top uncertain + top confident for text report
  graph_viz_every_n_epochs: 5
  graph_viz_top_k: 20              # top-K divergent edges for SCM tables
```

#### Output Structure

All outputs are saved under `{log_dir}/interpretability/epoch_{N}/{dataset}/`:

```
summary.json                      — full results dict (all analyzers)
report.txt                        — human-readable text report with sample explanations
uncertainty_histogram.png         — real vs fake uncertainty distribution
reliability_diagram.png           — ECE calibration plot
uncertainty_vs_confidence.png     — scatter: P(fake) vs uncertainty, colored by class
branch_evidence_ratios.png        — stacked bar of spatial/concept/causal contribution
branch_evidence_by_class.png      — side-by-side real vs fake evidence decomposition
rule_firing_rates.png             — grouped bar chart of 20 rule firing rates by class
concept_gate_distribution.png     — gate value histogram
causal_gate_distribution.png      — gate value histogram
disagreement_rates.png            — inter-branch disagreement bar chart
uncertainty_by_agreement.png      — uncertainty conditioned on branch agreement

# CCV-specific (when causal_branch.type = ccv):
ccv_forensic_radar.png            — 5-axis radar of forensic group anomalies
ccv_learned_constraints.png       — K=16 learned constraint activations by class
ccv_counterfactual_hist.png       — counterfactual mismatch distribution

# SCM-specific (when causal_branch.type = improved_scm):
scm/
  identity_real_vs_fake.png       — side-by-side adjacency heatmaps (106x106)
  identity_divergence.png         — |A_real - A_fake| heatmap
  identity_top_edges.png + .txt   — ranked table of top-K divergent edges
  forensic_structural_*.png/txt   — same for 62-node structural sub-graph
  forensic_noise_*.png/txt        — same for 75-node noise sub-graph
  forensic_spectral_*.png/txt     — same for 48-node spectral sub-graph
  subgraph_divergence.png         — bar chart of L1(A_real, A_fake) per sub-graph
```

#### Per-Sample Explanation Format

The text report includes multi-level explanations for the most uncertain and
most confident predictions:

```
  Sample 42:
    [EDL] Predicted FAKE (prob=0.912), uncertainty=0.087 (low), alpha=[1.19, 12.81]
    [Evidence] spatial=8.23 (61%), concept=3.41 (25%), causal=1.87 (14%)
    [Rules] Top violations: cr_happy_au12=0.831, cr_symmetry_conflict=0.672
    [CCV] top anomaly: patch_noise=2.31, top learned constraint: C7=0.89, counterfactual mismatch=4.71
    [Gates] concept_gate=0.720, causal_gate=0.150 -- concept branch dominant
    [Disagreement] spatial=FAKE, concept=FAKE, causal=FAKE -- AGREE, uncertainty=0.087
```
