---

## Explainability & Interpretability for NeurIPS

The NeSyDeFake framework provides **seven distinct levels of interpretability**, spanning from per-prediction uncertainty to fine-grained forensic decomposition. Unlike black-box detectors, every prediction can be accompanied by a human-readable explanation of *why* the model flagged a face as fake and *how confident* it is.

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

### Level 4: CCV Causal Branch — Three Interpretable Sub-Components

The CCV branch decomposes its evidence into three independently meaningful signals:

#### 4a. Learned Constraint Violations (K=16 features)
```python
class LearnedConstraintFunctions:
    # Shared trunk + K sigmoid heads
    # combined_features (B, 122) → LayerNorm → Linear(122→48) → GELU → Linear(48→16) → sigmoid
    # Output: (B, 16) violation scores in [0, 1]
```
These complement the 23 hand-coded rules by discovering *data-driven* attribute interactions that domain experts missed. Post-hoc analysis: identify which input features each learned constraint is most sensitive to (gradient-based feature importance per constraint head).

#### 4b. Forensic Anomaly Scores (5 semantically grouped scores)
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

#### 4c. Counterfactual Mismatch (1 score)
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

### Suggested Presentation for NeurIPS Paper

#### Quantitative Explainability Experiments
1. **Uncertainty calibration:** Expected Calibration Error (ECE) on in-domain vs OOD test sets; show that NeSy-EDL with IBDC produces better-calibrated uncertainty than standard EDL or softmax
2. **Branch ablation as explanation validation:** Disable concept/causal branches → which predictions change? If a prediction was "explained by" concept evidence, removing the concept branch should flip it
3. **Rule violation statistics:** For correctly detected fakes, which consistency rules fire most? Expected: expression-AU rules for expression-based fakes (F2F), boundary rules for face swaps (FS, DF)
4. **Forensic group specificity:** Per-manipulation-method forensic anomaly profiles (radar charts) showing that different deepfake methods break different forensic groups
5. **Gate adaptation on OOD data:** Show that feature-conditioned gates shift trust toward symbolic branches on unseen datasets (cross-dataset generalization through interpretable trust allocation)

#### Qualitative Examples
1. **Case study: correctly detected + well-explained** — decomposed evidence, top violated rules, forensic anomaly profile, counterfactual mismatch
2. **Case study: high uncertainty → abstain** — branches disagree, IBDC produces high uncertainty, flagged for human review
3. **Case study: failure mode** — model confident but wrong, analysis of which branch(es) failed and why

#### Figures
1. **Architecture diagram** with evidence flow arrows and named intermediate outputs
2. **Evidence decomposition bar chart** per prediction (spatial/concept/causal stacked)
3. **Consistency rule heatmap** across test set (rules × samples, color = violation magnitude)
4. **Forensic anomaly radar chart** per deepfake method
5. **Uncertainty reliability diagram** (ECE plot)
6. **Gate distribution violin plots** across in-domain vs OOD datasets
