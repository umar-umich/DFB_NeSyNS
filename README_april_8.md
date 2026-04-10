# NeSyDeFake Changes — April 8-9, 2026

## 1. Ablation 4: Two Alternative Causal Branch Designs

Created two new branch architectures for Ablation 4 comparison:

### 4a: Causal Constraint Verification (CCV) Branch
**File:** `training/networks/nesy_defake/ccv_branch.py`
**Config:** `training/config/detector/nesy_defake_ablation4_ccv.yaml`

Three neuro-symbolic components (~15K params):
- **Learned Constraint Functions**: K=16 soft constraints over 122-d combined features (shared trunk + independent sigmoid heads)
- **Forensic Anomaly Detector**: 5 per-group autoencoders (boundary, symmetry, patch_noise, srm_noise, spectral) with bottleneck=8
- **Counterfactual Predictor**: MLP predicts z(32) from semantic summary(228-d), mismatch with compressed CLIP features = manipulation signal

Evidence input: 23 rules + 16 learned + 5 anomaly + 1 counterfactual = 45-d -> MLP -> 2-d evidence

### 4b: Improved SCM Branch (Fixes A-D)
**File:** `training/networks/nesy_defake/improved_scm_branch.py`
**Config:** `training/config/detector/nesy_defake_ablation4_causal.yaml`

Fixes to original SimplifiedCausalBranch:
- **A) Nonlinear SCMs**: 1-hidden-layer MLP (SiLU), adjacency via |W2|@|W1| (NOTEARS-MLP)
- **B) Graph divergence loss**: -L1(A_real, A_fake) pushes real != fake structure
- **C) Split forensic sub-graphs**: 3 groups (structural/noise/spectral) instead of one 115-node graph
- **D) Label-conditioned reconstruction**: real SCM fits reals only, fake SCM fits fakes only

4 sub-graphs x 2 SCMs = 8 NonlinearSCMs, summaries 4x8=32-d -> MLP -> evidence

### Results (without VLM features)

| Config | FF++ AUC | CelebDF AUC | Avg AUC | Early Stop |
|--------|----------|-------------|---------|------------|
| **4a CCV** | **0.955** | **0.899** | **0.926** | epoch 24 |
| 4b ImprovedSCM | 0.947 | 0.888 | 0.914 | epoch 16 |

**CCV wins.** Key observations:
- CCV causal_gate grew 0.31 -> 0.587 (branch very useful)
- ImprovedSCM DAG penalty exploded to billions (DAGMA spectral radius issue)
- Fixed DAG explosion: row-normalize adjacency + clamp(max=100) + reduced weight 0.05 -> 0.01

## 2. Detector Integration for Multiple Branch Types

**File:** `training/detectors/nesy_defake_detector.py`

- `causal_branch.type` config field: `'ccv'`, `'improved_scm'`, or `'simple'`
- Passes `labels` to causal branch forward (needed for label-conditioned recon)
- Generic adjacency matrix copying (`A_*` keys vary by branch type)

## 3. FaceBench VLM Features Enabled

The precomputed features were extracted as `facebench_semantic/*.pt` (64-d VLM subset, not raw 211-d).

**Changes:**
- Fixed directory name: `facellava_semantic` -> `facebench_semantic` in ALL configs
- Fixed dimension: `precomputed_dim: 211` -> `64` in ablation 3, 4_ccv, 4_causal
- Set `semantic_attributes.enabled: true` in ablation 3, 4_ccv, 4_causal
- Updated dataset loading (`nesy_defake_dataset.py`): handles both 64-d (already subset) and 211-d (full FaceBench) gracefully

## 4. Consistent Hyperparameters Across Ablations 3/4

Applied to ablation 3 for fair comparison with ablation 4:
- `edl.kl_weight: 0.15` (was 0.1)
- `edl.avu_weight: 0.1` (was 0.0)
- `class_weights: [1.8, 0.6]` (was [1.4, 0.8])
- `evidence_gate.concept_init: -0.5` (was -1.0, gate barely moved)

## 5. DAG Penalty Explosion Fix

**File:** `training/networks/nesy_defake/improved_scm_branch.py`

The DAGMA acyclicity constraint h(A) = -log det(sI - A*A) + d*log(s) explodes when adjacency values grow such that spectral_radius(A*A) > s=1.0.

Fix:
```python
# Row-normalize adjacency so spectral radius stays < 1.0
A_norm = A / A.sum(dim=1, keepdim=True).clamp(min=1.0)
dag_penalty = (dagma_acyclicity(A_real_norm) + dagma_acyclicity(A_fake_norm)).clamp(max=100.0)
```
Also reduced `dag_penalty_weight: 0.05 -> 0.01` in config.

## 6. Novel NeSy-EDL: Neuro-Symbolic Evidential Deep Learning

**File:** `training/networks/nesy_defake/losses/nesy_edl_loss.py`

Three novel components extending standard EDL for multi-branch neuro-symbolic fusion:

### 6a. Confidence-Modulated Evidence Fusion (CMEF)

Replaces static scalar gates with per-sample dynamic weighting:

```
Standard:  e_total = e_s + sigma(g1) * e_c + sigma(g2) * e_a
CMEF:      e_total = e_s + sigma(g1)*phi(S_c)*e_c + sigma(g2)*phi(S_a)*e_a
```

where phi(S_b) = sigmoid((S_b - K) / tau) is a confidence gate based on per-branch Dirichlet strength. Branches that are uncertain about a sample contribute less evidence.

**Key property:** On OOD data, neural branch becomes uncertain -> symbolic branches with higher confidence get higher relative weight -> better generalization through dynamic neural-symbolic trust allocation.

### 6b. Per-Branch Auxiliary Supervision (PBAS)

Each branch receives its own lightweight EDL loss (weighted by `aux_weight=0.1`), ensuring independently meaningful evidence before fusion. Prevents dominant branch from suppressing gradient to weaker branches.

### 6c. Inter-Branch Disagreement Calibration (IBDC)

Novel loss term: when neural and symbolic branches disagree (measured by cosine distance between Dirichlet means), fused uncertainty should be HIGH. When they agree, it can be LOW.

```
d(x) = 1 - cos(p_neural(x), p_symbolic(x))
L_bdc = -d * log(u_fused) - (1-d) * log(1 - u_fused)
```

This is a binary cross-entropy calibrating uncertainty against inter-branch agreement. Standard EDL has no mechanism for uncertainty to reflect reasoning CONFLICT between evidence sources.

### Integration

- `edl.nesy_fusion: true` in config enables CMEF + PBAS + IBDC
- CMEF module replaces scalar concept_gate/causal_ev_gate
- NeSyEvidentialLoss replaces EvidentialLoss
- New logged metrics: `concept_conf`, `causal_conf`, `cmef_tau`, `edl_aux`, `edl_bdc`
- CMEF parameters added to causal_branch optimizer group

### Config (ablation4_ccv.yaml):
```yaml
edl:
  nesy_fusion: true
  aux_weight: 0.1
  disagreement_weight: 0.05
```

## Ablation Summary

| # | Config | Components | What It Tests |
|---|--------|-----------|---------------|
| 1 | ablation1 | CLIP + CE | Baseline: frozen CLIP linear probe |
| 2 | ablation2_edl | CLIP + EDL | +uncertainty-aware loss |
| 3 | ablation3_concept | CLIP + EDL + Concept(122-d + 23 rules) | +symbolic consistency rules |
| 4a | ablation4_ccv | CLIP + NeSy-EDL + Concept + CCV(45-d) | +learned constraints, anomaly, counterfactual, CMEF |
| 4b | ablation4_causal | CLIP + NeSy-EDL + Concept + ImprovedSCM | +nonlinear SCMs, DAG, CMEF |

All ablations 3+ now use:
- Full 122-d features: fast(58) + VLM(64) from FaceBench
- 83-d forensic features
- Class weights [1.8, 0.6]
- AVU calibration (avu_weight=0.1)
- kl_weight=0.15

Next prompt: Okay one more thing is that, the most expensive part of our features extraction process is the facebench_semantic  features. So for ablation4_ccv, I want to run it once without facebench features to check if it's making any  postive addition or just a burden on the framework. Additionally, now let's talk about the explainability/interpretability of the framework with ablation4_ccv as possibly the final architecture? What we have and can present in the paper?