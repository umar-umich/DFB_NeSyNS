
---

## Ablation Study Design (April 6–7, 2026)

We designed a progressive ablation study with 4 configs that incrementally add NeSy modules on top of the GenD spatial baseline. Each ablation builds on the previous one, isolating the contribution of each component.

### Common Settings (shared across all 4 ablations)

| Setting | Value |
|---------|-------|
| Backbone | CLIP ViT-L/14 (frozen, LayerNorms fine-tuned) |
| Branch | Spatial only (`active_branches: ['spatial']`) |
| Classifier | Linear probe (1024 → 2) |
| Optimizer | Adam, lr=3e-4, no weight decay |
| Scheduler | Cosine warmup (1 epoch) |
| Batch size | 256 train / 512 test |
| Epochs | 50 (early stopping patience=15) |
| Training | Source-paired (GenD recipe), augmentations enabled |
| Train set | FaceForensics++ (c23) |
| Test sets | FaceForensics++, Celeb-DF-v2 |

---

### Ablation 1: Spatial-Only Baseline (GenD Reproduction)
**Config:** `training/config/detector/nesy_defake_ablation1.yaml`

**Purpose:** Reproduce GenD_CLIP performance (~0.95 video AUROC on CDFv2) as the pure baseline.

**Key settings:**
- `ablation_spatial_only: true` — skips ALL NeSy modules (causal, SAE, semantic, fusion)
- Pure cross-entropy loss (no auxiliary losses)
- No class weights (balanced CE)
- No uniformity-alignment loss
- No semantic features loaded
- Pipeline: `CLIP → spatial_proj → Linear(1024, 2)`

**What it measures:** Raw CLIP spatial feature quality for deepfake detection with GenD training recipe.

---

### Ablation 2: Spatial + EDL Loss
**Config:** `training/config/detector/nesy_defake_ablation2_edl.yaml`

**Purpose:** Replace CE with Evidential Deep Learning (EDL) loss to measure the effect of Dirichlet-based uncertainty estimation, without changing the architecture.

**Key settings:**
- `ablation_spatial_only: true`, `ablation_mode: spatial_edl`
- EDL config: `num_classes=2`, `annealing_epochs=10`, `kl_weight=0.1`
- Model predicts **evidence** (via softplus) instead of raw logits
- Logs uncertainty, evidence_succ/fail metrics
- Architecture identical to Ablation 1 — only the loss changes

**Changes from Ablation 1:**
- CE → EDL loss (softplus evidence → Dirichlet → log-likelihood + annealed KL)

**What it measures:** Whether uncertainty-aware training improves calibration and cross-dataset generalization without architectural changes.

---

### Ablation 3: Spatial + Concept Branch + EDL
**Config:** `training/config/detector/nesy_defake_ablation3_concept.yaml`

**Purpose:** Add a neuro-symbolic concept evidence branch that operates on precomputed semantic features, producing independent evidence fused with spatial evidence via a learnable gate.

**Key settings:**
- `ablation_spatial_only: false`, `ablation_mode: concept_edl`
- **Concept branch:** `combined_dim=122` (fast(58) + vlm(64)), `rules_dim=23`, `hidden_dim=64`
- **Evidence gate:** `concept_init=-3.0` (sigmoid ≈ 0.05, starts near zero)
- Loads precomputed features: `fast_semantic` (58-d) + `facellava_semantic` refined to 64-d VLM = 122-d combined
- Consistency rules enabled (23-d)
- `use_refined_features: true`
- Concept branch LR = 0.001 (3.3x base lr for faster concept learning)
- **No gradient path from concept branch to CLIP** — fully independent

**Evidence fusion formula:**
```
spatial_evidence  = softplus(linear_head(CLIP_features))
concept_evidence  = softplus(concept_mlp(122-d combined + 23 rules))
total_evidence    = spatial_evidence + sigmoid(gate) * concept_evidence
alpha = total_evidence + 1
uncertainty = K / sum(alpha)
```

**Changes from Ablation 2:**
- Adds ConceptBranch (145-d → 64 → 2 evidence)
- Adds evidence fusion gate
- Loads semantic + fast_semantic + consistency rule features

**What it measures:** Whether structured semantic concept evidence (from fast extractors + VLM attributes + consistency rules) provides complementary signal to CLIP spatial features.

---

### Ablation 4: Spatial + Concept + Causal + EDL (Full NeSy)
**Config:** `training/config/detector/nesy_defake_ablation4_causal.yaml`

**Purpose:** Full neuro-symbolic pipeline with explainable causal graphs. Adds a simplified causal branch that learns structural causal models (SCMs) for identity and forensic sub-graphs.

**Key settings:**
- `ablation_spatial_only: false`, `ablation_mode: causal_edl`
- **Concept branch:** same as Ablation 3
- **Causal branch:** `backbone_dim=1024`, `z_causal_dim=32`, `curated_dim=51`, `rules_dim=23`, `forensic_dim=83`
- **Evidence gates:** `concept_init=-3.0`, `causal_init=-3.0` (both start near zero)
- **Forensic features enabled:** 83-d precomputed
- Causal branch LR = 0.001
- `interpretability.save_causal_graphs: true` for explainability
- DAG penalty: `sparsity_penalty=0.01`, `dag_penalty_weight=0.05`

**Causal sub-graphs (4 linear SCMs):**
| Sub-graph | Nodes | Composition |
|-----------|-------|-------------|
| Identity-Real | 106 | z(32) + curated(51) + rules(23) |
| Identity-Fake | 106 | z(32) + curated(51) + rules(23) |
| Forensic-Real | 115 | z(32) + forensic(83) |
| Forensic-Fake | 115 | z(32) + forensic(83) |

**Evidence fusion formula (3 sources):**
```
spatial_evidence = softplus(linear_head(CLIP_features))
concept_evidence = softplus(concept_mlp(122-d + 23 rules))
causal_evidence  = softplus(residual_mlp(differential SCM residuals))
total = spatial + gate1 * concept + gate2 * causal
```

**Changes from Ablation 3:**
- Adds SimplifiedCausalBranch with 4 linear SCMs
- Adds forensic features (83-d precomputed)
- Adds causal evidence gate
- Adds DAG acyclicity + sparsity penalty in loss

**What it measures:** Whether causal structure discovery (real vs fake graph divergence) provides additional explainability and detection signal beyond concept-level evidence.

---

### Ablation Summary Table

| Ablation | Loss | Concept Branch | Causal Branch | Semantic Features | Forensic Features | Evidence Sources |
|----------|------|----------------|---------------|-------------------|-------------------|------------------|
| 1 — Spatial baseline | CE | ✗ | ✗ | ✗ | ✗ | 1 (spatial logits) |
| 2 — + EDL | EDL | ✗ | ✗ | ✗ | ✗ | 1 (spatial evidence) |
| 3 — + Concept | EDL | ✓ (122+23→64→2) | ✗ | ✓ (122-d) | ✗ | 2 (spatial + concept) |
| 4 — + Causal (Full NeSy) | EDL + DAG | ✓ (122+23→64→2) | ✓ (4 SCMs) | ✓ (122-d) | ✓ (83-d) | 3 (spatial + concept + causal) |

### Additional Ablation: CLIP + DINOv2 Dual-Branch
**Config:** `training/config/detector/ablation_clip_dinov2.yaml`

A separate dual-branch ablation using both spatial (CLIP) and frequency (FAD-DINOv2) branches without any NeSy modules:
- `active_branches: ['spatial', 'frequency']`
- Attention fusion (`fused_dim=2048`, `projection_dim=1024`)
- Uniformity-alignment loss enabled (`alignment=0.1`, `uniformity=0.5`)
- Class weights `[0.7, 1.3]`, projection dropout (spatial=0.2, freq=0.1)
- Deep MLP classifier (512→256→2) instead of linear probe
- lr=1e-4, weight_decay=1e-4, 30 epochs
- Train on FaceForensics++_augmented