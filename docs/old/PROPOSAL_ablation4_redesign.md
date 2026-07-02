# Ablation 4 Redesign Proposal: Causal Evidence Branch

## Current Design — What's Wrong

The current `SimplifiedCausalBranch` has four Linear SCMs (identity real/fake, forensic real/fake) that compute `x_hat = Wx + b`, take the residual `x - x_hat`, and compute a differential `r_fake - r_real`. This differential goes through an MLP to produce evidence.

### Problem 1: Linear SCMs Cannot Capture Nonlinear Causal Relationships

Real causal chains are nonlinear:
- age_score → wrinkled_skin is a sigmoid/step function (wrinkles appear after ~40, not linearly)
- gender_score → beard is a threshold function (strongly nonlinear)
- expression → AU activation involves complex muscle coactivation patterns

A linear `W` matrix can only capture linear correlations. The residual `x - Wx - b` for a nonlinear relationship will always be large regardless of real/fake — it's measuring model misfit, not manipulation signal.

### Problem 2: No Incentive for Real ≠ Fake Graph Divergence

Both real and fake SCMs are trained with the same objective (minimize residuals via the classification loss backprop). Without explicit encouragement to learn *different* structures, they converge to similar solutions → `r_fake - r_real ≈ 0` → near-zero signal.

The `graph_divergence` loss weight is 0.0 in the config. The DAG + sparsity penalties regularize each SCM independently but don't push them apart.

### Problem 3: 115-Node Forensic Sub-Graph is Intractable

The forensic SCM has `d = 32 + 83 = 115` → a 115×115 weight matrix = 13,225 parameters per SCM, 52,900 for all four forensic SCMs. With the gated evidence fusion (gate starting at sigmoid(-1.5) ≈ 0.18), the gradient signal through the causal branch is weak. These SCMs are severely under-constrained.

### Problem 4: The Detection Signal Path is Too Indirect

Current: `combined_features → curated → SCM → residual → differential → MLP → evidence → gate → total_evidence`

That's ~6 transformations before the signal reaches the loss. Each step attenuates gradients. The concept branch (ablation 3) has a much shorter path: `features → rules → MLP → evidence`.

### Problem 5: Chicken-and-Egg with Previous Gate Design

The gate initialized at sigmoid(-3.0) ≈ 0.047 was essentially dead (stayed at 0.046 through all of ablation 3 training). We've fixed this to -1.0/-1.5, but the fundamental issue remains: if the causal branch doesn't produce useful signal early, the gate can't learn to open.

---

## Proposed Redesign: Causal Constraint Verification (CCV) Branch

### Core Insight

Instead of learning abstract causal structure (SCMs), directly **verify known causal constraints** and measure how much each sample violates them. This is the true neuro-symbolic approach: encode human knowledge as differentiable rules, then let the neural network learn which violations matter for detection.

The consistency rules in ablation 3 already do this partially (23 hand-coded rules). The redesign extends this with **learned constraint functions** that discover additional violations the hand-coded rules miss.

### Architecture

```
                                                  ┌─────────────────────┐
combined_features (B, 122) ──┐                    │ Constraint Violation │
                             ├─── concat ────────→│ Attention (CVA)      │──→ evidence (B, 2)
violations (B, 23) ──────────┘                    │                     │
                                                  └─────────────────────┘
forensic_features (B, 83) ─────→ Forensic         │
                                 Anomaly ──────────┘
                                 Detector (FAD)

spatial_raw (B, 1024) ─detach──→ Counterfactual ───┘
                                 Residuals (CR)
```

### Three Components

#### Component 1: Learned Constraint Functions (LCF) — extends the 23 hand-coded rules

The 23 consistency rules are hard-coded from domain knowledge. But there are likely violations we haven't thought of. LCF learns K additional soft constraints:

```python
class LearnedConstraintFunctions(nn.Module):
    """
    Learn K additional soft constraint functions over the combined feature space.
    Each constraint is a small MLP that outputs a scalar "violation score".

    Think of each as a learned version of rules like:
      cr_gender_beard = sigmoid(beard) * sigmoid(-gender_score)
    but the network discovers the feature interactions automatically.

    Total: K learned + 23 hand-coded = K+23 constraint scores.
    """
    def __init__(self, input_dim=122, num_constraints=16, hidden=32):
        # K parallel small MLPs: 122 → 32 → 1 each
        # Output: (B, K) learned violation scores
```

This gives us `23 + K` total constraint scores. Small K (8-16) keeps it tractable.

#### Component 2: Forensic Anomaly Detector (FAD) — replaces the 115-node forensic SCM

Instead of a massive linear SCM over forensic features, use a lightweight anomaly score. The key insight: forensic features for real faces should cluster tightly (consistent noise patterns), while fakes should be outliers.

```python
class ForensicAnomalyDetector(nn.Module):
    """
    Learns a compact "normal" representation of forensic features
    and measures deviation from it. Much simpler than a 115-node SCM.

    Architecture:
      forensic (B, 83) → encoder (83→32) → decoder (32→83)
      anomaly_score = ||forensic - decoded||² per feature group

    Groups (5 scores):
      - boundary gradients (6 features) → 1 score
      - blur/texture (6) → 1 score
      - symmetry + color (8) → 1 score
      - SRM + noise (27) → 1 score
      - FFT spectral (10) → 1 score

    Output: (B, 5) grouped anomaly scores
    """
```

5 interpretable anomaly scores instead of 115-node SCM residuals.

#### Component 3: Counterfactual Residuals (CR) — simplified, nonlinear

Instead of 4 linear SCMs, use a single nonlinear predictor that learns "given these semantic attributes, what should the CLIP features look like?" The residual between predicted and actual CLIP features is the manipulation signal.

```python
class CounterfactualPredictor(nn.Module):
    """
    Predicts what CLIP spatial features SHOULD look like given the
    semantic/forensic attributes. For real faces, the prediction is good
    (attributes match appearance). For fakes, there's a mismatch.

    This is a *counterfactual* question: "If a face truly had these
    attributes, what would it look like?"

    Architecture:
      semantic_summary (B, D) → MLP → predicted_z (B, 32)
      actual_z = compress(spatial_raw.detach())  # (B, 32)
      residual = ||actual_z - predicted_z||²     # scalar per sample

    Output: (B, 1) counterfactual mismatch score
    """
```

This replaces the entire real/fake SCM pair with a single predictor. The insight: we don't need separate real/fake models — a *good* predictor naturally has low residual for real faces and high residual for fakes.

### Evidence Fusion

All three components feed into a small attention-based fusion:

```python
class CausalEvidenceFusion(nn.Module):
    """
    Fuses multiple violation/anomaly signals into 2-d evidence.

    Input features (concatenated):
      - 23 hand-coded consistency violations
      - K learned constraint violations
      - 5 forensic anomaly scores
      - 1 counterfactual mismatch score
      Total: 23 + K + 5 + 1 = ~45 features

    Architecture:
      (B, 45) → LayerNorm → Linear(45, 64) → GELU → Dropout
              → Linear(64, 2) → softplus → evidence (B, 2)
    """
```

### Why This is Better

| Aspect | Current (Linear SCM) | Proposed (CCV) |
|--------|---------------------|----------------|
| **Nonlinearity** | Linear only | Nonlinear constraint functions |
| **Parameters** | ~180K (4 SCMs × 115²) | ~15K (small MLPs) |
| **Interpretability** | 115×115 adjacency matrix | Named violation scores |
| **Gradient path** | 6 steps to loss | 3 steps to loss |
| **Real/fake separation** | Hopes SCMs diverge | Built-in (anomaly = deviation from normal) |
| **Graph discovery** | Needs DAGMA penalty, EMA | Not needed — constraints are explicit |
| **Domain knowledge** | Implicit in feature selection | Explicit in constraint design |
| **Trainability** | Under-constrained SCMs | Small, focused modules |

### What We Lose (and Why It's OK)

1. **DAG adjacency matrices for explainability**: We lose the pretty causal graphs. But the learned constraint functions + anomaly scores are MORE interpretable — you can directly inspect which violations fire for each sample.

2. **Causal discovery in the formal sense**: We're no longer doing structure learning. But formal causal discovery needs much more data and interventional experiments than we have. The violation-based approach captures the same NeSy insight (manipulations break attribute consistency) without the SCM machinery.

3. **Real/fake SCM pairs**: The counterfactual predictor replaces this with a single model. If the predicted appearance matches actual appearance, it's real; if not, it's fake. This IS the causal question, just posed differently.

---

## Alternative: Keep SCM Architecture, Fix the Issues

If you want to keep the SCM-based approach for the paper's narrative, here are targeted fixes:

### Fix A: Replace Linear → Nonlinear SCM (1-hidden-layer MLP)
```python
class NonlinearSCM(nn.Module):
    def __init__(self, d, hidden=64):
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(),
            nn.Linear(hidden, d))
        # Adjacency via Jacobian or attention mask
```

### Fix B: Add explicit graph divergence loss
```python
# Encourage real and fake adjacency matrices to differ
div_loss = -F.l1_loss(A_real, A_fake)  # negative: maximize difference
# Or use cosine distance
div_loss = F.cosine_similarity(A_real.flatten(), A_fake.flatten(), dim=0)
```

### Fix C: Split 83-d forensic into smaller sub-graphs
Instead of one 115-node forensic sub-graph, use 5 small sub-graphs:
- Boundary (32 + 6 = 38 nodes)
- Texture (32 + 6 = 38 nodes)
- Symmetry+Color (32 + 8 = 40 nodes)
- Noise (32 + 27 = 59 nodes)
- Spectral (32 + 10 = 42 nodes)

Each is much more tractable, and you still get interpretable causal graphs.

### Fix D: Use label-conditioned SCM training
Only train the real SCM on real samples and fake SCM on fake samples during the loss computation (not just graph discovery):
```python
if labels[i] == 0:
    loss += mse(scm_real(x[i]), x[i])   # real SCM fits reals
else:
    loss += mse(scm_fake(x[i]), x[i])   # fake SCM fits fakes
```
This directly forces the two SCMs to specialize.

---

## Recommendation

**For the ablation study paper**: I recommend the **CCV (Causal Constraint Verification)** approach. It's:
1. More principled neuro-symbolically (explicit constraints + learned extensions)
2. More parameter-efficient (~10x fewer parameters)
3. More interpretable (named violation scores vs opaque adjacency matrices)
4. Much easier to train (shorter gradient paths, no DAG constraints needed)
5. Novel — no prior deepfake work uses learned constraint verification + counterfactual residuals

**If you prefer to keep SCMs**: Use fixes A+B+C+D together. The most impactful is Fix D (label-conditioned training) — without it the SCMs have no incentive to specialize.

---

## Expected Dimension Summary (CCV approach)

| Signal | Dimensions | Source |
|--------|-----------|--------|
| Hand-coded violations | 23 | ConsistencyRulesV7 |
| Learned constraints | 16 | LearnedConstraintFunctions |
| Forensic anomaly scores | 5 | ForensicAnomalyDetector (grouped) |
| Counterfactual mismatch | 1 | CounterfactualPredictor |
| **Total causal input** | **45** | Concatenated |
| Causal evidence output | 2 | CausalEvidenceFusion MLP |

Total trainable params: ~15-20K (vs ~180K current)
