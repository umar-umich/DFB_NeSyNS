# NeSy-DeFake — A Plain-English Walkthrough

_A presentation-friendly tour of what the framework does, why each piece
exists, and how everything fits together._

---

## 1. The problem, in one paragraph

Modern deepfake detectors trained on one dataset (e.g. FaceForensics++)
collapse when shown a new distribution (Celeb-DF-v2, DFDC, DFDCP…).
They memorise generator-specific artifacts that do not transfer.
Worse, they are overconfident on everything — real or fake, in-domain
or out-of-domain — so we cannot tell when to trust them.

We want a detector that (a) **generalises across datasets**, (b)
**tells us when it is unsure**, and (c) **explains why** it flagged a
face as fake. NeSy-DeFake layers a small symbolic/causal "second
opinion" on top of a strong vision backbone, and fuses the two using
evidential (Dirichlet) uncertainty.

---

## 2. The big picture (one slide)

```
                         ┌────────────────────────────────────────┐
                         │           Evidence Fusion              │
  image ─► CLIP (frozen) ─► spatial evidence  ─┐                  │
                                               ├─► total evidence │
  face attributes (VLM + rules) ──► concept ──┤   │               │
                       evidence                │   │               │
                                               │   ▼               │
  forensic features ──────────► causal ────────┘  α (Dirichlet)    │
                       evidence                    │               │
                                                   ▼               │
                                          probability + uncertainty│
                         └────────────────────────────────────────┘
```

Three information streams produce **evidence** for "real" vs "fake";
a learned fusion combines them; the fused Dirichlet gives both a
probability **and** a calibrated uncertainty.

---

## 3. Why "evidence" instead of logits?

Standard classifiers output logits → softmax → probabilities.
Softmax always sums to 1, so a confident-wrong model looks identical
to a confident-correct one. We replace softmax with **Evidential Deep
Learning (EDL)**:

- The network outputs non-negative **evidence** `e_k ≥ 0` per class.
- We interpret `α = e + 1` as a Dirichlet distribution over class
  probabilities.
- The total evidence `S = Σ α_k` encodes **how much** the model
  believes in any answer. Uncertainty `u = K / S` shrinks when
  evidence is high, grows when the model has "no opinion".

Payoff: out-of-distribution samples get low `S`, so the detector can
say **"I don't know"** instead of guessing with 0.999 confidence.

---

## 4. The three evidence sources

### 4.1 Spatial stream (always on)
- **CLIP ViT-L/14** — frozen except for its LayerNorms.
- Why CLIP? Large pre-training makes it robust to the low-level noise
  each new generator introduces. LayerNorm fine-tuning is the GenD
  recipe and keeps the representation light.
- A projection head (standard / residual / bottleneck / Houlsby) maps
  CLIP features to the classifier input. Houlsby-style adapters are
  the default — they add a small trainable bottleneck so the frozen
  backbone does not have to learn anything new.
- Produces `spatial_evidence = softplus(linear_head(CLIP))`.

### 4.2 Concept stream (Ablation 3+)
Humans spot fakes by reasoning about the face: "the teeth are
asymmetric", "the lighting on the left side does not match the
right". We encode that reasoning:
- **Fast semantic features (58-d)** computed from 3D landmarks:
  eye openness, nose symmetry, jaw width ratio, specular highlights,
  etc.
- **VLM semantic features (64-d)** — subset of FaceBench attributes
  precomputed by a Face-LLaVA model (disabled in the "noVLM"
  ablation to test whether the expensive VLM is actually needed).
- **Consistency rules (23-d)** — hand-written predicates like
  "left_eye_open ≈ right_eye_open", firing 1 if violated.
- All three concatenate into a symbolic feature vector, a tiny MLP
  converts it into `concept_evidence`.
- Why useful: these features are generator-agnostic — a deepfake
  that passes pixel statistics can still trip the symmetry rule.

### 4.3 Causal stream (Ablation 4)
The concept stream asks "does the face violate a rule?" The causal
stream asks **"is this face causally consistent with a real face?"**

We offer three implementations, selected by `causal_branch.type`:

- **`ccv` (Causal Constraint Verification — our novel contribution).**
  Learns `K = 16` smooth scalar constraint functions over
  `{spatial_features, attributes, rules, forensic_features}`. Each
  constraint is a small MLP whose output should be ≈0 on real faces
  and drift from 0 on fakes. A counterfactual predictor checks
  "if this face were real, would these features be plausible?" —
  residual becomes evidence.
- **`improved_scm`** — a nonlinear Structural Causal Model with
  identity and forensic sub-graphs, label-conditioned
  reconstruction, and a graph-divergence loss.
- **`simple`** — a linear SCM baseline.

All three produce `causal_evidence` plus diagnostic artefacts
(constraint violations, counterfactual residuals, DAG adjacency
matrices) used for **explainability**.

---

## 5. Evidence Fusion — the glue

The three streams each say "I see this much evidence for fake". How
should we combine them? Three strategies, all encapsulated in
`EvidenceFusion`:

### 5.1 Static scalar gates (simplest)
```
total = spatial + σ(g_c) · concept + σ(g_r) · causal
```
`g_c`, `g_r` are learnable scalars initialised negative so the
symbolic streams start ~30 % weight and climb as they prove useful.

### 5.2 Feature-conditioned gates (per-image)
A tiny `Linear(spatial_features → num_gates)` emits **per-image**
gates. Lets the model trust the causal branch more on
compression-heavy frames, the concept branch more on clear frontal
faces, etc.

### 5.3 Confidence-Modulated Evidence Fusion — CMEF (novel)
The predecessor of this project used a cross-attention mechanism
("CausalViolationAttentionFusion") that attended from fused
features over SCM residuals. That worked but lived outside the
evidence framework.

**CMEF** does the same idea **in EDL space**: each symbolic branch
reports its own Dirichlet confidence; CMEF weights branches by how
sharp their Dirichlet is, per sample. A branch that is unsure about
a given image is automatically down-weighted. Result:
- Symbolic noise on hard cases does not drown out a confident
  spatial prediction.
- Cheaper than attention (scalar weights, not softmax over
  residuals).
- Fully calibrated — weights act on `α`, so uncertainty stays
  consistent.

CMEF is paired with **PBAS** (per-branch auxiliary supervision — each
branch is trained on its own EDL head) and **IBDC** (inter-branch
disagreement calibration — the model is punished when the fused
Dirichlet disagrees with itself about which branch to trust).

---

## 6. The loss picture

- **EDL NLL** — negative log-likelihood of the Dirichlet on the
  target class.
- **EDL KL annealing** — early in training, keep the Dirichlet flat;
  as training progresses, let it sharpen. Prevents premature
  overconfidence.
- **AVU** (Accuracy-vs-Uncertainty) — penalises *confident-wrong*
  predictions specifically.
- **Aux + disagreement** (NeSy-EDL only).
- **DAG penalty** (Ablation 4) — keeps the learned causal graphs
  mostly acyclic.
- **Uniformity-Alignment** — GenD-style regulariser on L2-normalised
  embeddings. Alignment pulls same-class embeddings together,
  uniformity spreads the class clusters over the hypersphere. Big
  cross-dataset win in the GenD paper; we reuse it here on the
  spatial projection.

---

## 7. The ablation ladder

| Ablation        | Spatial | EDL | Concept | Causal | Purpose                             |
|-----------------|:-:|:-:|:-:|:-:|-------------------------------------|
| **1** spatial_ce | ✓ |   |   |   | Reproduce GenD-CLIP baseline (CE).  |
| **2** spatial_edl| ✓ | ✓ |   |   | Does EDL alone improve calibration? |
| **3** concept_edl| ✓ | ✓ | ✓ |   | Does symbolic reasoning help?       |
| **4a** causal_causal (ImprovedSCM) | ✓ | ✓ | ✓ | SCM | SCM explainability track.   |
| **4b** causal_ccv (CCV)            | ✓ | ✓ | ✓ | CCV | Our novel branch.            |
| **4c** ccv_novlm                   | ✓ | ✓ | ✓ | CCV | CCV without the expensive VLM. |

The ladder lets us show *each* module earns its keep — not just the
final system number.

---

## 8. Source-code layout (where each piece lives)

```
training/
├── detectors/
│   └── nesy_defake_detector.py          ← orchestrator (~500 lines)
├── networks/nesy_defake/
│   ├── foundation_models/               ← CLIP spatial extractor
│   ├── classifiers/
│   │   ├── multitask_head.py            ← classifier head
│   │   ├── projection_heads.py          ← standard/residual/bottleneck/Houlsby
│   │   └── feature_conditioned_gate.py  ← per-image evidence gate
│   ├── concept_branch.py                ← rule-based evidence MLP
│   ├── ccv_branch.py                    ← novel CCV branch
│   ├── improved_scm_branch.py           ← nonlinear SCM branch
│   ├── causal_branch_factory.py         ← picks the causal variant
│   ├── fusion/
│   │   └── evidence_fusion.py           ← CMEF / conditioned / static
│   └── losses/
│       ├── edl_loss.py                  ← vanilla EDL
│       ├── nesy_edl_loss.py             ← NeSy-EDL (CMEF + PBAS + IBDC)
│       └── unifalign.py                 ← Uniformity-Alignment (GenD)
├── trainer/trainer.py                   ← train / test loops
├── train.py                             ← entrypoint + optimizer groups
└── config/detector/nesy_defake_ablation*.yaml
```

The detector file is now an orchestrator — every reusable module has
its own file.

---

## 9. Training flow (what happens each step)

1. **Load batch** — source-paired real/fake pairs from the same
   original video (GenD recipe; reduces shortcut learning).
2. **Feature extraction** — CLIP produces `spatial_raw` (1024-d);
   precomputed concept + forensic features are fetched from disk.
3. **Projection** — projection head maps `spatial_raw` → `projected`
   (with optional Houlsby adapter).
4. **Spatial evidence** — classifier head → softplus → evidence.
5. **Concept / causal branches** — run only if active for the
   current ablation; each returns evidence plus diagnostics.
6. **Fusion** — `EvidenceFusion` combines the streams.
7. **Dirichlet** — `α = e + 1`; prediction = `α / S`; uncertainty =
   `K / S`.
8. **Loss** — EDL NLL + KL + AVU (+ CMEF aux + IBDC + DAG penalty +
   UA regulariser, depending on ablation).
9. **Backprop** — AdamW, cosine-warmup, frozen backbone except
   LayerNorms, per-group LRs (symbolic branches learn faster).

---

## 10. Why this is a publishable story

- **Generalisation**: UA loss + frozen CLIP + per-sample fusion
  gives cross-dataset robustness.
- **Calibration**: EDL produces honest uncertainty. Confident-wrong
  predictions drop.
- **Explainability**: the concept branch surfaces which of 23 rules
  fired on a misclassified frame; the causal branch returns
  constraint violations and counterfactual residuals.
- **Neuro-symbolic on modern deepfakes**: most NeSy work is on
  tabular / NLP data. We show it works on a real, competitive
  vision benchmark.
- **Clean ablations**: the ladder isolates each contribution.

---

## 11. What to show in a 10-minute talk

1. **Slide 1**: the cross-dataset problem (AUC drop chart from
   Ablation 1 baseline).
2. **Slide 2**: the big-picture diagram from Section 2.
3. **Slide 3**: what is "evidence" (Section 3 — one equation).
4. **Slide 4**: the three streams (Section 4).
5. **Slide 5**: CMEF vs attention-based fusion (Section 5.3).
6. **Slide 6**: ablation ladder table (Section 7).
7. **Slide 7**: cross-dataset results + uncertainty histogram.
8. **Slide 8**: explainability — show which rule / constraint fired
   for a real misclassified frame.
9. **Slide 9**: limitations — frame-level only, VLM cost,
   dependence on precomputed features.
10. **Slide 10**: next steps — video-level temporal module,
    end-to-end VLM training, more causal branches.

---

_File: `docs/README_april_17_simple.md` — last updated 2026-04-17._
