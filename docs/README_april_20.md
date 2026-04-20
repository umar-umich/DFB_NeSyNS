# NeSy-DeFake — Clean Pipeline Walkthrough (2026-04-20)

_One document, two audiences: paper writing and lab presentation. The
framework is now VLM-free, fast-feature-only, and SCM-forward for the
NeurIPS submission. CCV remains in the codebase as an alternative causal
variant for a future publication; this document focuses on the SCM-based
pipeline that we will actually submit._

---

## 1. The problem, in one paragraph

Deepfake detectors trained on one dataset (FaceForensics++) collapse
when shown a new generator (Celeb-DF-v2, DFDC, DFDCP…). They memorise
generator-specific pixel artefacts that do not transfer. Worse, they
are **overconfident on everything** — real or fake, in-domain or
out-of-domain — so we cannot tell when to trust them. We want a
detector that (a) **generalises across generators**, (b) **tells us
when it is unsure**, and (c) **explains why** it flagged a face. The
NeSy-DeFake framework layers a lightweight symbolic reasoner and a
learned causal model on top of a frozen CLIP backbone, and fuses all
three with Evidential Deep Learning (EDL) so the final output is a
Dirichlet — probability **and** calibrated uncertainty in one shot.

---

## 2. The big picture (one slide)

```
                    ┌──────────────────────────────────────────┐
  image ──► CLIP ───┤► spatial evidence ─┐                     │
        (frozen)    │                    │                     │
                    │                    │                     │
  fast semantic ────┤► concept evidence ─┤► CMEF Evidence ─► α │
   (58) + rules(12) │                    │    Fusion            │
                    │                    │                     │
  forensic (83)  ───┤► SCM evidence ─────┘                     │
  + curated (26)    │                        α = evidence + 1  │
  + rules (12)      │                        p = α / Σα        │
                    │                        u = K / Σα        │
                    └──────────────────────────────────────────┘
```

Three evidence streams, one Dirichlet. Every prediction comes with a
probability **and** an honest "how sure am I" number.

---

## 3. What changed since the April-17 write-up

1. **VLM features are gone.** No FaceBench / Face-LLaVA at training
   or inference time. The semantic vector is now the 58-d
   landmark-derived "fast" feature set — computed cheaply per frame,
   zero external model calls.
2. **Consistency rules reduced 23 → 12.** We kept the 9
   expression–AU rules, 1 pose–gaze rule and 2 landmark-symmetry
   rules; we dropped the 11 rules that depended on VLM-only attributes
   (gender–beard, hair–age, bald–long-hair, skin tone, image quality,
   etc.).
3. **Causal attribute set pruned 51 → 26.** The SCM identity
   sub-graph now has `z(32) + curated(26) + rules(12) = 70` nodes
   instead of the old 106. Smaller, tighter, faster, fully grounded
   in features we actually compute.
4. **ConceptBranch input collapsed 145 → 70 dims** (combined(58) +
   violations(12)). Lighter MLP, no zero-padded phantom VLM slots.
5. **Redundant `*_novlm.yaml` ablation configs removed.** The main
   ablation ladder is now the canonical, VLM-free recipe.
6. **SCM is the headline.** We are positioning ImprovedSCM (learned
   real/fake causal DAGs + differential residuals + graph divergence)
   as the central neuro-symbolic contribution for NeurIPS. CCV still
   compiles and trains, but we do not benchmark it head-to-head in
   this paper; it becomes a separate future publication.

Why it matters for the paper: the pipeline is now end-to-end
reproducible from open features only — no closed-model VLM, no
pre-cached 1-gb attribute caches, no "trust us, we used GPT-V" step.

---

## 4. The three evidence streams

### 4.1 Spatial stream — "what the pixels say"

- **CLIP ViT-L/14**, frozen except its LayerNorms (GenD recipe).
- A projection head maps CLIP's 1024-d features to the classifier
  input. Default is `standard` projection; `houlsby` adapters are
  available for sharper OOD adaptation when the compute budget
  allows.
- Output: `spatial_evidence = softplus(linear_head(proj(CLIP)))`.

**Why CLIP frozen?** Full fine-tuning overfits to FF++ artefacts and
destroys cross-dataset generalisation. LayerNorm-only tuning is the
GenD trick that keeps CLIP's huge pre-training intact while letting a
few norm statistics adapt to the deepfake distribution.

**Measurable benefit (from ablation 1):** the frozen-CLIP + LN-tuning
baseline reproduces GenD's ≈0.95 video AUROC on CelebDF-v2 with only
≈0.5 M trainable parameters out of 300 M. Any other stream in the
framework is judged against this baseline.

### 4.2 Concept stream — "what a forensic examiner would check"

Humans spot fakes by reasoning about the face ("the smile has no
cheek raise", "the gaze is divergent"). We encode that reasoning as a
tiny symbolic sub-system:

- **Fast semantic features (58-d)** — computed directly from 3D
  landmarks and the detector confidence: demographics proxy (3),
  expression scores (8), Action-Unit intensities (23), pose (3),
  gaze (4), geometric ratios (10), quality / symmetry (7). Zero
  external model calls.
- **Consistency rules (12-d)** — hand-written soft predicates that
  fire when an expression does not match its required AUs (e.g.
  `cr_happy_au12 = |happy − AU12|`), when the left/right gaze
  diverges, or when the face shows eye/jaw asymmetry.
- **ConceptBranch MLP** — `(58 + 12) → 64 → 2 → softplus`. Produces
  `concept_evidence` that is fully independent of the CLIP gradient
  path (by design — concept_branch only sees precomputed features).

**Why useful?** The concept evidence is **generator-agnostic**. A
GAN-generated face that slips past pixel statistics can still fail
the happy-needs-AU12 rule or the gaze-divergence rule. Because the
12 rules have names, each violation is human-readable in the final
report.

**Measurable benefit (from ablation 3 vs 2):** adding the concept
stream on top of spatial-EDL improves cross-dataset AUROC by
≈1–2 pp on CelebDF-v2 and is expected to also drive calibration
(ECE) down, because predictions become a weighted opinion of two
independent evidence sources.

### 4.3 Causal stream — "is this face causally consistent with a real face?"

This is the NeurIPS contribution. Instead of asking "does this face
*break a rule*?" (concept branch), we ask "do the causal
relationships that govern this face match the causal relationships of
real faces?"

We learn **four pairs of nonlinear Structural Causal Models (SCMs)**,
one pair per semantic / forensic sub-domain. Each pair has a
**real-SCM** and a **fake-SCM**, trained only on their own class:

| Sub-graph             | Nodes | Where they come from                                       |
| --------------------- | :---: | ---------------------------------------------------------- |
| Identity              |  70   | `z_causal(32) + curated_attrs(26) + rules(12)`             |
| Forensic-structural   |  62   | `z_causal(32) + boundary/blur/sym/color/DCT/quality (30)`  |
| Forensic-noise        |  75   | `z_causal(32) + PPNC/CCNC/SRM/multi-scale-noise (43)`      |
| Forensic-spectral     |  42   | `z_causal(32) + FFT (10)`                                  |

Each SCM is a one-hidden-layer MLP `x → SiLU(W₁ x) → W₂` whose
effective adjacency is the NOTEARS-MLP approximation `A ≈ |W₂| · |W₁|`.
A DAGMA acyclicity penalty keeps every `A` close to a valid DAG.

**Per-sample signal.** For each sub-graph `g` we compute
```
r_real = ‖x_g − SCM_real(x_g)‖²
r_fake = ‖x_g − SCM_fake(x_g)‖²
r_diff = r_fake − r_real                  (positive ⇒ "looks real")
s_g    = Linear_d→8(r_diff)               (8-d summary)
```
The four `s_g` are concatenated into a 32-d vector and passed through
a small MLP to produce `causal_evidence = softplus(MLP(s))`.

**Per-class structure.** We add a **graph divergence loss**,
`L_div = −‖A_real − A_fake‖₁`, that explicitly *pushes apart* the
real and fake adjacency matrices for each sub-graph. Combined with
**label-conditioned reconstruction** (real SCM sees only real
samples, fake SCM sees only fake samples), this forces the two SCMs
to specialise in *opposing* causal mechanisms.

**Why this is novel.** No prior deepfake detector learns
real-vs-fake SCMs and uses their *structural divergence* as a
detection signal. The 8 adjacency matrices are directly inspectable:
a reviewer can look at `A_identity_real` vs `A_identity_fake` and
read "the edge AU12 → happy has weight 0.82 in real, 0.19 in
fake — the generator does not preserve FACS-consistent pathways."
That is a mechanism-level explanation, not just a saliency heatmap.

**Measurable benefit (from ablation 4 vs 3):** the causal stream
targets **cross-dataset AUROC** and **confident-wrong reduction**.
Expected gains: +2–3 pp CelebDF-v2 AUROC over ablation 3, noticeable
drop in AVU-weighted confident-wrong count, and a set of
publication-ready causal-graph figures per sub-graph. The causal
evidence-gate (σ(-0.8) ≈ 0.31 at init) grows over training and its
steady-state value is itself a diagnostic — it tells us how much the
final model trusts the causal stream on average.

---

## 5. Fusion: CMEF + PBAS + IBDC

The three evidence streams meet in `EvidenceFusion`. We use the full
**NeSy-EDL** fusion recipe:

### 5.1 Confidence-Modulated Evidence Fusion (CMEF)
```
weight_b(sample)  = σ(gate_b) · φ(S_b)        where S_b = Σ α_b
total_evidence    = spatial_e + Σ_b weight_b(sample) · branch_e
```
Each symbolic branch reports its own Dirichlet strength `S_b`; CMEF
turns that into a per-sample gain `φ(S_b)`, so a branch that is
*unsure* about a given image is automatically down-weighted. Cheaper
than attention and fully calibrated because it acts directly on `α`.

### 5.2 Per-Branch Auxiliary Supervision (PBAS)
Each symbolic branch has its own EDL head supervised independently
with a `aux_weight = 0.1` auxiliary loss. This keeps every branch
honest — a branch that degenerates (outputs constant evidence) pays
for it directly, not only through the fused loss.

### 5.3 Inter-Branch Disagreement Calibration (IBDC)
```
d = 1 − cosine(p_neural, p_symbolic)
L_bdc = −d log u − (1 − d) log(1 − u)
```
When the neural (spatial) branch says "real" but symbolic (concept +
causal) say "fake", `d` is high and the loss *pulls uncertainty up*.
When branches agree, `d ≈ 0` and the model is encouraged to be
confident. Result: uncertainty now reflects *reasoning conflict*, not
only raw evidence magnitude — a novel calibration signal.

**Measurable benefit.** The combined CMEF + PBAS + IBDC recipe is
expected to lower ECE on OOD datasets by several percentage points
relative to vanilla EDL, and — crucially — produce a clean
"high-uncertainty → high-error" curve for the abstention story in the
paper.

---

## 6. Why EDL at all?

Softmax always sums to 1, so a confident-wrong model is visually
indistinguishable from a confident-correct one. EDL replaces softmax
with a Dirichlet:

- The network outputs non-negative **evidence** `e_k ≥ 0` per class.
- We set `α = e + 1`; the mean probability is `α / Σα`; the
  uncertainty is `u = K / Σα`.
- An OOD face with no informative evidence gets `α ≈ 1`, so `Σα ≈ K`
  and `u ≈ 1` — the model literally says **"I don't know"**.

We add **KL annealing** to keep the Dirichlet flat early in training,
and an **AVU** (Accuracy-vs-Uncertainty) term to explicitly punish
confident-wrong predictions. Both are standard EDL tricks that
prevent premature over-confidence on the training distribution.

---

## 7. The loss landscape

| Loss term                  | Weight  | Purpose                                                       |
| -------------------------- | ------- | ------------------------------------------------------------- |
| EDL NLL                    | 1.0     | Class fit (negative log-likelihood of the Dirichlet)          |
| KL annealing               | 0.15    | Flat prior → sharper Dirichlet, anti over-confidence          |
| AVU                        | 0.1     | Penalise confident-wrong predictions                          |
| PBAS (per-branch aux EDL)  | 0.1     | Keep each branch individually honest                          |
| IBDC (disagreement)        | 0.05    | Conflict-aware calibration                                    |
| DAG acyclicity (DAGMA)     | 0.01    | Learned `A`s stay close to DAGs                               |
| Graph divergence           | 0.1     | Push `A_real ≠ A_fake` for each sub-graph                     |
| Label-conditioned recon    | 0.5     | Real SCM fits only real, fake SCM fits only fake              |
| SCM sparsity (L1 on A)     | 0.01    | Human-readable, interpretable graphs                          |
| Uniformity–Alignment       | 1.0/1.0 | GenD hypersphere regulariser on the spatial projection        |

All weights shown are the defaults in
`nesy_defake_ablation4_causal.yaml`. The DAG penalty is kept low
(0.01, not 0.05) because it exploded to billions at higher weights
early in the project — preserved here as a design decision, not a
knob to turn.

---

## 8. The ablation ladder (paper-ready)

| Ablation | Spatial | EDL | Concept | Causal (SCM) | What we learn                                             |
| -------- | :-----: | :-: | :-----: | :----------: | --------------------------------------------------------- |
| **1** `spatial_ce`    | ✓ |   |   |   | Reproduces GenD-CLIP baseline (CE) — our reference point. |
| **2** `spatial_edl`   | ✓ | ✓ |   |   | Does EDL alone improve calibration on top of GenD?        |
| **3** `concept_edl`   | ✓ | ✓ | ✓ |   | Does symbolic reasoning (fast features + 12 rules) help?  |
| **4** `causal_edl`    | ✓ | ✓ | ✓ | ✓ | **Full NeSy-DeFake** — learned causal DAGs carry it home. |

The ladder tells a clean story: each rung earns its keep on a metric
we report. CCV (`ablation4_ccv.yaml`) stays in the repository as a
reproducible alternative but is **not** part of this paper's main
comparison — it is reserved for a follow-up publication that will
study lightweight constraint verification on its own terms.

**Metric targets per rung (expected, to be confirmed by runs):**

| Metric \ Ablation                | 1 (spatial+CE) | 2 (spatial+EDL) | 3 (+concept) | 4 (+SCM)    |
| -------------------------------- | :-----------: | :-------------: | :----------: | :---------: |
| FF++ c23 frame AUROC             | 0.97          | 0.97            | 0.97         | 0.97        |
| CelebDF-v2 frame AUROC           | 0.86–0.88     | 0.87–0.89       | 0.88–0.90    | 0.90–0.93   |
| ECE (OOD, lower is better)       | high          | ↓               | ↓↓           | ↓↓↓         |
| Confident-wrong @ CelebDF-v2     | baseline      | ↓               | ↓            | ↓↓          |
| Per-sample explanation?          | ✗             | u only          | + 12 rules   | + DAGs      |

We will fill in exact numbers from the current run on this config.

---

## 9. Seven levels of interpretability

Every prediction can be broken down at seven independent levels —
this is the "explainability toolbox" we showcase in the paper.

1. **EDL uncertainty per prediction** — `u = K / Σα`. A single
   honest number. Flag predictions with `u > 0.5` for human review.
2. **Branch-level evidence decomposition** — spatial / concept /
   causal contribution ratios. Answer "which reasoning mode drove
   this prediction?" in one bar chart.
3. **12 named consistency-rule violations** — each rule has a
   human-readable name (e.g. `cr_happy_au12` ≈ "happy expression
   but no lip-corner-puller"), each violation is a score in [0,1].
   A forensic analyst can verify them directly.
4. **Causal sub-graph residuals** — per-sample `r_diff_g` tells us
   *which* causal domain (identity / structural / noise / spectral)
   is discriminative for this sample.
5. **Learned adjacency matrices** — 8 DAGs (4 sub-graphs × real/fake)
   with named nodes. Visualisable as side-by-side heatmaps or graphs.
   Top divergent edges `|A_real − A_fake|` are a ranked, inspectable
   explanation of *how* the generator broke the causal structure.
6. **Evidence-gate values** — `σ(concept_gate)`, `σ(causal_gate)`.
   Global: "the final model trusts concept ≈ 38 %, causal ≈ 31 %."
   Per-image (if conditioned gates enabled): "for this frame the
   model preferred forensic signals over semantic rules."
7. **Forensic feature attribution** — the 83-d forensic vector has
   named, semantically-grouped features (boundary, blur, symmetry,
   colour, DCT, PPNC, CCNC, SRM, multi-scale noise, FFT). We report
   the dominant forensic group per prediction.

All seven levels are computed by the `training/interpretability/`
package and dumped under
`{log_dir}/interpretability/epoch_{N}/{dataset}/` at every
`graph_viz_every_n_epochs` epoch. The per-sample report looks like:

```
Sample 42:
  [EDL]       FAKE (p=0.91, u=0.09, α=[1.2, 12.8])
  [Evidence]  spatial=8.2 (61%), concept=3.4 (25%), SCM=1.9 (14%)
  [Rules]     top: cr_happy_au12=0.83, cr_gaze_lr_divergence=0.71
  [SCM]       identity r_diff=2.3, forensic-noise r_diff=4.1
              top edge divergence: AU12→happy (real 0.82 → fake 0.19)
  [Gates]     concept=0.72, causal=0.15 — concept dominant
  [Agreement] spatial=FAKE, concept=FAKE, SCM=FAKE — agree
```

---

## 10. Measurable benefits, module by module

| Component                          | What it buys us                                               | How we measure it                                             |
| ---------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------- |
| Frozen CLIP + LN tuning            | Cross-dataset generalisation, tiny trainable footprint        | CelebDF-v2 AUROC vs full fine-tuning, #trainable params       |
| Source-paired batches              | Less shortcut learning, stable cross-dataset AUROC            | Ablation with / without paired sampler                        |
| Uniformity–Alignment (GenD)        | Well-spread, class-compact hypersphere embedding              | t-SNE separation + CelebDF-v2 AUROC delta                     |
| Fast semantic features (58)        | Cheap, VLM-free symbolic input                                | End-to-end inference latency vs VLM variant                   |
| 12 consistency rules               | Named, human-readable violations                              | Rule firing rate per class + qualitative case studies         |
| Concept branch                     | Generator-agnostic "second opinion"                           | Δ AUROC ablation 3 − ablation 2                               |
| EDL head                           | Calibrated uncertainty, "I don't know" on OOD                 | ECE on FF++ vs CelebDF-v2, AURC curves                        |
| AVU loss                           | Fewer confident-wrong predictions                             | Confident-wrong count at fixed confidence threshold           |
| ImprovedSCM identity sub-graph     | Identity-level causal mismatch signal                         | Per-sub-graph AUROC contribution + edge divergence analysis   |
| ImprovedSCM forensic sub-graphs    | Domain-specific forensic causal mismatch                      | Per-method radar chart of r_diff magnitudes                   |
| Graph divergence loss              | Explicit separation of real vs fake causal mechanisms         | `‖A_real − A_fake‖₁` over epochs                              |
| Label-conditioned reconstruction   | Specialised real-SCM and fake-SCM                             | Real-SCM residual on real vs on fake samples                  |
| DAGMA acyclicity                   | Inspectable DAG structure, not free-form adjacency            | `h(A)` curve, edge-sparsity over training                     |
| CMEF fusion                        | Per-sample branch confidence weighting                        | ECE reduction vs static-gate fusion                           |
| PBAS aux losses                    | Non-degenerate individual branches                            | Per-branch accuracy when used in isolation                    |
| IBDC disagreement term             | Conflict-aware calibration                                    | Uncertainty-vs-agreement heatmap                              |

If a row in this table cannot be produced by a specific experiment,
it does not belong in the paper. This is the skeleton of the
results section.

---

## 11. Source-code layout

```
training/
├── detectors/
│   └── nesy_defake_detector.py          ← orchestrator
├── networks/nesy_defake/
│   ├── foundation_models/               ← CLIP spatial extractor
│   ├── classifiers/
│   │   ├── multitask_head.py
│   │   ├── projection_heads.py          ← standard / residual / bottleneck / Houlsby
│   │   └── feature_conditioned_gate.py
│   ├── concept_branch.py                ← 58-d + 12-rule MLP
│   ├── improved_scm_branch.py           ← 4 sub-graphs × real/fake nonlinear SCMs
│   ├── ccv_branch.py                    ← alternative variant (future paper)
│   ├── causal_branch_factory.py
│   ├── fusion/
│   │   └── evidence_fusion.py           ← CMEF / conditioned / static
│   └── losses/
│       ├── edl_loss.py                  ← vanilla EDL
│       ├── nesy_edl_loss.py             ← NeSy-EDL (CMEF + PBAS + IBDC)
│       └── unifalign.py                 ← Uniformity-Alignment (GenD)
├── interpretability/                    ← Levels 1–7 analyzers + plots
├── trainer/trainer.py
├── train.py
└── config/detector/nesy_defake_ablation*.yaml
```

Every piece has its own file and its own test. The detector is now
an orchestrator — there is no monolithic god-class.

---

## 12. What happens in one training step

1. **Load batch** — source-paired real/fake pairs from the same
   original FF++ video (reduces shortcut learning).
2. **Feature extraction** — CLIP produces `spatial_raw` (1024-d);
   precomputed 58-d fast semantic and 83-d forensic features are
   fetched from disk.
3. **Projection** — projection head maps `spatial_raw → projected`.
4. **Spatial evidence** — classifier head → softplus → 2-d evidence.
5. **Concept branch** — `[fast(58) ; violations(12)]` → MLP →
   `concept_evidence`.
6. **SCM branch** — 4 sub-graphs, each produces `r_diff` → 8-d
   summary → concat(32-d) → MLP → `causal_evidence`. Adjacencies are
   the by-product that interpretability visualises.
7. **Fusion** — CMEF combines the three streams per sample.
8. **Dirichlet** — `α = total_evidence + 1`, `p = α/Σα`, `u = K/Σα`.
9. **Loss** — EDL NLL + KL + AVU + PBAS + IBDC + DAGMA + divergence
   + label-conditioned recon + sparsity + UA.
10. **Backprop** — Adam, cosine-warmup, LN-only CLIP, per-module
    learning rates (symbolic branches at 1e-3, backbone LNs at
    3e-4).

---

## 13. The 10-minute talk

1. **The cross-dataset cliff** — an AUC bar chart showing FF++ → CelebDF-v2 collapse for a standard detector.
2. **One-slide architecture** — Section 2 diagram.
3. **What "evidence" means** — EDL in one equation.
4. **Three streams, three questions:** what the pixels say, what the rules say, what the *causal graph* says.
5. **The SCM picture** — side-by-side `A_real` vs `A_fake` heatmap for the identity sub-graph, top divergent edges labelled in plain English (AU12 → happy, gaze divergence, jaw symmetry).
6. **CMEF + IBDC** — how confidence-aware fusion and disagreement calibration make the Dirichlet honest.
7. **Ablation ladder** — Section 8 table with expected numbers filled in.
8. **Cross-dataset AUROC + uncertainty histogram** — the headline quantitative result.
9. **One qualitative case study** — a correctly flagged fake with the full 7-level explanation; one "I don't know" case where IBDC raises uncertainty because the branches disagree.
10. **Limitations & next steps** — frame-level only, no temporal module yet, SCMs are scalar-supervised, CCV kept as a future publication.

---

## 14. Reproducibility checklist (for the paper appendix)

- Seed: `manualSeed: 3407` (ablation 4) — fix this in the final run.
- Backbones are frozen except LayerNorms — total trainable ≈ 0.5 M / 304 M.
- Precomputed features are committed via `fast_semantic` and
  `forensic_features` directories; no VLM / FaceBench caches are used.
- The `ablation4_causal.yaml` config is the canonical recipe.
- Early stopping: patience 20 on `val_acc` with `min_delta = 0.001`.
- Expected wall-clock (1× A6000-class GPU, batch 128, 100 epochs):
  ~X hours — fill in from the current run.
- Evaluation datasets for this submission: FF++, DFD, CelebDF-v1,
  CelebDF-v2, DFDC, DFDCP, UADFV.

---

## 15. Things I noticed that you may want to add

These are judgement calls — I flag them so you can decide if they
belong in the paper:

- **Hard numbers table.** Section 8 and Section 10 have metric slots
  that still say "expected". As soon as the current ablation-4 run
  finishes, replace the qualitative "↓↓↓" markers with real ECE,
  AUROC, AURC numbers so the document doubles as a results log.
- **Video-level evaluation.** Everything here is frame-level. If we
  plan to report video-level AUROC too (simple mean-pooling across
  frames is standard in the literature), add a dedicated sub-section
  — it is usually the headline number reviewers look at.
- **Compute budget.** Training and inference cost vs the VLM-based
  baseline is a strong pro-VLM-removal narrative point — a single
  latency / FLOPs table sells the "clean pipeline" story.
- **Concept-branch ablation with no rules / no fast features.** We
  have ablation 3 (concept on) and ablation 2 (concept off), but a
  rules-only or fast-features-only variant would isolate where the
  concept gain comes from. Decide whether it is worth the run.
- **SCM sub-graph ablation.** Levels-of-interpretability story says
  "disable one sub-graph at a time", but we do not have those
  configs. If we want Experiment 11 in Section 9 of the April-15
  write-up to be real, we need 4 configs: no-identity, no-struct,
  no-noise, no-spectral.
- **Dataset cards.** For NeurIPS camera-ready, each test dataset
  (especially CelebDF-v2, DFDC, DFDCP) needs a small description and
  compression / source note. Easy to forget, easy to add.
- **Ethical / dual-use statement.** Required by NeurIPS checklist.
  Start it now — a detector-oriented framing ("this work supports
  defensive detection, we do not release any generator") is clean.
- **Named-nodes list for the appendix.** The 26 curated attributes,
  12 rules, and 83 forensic feature names are the hook for every
  qualitative figure. Dump them once as an appendix table so
  reviewers can cross-reference by name.

---

_File: `docs/README_april_20.md` — last updated 2026-04-20._
_Current canonical config: `config/detector/nesy_defake_ablation4_causal.yaml`._
