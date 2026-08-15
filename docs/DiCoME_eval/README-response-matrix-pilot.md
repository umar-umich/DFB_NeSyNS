# Response Matrix + Biometric-Integrity Pilot Plan

Operational README for the DISCERN v2 T-BIOM pilot. Supersedes all earlier versions of `README-dicome-component-swap-pilot.md` and `README-generation-process-pilot.md`. Companion to `README-dicome-reproduction.md` (eval) and `README-dicome-training-audit.md` (training verification).

---

## Working thesis (do not overclaim yet)

**Biometric-Integrity-Aware Evidential Reasoning under Heterogeneous Face Forgery Shift.**

Empirical motivation from P1 sub-suite: a globally optimized forensic representation can specialize toward particular manipulation regimes. Changing DiCoME's manifold bottleneck substantially trades face-swap performance against reenactment/talking-face performance, despite similar aggregate cross-dataset behavior.

**Important caveat**: the paper does NOT claim yet that DiCoME's per-sample MI penalty *causes* the specialization. That is a hypothesis until an MI-isolation experiment confirms it. Language throughout the pilot must read as "consistent with" not "caused by."

The final architecture is not committed. DiCoME is the experimental chassis for identifying useful forensic mechanisms. The pilot's deliverable is a **response matrix**, not a proposed architecture. The matrix informs the architecture — not the other way around.

---

## What's being built

Two things simultaneously:

1. **Empirical response atlas**: an operator × manipulation-family × behavior matrix showing what each mechanism sees, where it fails, and whether its failures complement others. Behavior includes AUC, score distributions, rescue/harm, uncertainty, and representation structure — not just AUC.

2. **A visualization + analysis package**: shared across all pilots so every mechanism is measured consistently and figures compose across the paper.

The final DISCERN v2 architecture is NOT being built in this pilot. Do not implement it. The atlas comes first.

---

## Pilot execution order

**Immediate (existing checkpoints, no retraining)**:
- **DF40 build + eval on P0-DS + P1a–P1d**. Highest priority — cheap, uses existing checkpoints, reveals whether Direction 2 (rate-distortion signatures) has empirical footing.

**Parallel next**:
- **P3a**: diffusion consistency (LaRE²-style latent/noise reconstruction error), single timestep, clean, no FIRE yet.
- **P4**: official LGrad (LGrad's provided pretrained network + target class scheme).

**Secondary (only if trivial to run concurrently)**:
- **P2a**: LDM first-stage AE reconstruction only (AEROBLADE-style), no pixel/latent/CLIP combination.

**Conditional (only after response matrix is built)**:
- **P1d-isolation**: separates multi-rate training from rate-response features
- **P1e**: MI-isolation to confirm the mechanism claim
- **P3b**: P3a + FIRE frequency decomposition
- **P2b**: P2a + pixel/latent/CLIP residual combination

Deferred entirely: FactorVAE, InfoVAE, VQ-VAE, Denoising Multi-β VAE, FSFM, flow reconstruction operators.

---

## Prerequisites

- Env `discern_ext` (renamed from `dicome` on 2026-08-12), repo at `../DiCoME`, self-trained P0-DS checkpoint in place.
- Existing checkpoints for P1a, P1b, P1c, P1d from prior runs.
- **New data needed**:
  - DF40 H5 build (FS, FR, EFS, FE splits + per-method breakdown; frame-only methods kept separate from video-level)
  - DigiFakeAV if practical (diffusion-based digital-human videos)
- **New downloads**:
  - SD 2.1 UNet + scheduler (for P3a)
  - Official LGrad pretrained network + target class scheme (for P4)
  - SD 2.1 VAE (for P2a, if run)
- Compute: 2× H200. DF40 eval on existing checkpoints ~1 day. Each new pilot: ~2 days.

---

## Contract (all pilots)

- 🟢 YOU-RUN: operator implementation, visualization/analysis package code, evidential head plumbing, config, verification, analysis, reporting, all plot generation.
- 🔴 UMAR-RUNS: training runs, eval runs, dataset conversions.
- 🟡 ASK-UMAR: any change to hyperparameters, seed, evaluation sources, or stacking pilots.
- No invented numbers. Every AUC and every plot value from an actual run.
- Do NOT modify DiCoME's central config or core model code. Work in `../DiCoME/experiments/pilot_{PILOT_ID}/`.
- Do NOT implement the final DISCERN v2 architecture in this pilot cycle. The response matrix comes first.

---

## Shared visualization + analysis package

All pilots use one shared package to enforce consistency. Build first, before running new pilots:

```
../DiCoME/experiments/common/analysis/
├── export_features.py       # sample-level feature/evidence/score export
├── response_matrix.py       # AUC delta heatmaps
├── score_distributions.py   # p_fake and vacuity distributions by family
├── rescue_harm.py           # per-sample rescue/harm quadrants + matrices
├── embedding_viz.py         # PCA + UMAP (skip t-SNE by default — see note)
├── representation_metrics.py # silhouette, DB, linear probe, HSIC, CKA
├── rate_response.py         # P1d-specific rate-distortion analysis
├── uncertainty_viz.py       # error-vacuity + risk-coverage
└── generate_report.py       # aggregates figures + numbers into ANALYSIS.md
```

**Sampling discipline** (mandatory for reproducibility):
- Fixed random seed for all sampling
- Same sample-ID list across pilots wherever possible
- Stratified sampling by real/fake and manipulation family
- For embeddings: 5–20K samples, not full DF40's 800K+ frames
- Save exact sample-ID list to `common/analysis/viz_cohort.csv` so every pilot uses the same visualization cohort

**t-SNE note**: skip t-SNE by default. ChatGPT's plan explicitly warns not to make scientific claims from t-SNE geometry alone. Use PCA for global structure and UMAP for neighborhood structure. Only add t-SNE if a specific claim genuinely needs it, and always alongside quantitative metrics — never t-SNE alone.

---

## Required outputs per pilot (essential only)

Umar's constraint: only meaningful visualizations, not garbage. Every pilot must produce:

**Core (mandatory)**:
- `response/auc_delta_family.png` — V1 AUC response heatmap by family. Paper-figure candidate.
- `response/auc_delta_method.png` — V1 heatmap by DF40 method. Paper-figure candidate.
- `distributions/score_by_family.png` — p_fake distributions per family (Real vs FS vs FR vs TF vs EFS vs FE). Explains AUC movement.
- `complementarity/rescue_harm.png` — CLIP vs operator rescue/harm stacked bars per family + scatter with p_fake_CLIP vs p_fake_operator. Justifies reasoning architecture.
- `complementarity/error_overlap.csv` — per-family error overlap and rescue rates as numbers.
- `features/sample_features.npz` + `metadata.csv` — raw data behind every plot so figures can be regenerated for the paper.
- `ANALYSIS.md` — verdict, response-matrix row, interpretation.

**Supplementary (generate but do not prioritize)**:
- `embeddings/pca_*.png` and `embeddings/umap_*.png` — feature-distribution visualization. PCA for global structure, UMAP for neighborhood. Use only if a clear scientific question hinges on representation geometry.
- `distributions/uncertainty_by_family.png` — vacuity distributions per family. Useful for the DISCERN uncertainty story.
- `complementarity/evidence_correlation.png` — evidence-space correlation matrix across operators.

**Pilot-specific (only where relevant)**:
- P1d only: `rate_distortion/` directory with curves_by_family.png, slope_distribution.png, rate_response_umap.png, rate_response.csv. Potential core contribution if patterns emerge.
- P3/P4 only: raw process-measurement export before evidential head, for later cross-operator comparison.

**Explicitly NOT required (skip unless a specific claim needs them)**:
- Reliability diagrams — calibration is not this paper's core
- Multiple UMAP variants for every representation — one per representation is enough
- t-SNE plots without quantitative backing
- Per-generator uncertainty plots — aggregate to family level unless generator-level tells a distinct story

---

## Baselines

Every downstream pilot's DELTA reports against **P0-DS**: 0.9644 CDFv2, 0.9419 DFD, 0.8825 DFDC, 0.8576 DFDCP, 0.9929 FF++ (from `TRAINING_REPRODUCTION.md`); CDFv3 face-swap-only 0.9516; Deepfake-Eval-2024 0.6918.

**P0-UW is not used as a baseline** — the earlier pilot classified P0-UW as harmful (CDFv3 −0.077, especially FaceReenact −0.123). All new pilots use DS-fusion (P0-DS-style) so operator effects are directly comparable with P0/P1.

---

## Success classification

Not by mean AUC. For each operator, record:
- Standalone AUC + CLIP+operator AUC (per family, per generator)
- Rescue-minus-harm on CLIP errors
- Error overlap and evidence correlation with CLIP
- Distribution structure (score, vacuity)
- Wild-dataset behavior (Deepfake-Eval-2024)

Then classify:

- **Generalist**: transferable improvement across multiple families with positive rescue-minus-harm.
- **Specialist**: strong process/family expertise with useful complementary errors (rescues CLIP on specific families without broad damage). Specialists are NOT failures — they may motivate applicability-aware reasoning.
- **Redundant**: little conditional information beyond CLIP.
- **Harmful**: negative transfer without compensating specialization.

---

## Evaluation hierarchy

Four axes, not one gate:

1. **Conventional cross-dataset**: CDFv2, DFD, DFDC, DFDCP. Continuity with literature.
2. **Main heterogeneous-forgery evaluation** (central):
   - CDFv3 / Celeb-DF++: FS / FR / TF, per generator
   - DF40: FS / FR / EFS / FE, per method, keeping frame-only and video-level methods separate
3. **Modern synthesis/digital-human**: DigiFakeAV if practical. Otherwise skip.
4. **Wild deployment**: Deepfake-Eval-2024. Prominent but not sovereign.

Optional: FakeParts only if the final method contains a local/structural integrity claim. Do not broaden into generic AIGC datasets to inflate dataset count.

---

## Per-pilot prompt template

Paste inside the code block. Fill `PILOT_ID`.

````markdown
# Task: Response-matrix pilot — PILOT_ID

## Pilot specification
- **Pilot ID**: DF40-eval-P0P1 / P3a / P4 / P2a / (later) P1d-isolation / P1e / P3b / P2b
- **Change from baseline**: [describe]
- **Comparison baseline**: P0-DS
- **Fusion**: DS-fusion (do not vary fusion in this pilot cycle)

Refer to `README-response-matrix-pilot.md` for the full plan.

## Prerequisite: shared analysis package

Before implementing the pilot, verify `../DiCoME/experiments/common/analysis/` exists with the modules listed in the README's "Shared visualization + analysis package" section. If not, create it first — it is a prerequisite for every pilot. Include a fixed `viz_cohort.csv` sample-ID list generated from a fixed random seed with stratified sampling by real/fake and manipulation family.

## Phase 1 — Implementation (YOU-RUN)

Create `../DiCoME/experiments/pilot_PILOT_ID/` with standard files. Pilot-specific implementation:

- **DF40-eval-P0P1**: no operator change. Build DF40 H5 following DiCoME's H5 conventions. Keep FS/FR/EFS/FE splits and per-method. Frame-only methods (no video structure — see writeup) get frame-level AUC only, not video-level. Evaluate existing P0-DS + P1a + P1b + P1c + P1d checkpoints on DF40. No retraining. Add DF40 columns to the response matrix.

- **P3a**: frozen SD 2.1 UNet + scheduler. LaRE²-style single-timestep latent reconstruction error. Choose ONE timestep (t=100 or t=250) and stick with it. Feed error map statistics to a fresh evidential head as the second view. No FIRE, no multi-timestep, no frequency decomposition.

- **P4**: official LGrad. LGrad's provided pretrained network + their exact target class scheme. Do not substitute. Any deviation must be labeled "LGrad-inspired" in `notes.md`.

- **P2a**: frozen LDM first-stage AE cycle. Clean AEROBLADE-style pixel-level reconstruction residual only. No pixel/latent/CLIP combination.

Sanity check: instantiate the modified model, print trainable params. P3a/P4/P2a add ~10K–100K on the operator's evidential head. Frozen components (SD UNet, LGrad backbone, LDM VAE) must stay frozen — verify `requires_grad`.

🟡 ASK-UMAR if the operator requires input-preprocessing changes.

## Phase 2 — Operator preparation (YOU-RUN)

Verify frozen models load and produce expected outputs on a small FF++ batch. Save 5 example intermediate outputs to `sanity/` for visual check. Skip for DF40-eval-P0P1 (no new operator).

## Phase 3 — Training run (YOU-RUN prepare, UMAR-RUNS launch)

For P3a / P4 / P2a: 20 epochs on FF++ c23, seed 42, batch 128, LR 1e-4. Same hyperparameters as P0-DS. Skip for DF40-eval-P0P1.

## Phase 4 — Evaluation (YOU-RUN prepare, UMAR-RUNS launch)

Evaluate on the full four-axis set:
- Conventional: CDFv2, DFD, DFDC, DFDCP (+ FF++ as in-domain sanity)
- Heterogeneous-forgery: CDFv3 per FS/FR/TF and per generator, DF40 per FS/FR/EFS/FE and per method
- Wild: Deepfake-Eval-2024
- Modern synthesis: DigiFakeAV if H5 exists, else skip

Results per source → `results/{source}.json`. Also export per-sample features/evidence/scores for the visualization cohort to `analysis/features/sample_features.npz` with `metadata.csv`.

## Phase 5 — Analysis, visualization, DELTA reporting (YOU-RUN)

Use `experiments/common/analysis/` modules. Generate CORE outputs (mandatory) and SUPPLEMENTARY outputs (only if a specific question warrants). Do NOT generate every possible plot — meaningful only.

Write `experiments/pilot_PILOT_ID/analysis/ANALYSIS.md`:

**Response matrix row**: AUC per source (append to `experiments/RESPONSE_MATRIX.md`).

**Complementarity numbers** (for P3a/P4/P2a):
- Per-family rescue-minus-harm
- Error overlap with CLIP
- Evidence correlation with CLIP
- HSIC/MI as supporting diagnostics, not hard success criteria

**Verdict classification**: Generalist / Specialist / Redundant / Harmful, justified from the response pattern.

**Interpretation** (15–25 lines): what the response pattern says about the operator, and what it implies for the biometric-integrity framing. Language must read as "consistent with" not "caused by" for any mechanism claim about MI/TC/rate-distortion.

## Explicit non-goals
- Do NOT implement the final DISCERN v2 architecture.
- Do NOT modify DiCoME's central config or core model code outside the pilot dir.
- Do NOT stack pilots.
- Do NOT change hyperparameters or seed.
- Do NOT run conditional pilots (P1d-isolation, P1e, P3b, P2b) before their gating pilot's verdict.
- Do NOT call P4 "LGrad" if any component was substituted — must be "LGrad-inspired."
- Do NOT generate visualizations that don't answer a specific analytical question (no garbage plots).

## Success criteria
- Operator implementation exists, self-contained.
- Trained checkpoint from full 20 epochs (except DF40-eval which uses existing).
- Four-axis evaluation completed.
- Response matrix row appended.
- Core visualizations generated with source data preserved.
- ANALYSIS.md exists with verdict and interpretation.
````

---

## Response matrix aggregation

After every pilot appends its row to `experiments/RESPONSE_MATRIX.md`, generate `experiments/response_matrix_visualization.md`:

- Master AUC-delta heatmap across all operators × all sources (paper-figure candidate)
- Cross-operator error-overlap matrix E_ij = P(O_i wrong ∧ O_j wrong)
- Cross-operator rescue matrix R_ij = P(O_i wrong, O_j correct)
- Correlation matrix Corr(p_fake_CLIP, p_fake_operator_1, ..., p_fake_operator_n)

These four matrices are the empirical basis for the architecture decision. Do not commit to an architecture before they exist.

---

## Expected directory layout

```
../DiCoME/experiments/
├── common/
│   └── analysis/            # shared visualization + analysis package
│       ├── viz_cohort.csv   # fixed sample-ID list for all pilots
│       └── [modules listed above]
├── pilot_DF40-eval-P0P1/    # existing checkpoints, DF40 columns
├── pilot_P3a/               # LaRE² diffusion consistency
├── pilot_P4/                # official LGrad
├── pilot_P2a/               # AEROBLADE LDM AE (secondary)
├── pilot_P1d-isolation/     # conditional
├── pilot_P1e/               # conditional
├── pilot_P3b/               # conditional
├── pilot_P2b/               # conditional
├── RESPONSE_MATRIX.md       # aggregated across all pilots
└── response_matrix_visualization.md  # master heatmap + cross-operator matrices
```

Each pilot dir has: `operator.py` (if new operator), `model.py`, `config.yaml`, `train.sh`, `checkpoints/`, `logs/`, `results/*.json`, `analysis/` subdirectory with plots and features, `ANALYSIS.md`.

---

## Decision point after immediate pilots complete

After DF40-eval + P3a + P4 finish, inspect the response matrix and cross-operator matrices. Decision tree:

| Observation | Direction |
|---|---|
| Rate-distortion structure clear on DF40 + at least one process operator (P3a or P4) produces complementary evidence | Best case: biometric-integrity reasoning + rate-distortion signatures + process consistency. Run P1d-isolation to confirm rate-distortion claim rigorously. |
| Rate-distortion structure absent, process operators complementary | Applicability-aware cross-operator reasoning. Skip P1d-isolation and P1e. |
| Rate-distortion structure clear, process operators redundant | Rate-distortion forensics + integrity-aware uncertainty. Run P1d-isolation to strengthen the claim. |
| Operators produce specialist expertise with structured trade-offs | Still useful — applicability-aware specialist reasoning. Skip conditional pilots and go straight to architecture design. |
| No operator produces meaningful conditional information | Stop adding architecture. Pivot to reliability under heterogeneous shift / process-diverse training / selective prediction. |

Only after this decision is made should the final DISCERN v2 architecture be designed and implemented.

---

## Timeline

- Shared analysis package build: 1 day
- DF40 build + eval on P0-DS + P1a-d: 2 days
- P3a: 2 days (implementation + train + eval + analysis)
- P4: 2 days (parallel with P3a on separate GPU if possible)
- P2a: 2 days (secondary)
- Response matrix + cross-operator matrices: 1 day
- **Decision point** — commit to a direction
- Conditional pilots (if triggered): 2 days each
- Architecture design + implementation: 5-7 days
- Multi-axis evaluation on final system: 2 days
- Writing: 5-7 days

Total: ~4 weeks realistic. The pilot phase itself (through decision point) is ~2 weeks.

---

## Known gotchas

- **DF40 frame-only vs video-level methods**: some DF40 methods (CollabDiff, MidJourney, deepfacelab, heygen, stargan, starganv2, styleclip, whichfaceisreal) have no video structure. Keep them separate — cannot share a column with video-level rows.
- **pixart excluded**: incomplete download per prior audit. EFSAll family aggregate reported pixart-excluded.
- **Language discipline**: any claim about MI penalty causing specialization must read as "consistent with" not "caused by" until P1e MI-isolation confirms.
- **P4 LGrad discipline**: substituting components while calling it LGrad conflates our contribution with theirs. Any deviation labeled "LGrad-inspired" in notes and paper.
- **Visualization sampling**: fixed sample-ID list mandatory across pilots for fair comparison. Do not draw fresh samples per pilot.
- **PCA vs UMAP vs t-SNE**: t-SNE skipped by default — use only for qualitative inspection alongside quantitative metrics. Never claim clusters "clearly separated" from t-SNE alone.
- **DFD gap**: baseline DFD is 0.9419 vs paper 0.982. Interpret DFD deltas against 0.9419.
- **Single-seed**: |Δ| in 0.005–0.015 needs second seed before conclusion. Larger deltas outside seed noise.
- **CDFv3 face-swap-only limitation**: still applies. Full CDFv3 coverage would fix this.
- **No final-architecture jump**: the strong temptation after seeing initial results is to start building DISCERN v2. Resist until the response matrix + cross-operator matrices are complete.

---

## What NOT to make the paper

Explicitly avoid these framings even if they're numerically supported:

- "MR-VAE improves DiCoME" — too small
- "CLIP + diffusion + LGrad gives better AUC" — reads as heterogeneous ensemble
- "Training with more datasets improves generalization" — weak architectural thesis
- "We beat DiCoME on Deepfake-Eval-2024" — one benchmark is not generalization

The paper's contribution is:
- The response matrix as an empirical characterization of forensic mechanism specialization
- The biometric-integrity decomposition framework for reasoning about which mechanism is applicable per sample
- Applicability-aware evidential reasoning that treats absence of operator-relevant evidence as evidence of authenticity rather than as ignorance
- Whatever architecture the response matrix empirically supports