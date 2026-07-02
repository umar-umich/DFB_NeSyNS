# NeSy-DeFake — Interpretability & Explainability
### State of the art as of 2026-04-22

This document complements `README_april_20.md` (which describes the
detector and training recipe). It focuses on the **interpretability
toolbox**: what the system emits at test time, where each artifact
lives on disk, how each level maps back to the seven-level taxonomy
in Section 9 of the April-20 write-up, and what each one buys the
paper.

The short version: the `training/interpretability/` package runs end
to end by default inside `training/test.py`. A single invocation of
the test script produces every figure the NeurIPS paper needs — from
the calibration reliability diagram up to the per-sample case-study
gallery and the cross-dataset `fig1_mosaic.png`.

---

## 1. One-line TL;DR

Running `python training/test.py --detector_path … --weights_path …`
now emits, **per test dataset**, a self-contained
`interpretability/` folder covering all seven levels of
explainability, plus a cross-dataset `fig1_mosaic.png` at the top of
the run. No extra flags required.

---

## 2. How it hangs together

```
training/
├── test.py                                   # eval entry point, auto-wires engine
└── interpretability/
    ├── engine.py                             # orchestrator (collect, analyze, visualize)
    ├── visualization.py                      # stateless plotting helpers
    └── analyzers/
        ├── base.py                           # BaseAnalyzer interface
        ├── edl_uncertainty.py                # Level 1
        ├── branch_evidence.py                # Level 2
        ├── consistency_rules.py              # Level 3
        ├── scm_analysis.py                   # Levels 4 + 5
        ├── gate_analysis.py                  # Level 6
        ├── disagreement.py                   # Bonus: NeSy inter-branch view
        ├── selective_prediction.py           # Risk–coverage / AURC
        ├── case_study.py                     # Per-sample paper-figure gallery
        ├── ccv_analysis.py                   # Level 4-CCV (reserved for future paper)
        └── tsne_analyzer.py                  # Opt-in embedding scatter
```

Lifecycle per analyzer:

1. `collect(preds, labels)` — called once per eval batch.
2. `analyze()` — called once after all batches, returns a dict of scalars.
3. `visualize(save_dir)` — writes PNGs / TXT under `save_dir`.
4. `explain_sample(idx)` — returns a one-line textual breakdown
   (used by the per-sample report).

The engine additionally takes:

- `collect_model_params(model)` — snapshot adjacency matrices from
  the SCM branch after inference.
- `set_image_paths(paths)` — feed image paths into
  `CaseStudyAnalyzer` so the gallery can render the actual frames.
- `set_method_labels(label_spe)` — feed per-sample specific-method
  labels (FF-DF, FF-F2F, FF-FS, FF-NT, …) into `SCMAnalyzer` for the
  per-method fingerprint radar.

---

## 3. Mapping README §9 levels → analyzers → files

| § | Level                                           | Analyzer                          | Artifacts                                                                                                                   |
| - | ----------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------|
| 1 | EDL uncertainty per prediction                  | `EDLUncertaintyAnalyzer`          | `uncertainty_histogram.png`, `reliability_diagram.png` (+ ECE in `summary.json`)                                            |
| 2 | Branch-level evidence decomposition              | `BranchEvidenceAnalyzer`          | `branch_evidence.png` (per-class stacked bar)                                                                               |
| 3 | 12 named consistency-rule violations             | `ConsistencyRuleAnalyzer`         | `rule_firing_rates.png`; optional `rule_slice_means.csv` + `rule_error_gap.png` via `--rule_error_analysis`                 |
| 4 | Per-sample causal sub-graph residuals `r_diff_g` | `SCMAnalyzer`                     | `scm/r_diff_radar.png`, `scm/r_diff_hist_{identity,forensic_*}.png`, `scm/dominant_subgraph_frequency.png`                  |
| 5 | Learned DAGs with named nodes                    | `SCMAnalyzer`                     | `scm/{sub}_side_by_side.png`, `scm/{sub}_graph.png`, `scm/{sub}_divergence.png`, `scm/{sub}_top_edges.txt`                  |
| 6 | Evidence-gate values                             | `GateAnalyzer`                    | `gate_distributions.png`                                                                                                    |
| 7 | Forensic feature attribution (per sub-graph)     | `SCMAnalyzer` (dominant sub-graph) | `scm/dominant_subgraph_frequency.png`, `scm/r_diff_radar_per_method.png`                                                    |

Additional analyzers that don't map to a specific level but round out
the story the paper wants to tell:

| Analyzer                            | What it adds                                                                         | Artifacts                                              |
| ----------------------------------- | -------------------------------------------------------------------------------------| ------------------------------------------------------ |
| `DisagreementAnalyzer`              | NeSy stake: how often do neural + symbolic branches disagree, and is `u` correlated? | `disagreement_rates.png`, `uncertainty_by_agreement.png` |
| `SelectivePredictionAnalyzer`       | "Flag for human review" story — AURC vs softmax-margin vs oracle                     | `risk_coverage_curve.png`, `confidence_vs_accuracy.png` |
| `CaseStudyAnalyzer`                 | Per-sample paper figure: frame + gauges + evidence + rules + radar                   | `case_study/{uncertain,confident_correct,confident_wrong}/*.png` |
| `TSNEEmbeddingAnalyzer` (opt-in)    | Embedding structure in `worst` / `best` / `random` slices                            | `tsne/tsne_{mode}.png`                                 |
| `CCVAnalyzer`                       | CCV branch analysis (reserved for future CCV-vs-SCM publication)                     | `ccv_forensic_radar.png`, `ccv_learned_constraints.png`, `ccv_counterfactual_hist.png` |

---

## 4. What the test-time output tree looks like

```
logs/test/<run_name>/
├── fig1_mosaic.png                           # cross-dataset paper-fig-1
├── hparams.yaml
└── <dataset_name>/                           # one per entry in test_dataset
    ├── metrics.csv
    ├── test_predictions.csv
    ├── misclassified_frames.csv
    ├── misclassified_videos.csv
    ├── test/
    │   ├── frame_metrics/
    │   │   ├── test_roc_frame.png
    │   │   ├── test_pr_curve.png
    │   │   ├── test_f1_curve.png
    │   │   ├── test_fpr_fnr_curve.png
    │   │   ├── test_confusion.png
    │   │   ├── test_confusion_norm.png
    │   │   └── test_probs_distribution.png
    │   └── video_metrics/
    │       └── …
    └── interpretability/
        ├── summary.json                      # all analyzer results, JSON
        ├── report.txt                        # human-readable report + per-sample explanations
        ├── uncertainty_histogram.png         # Level 1
        ├── reliability_diagram.png           # Level 1
        ├── branch_evidence.png               # Level 2
        ├── rule_firing_rates.png             # Level 3
        ├── gate_distributions.png            # Level 6
        ├── disagreement_rates.png
        ├── uncertainty_by_agreement.png
        ├── risk_coverage_curve.png
        ├── confidence_vs_accuracy.png
        ├── scm/
        │   ├── r_diff_radar.png              # Level 4
        │   ├── r_diff_radar_per_method.png   # Level 7 / per-method fingerprint
        │   ├── r_diff_hist_identity.png      # Level 4
        │   ├── r_diff_hist_forensic_structural.png
        │   ├── r_diff_hist_forensic_noise.png
        │   ├── r_diff_hist_forensic_spectral.png
        │   ├── dominant_subgraph_frequency.png
        │   ├── subgraph_divergence.png
        │   ├── identity_side_by_side.png     # Level 5 (A_real | A_fake | |diff|)
        │   ├── identity_graph.png            # Level 5 (networkx divergent-edge graph)
        │   ├── identity_divergence.png
        │   ├── identity_top_edges.txt
        │   ├── forensic_structural_*.png/.txt
        │   ├── forensic_noise_*.png/.txt
        │   └── forensic_spectral_*.png/.txt
        └── case_study/
            ├── uncertain/                    # top-K most-uncertain samples
            │   ├── 00_idx<n>.png
            │   └── …
            ├── confident_correct/            # top-K confident & correct
            └── confident_wrong/              # top-K confident & wrong
```

For a 2-dataset eval (FaceForensics++ + Celeb-DF-v2), the cross-dataset
`fig1_mosaic.png` at the root of the run pulls four tiles from each
dataset folder into a grid. Defaults are `reliability_diagram.png`,
`branch_evidence.png`, `scm/r_diff_radar.png`, `risk_coverage_curve.png`.

---

## 5. What each figure is for

### 5.1 EDL uncertainty (Level 1)

- `uncertainty_histogram.png` — density of per-prediction `u = K / Σα`
  split by class. If EDL is working, both modes should lie roughly on
  top of each other with the tail biased toward misclassified
  samples.
- `reliability_diagram.png` — binned confidence vs accuracy with the
  gap bars coloured. ECE is reported in the title and in
  `summary.json`.

Paper use: **Section 7 (loss landscape)** and **Section 8 (ablation
table)**. Expect ablation 2 onwards to produce noticeably lower ECE
than ablation 1.

### 5.2 Branch evidence (Level 2)

`branch_evidence.png` shows the mean sum of evidence produced by the
spatial / concept / causal branches, stacked by class. Answers "which
reasoning pathway drove the prediction on average?". For a healthy
NeSy fusion we want all three stacks visibly non-zero (otherwise PBAS
has collapsed) and the fake side to look different from the real
side (otherwise the branches are redundant).

Paper use: validates that the NeSy claim is real — the symbolic
branches are contributing, not just sitting idle.

### 5.3 Consistency-rule violations (Level 3)

- `rule_firing_rates.png` — mean violation score per named rule, real
  vs fake. Rules are sorted by `|fake − real|` in the JSON; the plot
  uses the natural order for stability.
- With `--rule_error_analysis`: `rule_slice_means.csv`,
  `rule_error_gap.png`, `rule_slice_summary.txt` — slice the same
  violation tensor by TN / FP / TP / FN and rank rules by the
  misleading-gap (`|FP − TN| + |FN − TP|`).

Paper use: table of top-K human-readable rules in the appendix; case
studies citing specific rule names (e.g. `cr_happy_au12`) in the main
text.

### 5.4 SCM per-sample residuals (Level 4)

- `scm/r_diff_radar.png` — four-arm radar, mean `r_diff` per
  sub-graph split by class. Clean visualization of which causal
  domains are discriminative.
- `scm/r_diff_hist_{identity,forensic_structural,forensic_noise,forensic_spectral}.png` —
  full per-sub-graph distributions.
- `scm/dominant_subgraph_frequency.png` — for each class, the
  fraction of samples in which each sub-graph "wins" the argmax.
- `scm/r_diff_radar_per_method.png` — **only emitted when `label_spe`
  is available**. One radar trace per specific method; a "per-method
  causal fingerprint" that argues the SCM is learning
  method-specific artefacts rather than a single generic signal.

Paper use: the r_diff radar is the cleanest one-figure argument that
"different causal sub-graphs catch different kinds of fakes".

### 5.5 SCM adjacency matrices (Level 5)

For each sub-graph {identity, forensic_structural, forensic_noise,
forensic_spectral}:

- `{sub}_side_by_side.png` — `A_real`, `A_fake`, and
  `|A_real − A_fake|` on a shared colour scale. This is the slide-ready
  figure for Section 13 of the April-20 talk.
- `{sub}_graph.png` — networkx-rendered directed graph induced on the
  top-K divergent edges, with blue arrows sized by `A_real` weight
  and red arrows sized by `A_fake` weight (parallel curved arrows so
  both are visible). Makes "the generator broke edge A → B" legible
  without reading a heatmap.
- `{sub}_divergence.png` — divergence-only heatmap with `hot`
  colormap (back-compat for older notebooks).
- `{sub}_top_edges.txt` — ranked text table for grep / citation.

Paper use: the identity sub-graph's side-by-side plot is the
single most-effective visual for the "causal structure differs"
claim.

### 5.6 Evidence gates (Level 6)

`gate_distributions.png` — overlay of `σ(concept_gate)` and
`σ(causal_gate)` across the whole dataset. With static gates it is
two vertical spikes; with conditioned gates it is two full
histograms.

Paper use: one sentence on the average "trust allocation" (e.g. "the
model trusts the concept branch 42% and the causal branch 31% on
Celeb-DF-v2").

### 5.7 Inter-branch disagreement (bonus)

- `disagreement_rates.png` — pairwise and overall disagreement
  between neural/concept/causal predictions.
- `uncertainty_by_agreement.png` — `u` distribution for agreeing vs
  disagreeing samples. A valid IBDC (calibration) signal should push
  `u` up when branches disagree.

Paper use: quantitative evidence that IBDC actually exploits
disagreement rather than ignoring it.

### 5.8 Selective prediction (new)

- `risk_coverage_curve.png` — selective risk vs coverage for three
  rankings: EDL `u` (blue, ours), softmax margin (orange, baseline),
  oracle (grey dashed, lower bound).
- `confidence_vs_accuracy.png` — accuracy on the top-K most-confident
  samples at 10 coverage levels.
- AURC, AURC\_softmax, AURC\_oracle, E-AURC in `summary.json`.

Paper use: operationalizes the "flag for human review" claim. Report
AURC against both the oracle and the softmax-margin baseline; the
gap vs softmax quantifies what EDL uncertainty adds over just reading
the probability margin.

### 5.9 Case-study gallery (new)

For each dataset the analyzer picks three groups:

- **uncertain** — top-K highest `u` samples (where the model is
  honest about not knowing)
- **confident\_correct** — top-K lowest `u` among correctly-classified
  samples (clean wins)
- **confident\_wrong** — top-K lowest `u` among mis-classified
  samples (most damaging failures)

Each sample is rendered as a five-panel PNG:

```
[ frame ]  [ p(fake) ]  [ evidence ]  [ top rules ]  [ r_diff radar ]
           [   u     ]
```

with a footer strip carrying the gate values.

Paper use: the `confident_wrong` group is the most useful for the
"limitations" paragraph; `uncertain` samples are the most useful for
defending the calibration claim (are the flagged samples actually
ambiguous?).

### 5.10 Cross-dataset mosaic

`fig1_mosaic.png` at the top of the run stitches one tile per column
from each dataset's interpretability folder. Defaults:

- Reliability diagram (calibration stability across datasets)
- Branch evidence (does the branch mix change cross-dataset?)
- SCM `r_diff_radar` (does the causal fingerprint shift?)
- Risk-coverage curve (does selective prediction still work OOD?)

Override the tile list by editing the call in `test.py`; the builder
is `interpretability.visualization.build_results_mosaic(...)`.

Paper use: one figure in Section 1 / Section 4 that argues the whole
pipeline carries across datasets rather than just on FF++.

---

## 6. Running it

The engine is on by default. Minimal invocation:

```bash
python training/test.py \
  --detector_path training/config/detector/nesy_defake_ablation4_causal.yaml \
  --weights_path  logs/train/<run>/best_video_auroc.pth
```

### Opt-in flags
- `--no_interpretability` — disable everything (fastest, GenD-style
  metrics only).
- `--tsne_mode {best,worst,random,all}` — enable the t-SNE embedding
  analyzer; also accepts `--tsne_top_k`, `--tsne_feature_key`.
- `--rule_error_analysis` — write the TN/FP/TP/FN rule-slice report.

### Config knobs (under `interpretability:` in the detector YAML)

| key                        | default | effect                                                         |
| -------------------------- | ------- | -------------------------------------------------------------- |
| `graph_viz_top_k`          | 20      | top-K for divergent-edge tables and graph renders              |
| `case_study_samples`       | 6       | K per group in the case-study gallery                          |
| `num_explain_samples`      | 10      | K used for the per-sample report                               |

---

## 7. Mapping to paper sections

| README-April-20 section                       | Figure(s) to use                                          |
| --------------------------------------------- | --------------------------------------------------------- |
| §1 problem framing                            | `fig1_mosaic.png` + one case study from `confident_wrong/`|
| §4 three evidence streams                     | `branch_evidence.png`, `rule_firing_rates.png`            |
| §5 fusion (CMEF + PBAS + IBDC)                | `gate_distributions.png`, `disagreement_rates.png`        |
| §6 why EDL                                    | `reliability_diagram.png`, `risk_coverage_curve.png`      |
| §7 loss landscape                             | reliability + AURC numbers from `summary.json`            |
| §8 ablation ladder                            | `summary.json` ECE / AURC across ablations                |
| §9 seven levels of interpretability           | one-shot walk through the gallery above                   |
| §10 measurable benefits                       | radar + per-method fingerprint + r_diff histograms        |
| §13 ten-minute talk, point 5 (SCM picture)    | `scm/identity_side_by_side.png`, `scm/identity_graph.png` |

---

## 8. What is still missing

These are explicit non-goals for the current test pipeline. They are
cheap to add if we decide they are needed:

- **Video-level aggregation of interpretability signals.** All
  analyzers operate at frame level. For a "per-video uncertainty"
  story, we would need to collapse `u`, `r_diff_g`, and the gate
  values by video id before plotting.
- **Temporal consistency view.** No per-frame-within-video plot
  exists; if a reviewer asks about temporal robustness, we need to
  sort by frame index within each `video_id` and plot `u(t)`.
- **Ablation-index comparison plots.** Each ablation writes its own
  run directory; there is no automatic "compare ablation 1/2/3/4 on
  the same axes" figure. A tiny script reading each run's
  `summary.json` and plotting a side-by-side reliability diagram
  would close this.
- **Node-wise attribution inside each SCM sub-graph.** We show
  top-edge divergence but do not yet rank *nodes* by incoming /
  outgoing divergence. Useful for the appendix.
- **CCV analyzer parity.** CCV outputs
  (`forensic_anomaly_radar.png`, learned-constraint bars,
  counterfactual histograms) only fire when `causal_branch.type ==
  'ccv'`. That remains reserved for the future CCV-vs-SCM paper; the
  SCM pipeline is the NeurIPS target.
- **Interactive HTML report.** Everything is static PNG / CSV / JSON
  today. If the paper uses a supplementary website, we can export
  the same data into a single HTML dashboard with two dozen extra
  lines in `engine.finalize`.

---

## 9. Reproducibility notes

- All figures are generated by the headless matplotlib `Agg` backend
  and use the stateless helpers in
  `training/interpretability/visualization.py`. Re-rendering is a
  pure function of `summary.json` plus the cached numpy arrays held
  on the analyzer instance — no model weights required after
  inference.
- Randomness in t-SNE is seeded via `interpretability.tsne.seed`
  (default 42). Networkx `spring_layout` seed is pinned at 7 inside
  `plot_divergent_graph` so the paper figure is deterministic.
- The `CaseStudyAnalyzer` writes the per-sample gallery only if
  `set_image_paths(...)` has been called (i.e., inside `test.py`).
  Running the engine standalone without image paths still writes a
  `case_study_no_images.txt` marker so silent failures are obvious.
- `SCMAnalyzer.set_method_labels` is optional; without it, the
  per-method radar is simply not emitted.

---

## 10. Quick sanity checks after a new run

1. Open `<run>/fig1_mosaic.png`. If any cell is blank, the
   corresponding dataset's interpretability folder is missing a tile.
2. Open `<dataset>/interpretability/summary.json` and confirm
   `edl_uncertainty.ece < 0.1` (ballpark) and
   `selective_prediction.E_AURC_uncertainty <
   selective_prediction.E_AURC_softmax`. If the EDL E-AURC is worse
   than softmax, the Dirichlet head is not actually adding
   information beyond the probability margin.
3. Open `scm/identity_side_by_side.png`. `A_real` and `A_fake` should
   look visibly different; if they look identical, the graph
   divergence loss has collapsed.
4. Open `case_study/confident_wrong/00_idx*.png`. The top rules
   panel should show at least one rule firing with a recognisable
   name; if it is empty, the consistency-rule detector has not been
   plumbed correctly for this ablation.

If any of these four sanity checks fail, the other figures are not
trustworthy; fix the check first before using the run for paper
numbers.
