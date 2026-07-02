## Implementation status update — 2026-05-02

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


___________________
Files created                                                                                                             
                                                                                                                              
  configs/ablations/full_defakenet_18rules.yaml                                                                             
  configs/ablations/no_ibdc.yaml                                                                                              
  configs/ablations/no_cmef.yaml          (requires patches/ablation_no_cmef.diff)                                          
  configs/ablations/no_pbas.yaml                                                                                              
  configs/ablations/no_causal.yaml                                                                                            
  configs/ablations/no_symbolic.yaml      (requires patches/ablation_no_symbolic.diff)                                        
  configs/ablations/visual_edl_only.yaml                                                                                      
  patches/ablation_no_cmef.diff                                                                                               
  patches/ablation_no_symbolic.diff                                                                                           
  scripts/train_all_ablations.sh          (chmod +x)                                                                          
  scripts/verify_ablation_configs.py                                                                                          
                                                                                   
  All seven configs pass the verifier (python scripts/verify_ablation_configs.py → exit 0).                                   
                                                                                                                            
  Retraining flag — YES                                                                                                       
                                                                                                                              
  The existing full-DeFakeNet checkpoint (most recent:                                                                        
  logs/train/nesy_defake_ablation4_causal_2026-04-24-01-59-25_exp/best_avg.pth) was trained on the 12-rule v7 set —           
  concept_mlp.1.weight.shape == [64, 70]. The new 18-rule retained set requires [64, 76]. The full DeFakeNet must be retrained
   (this is full_defakenet_18rules in the launcher) before any ablation comparison is meaningful. There is no usable        
  checkpoints/defakenet_full.pth; the current state lives under logs/train/....

  Recommended training order + rough GPU-hours                                                                                
                                                                                   
  The launcher is already ordered by paper-priority. GPU-hours below assume a single H200 / H100 at the existing batch size of
   128 and the existing nEpochs: 100 budget — actual budget per ablation depends on early-stopping (patience: 20) which     
  typically resolves around epoch 25–35 in your prior runs:                                                                   
                                                                                                                            
  ┌───────┬────────────────────────┬────────────────────┬─────────────────────────────────────────────────────┐               
  │ Order │        Ablation        │ Expected GPU-hours │                         Why                         │
  ├───────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────────────┤               
  │ 1     │ full_defakenet_18rules │ ~14–18 h           │ Full pipeline; new headline checkpoint (mandatory). │             
  ├───────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────────────┤
  │ 2     │ no_ibdc                │ ~13–17 h           │ Headline-claim ablation.                            │               
  ├───────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────────────┤               
  │ 3     │ no_symbolic            │ ~12–16 h           │ Concept stream skipped → modestly faster.           │               
  ├───────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────────────┤               
  │ 4     │ no_causal              │ ~10–14 h           │ No causal SCM pairs → notably faster.               │             
  ├───────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────────────┤               
  │ 5     │ no_cmef                │ ~14–18 h           │ Same architecture; only fusion math changes.        │             
  ├───────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────────────┤               
  │ 6     │ no_pbas                │ ~14–18 h           │ Same architecture; only loss term changes.          │             
  ├───────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────────────┤               
  │ 7     │ visual_edl_only        │ ~7–10 h            │ Backbone only; fastest.                             │             
  └───────┴────────────────────────┴────────────────────┴─────────────────────────────────────────────────────┘               
                                                                                                                            
  Total budget ≈ 80–110 GPU-hours end-to-end on one H200. If interrupted, rows 1–4 alone produce a defensible ablation table. 
                                                                                                                            
  Two commands the user runs                                                                                                  
                                                                                                                            
  │ 3     │ no_symbolic            │ ~12–16 h           │ Concept stream skipped → modestly faster.           │
  ├───────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────────────┤
  │ 4     │ no_causal              │ ~10–14 h           │ No causal SCM pairs → notably faster.               │
  ├───────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────────────┤
  │ 5     │ no_cmef                │ ~14–18 h           │ Same architecture; only fusion math changes.        │
  ├───────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────────────┤
  │ 6     │ no_pbas                │ ~14–18 h           │ Same architecture; only loss term changes.          │
  ├───────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────────────┤
  │ 7     │ visual_edl_only        │ ~7–10 h            │ Backbone only; fastest.                             │
  └───────┴────────────────────────┴────────────────────┴─────────────────────────────────────────────────────┘

  Total budget ≈ 80–110 GPU-hours end-to-end on one H200. If interrupted, rows 1–4 alone produce a defensible ablation table.

  Two commands the user runs

  patch -p1 < patches/ablation_no_cmef.diff
  patch -p1 < patches/ablation_no_symbolic.diff      # apply the two patches once
  python scripts/verify_ablation_configs.py          # confirms all 7 configs OK
  bash   scripts/train_all_ablations.sh              # trains every ablation in priority order

  After each run completes, copy/symlink logs/train/<config>_<timestamp>_exp/best_avg.pth → checkpoints/<config>.pth so the
  downstream evaluators (scripts/run_all.sh) find them.


d4bc741e-427d-4c18-bd32-4708f56efcec