# Ablation Probes — run manifest

Eight short (6-epoch) ablation probes derived from
`configs/ablations/full_defakenet_18rules.yaml` by `_generate_probes.py`.
Each isolates one component of the NeSy-DeFake pipeline. Regenerate with:

```bash
cd training && python config/detector/probes/_generate_probes.py
```

## Prerequisites (read first)

- **Conda env:** `dfb_nesy` — `/data/umar/miniconda3/envs/dfb_nesy/bin/python`.
- **Working directory:** the **repo root** (`/data/umar/Repos/DFB_NeSyNS`). Both the
  dlib landmark file (`preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat`,
  loaded by `dataset/fwa_blend.py`) and `configs/retained_predicates.yaml` are resolved
  **relative to the CWD**, so commands must be launched from the repo root.
- **Precomputed features on disk:** `fast_semantic` (58-d) and `forensic_features` (83-d)
  for FaceForensics++ and Celeb-DF-v2 (concept + causal branches consume these).
- All probes share: `nEpochs=6`, `save_epoch=6`, `early_stopping.enabled=false`,
  `tta.enabled=false`, `manualSeed=3407`, `test_dataset=[FaceForensics++, Celeb-DF-v2]`,
  `log_dir=logs/probes/<name>`.

Shorthand used below:
```bash
PY=/data/umar/miniconda3/envs/dfb_nesy/bin/python
```

## What each probe isolates

| probe | ablation_mode | key delta | isolates |
|---|---|---|---|
| `p1_spatial_ce`   | spatial_ce  | spatial-only, CE loss            | GenD spatial baseline |
| `p2_spatial_edl`  | spatial_edl | + EDL loss                       | value of evidential head vs CE |
| `p3_concept`      | concept_edl | substrate_mode=both              | full concept branch (substrate + rules) |
| `p3a_rules_only`  | concept_edl | substrate_mode=rules_only        | symbolic rules alone as concept input |
| `p3b_substrate`   | concept_edl | substrate_mode=substrate_only    | dense substrate alone (rules excluded from MLP) |
| `p4_full_scm`     | causal_edl  | causal_branch.type=improved_scm  | full NeSy (spatial + concept + SCM) |
| `p4a_causal_only` | causal_edl  | improved_scm, concept `evidence_in_fusion=false` | causal evidence with concept **evidence** removed from fusion (its violations still feed the SCM) |
| `p4b_full_ccv`    | causal_edl  | causal_branch.type=ccv           | CCV causal branch vs SCM |

## Launch commands

Run from the repo root. Each probe = **train** then **test on the saved checkpoint**.
The trainer writes to `logs/probes/<name>/nesydefake_hybrid_<timestamp>/`; the best
checkpoint is `best_FaceForensics++.pth` there.

```bash
# ---- p1_spatial_ce ----
$PY training/train.py --detector_path training/config/detector/probes/p1_spatial_ce.yaml
$PY training/test.py  --detector_path training/config/detector/probes/p1_spatial_ce.yaml \
    --weights_path logs/probes/p1_spatial_ce/nesydefake_hybrid_*/best_FaceForensics++.pth

# ---- p2_spatial_edl ----
$PY training/train.py --detector_path training/config/detector/probes/p2_spatial_edl.yaml
$PY training/test.py  --detector_path training/config/detector/probes/p2_spatial_edl.yaml \
    --weights_path logs/probes/p2_spatial_edl/nesydefake_hybrid_*/best_FaceForensics++.pth

# ---- p3_concept ----
$PY training/train.py --detector_path training/config/detector/probes/p3_concept.yaml
$PY training/test.py  --detector_path training/config/detector/probes/p3_concept.yaml \
    --weights_path logs/probes/p3_concept/nesydefake_hybrid_*/best_FaceForensics++.pth

# ---- p3a_rules_only ----
$PY training/train.py --detector_path training/config/detector/probes/p3a_rules_only.yaml
$PY training/test.py  --detector_path training/config/detector/probes/p3a_rules_only.yaml \
    --weights_path logs/probes/p3a_rules_only/nesydefake_hybrid_*/best_FaceForensics++.pth

# ---- p3b_substrate ----
$PY training/train.py --detector_path training/config/detector/probes/p3b_substrate.yaml
$PY training/test.py  --detector_path training/config/detector/probes/p3b_substrate.yaml \
    --weights_path logs/probes/p3b_substrate/nesydefake_hybrid_*/best_FaceForensics++.pth

# ---- p4_full_scm ----
$PY training/train.py --detector_path training/config/detector/probes/p4_full_scm.yaml
$PY training/test.py  --detector_path training/config/detector/probes/p4_full_scm.yaml \
    --weights_path logs/probes/p4_full_scm/nesydefake_hybrid_*/best_FaceForensics++.pth

# ---- p4a_causal_only ----
$PY training/train.py --detector_path training/config/detector/probes/p4a_causal_only.yaml
$PY training/test.py  --detector_path training/config/detector/probes/p4a_causal_only.yaml \
    --weights_path logs/probes/p4a_causal_only/nesydefake_hybrid_*/best_FaceForensics++.pth

# ---- p4b_full_ccv ----
$PY training/train.py --detector_path training/config/detector/probes/p4b_full_ccv.yaml
$PY training/test.py  --detector_path training/config/detector/probes/p4b_full_ccv.yaml \
    --weights_path logs/probes/p4b_full_ccv/nesydefake_hybrid_*/best_FaceForensics++.pth
```

`test.py` writes, per dataset, `metrics.csv` (contains `auroc_frame`, `auroc_video`) plus
the new `per_sample_<dataset>.csv` / `per_sample_video_<dataset>.csv` diagnostics.

## Post-hoc: CCV NaN fix + re-eval (added 2026-07-31)

`p4b_full_ccv` produced **NaN** `p_fake_causal` on **DFDC** and **no results** on
**DeepFakeDetection** — the CCV branch's counterfactual mismatch (`.sum()` over 32 dims,
unbounded) and forensic reconstruction error overflowed `softplus` on extreme OOD samples.
`training/networks/nesy_defake/ccv_branch.py` was hardened (no behavior change on
in-distribution inputs): bounded `log1p(mean)` mismatch, `nan_to_num`+clamp on anomaly /
learned-violation signals, evidence logits clamped to `[-30, 30]` before softplus, and a
warn-and-sanitize guard on non-finite evidence (logs the dataset + count). The fix lives in
the **model forward**, so it applies at inference — **re-eval the existing checkpoint, no
retrain needed** (the hardening added no parameters; the p4b checkpoint loads unchanged).

`p4b_fix_eval.yaml` is `p4b_full_ccv.yaml` with `test_dataset: [DFDC, DeepFakeDetection]`.
Run from the repo root; `--output_dir` keeps it separate from the original p4b test folder:

```bash
# ---- p4b_fix_eval — re-test the EXISTING p4b checkpoint on the 2 NaN datasets ----
# launched: 2026-07-31 __:__   done: ____-__-__   (fill in to track job status)
$PY training/test.py \
    --detector_path training/config/detector/probes/p4b_fix_eval.yaml \
    --weights_path logs/train/p4b_full_ccv_2026-07-30-12-41-00/best_avg.pth \
    --output_dir   logs/test/p4b_fix_eval
# → writes logs/test/p4b_fix_eval/{DFDC,DeepFakeDetection}/metrics.csv
#   + per_sample_{DFDC,DeepFakeDetection}.csv  (should now be NaN-free)

# ---- per-branch standalone AUC (offline, no rerun) ----
# launched: 2026-07-31 __:__   done: ____-__-__
$PY scripts/branch_auc.py --logs_dir logs/test --out logs/probes_summary/branch_auc.csv
# frame/video ROC-AUC of p_fake_spatial / p_fake_concept / p_fake_causal (and fused
# p_fake) for every run on disk; non-finite entries are dropped per-branch and counted.
```

After the re-eval, re-run `branch_auc.py` (or point `--run logs/test/p4b_fix_eval`) to get the
now-finite causal-branch AUC on DFDC / DeepFakeDetection.

## Expected runtime

**Ballpark only — measure the first run and replace these.** Assumes 1×H200, FF++ c23
(~700 train videos × 32 frames ≈ 22k frames, ~175 iters/epoch at batch 128), frozen CLIP
ViT-L/14. The frozen backbone forward dominates, so per-epoch time is similar across probes;
symbolic/causal branches add ~10–25%. Test evaluates FF++ + CDFv2 at 32 frames/video.

| probe | train (6 ep) | test (FF++ + CDFv2) | notes |
|---|---|---|---|
| p1_spatial_ce   | ~1.5–2 h | ~15 min | lightest (no EDL/branches) |
| p2_spatial_edl  | ~1.5–2 h | ~15 min | + EDL head |
| p3_concept      | ~1.5–2 h | ~15 min | + concept MLP |
| p3a_rules_only  | ~1.5–2 h | ~15 min | concept MLP (18-d input) |
| p3b_substrate   | ~1.5–2 h | ~15 min | concept MLP (58-d input) |
| p4_full_scm     | ~2–2.5 h | ~15–20 min | heaviest: + 4 nonlinear SCMs + DAG/recon penalties |
| p4a_causal_only | ~2–2.5 h | ~15–20 min | as p4 (concept evidence off) |
| p4b_full_ccv    | ~2–2.5 h | ~15–20 min | + CCV branch |

Full sweep (8 probes, train + test): **~16–20 h** end-to-end on one GPU; parallelize across
GPUs to compress.

## Results (compiled 2026-08-01)

Auto-compiled by `compile_results.py` from each probe's latest
`logs/test/<probe>_*/‹dataset›/metrics.csv`. Regenerate with:

```bash
$PY training/config/detector/probes/compile_results.py
# → logs/probes_summary/{all_metrics.csv, results_auc.*, results_headline.md, results_diagnostics.csv}
```

Headline (frame/video AUC). **p4b_full_ccv's DFDC + DeepFakeDetection are read
from the `p4b_fix_eval` re-eval** (CCV NaN fix) via a per-dataset override in
`compile_results.py`; all other cells come from each probe's main run.

| probe | FF++ frame AUC | CDFv2 frame AUC | CDFv2 video AUC |
|---|---|---|---|
| p1_spatial_ce   | 0.9328 | 0.8701 | 0.9200 |
| p2_spatial_edl  | 0.9395 | 0.8823 | 0.9349 |
| p3_concept      | 0.9398 | 0.8862 | 0.9409 |
| p3a_rules_only  | 0.9372 | 0.8886 | 0.9422 |
| p3b_substrate   | 0.9406 | 0.8940 | 0.9444 |
| p4_full_scm     | 0.9368 | 0.8867 | 0.9426 |
| p4a_causal_only | 0.9350 | 0.8876 | 0.9406 |
| p4b_full_ccv    | 0.9399 | 0.8931 | 0.9466 |

Full 6-dataset matrix (frame AUC) is in `logs/probes_summary/results_auc.md`;
every metric × (probe, dataset) with `run_dir` provenance is in `all_metrics.csv`.

### p4b_full_ccv — cross-dataset frame / video AUC (post-fix, complete)

| dataset | frame AUC | video AUC | source run |
|---|---|---|---|
| FaceForensics++   | 0.9399 | 0.9851 | p4b_full_ccv (main) |
| Celeb-DF-v2       | 0.8931 | 0.9466 | p4b_full_ccv (main) |
| Celeb-DF-v3       | 0.8668 | 0.9464 | p4b_full_ccv (main) |
| DFDCP             | 0.8400 | 0.8622 | p4b_full_ccv (main) |
| DFDC              | 0.8023 | 0.8282 | **p4b_fix_eval** |
| DeepFakeDetection | 0.8817 | 0.9219 | **p4b_fix_eval** |
