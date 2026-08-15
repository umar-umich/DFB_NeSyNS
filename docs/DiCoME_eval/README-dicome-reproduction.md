# DiCoME Reproduction on Our 6-Dataset Evaluation Setup

Operational README for reproducing Kang et al.'s DiCoME (ICML 2026, "Divide and Conquer: Reliable Multi-View Evidential Learning for Deepfake Detection") on our existing evaluation datasets. This is preparation for the DISCERN v2 T-BIOM extension — DiCoME is the current SOTA on our benchmarks and must be reproducible in-house before we can claim any head-to-head result.

---

## Why this task

DiCoME's paper Table 1 reports video-level AUC of `0.977 / 0.982 / 0.882 / 0.993 / 0.911 / 0.886` on CDFv2 / DFD / DFDC / DFo / WDF / CDFv3, beating GenD (WACV'26) and Effort (ICML'25). For any comparison in the T-BIOM extension, we need those numbers reproduced on our own data pipeline. Cross-comparison against results-in-their-paper is not sufficient because preprocessing differences alone can create 1–3 AUC of noise.

The task uses their released checkpoint `dicome-best.ckpt` from HuggingFace `kxl0825/DiCoME`. No training.

---

## Prerequisites

- **Repo**: DiCoME cloned at `../DiCoME/` relative to the Claude Code working directory. Their README already read; the prompt below assumes their documented workflow.
- **Data**: our 6 evaluation datasets already present in the project directory (paths defined in our existing data config).
- **Compute**: one GPU is sufficient for eval. Our 2× H200 (148 GB) is more than enough.
- **Disk**: H5 conversion can produce tens of GB per dataset. Estimate before launching Phase 3.
- **Env**: fresh conda env `dicome` will be created — do not reuse an existing env.
- **Credentials**: HuggingFace anonymous access is sufficient for the DiCoME checkpoint; no token needed.

---

## How to use

1. Paste the full prompt (below, section "Claude Code prompt") into Claude Code.
2. Let it run **Phase 1 (data discovery) and Phase 2 (env + checkpoint)** to completion. Both are lightweight and safe.
3. Review the discovery report at `../DiCoME/eval_adaptation/discovery.md` before allowing Phase 3.
4. Phase 3 (H5 conversion) is heavy — check the disk estimate the agent prints and launch the conversion script manually.
5. Phase 4 writes per-dataset configs and verifies loaders. Review these before Phase 5.
6. Phase 5 prepares the eval commands. Launch `run_eval_all.sh` manually.
7. After eval completes, the agent runs Phase 6 to aggregate and interpret.

Each phase has explicit gates. Do not skip them — the ASK-UMAR triggers are there for cases where silent assumptions would compromise the reproduction.

---

## Claude Code prompt

Paste everything from here to the end of the code block:

````markdown
# Task: Evaluate DiCoME on our 6 evaluation datasets (paper-style, separate per dataset)

## Goal
Evaluate DiCoME's released checkpoint (`dicome-best.ckpt`, HuggingFace `kxl0825/DiCoME`) on the 6 evaluation datasets we already have set up in this project. Report per-dataset video-level AUC and compare against their paper Table 1. Do NOT train — evaluation only.

## Context (from their README)
- Repo at `../DiCoME`. README already read; no need to re-summarize it, just work from it.
- Env: `conda create -n dicome python=3.12.3` then `pip install -r requirements.txt`. Their verified stack: torch 2.6.0+cu118, torchvision 0.21.0+cu118, lightning 2.5.0, transformers 4.50.0, peft 0.14.0.
- Weights: download `dicome-best.ckpt` from HuggingFace `kxl0825/DiCoME`, place at `../DiCoME/weights/dicome-best.ckpt`.
- Data format: H5 files + txt split files. Convert with `tools/data/folder_to_h5_dataset.py` and `tools/data/h5_to_split_txt.py`. Each txt line is an H5 key like `CDFv2/Celeb-synthesis/id0_id16_0003/000.png`.
- Config: `src/config/dicome_default.yaml`. `tst_h5_path` and `tst_files` control test data. Both can be single or dict form (dict = per-dataset).
- Test command: `python tools/train/train_dicome.py test weights/dicome-best.ckpt --config_path src/config/dicome_default.yaml`.
- Their README explicitly recommends **separate test runs per dataset for paper-style reporting**. Do that.
- Output: prediction CSVs under `runs/`. Video-level AUC is their reported metric.
- Backbone: `openai/clip-vit-large-patch14` (public, will auto-download via transformers).
- Compute: 2× H200, 148 GB. One GPU is plenty for eval.

## Contract
- 🟢 YOU-RUN: env setup, HF checkpoint download, discovery of our data layout, writing per-dataset configs, verification scripts, aggregation, reporting.
- 🔴 UMAR-RUNS: H5 conversion for each of the 6 datasets (potentially large), and the actual six `test` runs.
- 🟡 ASK-UMAR: any decision that would rename data paths, change split-file semantics, or require regenerating our extracted frames.
- No invented numbers. Every AUC in the final report comes from an actual run.
- Do NOT modify anything under our project's data directories. Write H5s and splits into `../DiCoME/eval_adaptation/data/` only.
- Do NOT modify DiCoME's model code or their `src/config/dicome_default.yaml`. Write our own per-dataset config copies under `../DiCoME/eval_adaptation/configs/`.

## Phase 1 — Data discovery (YOU-RUN)

1. Find our project's data config in the current working directory. Enumerate the 6 evaluation datasets by name, root path, frame layout (single directory of PNG/JPG per video? nested by identity?), and any split files we already maintain (train/val/test lists).
2. For each dataset, identify: (a) the real-face subset(s) and their directory names, (b) the fake subset(s) and their directory names, (c) the video-id → frames mapping, (d) which frames are test-set frames.
3. Map each dataset's directory structure to DiCoME's expected H5 key format `{DatasetName}/{split}/{video_id}/{frame}.png`. Write the mapping table to `../DiCoME/eval_adaptation/discovery.md`.
4. Compare our 6 datasets against DiCoME's paper 6 (CDFv2, DFD, DFDC, DFo, WDF, CDFv3). Note matches and mismatches. If any of our 6 aren't in their 6 (or vice versa), flag it and 🟡 ASK-UMAR whether to proceed with our set or match theirs.

Stop here for my review before Phase 2 if the mapping requires any assumption about frame-to-video attribution or class labels.

## Phase 2 — Env + checkpoint (YOU-RUN)

1. Create conda env `dicome` per their README. `conda create -n dicome python=3.12.3 -y`, activate, `pip install -r ../DiCoME/requirements.txt`.
2. Download `dicome-best.ckpt` from `https://huggingface.co/kxl0825/DiCoME/blob/main/dicome-best.ckpt` to `../DiCoME/weights/dicome-best.ckpt`. Verify file exists and is non-empty.
3. Sanity-check load: `python -c` in the `dicome` env that loads the checkpoint via their `train_dicome.py test` machinery or directly via torch, prints state-dict key count and total parameter count. Their paper reports ~0.868M trainable params on top of frozen CLIP ViT-L/14 (~304M frozen).
4. Run their `compileall` smoke test from the README: `python -m compileall -q src tools` inside `../DiCoME`.

## Phase 3 — H5 conversion (UMAR-RUNS, YOU prepare)

For each of our 6 datasets, prepare (but do not launch) the H5 conversion commands.

1. Write `../DiCoME/eval_adaptation/convert_all.sh` that calls `python tools/data/folder_to_h5_dataset.py --image_root <ours> --output_h5 ../DiCoME/eval_adaptation/data/h5/{DatasetName}.h5` for each dataset. Use one H5 per dataset (matches their per-dataset reporting recommendation).
2. Add `--overwrite` behavior explicitly to each command so re-runs are safe.
3. For each dataset, also write the split-generation command using `tools/data/h5_to_split_txt.py` with the correct `--group_index` (determine by inspecting our directory depth per dataset — this may differ; document per-dataset in `adaptation_notes.md`).
4. Print the full script and per-dataset disk-size estimate (multiply num_videos × frames_per_video × avg_frame_bytes). If any dataset's estimate exceeds 100 GB, flag it — H5 conversion time and disk space are real constraints.
5. Print the script for review. I will run it after checking size estimates and mapping choices.

## Phase 4 — Per-dataset configs (YOU-RUN)

After H5s exist (Phase 3 completes):

1. Under `../DiCoME/eval_adaptation/configs/`, write 6 YAML files, one per dataset. Each is a copy of `src/config/dicome_default.yaml` with only `tst_h5_path` and `tst_files` overridden to point at that dataset's H5 and split txts. Do NOT touch any other field (backbone, feature_dim, etc.).
2. Also write one combined-YAML `all_datasets.yaml` using the dict form for `tst_h5_path` and `tst_files` — for reference / potential future use, not for primary reporting.
3. Write `verify_configs.py` that loads each of the 6 configs via `src.config.load_config`, instantiates the test data loader, and pulls one batch. Report per-dataset: num real samples, num fake samples, tensor shapes. Fail loudly on any dataset that doesn't load.

🟡 ASK-UMAR if any dataset yields fewer real or fake samples than expected (a factor-of-2 or more mismatch to our known dataset sizes) — likely a key-format mismatch in the split txts.

## Phase 5 — Evaluation preparation (YOU-RUN, don't launch)

1. Write `../DiCoME/eval_adaptation/run_eval_all.sh` — six sequential invocations, one per dataset:
```bash
cd ../DiCoME
python tools/train/train_dicome.py test weights/dicome-best.ckpt \
  --config_path eval_adaptation/configs/{DatasetName}.yaml \
  2>&1 | tee eval_adaptation/logs/{DatasetName}.log
```
2. Ensure `../DiCoME/eval_adaptation/logs/` and `../DiCoME/eval_adaptation/results/` directories exist.
3. Print the full script for review. I will launch it.

## Phase 6 — Result extraction + reporting (YOU-RUN, after I run Phase 5)

After I've run the eval, prediction CSVs will be under `../DiCoME/runs/` per their README, plus my per-dataset logs under `../DiCoME/eval_adaptation/logs/`.

1. Inspect one CSV first. Confirm whether it contains frame-level predictions or already video-level. Their paper reports video-level AUC — if the CSV is frame-level, write `aggregate_video_auc.py` that groups by video_id (parsed from H5 key structure `{Dataset}/{split}/{video_id}/{frame}.png`) using mean pooling of frame scores, then computes ROC-AUC.
2. Aggregate to `../DiCoME/eval_adaptation/RESULTS.md`. Columns: Dataset | DiCoME paper AUC | Our reproduction AUC | Δ | Num videos | Num real | Num fake | Notes.
3. Paper reference numbers (Table 1 video-level AUC): CDFv2 0.977, DFD 0.982, DFDC 0.882, DFo 0.993, WDF 0.911, CDFv3 0.886.
4. For CDFv3 specifically: our current setup only includes face-swap. Note this as a limitation in the CDFv3 row — the paper's 0.886 was likely computed on their full CDFv3 splits.
5. Flag any dataset with |Δ| > 0.02 AUC — likely a data preprocessing mismatch. Give a one-line hypothesis about the likely cause (frame sampling, compression, split coverage) in the Notes column.
6. Short interpretation section (15–30 lines) at the end: which datasets reproduced cleanly, which showed gaps, what the gaps suggest about our preprocessing vs. theirs, and whether the reproduction is trustworthy enough to use as our DiCoME baseline number for DISCERN v2 comparisons.

## Explicit non-goals
- Do NOT train anything.
- Do NOT modify our project's data directories or `src/config/dicome_default.yaml`.
- Do NOT modify DiCoME's model code.
- Do NOT re-extract frames — use whatever frames we already have.
- Do NOT report numbers not produced by an actual run.
- Do NOT skip any of the 6 datasets silently. If one can't be evaluated (missing data, unresolvable format mismatch), fail loudly and 🟡 ASK-UMAR.

## Success criteria
- All 6 datasets converted to H5 with matching txt splits.
- All 6 evaluated with real video-level AUC.
- `RESULTS.md` with the comparison table and interpretation exists.
- Reproduction gaps > 0.02 documented with a likely cause.
````

---

## Expected directory layout after completion

```text
../DiCoME/
├── weights/
│   └── dicome-best.ckpt                     # Phase 2
├── eval_adaptation/
│   ├── discovery.md                         # Phase 1
│   ├── adaptation_notes.md                  # Phase 3
│   ├── convert_all.sh                       # Phase 3 (I run)
│   ├── verify_configs.py                    # Phase 4
│   ├── run_eval_all.sh                      # Phase 5 (I run)
│   ├── aggregate_video_auc.py               # Phase 6, if needed
│   ├── RESULTS.md                           # Phase 6
│   ├── data/
│   │   └── h5/
│   │       ├── CDFv2.h5
│   │       ├── DFD.h5
│   │       ├── DFDC.h5
│   │       ├── DFo.h5
│   │       ├── WDF.h5
│   │       └── CDFv3.h5
│   ├── configs/
│   │   ├── CDFv2.yaml
│   │   ├── DFD.yaml
│   │   ├── DFDC.yaml
│   │   ├── DFo.yaml
│   │   ├── WDF.yaml
│   │   ├── CDFv3.yaml
│   │   └── all_datasets.yaml
│   ├── logs/
│   │   └── {dataset}.log                    # from Phase 5
│   └── results/
│       └── {dataset}.json                   # aggregated from CSVs
└── runs/
    └── {dicome's own CSVs per test run}
```

---

## Interpreting the results

The RESULTS.md table gives per-dataset reproduction AUC and delta against the paper. Rough guidance for what the deltas mean:

- **|Δ| ≤ 0.005**: clean reproduction, trustworthy as a DiCoME baseline for DISCERN v2 comparisons.
- **0.005 < |Δ| ≤ 0.02**: minor preprocessing difference (frame count per video, compression setting, crop). Usable but note the caveat when reporting.
- **|Δ| > 0.02**: real mismatch. Investigate before using. Most likely causes: (a) different frame sampling — DiCoME's DeepfakeBench pipeline uses a specific frame extraction protocol we may not match; (b) different compression level (c23 vs c40); (c) different test-split definition; (d) split file schema mismatch causing missing samples.
- **Δ negative (our AUC > paper)**: possible but suspicious. Usually means fewer hard samples in our test split. Verify sample counts match.

For DISCERN v2 head-to-head reporting, use the reproduction AUC — not the paper AUC — as the DiCoME baseline. This eliminates preprocessing as a confound in any DISCERN v2 vs DiCoME comparison.

---

## Known limitations

- **CDFv3 coverage**: our current CDFv3 setup only includes face-swap. DiCoME's paper 0.886 was likely computed on their full CDFv3 splits (face-swap, face-reenactment, entire-face-synthesis, face-edit). The CDFv3 comparison in RESULTS.md will not be an apples-to-apples number. Getting the remaining three splits into our pipeline is a prerequisite before treating this row as a real baseline.
- **H5 disk cost**: 6 datasets × thousands of videos × 32 frames per video × ~50–200 KB per frame can easily exceed 100 GB total. Estimate before launching Phase 3 conversion.
- **DeepfakeBench pipeline drift**: DiCoME's numbers assume DeepfakeBench-style preprocessing. If our frames were extracted with a different face detector, alignment routine, or crop margin, small deltas are expected across all six datasets.

---

## Next steps after this task

1. If reproduction is clean (all |Δ| ≤ 0.02): use these numbers as the DiCoME baseline in the DISCERN v2 T-BIOM comparisons. Move to multi-backbone perception pilot (CLIP + DINOv2 + FSFM standalone probes).
2. If reproduction has systematic gaps (multiple datasets with |Δ| > 0.02): investigate the frame extraction pipeline before proceeding. A DiCoME baseline that doesn't match the paper is worse than no baseline — reviewers will ask.
3. Regardless: fix CDFv3 coverage to include all four splits, and re-run just CDFv3 evaluation once the full data is in place.
4. Add Deepfake-Eval-2024 (Chandra et al., 2025) to the evaluation set — DiCoME never evaluated on it despite citing it, which is one of the cleanest gaps for DISCERN v2 to fill.