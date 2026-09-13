# Evaluation protocol v2 (declared 2026-09-13, before any v2 selection result was viewed)

Applies to every run compared from now on: historical and reselected R0, arms A/B/C, and all later
seeds. Supersedes the undeclared newest-file checkpoint choice used by the first SBI drivers.

## 1. Checkpoint selection — `analysis/tbiom/select_checkpoint.py`

| item | rule |
|---|---|
| candidates | every checkpoint the run saved, including `last.ckpt`; epoch read from the checkpoint file |
| data | VALmix only (manifest below). No OOD test export is opened |
| score | **VALmix macro AUROC**: unweighted mean of video-level AUROC over `CDFv2val`, `DFDCPval`, `DFEval24val`. Video score = DS replay of per-branch opinions averaged over frames. Unrounded |
| ties | scores within **1e-6** of the best are tied; **earliest epoch** wins |
| FF++ val | AUROC logged as a **diagnostic only**; never excludes or selects a checkpoint |
| record | epoch, checkpoint path, **sha256**, score, per-domain AUROC and video counts, tied candidates, VALmix manifest hashes → `logs/tbiom/stage23/<arm>_selection_v2.json` |

The earlier selector (`stage23_select.py`, exported `p_fused`) is unchanged, so Stage 2/3/5 selections
remain reproducible. v2 scores the DS replay because every results table uses it.

**VALmix manifest** (`DISCERN_Ext/eval_adaptation/data/splits/VALmix/`), 1,350 videos:

| split file | videos | frames | sha256[:16] |
|---|---:|---:|---|
| CDFv2val-Celeb-real.txt | 160 | 5,116 | `6fa78224588b4157` |
| CDFv2val-Celeb-synthesis.txt | 225 | 7,200 | `d865ab2aa22b30d0` |
| CDFv2val-YouTube-real.txt | 65 | 2,079 | `c5c3b0c6410cef51` |
| DFDCPval-method_A.txt | 216 | 6,247 | `24612d7da6d51ae5` |
| DFDCPval-method_B.txt | 9 | 276 | `2821ff4b36fc6362` |
| DFDCPval-real.txt | 225 | 6,202 | `02e8583a2cd7e148` |
| DFEval24val-fake.txt | 225 | 6,477 | `c127e1d1332f028f` |
| DFEval24val-real.txt | 225 | 6,545 | `aefd8af611ad7860` |

Full hashes are written into each selection JSON. VALmix is an authorised development set; its
checked disjoint by video path from the CDFv2 (518), DFDCP (654) and DFEval24 (814) test splits: 0 shared videos in each.

## 2. Operating threshold

τ = EER of the video-level DS-replay score on **FF++ val**, computed for **the selected checkpoint**.
Never carried over from a predecessor checkpoint or another run.

## 3. Run naming

| name | meaning |
|---|---|
| **R0-historical** | the Phase F result as originally reported: epoch 5, chosen on FF++ val AUROC. Preserved unchanged |
| **R0-v2** | the same training run with its checkpoint reselected under §1 |
| A-v2, B-v2, C-v2 | SBI arms under §1 |

Old SBI numbers (newest-file rule) are labelled **pre-v2** and not used for decisions.

## 4. Corrected comparison table (per run)

- selected epoch, sha256, selection score, per-domain VALmix AUROC, tied candidates, FF++ val diagnostic
- τ (FF++ val EER)
- six OOD sets: video AUROC, FPR_real and TPR at τ, video counts
- held-out processing (jpeg q40, resize 0.60 on CDFv2 and DFDCP): **absolute** clean and processed
  FPR_real and TPR at τ, change from clean, processed AUROC
- **retrospective, labelled as such**: processed TPR at matched FPR (0.05 and 0.10)
- seed count; everything so far is one seed

## 5. Synthetic provenance (SBI arms, from 2026-09-13)

Every training item carries `sample_kind`, `parent_key`, `replaced_slot_key`, `synth_seed`, `proc_kind`,
`proc_value`. SBI items use the parent frame's `video`/`identity`/`frame` and a dedicated SBI `source`
id (100); the replaced slot is kept in `replaced_slot_key` and `idx`. Each SBI sample is exactly
regenerable from (parent, seed, processing) — verified bit-identical on 10 samples. The RNG is created per DataLoader worker.

Reproducibility required one more fix: albumentations 1.4.24 seeds every transform from OS entropy, and SBI builds
new transforms on each call, so seeding `random`/`np.random` alone did not reproduce a blend. Transforms now take
their seed from NumPy's global state and SBI's persistent pipelines are reseeded per sample (`sbi_synth.py`).
The shared `training/dataset/sbi_api.py` is unchanged. Check: `analysis/tbiom/sbi_provenance_check.py`.

Consequence for existing results: **arm C (and B's processing draws) were trained before this fix**,
with metadata inherited from the replaced slot and an RNG duplicated across the two workers. Those
checkpoints are kept and scored under v2 selection, but a retrained C would not be the same recipe.

## 6. Firewall notes

- Development-exposed evaluation data are listed in `decision-2026-09-13.md`.
- CDFv3 and DF40 are separate test sets; generator names overlap and the genuine pools may be the same
  178 videos (`manifests-cdfv3-df40.md`).
- Below-chance CDFv3 subsets are verified as label-correct, manifest-correct and column-correct; scores are
  never reversed on the basis of target results.
