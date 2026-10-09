# PE-Core-L14-336 temporal branch (copy of VJEPA2_TemporalBranch, 2026-09-30)

Identical to the V-JEPA 2 branch documented below except for the backbone. Same data, splits, Arm-B augmentation,
evidential loss, LoRA (r=8, alpha=16, dropout 0.1), optimiser, learning rates, 10 epochs, batch 32 (one pass, no
accumulation) and epoch selection. Same design as the G14 copy in ../PECORE_TemporalBranch.

| | V-JEPA 2 branch | this copy |
|---|---|---|
| backbone | `facebook/vjepa2-vitl-fpc64-256` (video, 0.3B) | `vit_pe_core_large_patch14_336` (timm, image, 0.32B), frozen |
| LoRA targets | `query`/`value`, 24 layers | fused `qkv`, all 24 blocks (0.79M params) |
| input | 16 frames at 256 px | the same 16 frames, each through PE-Core; stored 256-px crops upsampled to 336 px (bicubic, on GPU) |
| clip embedding | mean over space-time tokens | PE attention-pooled 1024-d per frame -> temporal branch `TemporalMixer` (proj 512 + learned frame position + 2-layer transformer, 8 heads) -> mean over frames (6.84M params, adapters' lr 2e-4) |
| head | LN -> 128 -> 2, softplus | same, on the 512-d clip embedding |

Checkpoints save LoRA A/B and the temporal module. `verify.py` refers to the V-JEPA 2 checkpoint and does not apply.

Smoke-tested (H200): full batch of 32 clips 3.7 s per step, 34.9 GB peak; inference 1.05 s per 32 clips.
Expect ~10 min per epoch (~7 train + ~3 dev scoring), ~2 h for 10 epochs, ~35 min for the final exports.

```bash
cd /data/kutub/conferences/DFD/PECORE_L14_TemporalBranch
PY=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 nohup $PY train.py --model pecore_l14 --lora --augment B --seed 42 --run-suffix _review > train.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 $PY test.py  --branch pecore_l14_lora_B_review_seed42 --model pecore_l14 --datasets CDFv3 CDFv2
```

---

# V-JEPA 2 temporal branch — standalone review copy

The temporal view of DISCERN: a frozen V-JEPA 2 ViT-L with LoRA adapters and an evidential head,
trained on FF++ c23 and exported as one opinion `(p, u)` per video. This folder is for reviewing and
re-running that branch on its own, without the rest of the project.

Packaged for review 2026-09-30. Everything in `code/` is a **byte-identical copy** of the live repo
(`MANIFEST.sha256` has the hashes; provenance table below). Nothing was rewritten or simplified for
this folder, because a rewrite would be a different thing from what produced our numbers.

## Start here

Take your own copy first — this one is shared and read-only to you:

```bash
cp -r /data/umar/Repos/VJEPA2_TemporalBranch ~/VJEPA2_TemporalBranch
cd ~/VJEPA2_TemporalBranch
```

The copy is self-contained: `verify.py` and every module resolve paths relative to their own
location, and the frame data is read from `DISCERN_Ext` in place. Only **training** needs one extra
step (`./setup_local.sh`), because the output path inside `train_branch.py` points into the original
repo, which you cannot write to. See *Retraining* below.

```bash
PY=/data/umar/miniconda3/envs/temporal_vjepa2/bin/python      # transformers >= 4.55 for VJEPA2Model

# 1. does the folder work at all -- deps, data, weights, hashes. no GPU, a few seconds
OMP_NUM_THREADS=4 $PY verify.py --check

# 2. CDFv3, and the reenactment slice specifically. no GPU, instant -- scored from the shipped opinions
OMP_NUM_THREADS=4 $PY verify.py --slices

# 3. does it reproduce OUR numbers -- loads the checkpoint, re-exports one set, diffs per video
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 $PY verify.py --rerun CDFv2 --workers 8
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 $PY verify.py --rerun CDFv3 --workers 8   # ~20 min, 5118 videos
```

`--rerun CDFv2` currently reproduces the shipped opinions to **1.1e-16 per video** (518 videos,
AUROC 0.7383). If that number moves, something in the environment or the data has changed, and
that is exactly what the check is for.

## Where everything is

There is no `model.py` / `dataset.py` / `train.py` split: the branch is **one 320-line file**,
`code/tbiom/temporal/train_branch.py`, exactly as it is in the live repo. Splitting it for this
folder would mean shipping code that never produced a result. `train.py` and `test.py` at the root
are three-line wrappers that add nothing — they exist only so the entry points are obvious.

| what you are looking for | where | lines |
|---|---|---|
| **data loader (train AND test)** | `code/tbiom/temporal/train_branch.py` → `class Clips` | 121–152 |
| — clip sampling: 16 frames, uniform, in order | `Clips.__getitem__` | 139–152 |
| — arm-B augmentation, train clips only | `Clips.__getitem__` (the `augment == "B"` branch) | 148–151 |
| — per-worker RNG, seeded from (run seed, worker id, torch seed) | `Clips._worker_rng` | 129–134 |
| **which videos a dataset name resolves to** | `videos_of` → `dataset_spec` | 110–118 |
| **model** | `class Branch` | 155–215 |
| — LoRA adapter (frozen base + zero-init B) | `class LoRALinear` | 65–89 |
| — where adapters are injected | `inject_lora` | 92–102 |
| — evidential head + softplus | `Branch.__init__` / `Branch.forward` | 193, 203–215 |
| **training loop** | `main` | 234–310 |
| — class-balanced sampler | `main` (WeightedRandomSampler) | 257–259 |
| — optimiser, cosine schedule | `main` | 260–265 |
| — loss (DISCERN evidential, KL annealed) | `main`, imported from DISCERN_Ext | 275 |
| — per-epoch selection on the v3.1 dev score | `main` | 277–292 |
| **test / inference** | `opinions` | 217–231 |
| — frame-order control (shuffled export) | `export_shuffled` | 312–316 |
| **test-only entry point** | `code/tbiom/temporal/export_branch.py` | whole file, 40 lines |
| hyperparameters (epochs, batch, LoRA r/alpha, lrs) | `train_branch.py` header | 49–59, 104 |

`Clips` is the data loader for **both** train and test — training passes `augment="B"`, evaluation
does not, and `opinions()` wraps the same class in a plain DataLoader. That is deliberate: train and
test read the same h5 crops through the same sampling path, so the only difference is augmentation.

### Entry points

```bash
$PY train.py --model vjepa2 --lora --augment B --seed 42 --run-suffix _review   # train
$PY test.py  --branch vjepa2_lora_B_seed42 --datasets CDFv3 CDFv2               # export opinions
$PY verify.py --rerun CDFv3                                                     # score OUR checkpoint
```

`test.py` loads weights from the *original* repo's run directory (`export_branch.py` line 18). To
evaluate the checkpoint shipped in this folder, use `verify.py --rerun`, which loads
`weights/best.pt` directly and diffs against the reference opinions.

## What is shipped, and what is not

| | |
|---|---|
| `code/` | the branch and everything it imports, byte-identical (9 files) |
| `weights/best.pt` | the development-selected checkpoint, **epoch 6**, 196 MiB — LoRA adapters + head only, the backbone is downloaded from HF |
| `weights/train_record.json` | per-epoch loss and dev score, and the full budget the run was fixed to |
| `reference/opinions/*.csv` | 96 exported opinion files `(v, y, p, u)` — every dev, test and DF40 set, plus the shuffled-frame controls |
| `reference/warmgate_e19_*.csv` | the three detector exports the TRAINING loop needs for epoch selection |
| **not shipped** | the h5 frame data (287 GiB) and the HF backbone. Both are read in place; see below |

### External paths this folder depends on

Data is **not** copied. `/data/umar/Repos/DISCERN_Ext` is read in place for:

- `eval_adaptation/data/h5/*.h5` — the stored 256-px face crops (287 GiB, world-readable)
- `eval_adaptation/data/splits/**` and `eval_adaptation/configs/run1auxedl_*.yaml` — a dataset NAME is
  resolved to an h5 plus split lists through these configs (`dataset_spec`)
- `src/losses/evidential_loss.py` and `src/dataset/training_arms.py` — two DISCERN modules imported
  directly. They are left where they are rather than vendored, since the data forces the dependency anyway

`verify.py --check` tests every one of these and names the missing one if it fails.

## Provenance of `code/`

The layout mirrors the source tree on purpose: each module does `sys.path.insert(0, <its own parent>)`,
so mirroring the directories lets every file be copied **verbatim** — no import shims, no edited paths.

| shipped | copied from (DFB_NeSyNS) |
|---|---|
| `code/tbiom/temporal/train_branch.py` | `analysis/tbiom/temporal/train_branch.py` |
| `code/tbiom/temporal/export_branch.py` | `analysis/tbiom/temporal/export_branch.py` |
| `code/tbiom/xclip/extract_xclip.py` | `analysis/tbiom/xclip/extract_xclip.py` |
| `code/tbiom/{video_id,health,symbolic_overlay,select_checkpoint,select_checkpoint_v3,exp1_fusion_audit}.py` | `analysis/tbiom/` |

`train_branch.py` is the whole branch: LoRA injection, the `Branch` model, the `Clips` loader, the
training loop, epoch selection and the export path. `export_branch.py` only replays a finished run.

## The recipe, as fixed before launch

Backbone `facebook/vjepa2-vitl-fpc64-256`, **fully frozen**. LoRA (r=8, alpha=16, dropout=0.1) on
`query`/`value` of every encoder layer — 786 K trainable — plus an evidential head
(LayerNorm → 128 GELU → 2, softplus), 134 K. Data is FF++ c23 train only, 16 frames per video sampled
uniformly in order over the stored crops, class-balanced sampling. Arm-B augmentation on training
clips only, one label-independent draw per clip: none 40%, JPEG q70/q50/q30 10% each, down-up resize
0.75/0.50/0.35 10% each. AdamW, wd 0.01, cosine to 1e-6, bf16, batch 32, 10 epochs, seed 42, lr 2e-4
adapters / 2e-3 head. Loss is DISCERN's evidential loss with KL annealed over the 10 epochs.
Selection is the v3.1 development score on VALmix + VALmix-v2 + FF++ val after every epoch, best kept,
ties to the earliest — **epoch 6 won, dev v3.1 0.8092**.

JPEG q40 and resize 0.60 are evaluation-only severities and never appear in training.

## Reference numbers

Video-level AUROC of the shipped opinions. Re-exporting any of these should land on its number.

| dataset | videos | AUROC | | dataset | videos | AUROC |
|---|---:|---:|---|---|---:|---:|
| FF++ val | 700 | 0.9904 | | FFIW | 3470 | 0.8011 |
| VALmix | 1350 | 0.7420 | | KoDF | 1395 | 0.9656 |
| CDFv2 | 518 | 0.7383 | | IDForge | 5974 | 0.8899 |
| CDFv3 | 5118 | 0.9263 | | DFDM | 1896 | 0.9798 |
| DFD | 3431 | 0.7967 | | DF40 tpsm-cdf | 825 | 0.9915 |
| DFDC | 4704 | 0.7009 | | DF40 danet-cdf | 825 | 0.9522 |
| DFDCP | 654 | 0.7408 | | DF40 sadtalker-cdf | 838 | 0.9629 |
| DFEval24 | 814 | 0.6732 | | DF40 blendface-cdf | 818 | 0.9371 |

The branch is deliberately lopsided: strong where motion carries the manipulation (reenactment,
KoDF, DFDM) and weak where it does not (CDFv2 0.738, DFDC 0.701). It is **not** a standalone
detector and is never used as one — it is fused with the appearance and off-manifold views.

## CDFv3, and the reenactment slice

`verify.py --slices` scores these from the shipped opinions in about a second — no GPU, no re-export,
because the subset is recoverable from the video id (the second-to-last path component).

| slice | videos | clean | shuffled | delta |
|---|---:|---:|---:|---:|
| CDFv3 (all) | 5118 | 0.9263 | 0.9271 | +0.0009 |
| **CDFv3 reenactment** | **3378** | **0.9741** | 0.9701 | −0.0040 |
| — FaceReenact only | 1578 | 0.9705 | 0.9672 | −0.0033 |
| — TalkingFace only | 1978 | 0.9769 | 0.9724 | −0.0045 |
| CDFv3 face swap | 1918 | 0.8383 | 0.8480 | +0.0097 |

**The slice definition is the one the fusion study uses** (`dsr_system.py`): every real video, plus
the fakes whose subset contains `FaceReenact` or `TalkingFace`. Reals are never filtered — a slice
changes which manipulations are asked about, not which genuine videos must not be flagged. Scope
filters are applied upstream in `videos_of`, so IP_LAP (lip-sync only) is already excluded from the
5,118; that is why the count is not the full corpus.

This is the branch doing what it is for: **0.974 on reenactment against 0.838 on face swap**, a
13-point split on the same real pool and the same weights. The fused system leans on the appearance
and off-manifold views for swaps and on this branch for reenactment.

To re-derive the slice yourself rather than trust `verify.py`:

```python
import pandas as pd
from sklearn.metrics import roc_auc_score
d = pd.read_csv("reference/opinions/CDFv3.csv")
d["subset"] = d.v.str.split("/").str[-2]
r = pd.concat([d[d.y == 0], d[(d.y == 1) & d.subset.str.contains("FaceReenact|TalkingFace")]])
print(len(r), roc_auc_score(r.y, r.p))        # 3378  0.9741
```

To regenerate the opinions first and then slice them, `--rerun CDFv3` writes the same columns and
takes about 20 minutes for 5,118 videos.

## Two things worth a reviewer's attention

**1. Shuffling the frame order barely changes the score.** Same weights, frames in a fixed shuffled
order — the branch's own order-sensitivity control. Note the reenactment slice above moves only
−0.004, and CDFv3 face swap gets *better* by +0.010 when the order is destroyed:

| dataset | clean | shuffled | delta |
|---|---:|---:|---:|
| FF++ val | 0.9904 | 0.9869 | −0.0035 |
| VALmix | 0.7420 | 0.7347 | −0.0073 |
| VALmix-v2 | 0.8214 | 0.7966 | −0.0248 |
| CDFv3 | 0.9263 | 0.9271 | **+0.0009** |
| KoDF | 0.9656 | 0.9633 | −0.0023 |

If the branch were reading motion, destroying temporal order should cost more than this. On CDFv3 it
costs nothing at all. Either V-JEPA 2's features are largely order-invariant here, or the branch is
acting as a per-frame appearance ensemble that happens to be complementary to the CLIP view. We have
not resolved which, and it is the single most useful thing a second reader could pull on.



## Retraining from scratch

`train_branch.py` carries two absolute paths into the original repo: the run output directory, which
is **not writable by anyone else**, and the detector CSVs used for epoch selection. `setup_local.sh`
rewrites exactly those two lines in your copy, prints the diff, and regenerates `MANIFEST.sha256`
so `verify.py --check` still passes. It refuses to run against the shared master copy.

```bash
cd ~/VJEPA2_TemporalBranch
./setup_local.sh                 # or: ./setup_local.sh /scratch/kutub/vjepa_runs

cd code/tbiom/temporal
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 $PY train_branch.py \
    --model vjepa2 --lora --augment B --seed 42 --run-suffix _review
```

About 10 epochs over 3,600 FF++ videos, then it exports dev, test and control opinions.
`--run-suffix` keeps a rerun from colliding with an earlier one of your own.

After `setup_local.sh`, `train_branch.py` is the only file that differs from the live repo, and the
manifest header says so. Everything else stays byte-identical.

Bit-exact reproduction of a *training* run is not expected — sampler order, cuDNN kernels and bf16
reduction order all vary. Re-exporting from the shipped checkpoint IS expected to be exact, which is
what `--rerun` tests.

## Things I would flag before you read the code

None of these are fixed in this folder on purpose — see *Why the code is unchanged* below.

**The shipped checkpoint is 98% frozen backbone — fixed for new runs, 2026-09-30.**
`weights/best.pt` is 196 MiB for a branch with 786 K trainable parameters, because `LoRALinear`
holds the frozen `nn.Linear` as a submodule and `trainable_blocks.state_dict()` serialised
`base.weight`/`base.bias` too:

| stored in `trainable_blocks` | params | size |
|---|---:|---:|
| LoRA A/B (actually trained) | 0.79 M | 3.0 MiB |
| frozen base weights (identical to the HF backbone) | 50.38 M | 192 MiB |

Size was the visible symptom; the real problem is that such a file is a backbone **snapshot** —
loading it overwrites the freshly downloaded HF weights with those captured at training time, so a
backbone version change would be masked rather than raised.

`train_branch.py` now has `trainable_state()` (save the adapters only on the LoRA arm; the block arm
genuinely trains its modules, so it is unchanged) and `load_trainable()`, which accepts **both**
layouts and prints which one it read. Verified before committing: this 196 MiB checkpoint still
reproduces CDFv2 to 1.11e-16 over 518 videos, a slim re-save is **3.5 MiB** and reproduces the
old-format load to **0.00e+00**, and a checkpoint with a layer removed is now rejected by an
assertion instead of scoring quietly.

`weights/best.pt` is deliberately left in the old format: it is the artefact of record for the
deployed run, and `load_trainable` reads it. New runs will write ~3.5 MiB. Say the word if you would
rather have the converted file — the conversion is verified exact, it just replaces a run artefact.

**The real/fake label is a substring test.** `videos_of` (line 117) assigns
`0 if "real" in <subset>.lower() else 1`. It is correct for every corpus here, but a subset whose
name happened to contain "real" would be silently labelled genuine. Worth a glance if you add data.

**Eight backbones, one deployed.** `MODELS` (line 49) carries VideoMAE ×3, TimeSformer, ViViT,
X-CLIP and V-JEPA 2 ×2 from the architecture sweep. Only `vjepa2 --lora --augment B` was trained and
exported; the other branches of `Branch.__init__` and `inject_lora` are dead weight when reading,
and untested in this folder.

**Density.** `train_branch.py` packs multiple statements per line with `;` in the constructors and
the training loop, and `main()` is 76 lines covering config, model build, loader, optimiser, the
epoch loop, selection and six export passes. It is the least pleasant part to read; the file:line
map above is meant to compensate.

### Why the code is unchanged

Everything in `code/` is byte-identical to the live repo, and I would keep it that way for a review.
Reformatting or splitting it would mean you are reading code that never produced a result, and the
provenance check (`diff` against the repo, plus `MANIFEST.sha256`) would stop meaning anything. The
fixes above belong in the live repo, where they can be made once and re-run — say the word and I
will do the checkpoint one, which is the only change with a practical payoff.

## Known rough edges

- Data paths are absolute `/data/umar/Repos/DISCERN_Ext/...`, which is fine on this machine and
  needs editing anywhere else. Training output paths are handled by `setup_local.sh`.
- `extract_xclip.py` imports `CLIPModel`/`XCLIPModel` at module level, so importing the loader pulls
  in CLIP weights metadata that the V-JEPA path never uses. Harmless, just slow to import.
- `train_branch.py` carries eight backbone definitions (VideoMAE, TimeSformer, ViViT, X-CLIP, …) from
  the model sweep. Only `vjepa2` with `--lora --augment B` is deployed; the rest are untested here.
- The dataset name → h5 indirection through `run1auxedl_*.yaml` is inherited from the detector's eval
  configs. A new evaluation set needs one of those configs before it can be named to the exporter.

## Questions worth asking of this code

1. Is the frame-order control above telling us the branch is not temporal? What would settle it?
2. `Clips.__getitem__` samples `np.linspace(0, len(ks) - 1, 16)` over the stored crops. For a video
   with fewer than 16 stored frames this repeats frames. Is that the right behaviour, or should
   short videos be dropped?
3. Video identity comes from `video_id.py` — the directory containing the frame, with a documented
   fallback for flat image sets. That rule has twice reversed a conclusion in this project when it
   was wrong; `python code/tbiom/video_id.py --check` runs its regression test.
4. Epoch selection uses the v3.1 development score, which mixes three dev sets. Epoch 6 of 10 won.
   Is the selection stable, or is it picking noise? `weights/train_record.json` has every epoch.
