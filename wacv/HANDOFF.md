# FPAD / WACV — session handoff, parked 2026-08-25

Everything is committed; `git status` is clean apart from files listed below. Branch
`discern-v2-phase2`, 49 commits ahead of `main`. Nothing is running.

Env for everything: `/data/umar/miniconda3/envs/dfb_nesy/bin/python`. `base` has no pyarrow and
cannot read the per-sample parquets. Never `tail` the `logs/fpad/*.log` training logs directly —
they are `\r`-delimited tqdm streams; pipe through `tr '\r' '\n'` and grep for the epoch lines.

---

## The outcome: FPAD's standalone attempt was stopped on evidence

Umar's decision rule was three axes, all required to pass. **Two failed**, including the
non-circular one, so the standalone WACV attempt is closed. **Do not build a B6.**

| axis | result |
|---|---|
| 4a faithfulness | **FAIL** — deletion mostly *raises* confidence; insertion negative on 4 of 5 |
| 4b localization vs masks | **FAIL** — AUPRC drops on every manipulation; NeuralTextures falls below chance |
| 4c detection | pass — no collapse, but no gain either |

The mechanism is fully diagnosed, and this is the result worth writing:

* **Adaptation concentrates in the final block and gets worse with training.** Layers 4–20 shrink
  while layer 24 grows; the ratio rises every epoch (ordinary 48× → 63×, preserve 249× → 330× over
  five epochs). A six-dimensional "trajectory" with one dimension 300× the rest is a scalar.
* **The profile is a level, not a shape.** Real and fake mean shapes correlate at **0.9999**
  (ordinary), 0.9437 (preserve).
* **`L_preserve` does buy something real, and it is spatial.** The preservation map localises at
  1.4–2.2× chance on all four FF++ manipulations; the unpreserved map is *at or below* chance on
  two of them.
* **But the layer that localises is not the layer that classifies.** L20 localises at 2.64× chance,
  L24 at 1.11×. Trained to classify, B5's learned layer weights converged to **96.8% on L24 and
  0.2% on L20** — it had L20 available and suppressed it, because L24 carries the adaptation
  *magnitude* and magnitude drives classification.

Recommended framing, in Umar's words: workshop-scale analysis result plus a T-BIOM supporting
branch. The v3 brief (below) already carries these as settled constraints.

## What is still runnable, and worth finishing

Both are queued/fixed and cost nothing; they firm up numbers for whichever venue this goes to.

1. **Uncapped mechanism validation at epoch 9.** Both students completed all 10 epochs
   (`logs/fpad/studentA_{ordinary,preserve}_seed42/epoch_009.pth`). The only read so far is
   **epoch 1, capped**, and it returned BORDERLINE by a margin of *zero layers*. Re-score
   uncapped and re-run:
   ```
   # score both students at the SAME epoch — pinning --epoch is mandatory, not optional
   for arm in ordinary preserve; do
     $PY training/score_fpad.py --student logs/fpad/studentA_${arm}_seed42 --epoch 9 --seed 42 \
        --datasets FaceForensics++ --split val --output logs/fpad/score/${arm}_ffppval_ep9
     $PY training/score_fpad.py --student logs/fpad/studentA_${arm}_seed42 --epoch 9 --seed 42 \
        --datasets danet_cdf mcnet_cdf tpsm_cdf facevid2vid_cdf sadtalker_cdf --df40 \
        --output logs/fpad/score/${arm}_df40dev_ep9
   done
   $PY analysis/fpad/stage2_gate.py --ordinary <both ep9 parquets> --preserve <both ep9 parquets> \
      --runs logs/fpad/studentA_ordinary_seed42 logs/fpad/studentA_preserve_seed42 --out wacv
   ```
2. **The official FS-VFM linear probe.** It ran and **failed**: `unrecognized arguments:
   --local-rank=1`. torch ≥ 2.0's `torch.distributed.launch` passes `--local-rank` (hyphen) while
   the 2024-era FSFM code declares `--local_rank` (underscore). The flag is irrelevant either way,
   because `FSFM-CVPR25/fsvfm/util/misc.py:233` reads `LOCAL_RANK` from the **environment**.
   `queue_fsvfm_lp.sh` now uses `torchrun`, which sets that env var and preserves effective batch
   128 × 2 = 256 (what their `blr` assumes). Re-run directly, no waiting:
   ```
   EPOCHS=10 GPUS=1,2 bash analysis/fpad/queue_fsvfm_lp.sh    # skips the wait; students are done
   ```
   The ImageFolder tree is already built and verified at `/data/umar/Datasets/fsvfm_lp`
   (615,605 symlinks, 2.2 GB, FF++ 23,039 real / 92,159 fake).

## Bugs found this session that live outside FPAD

These affect the shared codebase and are already fixed and committed. Worth knowing because two of
them silently corrupted results rather than erroring.

* **Dataset order was not reproducible.** `abstract_dataset.py:346` shuffles with the *global*
  `random`, and the trainers seeded only torch and numpy. The same 115,198 FF++ frames hashed to
  two different orders across processes. **This affected `train_v1.py` too** — V1's shipped run was
  not reproducible in its data order. Both trainers and both scorers now seed `random`.
* **`MatchedAugment` used Python's `hash()`**, which is per-process randomised. Two runs of one seed
  augmented the same pair differently. Now `crc32`.
* **Frame subsampling returned contiguous frames.** `abstract_dataset.py` computed
  `step = total_frames // frame_num` *after* setting `total_frames = frame_num`, so `step` was
  always 1. Dormant at `frame_num == 32` (V1 unaffected), but it fired the moment we subsampled.
  Now genuinely even-spaced.
* **DF40 path resolution 60/81 → 77/81.** `heygen` — the brief's harm probe — resolved at 0.0, as
  did both `e4e` arms, `styleclip`, `whichisreal`. Four distinct causes, none missing data.
* **The scorers were unseeded**, so the JPEG probe compared different frames with and without
  compression — 132 of 1,280 shared, running at a tenth of its intended power while looking fine.

## Artifacts

```
wacv/REPO_MAP.md          stage 0 discovery, items 1-8
wacv/STAGE2_GATE.md       mechanism validation, epoch 1 capped, BORDERLINE
wacv/TERMINOLOGY.md       "gate" must not reach the paper — collides with the learned q_b
phase2/DF40_SPLIT.md      the sealed 38 Dev / 35 Holdout split and the exact zero-shot wording
phase2/REPO_MAP.md        phase-2 discovery
logs/fpad/studentA_*      both students, 10 epochs each
logs/fpad/B5_preserve_proto/    the B5 prototype that failed 4a and 4b
logs/fpad/localization/   mean-pool and B5 localization + faithfulness
```

## Next direction

**`docs/DiCoME_eval/discern-v2-phase2-v3.md`** — a resume brief for the Phase-2 / T-BIOM
architecture, written post-FPAD. Read it first; it supersedes `fpad-wacv-brief.md`.

It already carries the FPAD post-mortem as constraints: the FS-VFM branch becomes a *direct adapted
discriminative expert* (P_R and the depth-trajectory framings are archived as falsified), rescue is
computed fresh against the actual anchor's actual errors, and localization work must be
mask-supervised and decoupled from the detection objective.

Two things in it that need attention early, because both are places this project has been bitten:

* It specifies **VALmix diverse-validation for checkpoint selection**, and states the cost —
  Celeb-DF-v2, DFDCP and Deepfake-Eval-2024 lose the strict zero-shot label. The Stage-2 work here
  already spent Celeb-DF-v2 *reals* as the domain axis (178 videos / 2,841 frames, recorded in
  `wacv/stage2_gate.json` under `provenance_cost`); that is additive to the VALmix cost, not
  covered by it.
* It specifies a **firewall**: diverse-validation labels select checkpoints and nothing else.
  Gates, risk model and thresholds stay on FF++ `VAL_meta`. `training/stage567.py` already defaults
  to `--val-protocol ffpp` and records which protocol was used, so it complies as written — but
  the `diverse` option exists and would silently break the firewall if selected.
