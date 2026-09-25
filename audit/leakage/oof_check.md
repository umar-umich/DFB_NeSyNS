# Task 3 — out-of-fold opinions and the symptom check

## 3a. Provenance of each view's cached opinions

| view | checkpoint behind the cached opinions | trained on | selected on | note |
|---|---|---|---|---|
| APP | `DISCERN_Ext/runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt` | FFpp_train (5 manifests, from warm_dicome_gate_seed42.yaml trn_files) | VALmixV2 (v3.1 score) | frozen at the DiCoME warm start (freeze_views: true) |
| OM | `same checkpoint as APP` | FFpp_train | VALmixV2 (v3.1 score) | frozen at the DiCoME warm start |
| T | `DFB_NeSyNS/logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt` | FFpp c23 train only (3,600 videos, per train_branch.py) | v3.1 development score on VALmix + VALmixV2 + FFpp_val | LoRA adapters + evidential head only |

**The central intersection.** Every view was trained on FF++ c23 *train* and the trust weights are fitted on the development pool (VALmix, VALmix-v2, FF++ *val*). Those are disjoint splits of FF++ plus material from other sources, so at video level the training data and the weight-fitting data do not intersect — `overlap_video.csv` shows zero between `TRAIN FFpp c23` and both `DEV POOL` rows.

**What does intersect.** Checkpoint *selection* for all three views used VALmix-v2, which is also part of the trust-weight fitting pool. Selection and fitting therefore share data. That is not test leakage, but it does mean the fitted weights are applied to views chosen on the same videos, so the development numbers are optimistic as an estimate of held-out fusion quality. This is the stacking risk the brief names, and it is present by construction.

## 3b. Symptom check — per-view AUROC and mean vacuity

No threshold is set; the numbers are reported so a reader can judge.

| view | dev pool AUROC | dev pool mean vacuity | FF++ val AUROC | FF++ val vacuity | CDFv3 AUROC | CDFv3 vacuity |
|---|---:|---:|---:|---:|---:|---:|
| T | 0.8452 | 0.1456 | 0.9904 | 0.0528 | 0.9263 | 0.1601 |
| APP | 0.8276 | 0.2343 | 0.9953 | 0.1565 | 0.8576 | 0.2259 |
| OM | 0.8455 | 0.2902 | 0.9955 | 0.1362 | 0.8801 | 0.3079 |

The development pool includes FF++ val, so the two leftmost columns are not independent of each other. A view whose development vacuity sat far below its FF++-val vacuity would be the signature of having been fitted on development material; read the table with that in mind rather than against a threshold.

## 3c. Were the development numbers computed out of fold?

| split | fitting videos | selection videos | video overlap |
|---|---:|---:|---:|
| 0 | 2911 | 2903 | 0 |
| 1 | 2971 | 2843 | 0 |
| 2 | 2942 | 2872 | 0 |
| 3 | 2965 | 2849 | 0 |
| 4 | 2846 | 2968 | 0 |

**PASS** — the weights are fitted on the fitting half of each fold and scored on the held-out half, with no video in both. Verified by intersecting the two index sets that `dsr_system.splits` actually returns, not by reading the fold code.

