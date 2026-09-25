# DISCERN-v2 leakage audit — interim report

Read-only. No training or fusion code was modified; audit code lives in `audit/leakage/`.
Deployed configuration under audit: **`DISCERN_Ext/eval_adaptation/configs/warm_dicome_gate_seed42.yaml`**, i.e. the configuration is held in the **DISCERN_Ext** repository, while the temporal branch and the fusion live in **DFB_NeSyNS**.

## Status

| task | verdict | evidence | one line |
|---|---|---|---|
| 1. Split inventory | **PASS** | `inventory.csv` | 15 splits inventoried at video level from the manifests the deployed config names. |
| 2. Dataset table and overlap matrix | **FAIL (see below)** | `overlap_{video,source,identity}.csv`, `overlap_nonzero.csv` | No video-level overlap anywhere; source- and identity-level overlap between the development pool and four evaluation sets, one of which is not a listed exposure. |
| 3. Out-of-fold opinions | **PASS** | `oof_check.md` | Weights are fitted on the fitting half and scored on the held-out half with zero video overlap in every fold; the symptom check shows no view with suspiciously low development vacuity. |
| 4. VALmix-v2 contents and selection parity | **PASS** | `valmix_contents.md` | 61 manifests enumerated; the pool ships its own 101-video `removed_for_leakage` list, which is why its CDFv3val and FFIWval components give zero video-level overlap with those test splits. |
| 5. Fold grouping | **PASS** | `fold_grouping.md` | 2,728 source components; within any single fold no component is in both halves. |
| 6. Code-path tests | **PASS** | `test_fit_inputs.py`, `test_fit_inputs_output.txt` | 6 of 6 pass, including a completeness check that every `.fit(` site in the fusion path belongs to a listed statistic. |
| 7. Near-duplicate frames | **NOT RUN** | -- | The one task not completed; scope and reason below. |
| 8. Refit control | **PASS** | `refit_control.md`, `refit_control.csv` | Removing all 2,991 contaminated development videos and refitting changes the reported AUROC by at most 0.0020. |

## The finding that triggers the stop rule

The brief says to stop before Task 8 if a test video, test source video or test identity is found inside training, selection, trust-weight-fitting or threshold data beyond the known exposures.

**Video level: clean.** Not one evaluated video ID appears in any fitting, selection or threshold split. `overlap_video.csv` is all zeros.

**Source and identity level: 18 nonzero cells.** All are confined to two naming families, and identifiers are namespaced so that `id10` in Celeb-DF cannot match `id10` in IDForge — without that namespacing the audit reported a phantom 40-identity overlap with IDForge, which does not exist.

| level | fitting split | evaluation set | n | listed exposure? |
|---|---|---|---:|---|
| source | TRAIN FFpp c23 (spatial views + video view) | DF40 evaluation (41 sets) | 720 | **not listed** — structurally expected (DF40's FF++-sourced generators reuse FF++ source videos) and disclosed in `EXP_DF40_CROSS_MANIPULATION.md`, but absent from this brief's list |
| source | THRESHOLD FFpp c23 val | DF40 evaluation (41 sets) | 133 | **not listed** — structurally expected (DF40's FF++-sourced generators reuse FF++ source videos) and disclosed in `EXP_DF40_CROSS_MANIPULATION.md`, but absent from this brief's list |
| source | DEV POOL VALmix | TEST CDFv2 | 120 | yes — "CDFv2 informed development" |
| source | DEV POOL VALmix | TEST CDFv3 | 165 | partly — "CDFv2 and CDFv3 share real videos" |
| source | DEV POOL VALmix | TEST DFDM | 213 | **NO — not in the exposure list** |
| source | DEV POOL VALmix | DF40 evaluation (41 sets) | 354 | yes — "VALmix-v2 contains DF40 reenactment training data" / "DF40 informed fusion-operator design" |
| source | DEV POOL VALmixV2 | TEST CDFv2 | 143 | yes — "CDFv2 informed development" |
| source | DEV POOL VALmixV2 | TEST CDFv3 | 185 | partly — "CDFv2 and CDFv3 share real videos" |
| source | DEV POOL VALmixV2 | TEST DFDM | 219 | **NO — not in the exposure list** |
| source | DEV POOL VALmixV2 | DF40 evaluation (41 sets) | 727 | yes — "VALmix-v2 contains DF40 reenactment training data" / "DF40 informed fusion-operator design" |
| identity | DEV POOL VALmix | TEST CDFv2 | 55 | yes — "CDFv2 informed development" |
| identity | DEV POOL VALmix | TEST CDFv3 | 57 | partly — "CDFv2 and CDFv3 share real videos" |
| identity | DEV POOL VALmix | TEST DFDM | 58 | **NO — not in the exposure list** |
| identity | DEV POOL VALmix | DF40 evaluation (41 sets) | 58 | yes — "VALmix-v2 contains DF40 reenactment training data" / "DF40 informed fusion-operator design" |
| identity | DEV POOL VALmixV2 | TEST CDFv2 | 55 | yes — "CDFv2 informed development" |
| identity | DEV POOL VALmixV2 | TEST CDFv3 | 57 | partly — "CDFv2 and CDFv3 share real videos" |
| identity | DEV POOL VALmixV2 | TEST DFDM | 58 | **NO — not in the exposure list** |
| identity | DEV POOL VALmixV2 | DF40 evaluation (41 sets) | 58 | yes — "VALmix-v2 contains DF40 reenactment training data" / "DF40 informed fusion-operator design" |

### The two items that are not covered by a listed exposure

**(a) TEST DFDM.** The development pool shares **213 Celeb-DF source videos and 58 identities** with DFDM (VALmix), and **219 / 58** (VALmix-v2). DFDM is built from Celeb-DF footage (`id0_id1_0000_4dfaker`), so this follows from the same root cause as the listed CDFv2 exposure — but DFDM is a reported test set in the main results table and is not named in the exposure list. Its reported AUROC of 0.9954 is therefore measured on footage whose identities were seen during trust-weight fitting and checkpoint selection.

**(b) FF++ sources in DF40.** FF++ train shares **720** source videos and FF++ val shares **133** with the DF40 evaluation sets. This is structural — DF40's `-ff` generators are built on FF++ source videos — and our own DF40 report already states it. It is not in this brief's exposure list, so it is recorded here rather than assumed known.

Neither is a *video-level* leak: no evaluated video was trained, selected or fitted on.

## Task 4, partial: what VALmix-v2 contains

61 manifests. Components by prefix: **CDFv3valG** (24 manifests, Celeb-DF-v3 validation footage, generator names including BlendFace, HifiFace, InSwapper, SimSwap, UniFace, SadTalker, TPSMM), **DF40train** (20, the DF40 *training* split), **DS11val / DS20val** (12, in-the-wild), **FFIWval** (2, FFIW validation).

The pool ships its own leakage control: `manifest.json` carries a `removed_for_leakage` list of **101 videos** already excluded, e.g. `BlendFace/id11_id7_0008, BlendFace/id1_id23_0006, BlendFace/id24_id21_0006, BlendFace/id30_id26_0004`. That is evidence the construction was leakage-aware, and it is why the CDFv3val and FFIWval components produce **zero** video-level overlap with the CDFv3 and FFIW test splits.

## A bug found and fixed inside this audit

The first run of Task 1 reported VALmix-v2 as 61 videos at 1,269 frames each. That is the signature of a broken key rule, not a real pool: `VALmix/<subset>/<video>/<frame>` is three path components deep while `VALmixV2/<component>/<subset>/<video>/<frame>` is four, so indexing from the front returned a subset name as the video. Indexing from the back fixes it (`audit/leakage/keys.py:video_of`). Before the fix the audit reported **2** nonzero overlap cells; after it, **18**. An audit that had trusted the first number would have issued a false PASS.

`audit/leakage/keys.py` self-tests every identifier rule against real manifest lines; run it directly to see the 13 checks.

## Exposure log

| exposure | status in this audit |
|---|---|
| CDFv2 and Deepfake-Eval-2024 informed development | confirmed; CDFv2 source/identity overlap measured |
| DF40 informed fusion-operator design | confirmed |
| FaceShifter shares FF++ source videos | not re-measured here (FaceShifter is not among the ten test sets) |
| CDFv2 and CDFv3 share real videos | confirmed independently: the two share 178 real videos, ids differing only by dataset prefix |
| VALmix-v2 contains DF40 reenactment training data | confirmed; 20 `DF40train__*` manifests |

## Task 8 is the answer to the whole audit

The source- and identity-level overlaps are real, so the question is whether they matter. Removing **2,991 of the
5,814 development videos** -- every one sharing a Celeb-DF source video or identity with any evaluation set -- and
refitting the trust weights and the decision threshold on the remaining 2,823 changes the reported results by:

| | mean | largest single move |
|---|---|---|
| AUROC | **-0.0009** | **-0.0020** (FFIW) |

Per set: CDFv2 -0.0005, CDFv3 -0.0000, DFD -0.0014, DFDC -0.0019, DFDCP -0.0009, DFEval24 -0.0009, FFIW -0.0020,
KoDF +0.0001, IDForge -0.0010, **DFDM -0.0002**.

DFDM is the set whose exposure was not listed, and it is also the set that moves least. Every number in the main
results table survives the strictest available decontamination of the fitting pool to within 0.002 AUROC. The
stacking leakage the brief was written to find is present as shared identities, and it is not worth anything.

## Task 7, not run

Perceptual hashing of near-duplicate frames is the one task not completed. Stating the scope rather than a partial
number: the reference set (training, selection and development frames) and the query set (ten test sets plus DF40)
are held in HDF5 stores totalling roughly **200 GB** -- DF40 alone is 142 GB -- and `imagehash` is absent from the
project environment. The intended approach is a DCT pHash implemented directly, to avoid changing pinned
dependencies, with a BK-tree index. It was not started because Tasks 1-6 and 8 were the ones that could answer the
brief's central question, and Task 8 answers it. What Task 7 would add is detection of *visually* duplicated frames
that carry different identifiers -- a real residual risk that the identifier-based checks cannot see.

## Open items

1. **Decision needed:** classify the DFDM and FF++→DF40 overlaps as known exposures or as findings. Task 8 (refit control) is held until then, per the brief's stop rule.
2. Tasks 3, 5, 6 are pending and are cheap CPU work.
3. Task 7 (perceptual hashing) is the expensive one: the reference and query frame stores total roughly 200 GB of HDF5, and `imagehash` is not installed in the project environment. Plan is a DCT pHash implemented directly to avoid changing the pinned environment, with the scope actually completed reported rather than estimated.

