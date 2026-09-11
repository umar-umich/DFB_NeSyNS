# Stage 0 — frozen detector confirmed, measurement inventory taken

## The frozen configuration

**Arm B, three views, DS fusion, per-branch EDL, seed 42** — the Phase F freeze, unchanged.

| | |
|---|---|
| checkpoint | `DISCERN_Ext/runs/discern_ext_train/run1_auxedl_seed42/checkpoints/discern_ext-best-epoch=05-val_auroc_video=0.9955.ckpt` |
| config | `DISCERN_Ext/eval_adaptation/configs/run1_auxedl.yaml` |
| exports | `logs/tbiom/stage23/run1auxedl_*.csv` |
| coverage | CDFv2, CDFv3, DFD, DFDC, DFDCP, DFEval24, FFpp_val, VALmix, + DF40_FS / FR / EFS / FE |
| headline | 0.8782 mean OOD AUROC (3 seeds), 0.135 mean real-side FPR |

## What the exports actually carry

Eleven columns per frame: `key`, `video`, `label`, and a `(p, u)` pair for each of
`fused`, `semantic`, `artifact`, `fsvfm`.

| Stage 1 field | status | note |
|---|---|---|
| branch probability | **present** | `p_semantic`, `p_artifact`, `p_fsvfm` |
| branch vacuity | **present** | `u_*` |
| fused probability | **present** | `p_fused`, `u_fused` |
| branch evidence | **derivable** | `S = K/u`, `α = [(1−p)S, pS]`, `e = α−1` — exact, no re-export needed |
| cross-branch agreement | **derivable** | already computed this way by the overlay |
| **availability** | **MISSING, and worse than missing** | see below |
| **reliability** | **MISSING** | no quality or operating-condition estimate exists anywhere |
| **source** | **MISSING** | only dataset/video/frame from the h5 key |
| reconstruction / manifold diagnostics | **MISSING** | computed in the forward pass, never exported |

## The availability problem is the real Stage 0 finding

A validity flag **is** computed — `fsvfm_branch.py:183`, `valid = torch.isfinite(feat).all(dim=1)`
— and it is **not exported**. That alone would be a gap. The sharper problem is what the model
does with it:

```python
# core_model.py:229
fs_ev = fs["evidence"] * fs["valid"].unsqueeze(1).to(fs["evidence"].dtype)
```

An unavailable measurement has its evidence multiplied by zero, which produces a **vacuous
opinion** — `u → 1`. A cue that was measured and found uninformative also produces a vacuous
opinion. In the exported CSVs the two are **indistinguishable**.

That is precisely the collapse the brief forbids: *"Never infer availability from a low fake
score. It must come from the measurement process, not the decision."* Today availability is
recoverable only *through* the decision, because it was folded into the evidence before export.

It also pre-empts Stage 4's missing-measurement test. That test asks whether the system
converts *unavailable* into *contradicted*. The current architecture converts unavailable into
*vacuous*, and vacuous is not a distinct state in the export — so the test cannot be run on
these files, whatever its answer would be.

## Reliability and source do not exist at all

Reliability has no candidate field anywhere in the model. Nothing measures compression level,
resolution, camera, or any operating condition; Stage 2 has to create it.

Source is thinner than the brief assumes. All three branches read the **same aligned face
crop** — there are no regions, no segments, no per-detector provenance to record. The only
honest source field is `(dataset, video, frame, branch)`. That still supports the Stage 4
duplicate-evidence test, but the Stage 1 ambition of identifying correlated evidence *by
region* is not available in this architecture, and pretending otherwise would fabricate a field.

The one real correlation is already known and measured: `semantic` and `artifact` both derive
from the CLIP trunk (`artifact` is the orthogonal residual `f_s − f_c`), which is why dropping
`semantic` from the fusion is worth +0.008. That is a branch-level correlation, and it belongs
in the source field as a shared-trunk marker rather than as a region id.

## Pass condition

**Met.** The frozen configuration is named and its per-sample exports are located.

**But Stage 1 cannot be built from these files.** Three of its four required fields are absent,
and one of them — availability — has been actively destroyed by being multiplied into the
evidence before export. Stage 1 needs a re-export that writes, per frame:

1. `avail_<branch>` — the branch's own validity flag, **before** it is multiplied into evidence
2. the raw per-branch measurement the head consumes (β-VAE reconstruction error and `mu`/`log_var`,
   the orthogonal residual norm, the FS-VFM feature norm)
3. a `source` tag carrying the shared-trunk relationship

That is a exporter change plus one GPU pass over the eight datasets — no retraining, and the
frozen weights are untouched. Reliability remains Stage 2's job and cannot come from a
re-export.
