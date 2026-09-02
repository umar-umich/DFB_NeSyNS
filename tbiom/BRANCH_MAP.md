# Stage 0 — Branch map

Every path resolved from the tree, not from the earlier docs. Two chassis are in play and the
distinction is the main finding of this stage, so it is stated first.

| chassis | root | what it is | views today |
|---|---|---|---|
| **DISCERN_Ext** | `/data/umar/Repos/DISCERN_Ext` | the retrainable DiCoME fork. Branch 1 + Branch 2 live here. | 2 (semantic, artifact) |
| **discern_v2 (V1)** | `DFB_NeSyNS/training/networks/discern_v2/` | the separate NeSy V1 model. **Branch 3 (P2a) lives here.** | process, reference, applicability |

Branch 3 is not absent — it is in the *other* codebase. Stage 5 is therefore a port, not a
build-from-nothing. Details under "Branch 3" below.

---

## Branch 1 — semantic (CLIP + LoRA)

| what | path | detail |
|---|---|---|
| encoder | `DISCERN_Ext/src/encoders/clip_encoder.py:37-46` | `LoraConfig` -> `get_peft_model` |
| LoRA params | `DISCERN_Ext/src/config.py:41-44` | `target_modules=["q_proj","v_proj"]`, `rank=8`, `alpha=16`, `dropout=0.1` |
| head | `DISCERN_Ext/src/modules/evidential_head.py` | `EvidentialHead(feature_dim=64, num_classes=2)` |
| norm | `core_model.py:57` | `semantic_norm = LayerNorm(64)` |
| EDL loss | `DISCERN_Ext/src/losses/evidential_loss.py:35` | `evidential_loss_discern_ext` |

The peft version is pinned at 0.14.0 in `discern_ext` because every P0/P1 checkpoint was trained
under that LoRA implementation. **Do not touch that env.**

## Branch 2 — manifold / rate

| what | path | detail |
|---|---|---|
| β-VAE projector | `DISCERN_Ext/src/modules/beta_vae_aligned.py:7` | `BetaVAEWithAlignment`, forward at `:54` |
| alignment/VAE loss | `DISCERN_Ext/src/losses/aligned_vae_loss.py:23` | `aligned_vae_loss_func`; already detaches `z_original` internally |
| loss weights | `config.py:65-69` | `beta_kld=4.0`, `lambda_align=1.0`, `lambda_vae=0.1` |
| MR-VAE projector | `DISCERN_Ext/src/modules/mrvae_projector.py:45` | `MRVAEProjector`, `has_rate_response=True` |
| FiLM | `mrvae_projector.py:33` | `film_enc` (`:54`), `film_dec` (`:58`); starts at identity (γ=δ=0) |
| **rate response** | `mrvae_projector.py:84` | `rate_response(x) -> (B,K)` distortion across `BETA_GRID` |
| `BETA_GRID` | `mrvae_projector.py:28` | `(0.1, 0.32, 1.0, 3.16, 10.0)`, `RATE_RESPONSE_K = 5` |
| artifact construction | `core_model.py:129-135` | orthogonal residual, then `cat([f_a, rate])` when enabled |
| purification | `core_model.py:159-177` | `f_r = f_s - f_c`; project off `f_s`; return `f_a` |
| switches | `config.py:82-83` | `manifold_projector: beta_vae\|mrvae`, `artifact_extra_dims: int` |

`artifact_extra_dims=5` gives the 69-d artifact head (64 + 5) that matches P1d's saved
`artifact_norm` width exactly. **This is precisely the Stage 3 wiring already present**: the
concat point at `core_model.py:132-135` is generic — it fires on
`has_rate_response and artifact_extra_dims`, not on the projector class. Stage 3 (β-VAE artifact
+ MR-VAE rate curve) needs the rate operator made available *alongside* the β-VAE projector
rather than a new concat path.

## Branch 3 — generative process (P2a) — **found, in the other chassis**

| what | path | detail |
|---|---|---|
| **SDXL-VAE residual operator** | `DFB_NeSyNS/training/networks/discern_v2/process_residual.py:64` | `ProcessResidualOperator`; `build_operator()` at `:189` |
| statistics | `process_residual.py:47-55` | K=6: `mse_mean, mse_std, mse_max, mse_p90, lpips_proxy, center_ratio` |
| **evidence branch** | `DFB_NeSyNS/training/networks/discern_v2/process_branch.py:62` | `ProcessEvidenceBranch`; `forward` at `:138`, `dirichlet` at `:156` |
| frozen calibration | `configs/discern_v2/process/process_stats.pt` (+ `.json`) | fit on FF++ **real train** frames only |
| fit script | `analysis/discern_v2/fit_process_stats.py` | reals-only enforced in code |
| VAE weights | `/data/umar/Repos/DiCoME/eval_adaptation/data/models/sdxl-vae/` | 335 MB, present |
| existing attach point | `training/networks/discern_v2/discern_v1_model.py`, `branches.py` | V1 chassis, **not** DISCERN_Ext |

`ProcessEvidenceBranch.forward` already returns `{evidence, feature, valid, raw_stats}` — the
`evidence` tensor is exactly the interface DS fusion needs, so no head rewrite is required.

## Fusion

| what | path | detail |
|---|---|---|
| DS combination | `core_model.py:67-110` | `DS_Combin` + `_combine_two_opinions` |
| view assembly | `core_model.py:116-157` | returns `(fused, semantic, artifact, f_s, f_c, z, mu, log_var)` |
| module wrapper | `DISCERN_Ext/src/model/discern_ext_module.py` | Lightning module, test/val steps |

## Evaluation

| what | path | detail |
|---|---|---|
| **video identity** | `analysis/tbiom/video_id.py` | the single grouping rule; `--check` runs 49 cases / 12 datasets |
| health dashboard | `analysis/tbiom/health.py:70` | `dashboard()` -> AUROC, EER, `fpr_real_at_tau`, `delta_rf`, `mean_p_on_real` |
| τ convention | — | frozen once per model+readout on **its own FF++ val**, applied unchanged |
| per-view exporter | `DiCoME/eval_adaptation/export_views.py` (mirrored in `analysis/tbiom/dicome_ablation/`) | writes `p_fused/p_semantic/p_artifact` + uncertainties |

---

## Pass condition — met, with two things to report before proceeding

All paths resolve, **P2a module and wiring included**. Two facts change the shape of Stage 5 and
are reported here as the brief requires.

**1. `DS_Combin` hard-rejects a third view.** `core_model.py:74` raises
`ValueError("DS_Combin expects exactly two Dirichlet alpha tensors.")`, and
`_combine_two_opinions` is written for exactly two opinions with `for view_idx in range(2)` and a
2-way `bmm`. Branch 3 cannot be attached without extending it. This is mechanical, not a design
question: the DS orthogonal sum is associative, so three views fold as
`combine(combine(sem, art), proc)`. Stage 6's "standard DS" arm is that fold. Flagged because it
is a code change to the fusion operator, which Stage 5 would otherwise land silently.

**2. Branch 3 must be ported across chassis.** P2a's operator, branch, frozen statistics and
SDXL-VAE weights all exist and are validated, but they live in the V1 model tree and expect V1's
`ResidualCalibrator`/`DirichletState`. DISCERN_Ext has no third-view attach point at all. The
port is the work of Stage 5, and it carries one hazard the source file names explicitly:

> **Caching hazard** (`process_residual.py:20-35`). The residual is a property of one particular
> image. DISCERN_Ext's datamodule applies CLIP normalisation; the AE expects [-1,1]. Pairing a
> cached clean residual with an independently augmented visual input has bitten this project
> before. `assert_cache_safe()` exists to make the rule checkable. **No caching in Stage 5.**

Neither blocks Stage 1, 2, 3 or 4, which touch Branch 2 only.
