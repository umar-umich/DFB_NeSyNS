# V1_BUILD_REPORT — DISCERN v2 Phase 2

Deliverable for build spec §26. Everything below is either measured on this machine or traced to
`file:line` in validated code. Cells that need a run say `TODO(run)`; no number here is asserted
from memory.

Build commits: `ffbd03f` … `8f516a4` (9 commits on `discern-v2-phase2`, HEAD `8f516a4`).

---

## 1. Confirmed paths

| item | value | how confirmed |
|---|---|---|
| DISCERN repo | `/data/umar/Repos/DFB_NeSyNS` | `../DISCERN/` does not exist; this repo holds the v2 code the spec describes |
| DiCoME clone | `/data/umar/Repos/DiCoME` | read directly for every §3/§10 value below |
| FSFM clone | `/data/umar/Repos/FSFM-CVPR25` | cloned at branch `FSVFM-extension-R1`, commit `b84aafa2507c30bb4986564de8262969647e450c` |
| env | `dfb_nesy` | torch 2.9.1+cu128, transformers 4.44.2, peft 0.14.0, timm 1.0.25, dlib 20.0.0 |

---

## 2. FS-VFM artifact (§4)

Downloaded from HF `Wolowolo/fsfm-3c`, `pretrained_models/FS-VFM_ViT-L_VF2_600e/`.

| property | value |
|---|---|
| file | `weights/FS-VFM/checkpoint-599.pth` (4,369,346,513 bytes) |
| sha256 | `3fd99324c998ee06e4daa7f9d845e73f65f1628dee741b2c98664d566526bc87` |
| `epoch` | 599 |
| `args.model` | `fsfm_vit_large_patch16` |
| `args.input_size` / `mask_ratio` / `norm_pix_loss` | 224 / 0.75 / True |
| `args.data_path` | `VGG-Face2` |
| normalization | mean `[0.54822075, 0.42340535, 0.36546516]`, std `[0.27891761, 0.24385408, 0.23493893]` — read from the shipped `pretrain_ds_mean_std.txt`, **not** ImageNet |
| state-dict | 296 encoder tensors load; missing only `fc_norm.{weight,bias}`, `head.{weight,bias}`; 149 unexpected (pretraining decoder / `mask_token` / `norm`) |

Two findings worth recording:

- **The `600e` / `-599` mismatch is the 0-based final index, as §4 anticipated** — but the
  pretraining `args.epochs` is **800**, so the released artifact is epoch 599 of a longer
  configured schedule, not the end of a 600-epoch run.
- **Two FS-VFM checkpoints already existed locally** (`GenD_NeSy/weights/FS-VFM/*.pth`) and are
  **not usable**: their args show `finetune=.../FS-VFM_ViT-L_VF2_600e/checkpoint-599.pth`,
  epoch 9 of 10, FF++-c23 fine-tuned with a classification head. Using one as "the frozen
  real-face prior" would place an FF++-supervised detector inside the branch that is supposed to
  carry an independent bona-fide prior.

**Pooling.** In the official `global_pool` default the ViT deletes `norm` and creates `fc_norm`,
which a pretraining checkpoint does not contain — so frozen, `fc_norm` is LayerNorm at default
init and the pretrained final norm is discarded. `cls` pooling keeps that norm and reads CLS.
`global_pool` is the V1 default because it is what the authors' downstream path and released
fine-tuned checkpoints use; `cls` is exposed as the logged alternative.

---

## 3. Recipe items borrowed from DiCoME (§3, §10)

Every value read from the clone, not from the spec's prose. All four LoRA values match the form
§3 said to expect.

| item | value | source |
|---|---|---|
| LoRA target modules | `["q_proj", "v_proj"]` | `src/config.py:41` |
| LoRA rank / alpha / dropout / bias | 8 / 16 / 0.1 / `none` | `src/config.py:42-45` |
| backbone | `openai/clip-vit-large-patch14` | `src/config.py:61` |
| feature projection | `Linear(1024 -> 64)` | `src/encoders/clip_encoder.py:52-56` |
| evidence head | `Linear(64) -> ReLU -> Linear(K) -> softplus` | `src/modules/evidential_head.py` |
| pre-head LayerNorm | `LayerNorm(feature_dim)` | `src/model/core_model.py:47` |
| optimizer | AdamW, lr 1e-4, betas (0.9, 0.999), wd 0.01 | `src/config.py:70-77` |
| weight-decay exclusion | names containing `bias`, `norm`, `bn` | `src/model/dicome_module.py:503-517` |
| scheduler | CosineAnnealingLR, per step, `eta_min` 1e-6 | `dicome_module.py:522-535` |
| augmentation | flip 0.5 → affine(10, 0.1, 0.9–1.1) → blur k(3,7) σ(0.1,2.0) → jitter(0.2, 0.2) | `src/dataset/base.py:67-78` |
| DS combination | conflict = off-diagonal belief mass; `b_f = (b·b + b·u + b·u)/(1−C)` | `src/model/core_model.py:57-99` |

Resolved §10 questions: the reproduced code **does** exclude bias/norm from weight decay, so that
exclusion is inherited rather than assumed; and blur/jitter are **unconditional** (every training
image), which is the delta against our 10%-of-batches version.

**The 64-D bottleneck is kept.** §3 allows this only if inspection shows it is recipe rather than
a consequence of Geometric View Purification. It is: the projection lives in the encoder, is
applied to CLIP's pooler output before any view-specific structure exists, and feeds the semantic
head directly. GVP consumes `f_s` but did not create it. Revisiting it is a §24 iterate item.

**Not ported (§25):** DiCoME's β-VAE semantic manifold and orthogonal-projection artifact view.

---

## 4. Parameter counts

| component | trainable | frozen |
|---|---:|---:|
| Branch A: LoRA | 786,432 | 303,179,776 (CLIP) |
| Branch A: projection / norm / head | 65,600 / 128 / 4,290 | — |
| Branch B: FS-VFM encoder | 0 | 303,301,632 |
| Branch B: `P_R` (latent 128, hidden 256) | 0 after Stage A | 591,488 |
| Branch B: evidence head | 65,858 | — |
| §4.2 direct probe (control) | 65,730 | — |
| Branch C: SDXL VAE | 0 | 83,653,863 |
| Branch C: head | 290 | — |
| Applicability gate (per specialist) | 321 | — |
| Risk model | 5 | — |

Stage B trainable total: **922,598** = 856,450 (Branch A) + 65,858 (reference head) + 290
(process head). The §4.2 direct probe adds a further **65,730**, giving **988,328** parameters
optimised in Stage B — it is trained so the control is fair, but its output never enters fusion.

---

## 5. Staged sequence (§9)

| stage | what trains | what is frozen | artifact |
|---|---|---|---|
| Pre | nothing | — | §8 parity report |
| A | `P_R` only, on FF++ real FS-VFM features, cosine | FS-VFM | `reference_C3_ae_cosine.pt` |
| — | process standardisation (statistics only) | VAE | `process_stats.pt` |
| B | LoRA + `H_sem`, `H_ref`, `H_proc`, `e_direct` | FS-VFM, `P_R`, VAE | per-epoch checkpoints |
| C | nothing | all | selected checkpoint |
| D | `q_ref`, `q_proc` on VAL_meta (5-fold) | all experts | gates + out-of-fold `q` |
| E | risk model on out-of-fold `q` | all | frozen `DeferPolicy` |
| F | nothing | all | results + instrumentation |

---

## 6. Partitions

FF++ train: 115,198 frames (23,039 real / 92,159 fake) at `frame_num.train = 32`, measured.

Meta split (§12), measured on the real FF++ **val** split:

| | videos | real | fake | identities |
|---|---:|---:|---:|---:|
| VAL_select | 420 (60.0%) | 84 | 336 | 84 |
| VAL_meta | 280 (40.0%) | 56 | 224 | 56 |

700 videos → 70 identity groups. Per-subset balance is exact (84 of each of FF-real/DF/F2F/NT/FS
in VAL_select; 56 of each in VAL_meta). Folds: 60/60/60/50/50 videos. **Zero identities shared**
across partitions or folds.

---

## 7. Formulas implemented

```
alpha = e + 1 · S = sum(alpha) · p = alpha/S · u = K/S · belief_k = e_k/S     (§7)
assert sum_k belief_k + u == 1                                                (§7, enforced)
belief'_b = q_b · belief_b · u'_b = (1-q_b) + q_b·u_b                         (§14.2)
branch_valid_b = 0  ->  belief' = 0, u' = 1                                   (§14.1)
DS: C = sum_{i!=j} b_a,i·b_b,j · b_f = (b·b + b·u + b·u)/max(1-C, eps)        (§14.2)
S_f = K/clamp(u_f) · e_f = belief_f·S_f · alpha_f = e_f + 1                   (§16)
V = u_f · w_b = q_b(1-u_b) · C = sum w_b w_c JS(p_b,p_c)/(sum w_b w_c + eps)
A = 1 - mean_{b in {ref,proc}} w_b                                            (§17)
t_b = 1[CE(p_sem,y) - CE(p_sem(+)b^DS, y) > delta], delta = 0                 (§13)
risk = sigma(w·[V, C, A, |p_f - 0.5|] + b)                                    (§18)
```

**One departure from DiCoME, and it is a finding.** The epsilon guard §14.2 asks for is necessary
but *not sufficient*. At **total** conflict every numerator is zero as well, so the clamp returns
a zero "opinion" whose belief and vacuity sum to 0 — finite, but not an opinion, and it silently
corrupts `p`, `V` and every risk feature. Dempster's rule is genuinely undefined there. V1 falls
back to the **vacuous** opinion (the views annihilate, so nothing is known) and flags the sample;
`C`, computed from the pre-fusion opinions, still reports that informed experts disagreed. That
separation is exactly what §17 exists to provide.

---

## 8. Guards passed

See `V1_GUARDS_REPORT.md` for the full list. Summary: **121 tests pass**, 1 skips without an
env var.

| suite | result |
|---|---|
| `test_fsvfm` | 11 |
| `test_semantic_branch` | 9 |
| `test_process_branch` | 7 |
| `test_ds_fusion` | 20 |
| `test_v1_model` | 12 (+1 skip) |
| `test_applicability_gate` | 13 |
| `test_risk_model` | 13 |
| `test_reference` | 13 |
| `test_discern_v2` / `test_applicability` / `test_integration` | 23 / 21 / 12 |

---

## 9. Parity decision (§6, §8)

**§6 numerical parity: exact.** Our `FrozenFSVFM` reproduces the official FSFM-CVPR25 downstream
feature with max absolute difference **0.0** and cosine **1.0**, compared against the official
`models_vit` built and loaded from the clone by the official recipe.

**§8 crop parity: feature-side agreement is high, probe-side pending a real run.** On 64 paired
FF++ frames, native DLIB+30% and aligned crops agree at mean cosine **0.9942** (min 0.9885) in the
frozen representation. The probe arm is `TODO(run)` at a usable subset size — see
`V1_PARITY_REPORT.md`.

---

## 10. Deviations from the spec

| § | deviation | reason |
|---|---|---|
| §8 | reference branch reads the **aligned** crop, not the native DLIB+30% default | Umar's decision, 2026-08-18: avoids a second crop cache across every split. The parity control measures the cost. Native crops are unreachable for DFDC/DFDCP/UADFV regardless — no source frames locally |
| §4.1 | `P_R` hidden 256 / latent 128 rather than the module default 32 | the default was sized for DiCoME's 64-d feature; on 1024-d it is a 1024→32 bottleneck that dominates `r_ref`. 🟡 ASK-UMAR |
| §14.2 | vacuous fallback at total conflict, beyond the epsilon guard | Dempster's rule is undefined there; see §7 above |
| §12 | source-disjointness by union-find over **every** id in a video name | `_extract_source_video` keeps only the first id, so `802_885` would leave identity 885 free to cross the split boundary |

---

## 11. Not yet built

**Stage B has no training entry point.** All three branches, the fusion layer, the gates and the
risk model exist and are tested, but nothing yet wires them to `training/train.py`'s loop with the
three-way preprocessing, the per-branch EDL losses and per-epoch checkpointing. `V1_RUN_COMMANDS.md`
marks that step accordingly rather than printing a command that does not run.
