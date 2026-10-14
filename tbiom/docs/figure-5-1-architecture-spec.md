# Figure 5.1 — exact architecture specification, read from the code

Every value below was read from the repository on 2026-10-11 and carries its file path and line
number. Values that are not present in code or configs are marked **not found** rather than
inferred. Two repositories are involved: `DFB_NeSyNS` (this one) and `DISCERN_Ext`, which is where
the deployed LFS/off-manifold branch was trained.

Paths are relative to `/data/umar/Repos/DFB_NeSyNS` unless the line says `DISCERN_Ext`.

---

## 1. APP branch

| item | value | source |
|---|---|---|
| Backbone checkpoint | `vit_pe_core_large_patch14_336` (timm), key `pe_core_l` | `upgrade/extract_spatial.py:63` |
| Input resolution | **336** | `upgrade/extract_spatial.py:63`; used at `upgrade/adapt_backbone.py:320,331` |
| Upsampling | stored crops are 256, so input is 100% upsampled by 1.31x | `upgrade/extract_spatial.py:57-62` |
| Adaptation | **LoRA + LayerNorm** (recipe `LN`) | `upgrade/branchfusion/branches.py:59`; `upgrade/adapt_backbone.py:118-130` |
| LoRA rank / alpha / dropout | **8 / 16 / 0.1** | `upgrade/adapt_backbone.py:48-50`; run record `analysis/upgrade/adapt/pe_core_l_LN_s0.json` |
| LoRA target modules | `q_proj`/`v_proj`, or the fused `qkv` Linear | `upgrade/adapt_backbone.py:93-97` |
| Fused-qkv consequence | timm ViTs fuse qkv, so for PE-Core the **key** projection is adapted too (documented deviation) | `upgrade/adapt_backbone.py:88-91` |
| Embedding used | timm pooled output (`num_classes=0`), readout `"pooled"` | `upgrade/extract_spatial.py:63`; `upgrade/adapt_backbone.py:186-190` |
| Embedding dimension | **1024** | `analysis/upgrade/adapt/pe_core_l_LN_s0.json` (`"dim": 1024`) |
| Evidence head | `Linear(1024 -> 256) -> GELU -> Dropout(0.1) -> Linear(256 -> 2)` | `upgrade/train_head.py:42` |
| Evidence activation | **softplus** | `upgrade/train_head.py:45` |

### Nuisance suppression — NOT in the deployed branch

The deployed A branch applies the head directly to cached features with no projection
(`upgrade/branchfusion/branches.py:59-60`, reader at `upgrade/branchfusion/branches.py:~84-107`).
The audit states it outright: *"B0 carries NO nuisance suppression, so none is added."*
— `tbiom/reports/branch_fusion/audit.md:10` and `:297`.

A standalone study exists and was never adopted (`upgrade/part4_suppression.py`):

| item | value | source |
|---|---|---|
| Paired differences | `Delta_{i,a} = z(T_a(x_i)) - z(x_i)` over 6 processing settings | `upgrade/part4_suppression.py:38,83-101` |
| Settings | `jpeg:70, jpeg:50, jpeg:30, resize:0.75, resize:0.50, resize:0.35` | `upgrade/part4_suppression.py:38` |
| Retained rank q | top-k at 90% explained variance, capped at 32; **realised rank 32** at 0.747 variance | `upgrade/part4_suppression.py:40,112-119`; `analysis/upgrade/part4_subspaces_pe_core_l.json` |
| Readout the basis was fitted on | `last+mid+half`, 3072-d | `analysis/upgrade/part4_subspaces_pe_core_l.json` |
| Stage of estimation | fitted on **FF++ train only** | `upgrade/part4_suppression.py:16` |
| Standardization statistics source | **not found** | — |

---

## 2. LFS branch

Branch `O` is read from delivered CSVs (`upgrade/branchfusion/branches.py:63-64`); the
implementation lives in `DISCERN_Ext`.

| item | value | source |
|---|---|---|
| CLIP checkpoint | `openai/clip-vit-large-patch14` | `DISCERN_Ext/src/config/discern_ext_default.yaml:7`; default at `DISCERN_Ext/src/encoders/clip_encoder.py:17` |
| Input resolution | **not found** — never set; comes from `CLIPProcessor.from_pretrained` | `DISCERN_Ext/src/encoders/clip_encoder.py:28,71-72` |
| Feature used | `outputs.pooler_output` — final-layer post-LN pooled CLS, not an intermediate layer or token | `DISCERN_Ext/src/encoders/clip_encoder.py:81` |
| Raw dimension | `clip_vision_model.config.hidden_size` | `DISCERN_Ext/src/encoders/clip_encoder.py:31` |
| Projection | `nn.Linear(hidden -> 64)` | `DISCERN_Ext/src/encoders/clip_encoder.py:56` |
| Working feature dim | **64** | `DISCERN_Ext/src/config/discern_ext_default.yaml:8` |
| LoRA rank / alpha / dropout | **8 / 16 / 0.1** | `DISCERN_Ext/src/config/discern_ext_default.yaml:16-18` |
| LoRA target modules | **`q_proj`, `v_proj`** | `DISCERN_Ext/src/config/discern_ext_default.yaml:13-15` |

### Manifold projector — beta-VAE, not MR-VAE

`manifold_projector` defaults to `"beta_vae"` and the default config does not set it
(`DISCERN_Ext/src/model/core_model.py:40-48`).

| item | value | source |
|---|---|---|
| beta-VAE encoder | `Linear(64 -> 32) -> ReLU`; `fc_mu`, `fc_log_var` each `32 -> 32` | `DISCERN_Ext/src/modules/beta_vae_aligned.py:21-29` |
| beta-VAE decoder | `Linear(32 -> 32) -> ReLU -> Linear(32 -> 64)` | `DISCERN_Ext/src/modules/beta_vae_aligned.py:32-36` |
| Hidden dim | **32** | `DISCERN_Ext/src/modules/beta_vae_aligned.py:21` |
| Latent dim | **32** | `DISCERN_Ext/src/config/discern_ext_default.yaml:21` |
| beta-VAE latent mean at inference | **not found** in the lines read | — |
| MR-VAE (present, NOT selected) | same widths plus FiLM; `encoder_mlp 64->32 + ReLU`, `fc_mu/fc_log_var 32->32`, `decoder_in 32->32 + ReLU`, `decoder_out 32->64` | `DISCERN_Ext/src/modules/mrvae_projector.py:50-59` |
| MR-VAE latent mean at inference | **yes** — `if not self.training: return mu` | `DISCERN_Ext/src/modules/mrvae_projector.py:71-74` |
| Residual | `f_r = f_s - f_c`, then the component along `f_s` removed | `analysis/tbiom/temporal/residual_head.py:33-37` |
| Evidence head | `Linear(64 -> 64) -> ReLU -> Linear(64 -> 2)`, softplus | `DISCERN_Ext/src/modules/evidential_head.py:11-16` |
| Head input width | `artifact_dim = feature_dim + artifact_extra_dims = 64` | `DISCERN_Ext/src/model/core_model.py:113,115` |

---

## 3. TD branch

Trainer: `tbiom/handoff/PECORE_L14_TemporalBranch/code/tbiom/temporal/train_branch.py` (abbreviated
`train_branch.py` below).

| item | value | source |
|---|---|---|
| Checkpoint | `facebook/vjepa2-vitl-fpc64-256` | `train_branch.py:50` |
| Frames per clip | **16**, sampled uniformly IN ORDER over stored crops | `train_branch.py:50`; `train_branch.py:193` |
| Token dimension | **1024** | `train_branch.py:50` |
| Input resolution | not in repo code; HF config gives `crop_size` / `image_size` **256** | `~/.cache/huggingface/hub/models--facebook--vjepa2-vitl-fpc64-256/.../config.json` |
| Patch size | not in repo code; HF config **16** | same |
| Tubelet size | not in repo code; HF config **2** | same |
| Pretrained frames_per_clip | HF config **64** (branch feeds 16) | same |
| Resulting token count | **not found** — never computed in the repo; not derived here | — |
| Pooling over tokens | mean: `.last_hidden_state.mean(1)`, `skip_predictor=True` | `train_branch.py:316` |
| LoRA rank / alpha / dropout | **8 / 16 / 0.1** | `train_branch.py:68` |
| LoRA target modules | scope `"encoder"`, leaves **`{query, value}`** | `train_branch.py:70`; applied `train_branch.py:102-112` |
| Evidence head | `LayerNorm(1024) -> Linear(1024 -> 128) -> GELU -> Linear(128 -> 2)` | `train_branch.py:286` |
| Evidence activation | **softplus**, applied in `forward` | `train_branch.py:322` |

---

## 4. Opinions and fusion

### Evidence to belief and vacuity

```
alpha = e + 1
S     = sum(alpha)
p     = alpha_fake / S
u     = K / S          with K = 2
```
— `train_branch.py:390-392`, docstring `train_branch.py:366-373`; `K = 2.0` at
`upgrade/branchfusion/opinions.py:42`.

### (p, u) to a mass function

```
b1 = p - u/2
b0 = 1 - p - u/2
m(Theta) = u
```
— `analysis/tbiom/grouped/ops.py:33-38`. The same function returns a mask of rows violating
`u/2 <= p <= 1 - u/2`.

### Frame pooling

| item | value | source |
|---|---|---|
| Reduction | unweighted mean over a video's frames, applied to `p` and `u` separately | `upgrade/branchfusion/common.py:214-222`; called at `upgrade/branchfusion/branches.py:~105-106` |
| Frames per video at inference | **not a configured value**; `frames_of` applies no per-video cap | `upgrade/adapt_backbone.py:232-237` |
| Observed in cache | 30-32 frames per video (measured on `logs/upgrade/feats/pe_core_lLN_CDFv2.npz`: 518 videos, 16572 frames) | — |

### Reliability coefficients (deployed subset A+V+O)

| branch | coefficient |
|---|---|
| A (APP) | **1.00** |
| V (TD) | **0.25** |
| O (LFS) | **0.75** |

Source: `analysis/branchfusion/test_eval.json`, row `A+V+O`, operator `F8`,
`tau = 0.5226902760177873`, `tau_source = "FFpp_val"`.

| item | value | source |
|---|---|---|
| Application | **evidence scaling** `e' = r * e`, in the evidence domain, then back to an opinion | `analysis/tbiom/grouped/ops.py:53-57` |
| Convention constant | `DISCOUNT_DEFAULT = "evidence_scaling"` (not belief discounting) | `upgrade/branchfusion/opinions.py:52` |
| How fitted | exhaustive search over grid `(0, 0.25, 0.5, 0.75, 1.0)`, at least one branch anchored at 1 | `upgrade/branchfusion/opinions.py:293-302,323-326` |

### Direct multisource PCR6+

| item | value | source |
|---|---|---|
| Function | `fuse(masses, plus=True, attribute=False)` | `analysis/tbiom/pcr6p/ops_pcr6p.py:107-138` |
| Input | `(n, S, 3)` array of `(b1, b0, u)` per source | `analysis/tbiom/pcr6p/ops_pcr6p.py:108` |
| Returns | fused `(n, 3)` opinion `(m(F), m(G), m(Theta))` | `analysis/tbiom/pcr6p/ops_pcr6p.py:137-138` |
| With `attribute=True` | also `{"K_TS", "K_AO", "K_all", "K"}` — conflict mass by source pair | `analysis/tbiom/pcr6p/ops_pcr6p.py:116,138` |
| `plus` flag | selects PCR6+ (eq. 26) over PCR6 (eq. 15) | `analysis/tbiom/pcr6p/ops_pcr6p.py:110-111` |
| Wrapper | operator `F8` | `upgrade/branchfusion/opinions.py:65,151,243` |

### Final quantities

| item | formula | source |
|---|---|---|
| Fake probability | pignistic `p = clip(m(F) + m(Theta)/2, 0, 1)` | `analysis/tbiom/grouped/ops.py:60-61` |
| Fused vacuity | third component `m(Theta)` of the fused opinion | `analysis/tbiom/pcr6p/ops_pcr6p.py:137` |
| Conflict K | accumulated product mass over conflicting focal-element tuples | `analysis/tbiom/pcr6p/ops_pcr6p.py:115,122-125` |
| K reported as | `conflict_mean`, `conflict_max` | `upgrade/branchfusion/opinions.py:252-253` |

### Decision threshold

EER point computed on the **FF++ validation rows inside development only**, never on a test set:
`tau_source="FFpp_val"`, `tau = float(eer(dev.y[ff], pdev[ff])[1])` — `upgrade/branchfusion/subsets.py`
(`evaluate_test`); `eer` imported from `health` at `upgrade/branchfusion/subsets.py:57`.

---

## 5. Labels and naming

| item | value | source |
|---|---|---|
| Focal elements | `FAKE, GEN, THETA = 0b01, 0b10, 0b11` (bit sets over {fake, genuine}) | `analysis/tbiom/pcr6p/ops_pcr6p.py:56` |
| Head class order | **index 1 is fake** | `upgrade/branchfusion/branches.py:~102`; `train_branch.py:369` |
| Config | `num_classes: 2`, `binary_labels: true` | `DISCERN_Ext/src/config/discern_ext_default.yaml:5-6` |
| Fusion branch names | `A`, `V`, `O`, plus `M`, `P`, `S`, `R`, `Rmean` | `upgrade/branchfusion/branches.py:58-76` |
| PCR6+ view names | `VIEWS = ("T", "APP", "OM")` | `analysis/tbiom/grouped/ops.py:29` |
| "LFS" / "TD" | **do not appear as code identifiers** | — |

Mapping for the figure: **APP -> A**, **LFS -> O** (code calls it OM / `legacy_om`),
**TD -> V** (code calls it T).

---

## Mismatches between the code and the proposal claims

1. **APP resolution.** Proposal says 448. Code uses **336**, and the input is 100% upsampled from
   256 crops (1.31x).
2. **APP adaptation.** Proposal says layer-norm only. The deployed branch uses **LoRA + LayerNorm**
   (recipe `LN`).
3. **APP LoRA targets.** Even described as q/v, PE-Core's fused `qkv` Linear means the **key**
   projection is adapted as well.
4. **LFS projector.** Proposal says MR-VAE. The default and deployed projector is the **beta-VAE**.
   MR-VAE is implemented but requires `manifold_projector: mrvae`, which the default config does
   not set. The 32-d latent is correct for both.
5. **LFS resolution.** Proposal says 224x224. **No resolution is set anywhere in code or config** —
   it is whatever `CLIPProcessor` defaults to. Verify before stating it.
6. **TD head width.** Proposal says "LayerNorm 256 GELU 2". Code is **LayerNorm -> 1024->128 ->
   GELU -> 128->2**; the hidden width is 128.
7. **TD "16 crops at 256x256".** 16 frames is correct. 256 comes from the HF checkpoint config,
   not repo code. The checkpoint is `fpc64`, pretrained at 64 frames per clip while the branch
   feeds 16 — worth stating if the figure shows the clip length.
8. **LFS head activation.** Hidden width 64 is correct, but the activation is **ReLU** where APP
   and TD use GELU. If the figure shows one activation for all three heads, that is wrong.
9. **Nuisance suppression.** If Figure 5.1 draws a suppression block in the APP branch, it
   contradicts the audit: the deployed system carries none.
10. **Branch naming.** APP / LFS / TD appear nowhere in the code. Figure labels will not match any
    identifier a reader can grep; the mapping table above is needed in the caption or the text.

### Not verified — do not state these without checking

- LFS input resolution (never set in code or config).
- Whether the beta-VAE uses the latent mean at inference (MR-VAE does; beta-VAE not found).
- TD token count (never computed in the repo; not derived here).
- Standardization statistics source for the suppression study.
