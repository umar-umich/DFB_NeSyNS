# Manifold projectors in DiCoME/P0-DS — code, mechanism, and what has been tried

Written to be handed to someone with no session context, for architecture planning. Every number
was measured in this repo.

---

## 1. Where the code is

Two repos are involved. The pilots were **implemented** in `DFB_NeSyNS` and **trained into**
`DiCoME`'s run tree.

### The projectors

| what | path |
|---|---|
| **All projector variants** (β-VAE, deterministic AE, β-TCVAE, WAE, **MR-VAE**) | `DFB_NeSyNS/training/networks/discern_v2/projectors.py` |
| `MRVAEProjector` + `FiLM` + `BETA_GRID` | same file, lines ~222–300 |
| Rate branch built on the MR-VAE operator | `DFB_NeSyNS/training/networks/discern_v2/rate_branch.py` |
| DiCoME's own β-VAE (the one inside P0-DS) | `DiCoME/src/modules/beta_vae_aligned.py` |
| **Port of MR-VAE into the fork** | `DISCERN_Ext/src/modules/mrvae_projector.py` |

### Where it plugs in

| what | path |
|---|---|
| Model, view construction, DS fusion | `DiCoME/src/model/core_model.py` (`forward`, `_geometric_view_purification`) |
| Lightning module, loss composition | `DiCoME/src/model/dicome_module.py` (`_compute_loss`) |
| The VAE/alignment objective | `DiCoME/src/losses/aligned_vae_loss.py` |
| Evidential (EDL) loss | `DiCoME/src/losses/evidential_loss.py` |
| Training entry point | `DiCoME/tools/train/train_dicome.py` (fork: `tools/train/train_discern_ext.py`) |
| Config schema | `DiCoME/src/config.py` |

### Training configs and checkpoints

| what | path |
|---|---|
| P0-DS recipe (the baseline) | `DiCoME/eval_adaptation/configs/train_ffpp.yaml` |
| P0-DS checkpoints | `DiCoME/runs/dicome_train/ffpp_reproduce_seed42/checkpoints/` |
| **P1d (MR-VAE) checkpoints** | `DiCoME/runs/pilots/P1d/checkpoints/` |
| All pilot results | `DFB_NeSyNS/docs/DiCoME_eval/RESULTS_ALL_PILOTS.md` |
| Fork with the MR-VAE flag wired | `DISCERN_Ext/` (`manifold_projector: beta_vae\|mrvae`) |

In the fork, `manifold_projector: mrvae` + `artifact_extra_dims: 5` reproduces P1d. Default is
`beta_vae`, so nothing existing changed.

---

## 2. How MR-VAE differs from β-VAE — mechanism

Both sit in the same place: they reconstruct the 64-d CLIP semantic feature `f_s` to a
manifold-consistent `f_c`. The residual `f_r = f_s − f_c` is projected onto the orthogonal
complement of `f_s` to give the **artifact view** `f_a`. Two EDL heads (semantic, artifact) are
then combined by Dempster-Shafer inside the model.

### β-VAE (DiCoME, P0-DS)

```
f_s (64) → Linear(64→32) → ReLU → {mu(32), log_var(32)}
                                → z ~ N(mu, sigma)
                                → Linear(32→32) → ReLU → Linear(32→64) → f_c
```

One reconstruction, at one implicit rate. Loss:

```
L_vae = lambda_align * (1 − cos(f_s.detach(), f_c))  +  beta_kld * KL(q(z|f_s) || N(0,I))
```

with `beta_kld = 2.0`, `lambda_align = 1.0`, and `L_total = L_evidential + 0.7 * L_vae`.

### MR-VAE (P1d)

```
f_s (64) → Linear(64→32) → ReLU → FiLM(log beta) → {mu, log_var}
                                                 → z
                                                 → Linear(32→32) → ReLU → FiLM(log beta) → Linear(32→64) → f_c
```

Three differences that matter:

1. **beta is an input, not a constant.** It enters through FiLM — `h * (1 + gamma) + delta` with
   `gamma, delta = Linear(log beta)` — so it *rescales* the representation multiplicatively. A
   shift alone could not express "how much information survives". FiLM is initialised to identity
   (`gamma = delta = 0`), so at init the model is an ordinary unconditioned VAE.

2. **beta is sampled log-uniformly during training** over `BETA_RANGE = (0.1, 10.0)`, because the
   rate-distortion curve is roughly linear in `log beta`; uniform sampling would undertrain the
   low-beta end where the signal is expected to live.

3. **It exposes a per-sample rate response.** `rate_response(f_s)` returns a K-point curve of
   reconstruction distortion across `BETA_GRID = (0.1, 0.32, 1.0, 3.16, 10.0)` — how fast the
   feature degrades as the latent is squeezed harder.

### The part that is easy to miss

**P1d is not just a projector swap.** Its artifact branch is **69-d = 64 (orthogonal residual) +
5 (the rate response)**. The rate curve is concatenated onto the artifact feature and fed to the
artifact head. That concatenation is the pilot's actual hypothesis — the claim is that *how a
feature degrades under compression* is forensically informative, not merely that a multi-rate
decoder reconstructs better. Any redesign should treat "multi-rate projector" and "rate curve as
evidence" as two separable choices; P1d changed both at once.

---

## 3. What has been tried, and what came closest to P0-DS

All trained identically: FF++ c23, seed 42, batch 128, LR 1e-4, 20 epochs. Video AUROC.

| source | **P0-DS** (β-VAE) | P1a det. AE | P1b β-TCVAE | P1c WAE | **P1d MR-VAE** | **P2a AEROBLADE** | P3a LaRE2 | P4 LGrad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FF++ | 0.9929 | 0.9924 | 0.9902 | 0.9892 | 0.9924 | 0.9928 | 0.9914 | 0.9897 |
| CDFv2 | **0.9644** | 0.9396 | 0.9434 | 0.9421 | 0.9495 | 0.9597 | 0.9641 | 0.9458 |
| CDFv3 | **0.9516** | 0.9100 | 0.8950 | 0.8902 | 0.8852 | 0.8969 | 0.8664 | 0.8840 |
| DFD | 0.9419 | 0.9320 | 0.9255 | 0.9346 | 0.9358 | 0.9362 | **0.9482** | 0.9309 |
| DFDC | **0.8825** | 0.8613 | 0.8563 | 0.8627 | 0.8705 | 0.8660 | 0.8664 | 0.8615 |
| DFDCP | 0.8576 | 0.8879 | 0.8814 | 0.8713 | **0.8959** | 0.8854 | 0.8686 | 0.8708 |
| DFEval24 | **0.6918** | 0.6494 | 0.6514 | 0.6560 | 0.6645 | 0.6811 | 0.6685 | 0.6603 |
| **mean Δ** | — | −0.0157 | −0.0199 | −0.0195 | **−0.0127** | **−0.0092** | −0.0156 | −0.0199 |

**Nothing beat P0-DS on mean AUROC.** Closest: **P2a (−0.0092)**, then **P1d (−0.0127)**.

- **P2a — AEROBLADE**: adds a THIRD view from a frozen SDXL-VAE reconstruction residual. Closest
  overall, and nearly matches P0-DS on CDFv2 (0.9597) and FF++. It is an extra view, not a
  projector swap, so it is orthogonal to the MR-VAE question and could combine with it.
- **P1d — MR-VAE**: only pilot that beat P0-DS anywhere by a wide margin — **DFDCP +0.0383**.
- **P3a — LaRE2**: best DFD of any arm (0.9482) and near-P0-DS on CDFv2 (0.9641), but worst CDFv3.
- **P1a/P1b/P1c** (deterministic AE, β-TCVAE, WAE) all lose 0.016–0.020. Simply varying the VAE
  family does not help; that space looks explored.

Note **every** variant beats P0-DS on **DFDCP**, where P0-DS is unusually weak (0.8576). That is a
pattern worth explaining rather than averaging away.

---

## 4. The finding that reopened this

The pilot table is AUROC-only; it predates the health dashboard. Re-scored on real-side metrics,
with τ frozen per anchor on its own FF++ val:

| dataset · readout | MR-VAE AUROC / **FPR_real** | β-VAE AUROC / **FPR_real** |
|---|---|---|
| CDFv2 · fused | 0.9505 / **0.067** | **0.9646** / 0.129 |
| DFDCP · fused | **0.8957** / **0.252** | 0.8573 / 0.391 |
| VALmix · fused | 0.8843 / **0.175** | **0.8852** / 0.213 |
| CDFv2 · artifact | 0.9513 / **0.051** | **0.9685** / 0.073 |
| VALmix · artifact | 0.8742 / **0.107** | **0.8789** / 0.127 |

MR-VAE has **lower real-side FPR in every pairing measured**, and wins both axes on DFDCP. The
lowest FPR of any anchor in this project is MR-VAE artifact at 0.051 (CDFv2).

**Not yet established**: only 3 of 7 datasets re-scored. P1d's large AUROC losses on CDFv3
(−0.066) and DFEval24 (−0.027) are NOT re-measured on the operational axis. If those carry an FPR
penalty too, this reverses.

---

## 5. Open questions for the architecture discussion

1. **Separate the two changes P1d conflated.** Multi-rate projector *without* the rate concat,
   and β-VAE *with* a rate-like feature, are both unrun. Which half carries DFDCP's +0.038?
2. **Why does CDFv3 punish every variant?** P0-DS 0.9516 → best alternative 0.8969. CDFv3 is our
   face-swap-only subset (8 generators). Something about the β-VAE specifically suits it.
3. **Is the 5-point grid right?** `BETA_GRID` and `hidden_dim = 32` are inherited defaults. The
   32-d hidden was sized for a 64-d feature; on a wider encoder it dominates reconstruction error,
   which *is* the rate response.
4. **Can P2a and P1d combine?** They change different things — third view vs projector — and are
   the two closest arms.
5. **Checkpoint selection.** Every pilot, P0-DS included, selected on saturated FF++ val AUROC
   (all above 0.995, ranking among them is noise). P1d's epochs 3 and 13 have never been
   evaluated on anything else.
