# Branch 3 and the Three-View Framework — experimental report

**Scope.** Everything from Stage 5's failed generative-process branch through Run 1's four
trained configurations. Protocol is constant throughout: FF++ c23, seed 42, batch 128, 20 epochs,
video-level AUROC under the single video-identity rule (`analysis/tbiom/video_id.py`), and `tau`
frozen per model at its own EER on FF++ val and applied unchanged everywhere.

**Means are over six OOD sets** — Celeb-DF-v2, Celeb-DF-v3, DFD, DFDC, DFDCP,
Deepfake-Eval-2024. VALmix is the development split: reported, never averaged in, and it is the
only data F2's fusion weights ever see.

**Reference.** P0-DS, the two-view retrainable DiCoME chassis (CLIP semantic + β-VAE artifact),
mean OOD AUROC **0.8633**, mean real-side FPR **0.247**.

---

## 1. Headline

| arm | Branch 3 | fusion | loss | mean AUROC | vs P0-DS | mean FPR_real |
|---|---|---|---|---:|---:|---:|
| P0-DS | — (two views) | DS | EDL | 0.8633 | — | 0.247 |
| armA | LoRA student | DS | EDL, **fused-only** | 0.8301 | −0.0332 | **0.169** |
| **armB** | LoRA student | DS | EDL **+ per-branch** | **0.8729** | **+0.0096** | 0.177 |
| 1c | FSFM released FT | DS | EDL + per-branch | 0.8620 | −0.0013 | 0.256 |
| 1d | FSFM released FT | mean logits | **cross-entropy** | 0.8689 | +0.0056 | 0.274 |

**armB is the configuration that clears the bar.** It also satisfies the real-side condition —
FPR does not regress on Celeb-DF-v2 (0.067 vs 0.129), Celeb-DF-v3 (0.067 vs 0.129) or DFDC
(0.228 vs 0.317). With F1 or F2 fusion it reaches **+0.0145**.

### Per dataset

| dataset | P0-DS | armA | **armB** | 1c | 1d |
|---|---:|---:|---:|---:|---:|
| Celeb-DF-v2 | **0.9646** | 0.8638 | 0.9523 | 0.9555 | 0.9461 |
| Celeb-DF-v3 | 0.8409 | 0.8626 | **0.8888** | 0.8807 | 0.8782 |
| DFD | 0.9421 | 0.8812 | 0.9376 | 0.9308 | **0.9431** |
| DFDC | 0.8828 | 0.8559 | **0.8839** | 0.8705 | 0.8797 |
| DFDCP | 0.8573 | 0.8381 | **0.8991** | 0.8675 | 0.8801 |
| Deepfake-Eval-2024 | **0.6922** | 0.6789 | 0.6755 | 0.6669 | 0.6863 |
| VALmix *(dev)* | **0.8852** | 0.8276 | 0.8823 | 0.8766 | 0.8742 |

armB's profile is large gains where P0-DS is weakest (**CDFv3 +0.0479**, **DFDCP +0.0418**) and
small losses where it is strongest. That anti-correlation is the signature of a genuinely
complementary third view rather than a uniformly better model.

---

## 2. Stage 5 — the generative-process branch failed, and how

Branch 3 was originally an AEROBLADE-style frozen SDXL-VAE reconstruction residual: encode/decode
cycle, six pooled error statistics (`mse_mean`, `mse_std`, `mse_max`, `mse_p90`, `lpips_proxy`,
`center_ratio`), then an evidential head.

**It contributed ±0.0001 AUROC.** Measured by inverting each view's exported `(p, u)` back to its
Dirichlet opinion (`S = K/u`) and replaying the fusion with and without it; the replay reproduces
the exported `p_fused` to **2.15e-07**, so the counterfactual is exact.

| domain | 2-view | 3-view | Δ | Branch 3 alone | mean u |
|---|---:|---:|---:|---:|---:|
| CDFv2val | 0.9051 | 0.9051 | −0.0000 | 0.5101 | 0.9978 |
| DFDCPval | 0.9535 | 0.9536 | +0.0000 | **0.6038** | 0.9934 |
| DFEval24val | 0.5699 | 0.5697 | −0.0001 | 0.5021 | 0.9958 |

`u ≈ 0.996` means evidence ≈ 0 — a **vacuous** opinion, which is the **identity element** of the
DS orthogonal sum. The branch was not outvoted; it was ignored by the algebra.

**The input was never the problem.** A logistic probe on the same six statistics reaches
**0.7047** on FF++ val (`center_ratio` alone 0.6661). The head was the problem — and giving it
**89× more capacity made it worse** (u 0.871 → 0.996). A better head learns to be silent more
precisely, which rules out capacity, initialisation and conditioning.

### The controlled contrast

The V1 anchor's exports carry `p_proc` / `u_proc` — an independent implementation of the same
AEROBLADE idea in a different framework. It is **not** vacuous:

| framework | supervision | mean u_proc | process AUROC |
|---|---|---:|---:|
| DISCERN-Ext Stage 5 | fused-only | 0.996 | 0.51–0.60 |
| **V1** | fused **+ per-branch EDL** | **0.351** | **0.621** (0.710 DFDCP) |

`discern_v1_model.py:276` — `auxiliary_losses()`, *"so each branch is individually usable."* That
pointed directly at per-branch supervision as the intervention, and Run 1 tested it.

**Not a controlled experiment**, and recorded as such: V1 also differs in fusion operator (CCF,
where a low-mass view is not the identity), branch composition, head construction and
applicability discounting.

### Alternative explanations, kept open

Correlated errors; evidence-scale mismatch (0.12–0.17 against 0.9 is an order of magnitude);
the EDL loss's own penalty on confident error, which makes low evidence the safe policy for a
0.6-accuracy view; sequential DS chaining, where the third view meets an already-sharpened
opinion; gradient domination, since the CLIP heads share a trunk and receive gradient through two
paths. **Initial-scale dominance** was added by Run 1 and is the one this evidence most directly
implicates.

### P2a's pilot, re-read

P2a motivated Branch 3 with `auc_operator` 0.87–0.97. The same rows report
**`error_overlap = 1.0`, `rescue = 0`, `harm = 77`** at threshold 0.5 — coexisting only if its
operator scores were also clustered just above 0.5. P2a used **no auxiliary operator loss**
either. So its third view was very likely vacuous too, and its headline came from the two CLIP
branches.

**Confidence: inference, not measurement.** P2a exported semantic/artifact/fused evidence and
uncertainties for all eight datasets but no operator evidence and no `u_operator`. Its
checkpoints survive at `DiCoME/runs/pilots/P2a/checkpoints/`, so the direct measurement is
runnable and has not been run.

---

## 3. Branch-3 candidate search

A transfer bar was established after Stage 5 and is now scripted
(`analysis/tbiom/probe_branch3_candidates.py`): **fit a probe on FF++ frames only, test zero-shot,
report frame and video AUROC.** Minutes on cached features, no GPU. Nothing is built as a branch
until it clears this.

### Rejected — 83-d hand-crafted forensic features

Six families (region forensics, PPNC, CCNC, SRM, multi-scale noise, spectral FFT), precomputed
for 29 splits. Within-dataset cross-validation makes them look excellent and **those numbers are
not comparable to anything here**, because each fold trained on the same dataset it tested on:

| set | within-dataset CV | **FF++-only fit, zero-shot (video)** |
|---|---:|---:|
| FF++ c23 | 0.9543 / 0.9661 | — |
| Celeb-DF-v2 | 0.9613 / **0.9804** | **0.6294** |
| Celeb-DF-v3 FaceSwap | 0.9210 / 0.9412 | **0.5006** |
| DFDCP | 0.9382 / **0.9819** | 0.5574 |
| DFD | — | 0.5667 |
| UADFV | — | 0.5131 |

0.95 → 0.50 is the signature of features fingerprinting **acquisition** — compression, camera,
preprocessing — not manipulation. Celeb-DF-v3 lands at exactly chance. Rejected.

### Promoted — FS-VFM

Two candidates, both built on FS-VFM ViT-L/16 (VGGFace2, ~3M real faces, masked autoencoding).

**(b) our LoRA preservation student** (`logs/fpad/studentA_preserve_seed42/epoch_009.pth`),
promoted on **rescue-minus-harm margin +0.141, positive on 5 of 8**, with near-peer AUROC
(0.8667 against the CLIP anchor's 0.8893). Promoted on the margin, not the mean.

**(c) FSFM's released fine-tuned ViT-L** — `FT_on_FF++_c23_32frames`, fine-tuned from the same
`checkpoint-599` base on `DS_FF++_all_cls/c23`, which is our training corpus, so no OOD set was
seen. Probed standalone through **our** pipeline before being wired:

| dataset | FT (ours) | FT (theirs) | P0-DS | Δ vs P0-DS | our student |
|---|---:|---:|---:|---:|---:|
| Celeb-DF-v2 | 0.9199 | 0.9517 | **0.9646** | −0.0447 | 0.8713 |
| Celeb-DF-v3 | 0.8750 | — | 0.8409 | **+0.0341** | — |
| DFD | 0.8651 | 0.9717 | **0.9421** | **−0.0770** | 0.8877 |
| DFDC | 0.8645 | 0.8775 | **0.8828** | −0.0183 | 0.8552 |
| DFDCP | **0.9051** | 0.9335 | 0.8573 | **+0.0478** | 0.8397 |
| Deepfake-Eval-2024 | 0.6861 | — | 0.6922 | −0.0061 | 0.6821 |
| VALmix | 0.8758 | — | 0.8852 | −0.0094 | — |
| **mean** | **0.8559** | | 0.8664 | −0.0105 | | 

Our **frame** AUROC on Celeb-DF-v2 is 0.8613 against their reported 0.8764 — agreement to 0.015,
which is what validates the port. **DFD does not reproduce**: ours 0.8651 against their 0.9717,
a 0.107 gap. Protocol difference (frame sampling and video aggregation), not diagnosed.

Two implementation points that fail silently if wrong and were therefore handled explicitly:
the encoder needs **its own normalisation** (mean 0.548/0.423/0.365), so CLIP's is undone and
FS-VFM's applied; and the released checkpoint carries **no class-name mapping** — the fake class
is **index 0**, determined empirically (index 1 gives AUROC 0.139, cleanly inverted). Their
classification head is not reused: a trainable `EvidentialHead` reads the 1024-d representation
so Branch 3 emits Dirichlet evidence like the other two views.

---

## 4. Run 1, question 1 — does per-branch supervision keep the experts alive?

**Yes, and it is the largest effect measured.** Per-branch vacuity by epoch:

| epoch | armA semantic | armA artifact | armA fsvfm | armB semantic | armB artifact | armB fsvfm |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.756 | 0.618 | 0.092 | 0.330 | 0.384 | 0.097 |
| 2 | 0.983 | 0.952 | 0.015 | 0.148 | 0.142 | 0.017 |
| 9 | 0.9998 | 0.9991 | 0.024 | 0.071 | 0.074 | 0.034 |
| 19 | **0.9997** | **0.9966** | 0.029 | **0.049** | **0.051** | 0.043 |

Under fused-only supervision the **two CLIP incumbents collapsed and the newcomer took over** —
the opposite of Stage 5, where the third view died. The pathology is therefore **winner-take-all,
not "weak views die"**, and which view survives tracks **initial evidence scale**: FS-VFM entered
at u = 0.09 against 0.76 and 0.62, and by epoch 2 the incumbents were past 0.95.

This must not be stated as "DS cannot support more than two views." It is a **multi-view
optimisation pathology in this DS-plus-fused-only setup**, with the alternatives in §2 open.

Standalone branch AUROC shows the cost:

| arm | semantic | artifact | fsvfm |
|---|---:|---:|---:|
| armA fused-only | 0.5539 | **0.3906** | 0.8301 |
| **armB aux-EDL** | **0.8657** | **0.8686** | 0.8314 |
| 1c | 0.8421 | 0.8509 | 0.8539 |
| 1d simple-CE | 0.7975 | 0.7983 | 0.8496 |

armA's artifact view at **0.3906 is below chance**, because a vacuous head's probability ordering
carries no information. **armA → armB is +0.0428 mean AUROC.**

### The simple-CE baseline

1d strips the evidential apparatus entirely — `LinearHead` logits, mean-of-logits fusion, plain
cross-entropy — with an identical head trunk so the comparison isolates the parameterisation, not
capacity. It lands at **+0.0056**: it partially reproduces the effect but lets both CLIP views
decay to **0.798** (against armB's 0.866/0.869) and has the **worst real-side FPR of any arm,
0.274, worse than P0-DS's 0.247**.

So per-branch supervision is the load-bearing change, and the evidential parameterisation adds a
further **+0.0040** on top — inside the noise band on its own.

---

## 5. Run 1, question 2 — which simple fusion?

Means over six OOD sets. F2 is a global non-negative softmax weighting fitted on **VALmix only**
and applied unchanged; no OOD export is opened during fitting.

| arm | F1 averaging | F2 learned | F3 DS | F2 weights (sem / art / fsvfm) | verdict |
|---|---:|---:|---:|---|---|
| armA | 0.8298 | 0.8301 | 0.8301 | 0.218 / 0.000 / 0.782 | no difference |
| **armB** | 0.8778 | 0.8774 | 0.8768 | 0.000 / 0.809 / 0.191 | **no second defect** — spread 0.0010 |
| 1c | 0.8708 | **0.8778** | 0.8701 | 0.000 / 0.536 / 0.464 | **F2 +0.0077**, clears the band |
| 1d | 0.8648 | 0.8702 | 0.8639 | 0.116 / 0.255 / 0.629 | F2 +0.0063 |

**For armB the operator choice is noise.** DS is not the bottleneck there, and a 0.0005 gap says
nothing. *(An earlier version of `fusion_replay.py` auto-reported that gap as "a second problem
isolated"; the threshold is now explicit at ~0.005 and all four reports were regenerated.)*

F2 zeroes the semantic weight on armB, 1c and armA — unsurprising, since semantic and artifact
both derive from the same CLIP trunk and the artifact view dominates it.

---

## 6. The finding worth the paper: complementarity does not convert

FS-VFM against the incumbent **pair** (rescue = P(FS-VFM right | pair wrong), thresholds at each
view's own FF++ val EER):

| arm | mean margin | positive on | framework AUROC |
|---|---:|---:|---:|
| armB (student) | +0.120 | 6/7 | **0.8729** |
| **1c (FSFM FT)** | **+0.335** | **7/7** | 0.8620 |

1c per dataset: CDFv2 rescue 0.712 / harm 0.122 / margin **+0.590** / overlap 0.288; DFDCP
+0.408; DFDC +0.394; VALmix +0.349; DFD +0.277; CDFv3 +0.250; DFEval24 +0.077.

**The more complementary third view produces the worse framework under DS.** On Celeb-DF-v2 1c's
Branch 3 rescues **71%** of the pair's errors at 12% harm, and fused CDFv2 still lands below
P0-DS. F2 recovers most of that gap (0.9660 against DS's 0.9582).

So the complementarity is real and **DS is failing to convert it**. That is the measured
motivation for a conflict-aware operator — on the arm with the most to convert — rather than an
assumed one.

**Corollary: standalone strength did not predict framework contribution.** The FSFM checkpoint
beat P0-DS standalone on DFDCP (+0.048) and CDFv3 (+0.034), yet in-framework 1c sits below armB,
whose Branch 3 is the weaker student.

---

## 7. Related correction — the CDFv3 discrepancy

The pilot table's P0-DS Celeb-DF-v3 cell (0.9516) was scored on the **FaceSwap family alone**:
re-scoring that checkpoint on FaceSwap only gives **0.9515**, against **0.8409** on the full
5,418-video set. P1d's cell used the full set, which is why it reproduced while P0-DS appeared
0.11 off. **The row mixed bases.**

Correcting it shifts every pilot by `0.1107 / 7 = +0.0158`, so **P2a (+0.0066) and P1d (+0.0031)
both beat P0-DS** and P3a/P1a tie — "nothing beat P0-DS" was an artifact of one mis-scoped cell.
Independent cross-check: the correction predicts P1d at +0.0031; direct re-measurement gave
+0.0033.

Underneath: **P0-DS is best on FaceSwap (0.9515) and worst on FaceReenact (0.7874) and
TalkingFace (0.7849)** — strong on exactly the family it was trained for. Only P0-DS and P1d
could be verified; the other six pilots' cells are unverified.

---

## 8. What is not established

1. **Arm ordering needs a confirming seed.** armB's +0.0096 against P0-DS clears the ±0.01 band;
   every arm-vs-arm difference (armB > 1d > 1c, margins 0.004–0.011) does not. Under §22 the
   ordering is an observation, not a claim. One seed-43 run of armB (~5 h) settles the headline.
2. **Deepfake-Eval-2024 is the one set where every arm loses to P0-DS** (best 0.6863 against
   0.6922). Nothing here helps on modern in-the-wild data.
3. **DFD does not reproduce FSFM's reported standalone number** (0.8651 vs 0.9717).
4. **Checkpoints for armB, 1c and 1d were taken at best FF++ val AUROC**, not by the VALmix
   macro-AUROC protocol used earlier. armB's candidates were within 0.0001, so it cannot move the
   headline, but it is a deviation.
5. **P2a's third-view vacuity is inference**, not measurement.
6. **The frozen-FS-VFM probe (Experiment 1 candidate (a)) was never run** — superseded by the
   fine-tuned checkpoint, and skipped deliberately.
7. **No configuration has been frozen.** armB clears the bar; nothing is yet pinned as canonical.

---

## 9. Reproduction

| what | where |
|---|---|
| training configs | `DISCERN_Ext/eval_adaptation/configs/run1_{fusedonly,auxedl}.yaml`, `run1c_ft.yaml`, `run1d_simplece.yaml` |
| Branch 3 | `DISCERN_Ext/src/modules/fsvfm_branch.py`, `fsvfm_encoder.py`, `fsvfm/` |
| process branch (parked) | `DISCERN_Ext/src/modules/process_branch.py`, `process_residual.py` |
| N-view DS fusion | `DISCERN_Ext/src/model/core_model.py` — `DS_Combin`, associative left fold |
| aux EDL + vacuity logging | `DISCERN_Ext/src/model/discern_ext_module.py` |
| per-view exporter | `DISCERN_Ext/eval_adaptation/export_views.py` |
| standalone FT probe | `DISCERN_Ext/tools/probe_fsvfm_ft.py` |
| input-or-head diagnostic | `DISCERN_Ext/tools/probe_process_stats.py` |
| transfer bar | `analysis/tbiom/probe_branch3_candidates.py` |
| branch contribution replay | `analysis/tbiom/branch3_contribution.py` |
| fusion replay F1/F2/F3 | `analysis/tbiom/fusion_replay.py` |
| Experiment 1 gate | `analysis/tbiom/exp1_fsvfm_gate.py` |
| video-identity rule | `analysis/tbiom/video_id.py` (`--check`: 49 cases, 12 datasets) |
| exports | `logs/tbiom/stage23/{run1fusedonly,run1auxedl,run1cft,run1dce}_*.csv`, `logs/tbiom/ftprobe/` |

Environments: training and export for the FS-VFM arms run in **`discern_fsvfm`**, a clone of
`discern_ext` plus `timm --no-deps`. **`discern_ext` is unmodified** — its peft 0.14.0 and
diffusers 0.32.2 pins are load-bearing for every P0/P1 checkpoint. Both envs carry the same peft,
so the LoRA maths is identical rather than merely similar.

The 3.4 GB FSFM checkpoint (`weights/FS-VFM-FT/`) is gitignored — fetched from
`Wolowolo/fsfm-3c`, not source.
