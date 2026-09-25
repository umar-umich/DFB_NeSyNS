# Did the identity overlap change checkpoint selection?

## 1. Branch training is clean, and that is the part that would force a retrain

| level | FF++ c23 train x the ten test sets | FF++ c23 train x DF40 evaluation |
|---|---|---|
| video | **0** | 0 |
| source video | **0** | 720 |
| identity | **0** | 0 |

All three branches are trained on FF++ c23 train only. It shares no video, no source video and no identity with any of the ten test sets, so **no branch was trained on test material and no branch needs retraining on that account.** The single training-side overlap is 720 FF++ source videos reappearing inside DF40's FF++-sourced generators, which is a property of DF40 and affects DF40 numbers only.

## 2. Checkpoint selection is the step that saw related identities

All three branches were selected on the v3.1 score over VALmix + VALmix-v2 + FF++ val, and that pool shares Celeb-DF identities with CDFv2, CDFv3, DFDM and DF40. The test below re-runs the selection on the clean subset only.

### Spatial branch (Appearance and Off-manifold)

| checkpoint | v3.1 on the full pool | v3.1 on the clean subset | videos full / clean |
|---|---:|---:|---|
| e13 | 0.8188 | 0.7646 | 5814 / 2823 |
| e17 | 0.8188 | 0.7646 | 5814 / 2823 |
| e19 | 0.8188 | 0.7646 | 5814 / 2823 |

The three rows are identical, and that is not a bug. Because the deployed recipe sets `freeze_views: true`, the
Appearance and Off-manifold heads are frozen at the DiCoME warm start and do not move between epochs. Verified
directly on the cached exports: across epochs 13, 17 and 19 the columns `p_semantic`, `u_semantic`, `p_artifact` and
`u_artifact` are **bit-identical** (max |difference| exactly 0), while only `p_fsvfm_raw` (max 1.88e-01) and
`gate_fsvfm` (max 1.01e-01) change -- and the face-prior branch those belong to is **excluded from the fusion**.

**So checkpoint selection cannot have affected the two spatial views at all.** The question is not whether a clean
selection would pick the same epoch; it is that for the views this system actually uses, every retained epoch is the
same function. Selection contamination is void for the spatial branch.

Winner on the full pool: **e13**. Winner on the clean subset: **e13**. **Unchanged** — the deployed checkpoint is the one a leakage-free selection would also have chosen, so there is nothing to re-select.

### Temporal branch (V-JEPA 2)

**FLAG — cannot be re-selected from retained artefacts.** Only the selected epoch's weights (`best.pt`) and its exported opinions were kept; the per-epoch checkpoints were not. Re-selecting on a clean subset would require re-exporting per-epoch opinions, which needs the discarded checkpoints.

What the retained record does show is the shape of the selection curve, from `train_record.json`:

| epoch | dev v3.1 | swap | reenactment | in-the-wild |
|---|---:|---:|---:|---:|
| 0 | 0.6788 | 0.6783 | 0.7333 | 0.6250 |
| 1 | 0.7345 | 0.7265 | 0.8081 | 0.6690 |
| 2 | 0.7915 | 0.7949 | 0.8683 | 0.7114 |
| 3 | 0.7946 | 0.7925 | 0.8864 | 0.7049 |
| 4 | 0.8088 | 0.8064 | 0.8984 | 0.7217 |
| 5 | 0.8019 | 0.7895 | 0.9093 | 0.7070 |
| 6 **<- selected** | 0.8092 | 0.7996 | 0.9113 | 0.7166 |
| 7 | 0.8072 | 0.7998 | 0.9077 | 0.7142 |
| 8 | 0.8055 | 0.7983 | 0.9035 | 0.7148 |
| 9 | 0.8063 | 0.7993 | 0.9039 | 0.7157 |

The selected epoch 6 leads the runner-up by 0.0004 on the full pool. Task 8 showed that removing the contaminated half of that pool moves a *fused* score by at most 0.0020; a margin of 0.0004 is smaller than that, so the selected epoch cannot be assumed to survive a clean re-selection.

## 3. Verdict on retraining

| question | answer |
|---|---|
| Was any branch trained on test material? | **No** — zero overlap at all three levels |
| Do the branches need retraining? | **No**, on this evidence |
| Was the spatial checkpoint choice affected? | **It cannot be.** The Appearance and Off-manifold views are frozen and bit-identical across every retained epoch; only the excluded face-prior branch varies |
| Was the temporal checkpoint choice affected? | **Unresolved, and the margin is too thin to wave away.** Epoch 6 leads the runner-up by only 0.0004 on the development v3.1 score, while Task 8 showed that decontaminating the pool moves a fused score by up to 0.0020. A clean re-selection could therefore pick a different epoch. The per-epoch checkpoints were not retained, so this cannot be settled without re-training the branch |
| Is joint training indicated by the leakage? | **No.** Joint training would change the architecture, not the exposure. The exposure is in selection and weight fitting, and Task 8 already shows refitting those on clean data moves results by at most 0.0020 |

The one methodological change the audit does support is unrelated to leakage: selection and trust-weight fitting share the same pool, which makes development numbers optimistic as held-out estimates. Separating a selection pool from a weight-fitting pool would fix that, and it costs no retraining.

