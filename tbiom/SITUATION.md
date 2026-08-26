# Where DISCERN v2 / T-BIOM actually stands — 2026-08-26

Written to be handed to someone with no session context. Every number here was measured in this
repo; nothing is carried over from a paper or a previous table.

---

## 1. The short version

Three things are true at once, and they pull in different directions.

1. **The multi-expert architecture has no empirical support.** The membership gate was run twice,
   on two independent out-of-domain bases, and dropped every candidate expert both times.
2. **Our CLIP anchor is a weaker detector than the DiCoME checkpoint it was ported from** — by
   3–5 AUROC points cross-dataset. We have been measuring complementarity against a handicapped
   baseline.
3. **The FF++ ⊕ DF40 training run collapsed** in a way that is fully diagnosed and is a property
   of the training corpus, not of the code. It has been stopped.

The open question is no longer "which experts join the anchor". It is **what the anchor should
be**, and whether the paper's contribution is an architecture or a reliability result.

---

## 2. What is trained today, and is it the framework or an ablation?

**It is the complete framework, and that is now inconsistent with our own findings.**

The V1 model (`training/networks/discern_v2/discern_v1_model.py`) is:

| branch | what it is | trainable |
|---|---|---|
| `semantic` (the anchor) | CLIP ViT-L/14 + LoRA (q_proj, v_proj, r=8) → 64-d bottleneck → EDL head | ~856 K |
| `reference` | FROZEN FS-VFM ViT-L/16 + reals-only autoencoder + evidence head | head only |
| `process` | FROZEN SDXL-VAE + 6 named statistics + evidence head | 290 params |
| `direct_probe` | a control on the reference branch, logged, never fused | small |

On top sit DS/CCF fusion, cross-fitted applicability gates, and a logistic risk model driving
Real/Fake/Defer.

The stopped run trained `loss_sem`, `loss_ref`, `loss_proc`, `loss_direct` — all four heads,
simultaneously, with `fused_loss_weight: 0.0` so the reasoning layer is applied but not trained.

**The inconsistency:** Stage 1 tested the v3 replacements for exactly the `reference` and
`process` specialists (the FS-VFM students, and the MR-VAE rate branch) and dropped all of them.
Yet the config we trained still has `reference.enabled: true` and `process.enabled: true`. We are
training a three-branch model whose second and third branches our own gate says should not exist.

That is not a bug — the run predates nothing, both decisions were made in the same week — but it
does mean **the current training arm is not the architecture Stage 1 implies.**

---

## 3. Stage 1: the membership gate, run twice, same answer

The gate asks three things of a candidate expert, against a freshly-scored CLIP anchor:
`P(expert right | anchor wrong)` vs `P(expert wrong | anchor right)`; the honest ceiling
`1 − P(both wrong)`; and how much of the ceiling-minus-anchor headroom a **label-free** cross-fit
gate can actually recover. The bar is 25% recovery.

### On DF40-Dev (36 generator methods, 14,631 videos, 75% intersection)

| expert | rescue | harm | margin | recovered | video AUROC |
|---|---:|---:|---:|---:|---|
| `fsvfm_preserve` | 0.275 | 0.190 | +0.085 | 6% | 0.8501 → 0.8597 |
| `fsvfm_ordinary` | 0.254 | 0.196 | +0.058 | 6% | 0.8501 → 0.8589 |
| `mrvae_rate` | 0.734 | 0.439 | +0.295 | 1% | 0.8501 → 0.8264 |

### On VALmix (3 real-world domains, 1,350 videos, 100% intersection)

| expert | rescue | harm | margin | recovered | gate AUROC | video AUROC |
|---|---:|---:|---:|---:|---:|---|
| `fsvfm_preserve` | 0.360 | 0.158 | +0.202 | 7% | 0.647 | 0.8719 → 0.8753 |
| `fsvfm_ordinary` | 0.367 | 0.161 | +0.206 | 10% | 0.667 | 0.8719 → 0.8773 |
| `mrvae_rate` | 0.395 | 0.526 | **−0.131** | −2% | 0.285 | 0.8719 → 0.8564 |

**The important detail.** On VALmix the *rescue* half got markedly stronger — margins nearly
quadrupled, and all three domains clear the bar individually (CDFv2val +0.248, DFDCPval +0.385,
DFEval24val +0.119). The *realizable* half did not, and the gate's own AUROC against its target
fell (0.647/0.667 vs 0.739/0.722 on DF40).

So this is **not** "DF40 was a hostile probe". The complementarity is real, and larger on
real-world data than on the generator zoo. What fails is the ability of a label-free gate to
locate it. That is precisely the distinction the gate exists to draw, and it now holds across 36
generator methods and three real-world domains.

`mrvae_rate` fails outright rather than marginally: negative rescue margin, harmful in all three
domains, gate AUROC 0.285 (below chance), and fusion costs 0.0154 AUROC. It is dropped on direct
evidence, not merely inferred from the earlier operator-inertness finding.

---

## 4. The anchor is weaker than the DiCoME checkpoint it was ported from

This is the finding that most changes what to do next.

| dataset | **DiCoME released ckpt** | **our CLIP anchor** | delta |
|---|---:|---:|---:|
| FF++ (in-domain) | 0.9905 | 0.9909 | +0.0004 |
| Celeb-DF-v2 | **0.9729** | 0.9225 | **−0.0504** |
| DFDC | **0.8822** | 0.8477 | **−0.0345** |
| DFD | **0.9392** | 0.9222 | −0.0170 |
| DFDCP | 0.8799 | 0.8912 | +0.0113 |

DiCoME's column is our own reproduction of their released `dicome-best.ckpt` on our pipeline
(`/data/umar/Repos/DiCoME/eval_adaptation/RESULTS.md`), so preprocessing is not the confound —
it reproduces their paper to −0.0041 on CDFv2 and +0.0002 on DFDC.

Two consequences:

- **Every complementarity number in §3 was measured against a handicapped anchor.** A stronger
  anchor makes rescue *harder* to demonstrate, so a stronger anchor would not rescue the experts —
  it would bury them further. The Stage-1 verdict is therefore robust to this, and arguably
  understated.
- **Our headline detector is currently below the SOTA baseline we are supposed to beat.** No
  amount of reliability machinery fixes a comparison that starts 5 points down on Celeb-DF-v2.

Note also our anchor is *itself* stronger than both FS-VFM students on 6 of 8 benchmarks, which
is consistent with nothing clearing the gate.

### Full cross-dataset table (all FF++-trained, video AUROC)

| model | FF++ | CDFv2 | CDFv1 | DFDCP | DFDC | DFD | DFEval24 | UADFV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP anchor (ours) | 0.9909 | 0.9225 | 0.8939 | 0.8912 | 0.8477 | 0.9222 | 0.6357 | 0.9958 |
| FS-VFM preservation | 0.9867 | 0.8713 | 0.8285 | 0.8397 | 0.8552 | 0.8877 | 0.6821 | 0.9825 |
| FS-VFM ordinary | 0.9864 | 0.8758 | 0.8646 | 0.8259 | 0.8511 | 0.8884 | 0.6782 | 0.9858 |

Deepfake-Eval-2024 at 0.64 is the honest number for in-the-wild media and is where every model
here is weakest.

---

## 5. The FF++ ⊕ DF40 run: what happened and why

Stopped at epoch 3 of 10 (9 h of ~26 h). Checkpoints 0–2 kept.

### Training signal

| epoch | train loss | val loss | FF++ val video AUROC |
|---:|---:|---:|---:|
| 0 | 1.786 | 2.285 | 0.9928 |
| 1 | 1.729 | 2.378 | 0.9848 |
| 2 | 1.712 | 2.583 | 0.9829 |

Train loss falling, validation loss rising monotonically from epoch 0. Pairing worked perfectly
(`pair_fraction 1.0`, `matched_aug_fraction 1.0`).

### Out-of-domain result at epoch 2 (all eight benchmarks)

| dataset | FF++-only anchor | FF++ ⊕ DF40 ep2 | delta |
|---|---:|---:|---:|
| FF++ (in-domain) | 0.9909 | 0.9926 | +0.0017 |
| Celeb-DF-v2 | 0.9225 | 0.7503 | **−0.1722** |
| Celeb-DF-v1 | 0.8939 | 0.8306 | −0.0633 |
| DFDCP | 0.8912 | 0.7247 | **−0.1665** |
| DFDC | 0.8477 | 0.7628 | −0.0849 |
| DFD | 0.9222 | 0.8414 | −0.0808 |
| Deepfake-Eval-2024 | 0.6357 | 0.6476 | +0.0119 |
| UADFV | 0.9958 | 0.9354 | −0.0604 |

**Seven of eight cross-dataset benchmarks down; only in-domain FF++ up.** The single exception,
Deepfake-Eval-2024 at +0.0119, is the benchmark where every model here is near chance anyway.

AUROC understates it. The score distributions show the actual failure:

| dataset | anchor separation | ep2 separation | ep2 mean p(fake) on REAL |
|---|---:|---:|---:|
| FF++ | +0.864 | +0.769 | 0.205 |
| Celeb-DF-v2 | +0.531 | **+0.000** | **0.997** |
| Celeb-DF-v1 | +0.523 | **+0.000** | **0.997** |
| DFDCP | +0.491 | +0.106 | 0.879 |
| DFDC | +0.459 | +0.096 | 0.892 |
| DFD | +0.570 | +0.334 | 0.575 |
| Deepfake-Eval-2024 | +0.140 | +0.070 | 0.854 |
| UADFV | +0.899 | +0.091 | 0.906 |

On both Celeb-DF sets the model emits **0.997 for real and fake alike**. It calls every
unfamiliar video fake with 99.7% confidence. The residual AUROC is rank noise inside a saturated
region, not detection. It is not a broken forward pass — FF++ still separates cleanly.

The pattern across all eight is one quantity: **mean p(fake) on REAL videos**. It is 0.205 on
FF++, whose reals the model trained on, and 0.575–0.997 everywhere else. UADFV is the sharpest
illustration — separation collapses from +0.899 to +0.091 purely because its reals become
unrecognisable to the model.

### Diagnosis

The training manifest is **720 real videos against 23,668 fakes (1:33), and every fake is derived
from those same 720 reals.** The FF++-only arm was 1:4.

The paired sampler keeps each *batch* balanced by cycling reals, so class imbalance is not the
issue. **Real diversity is.** The model sees the same 720 real capture conditions ~33 times per
epoch and needs only to learn "these are real, everything else is fake". That rule is perfect in
training and transfers to zero new-real domains.

This is the measured form of the diverse-real-corpus argument.

### Why our manifest has this property and DF40's protocol does not

DF40's own training protocol includes the `_cdf` method arms, which bring **Celeb-DF reals** into
training. We deliberately excluded them (`*_ff.json` only) so Celeb-DF would stay zero-shot. That
choice is exactly what starved the real side.

**The tension to resolve:**

- Add Celeb-DF reals → collapse fixed, but Celeb-DF-v1/v2 stop being zero-shot, and they are two
  of our headline benchmarks (and CDFv2 is where DiCoME is strongest).
- Keep FF++ reals only → Celeb-DF stays clean, but 720 reals against 32 fake method families.

Two ways to keep both:

1. **A real-only corpus we never test on.** `/data/umar/Datasets/ffhq256_subset` holds 8,750 real
   face images, in no test set and no VALmix domain — ~12× the current real diversity at zero
   contamination cost. Caveat: stills, not video.
2. **Cap fakes per real identity.** Sample ~4 DF40 methods per real per epoch, rotating which, so
   all 32 methods are seen across the run at an FF++-like exposure ratio. No new data, no
   contamination; costs per-epoch method coverage.

Neither has been implemented. SBI is now available as a third lever (see §6) and would add
synthetic fakes derived from *whatever* reals we supply, which interacts with both.

---

## 6. Infrastructure that is built, verified, and idle

- **SBI is unblocked.** It was unrunnable here: `sbi_api` needs dlib's 81 points and our
  preprocessing stored 5. Extracted over FF++ youtube c23 at 99.83% coverage (31,945/32,001
  frames), on the existing crops so pixels match training. Verified 6/6 seeds produce a blend
  changing 15–28% of pixels. Wired to a config key with an assert on point count.
- **VALmix is a real manifest.** 1,350 videos / 40,142 frames over Celeb-DF-v2, DFDCP and
  Deepfake-Eval-2024, balanced 450/450/450, zero overlap with all three test lists (re-verified on
  the final manifest). It previously existed only as a 5 GB HDF5 in another repo that nothing here
  could read. Adopting it makes those three corpora **domain-seen**; CDFv1, CDFv3, DFD, DFDC and
  UADFV stay strictly zero-shot.
- **DF40 training data resolves** from a colleague's world-readable tree — 31 methods, 74 GB, no
  download, no disk cost (`/data` is at 100% with ~129 GB free).
- **`DF40_all.json` is unusable as a training split** and we do not use it: it leaks FF++ *test*
  identities into its own train split (4,031 e4e images, 258 simswap/inswap videos) and its e4e
  paths do not resolve at all. Our manifest is built from per-method files and re-verifies zero
  test identities on its final contents.
- **`stage_de`'s `diverse` protocol is renamed** to `ffpp_cdf2_TESTCONTAMINATED` and gated behind
  an explicit flag — it calibrated gates, risk model and thresholds on Celeb-DF-v2's val split,
  which IS its test split (518/518).
- **Dataset hygiene worth knowing:** for DFDCP, Celeb-DF-v1/v2 and UADFV the shipped `val` split
  IS the shipped `test` split; DFDC ships 2 val videos against 4,704 test; Celeb-DF-v1 has 92 of
  its 100 test videos in its own train list. FF++ is clean (0 overlap in all three directions).

---

## 7. The decisions on the table

### A. What should the anchor be?

The proposal on the table is to **use the DiCoME detector as-is as the anchor**, get good results,
then swap in a comparable anchor later. Against the evidence:

**For.** It is 5 points better on CDFv2 and 3.5 on DFDC than our port. It removes "your baseline is
weak" as a reviewer objection. It makes any reliability contribution stand on its own rather than
being confounded with a re-implementation gap.

**Against / to check.** DiCoME is itself a *multi-view evidential* detector — it already does DS
fusion of two views inside the CLIP manifold. Using it whole as our "anchor branch" means our
architecture wraps a fusion model in another fusion layer, and the paper has to say clearly what
the outer layer adds. It also means we cannot ablate the anchor's internals. And a frozen released
checkpoint cannot be retrained on FF++ ⊕ DF40, so the combined-corpus arm would have to be a
separate story.

**The honest framing:** our port and DiCoME differ, and we have not diagnosed why. Before adopting
either, it is worth knowing whether the 5-point CDFv2 gap is a training-recipe gap we could close
or an architecture gap we cannot.

### B. Is the paper an architecture or a reliability result?

Stage 1's answer, twice over, is that there is no second expert worth having. The brief's own
fallback says that collapses the contribution to **selective prediction and abstention on a single
strong anchor**, with risk-coverage curves rather than a rescue table — and explicitly notes this
loses the V/C/U_sup decomposition, which needs at least one complementary specialist to exist.

Worth weighing against §3's finding that rescue *is* present and larger on real-world data. The
gap is between oracle complementarity and gate-realizable complementarity. A different research
question — "why can't a label-free gate find complementarity that demonstrably exists?" — is a
real question, and we have unusually clean evidence for it.

### C. What to do with the combined corpus

Blocked on §5's tension. Rebuild is understood but not implemented, and the choice between adding
FFHQ reals, capping fakes per real, using SBI, or admitting Celeb-DF reals changes what stays
zero-shot.

---

## 8. What is running right now

- **Training: stopped.** Checkpoints 0–2 kept in `logs/tbiom/v1_ffpp_df40_seed42/`.
- **Epoch-2 OOD evaluation: 4 of 8 done** (FF++, CDFv2, CDFv1, DFDCP — all reported above). DFDC
  and DFD are mid-run; Deepfake-Eval-2024 and UADFV are queued. They will not change the
  conclusion.
- Everything else is idle. GPU 3 is otherwise free.

Reports: `tbiom/STAGE1_MEMBERSHIP.md` (DF40), `tbiom/valmix/STAGE1_MEMBERSHIP.md`,
`tbiom/CROSSDATASET.md`, `tbiom/VALMIX.md`, `tbiom/FFPP_DF40_MANIFEST.md`,
`tbiom/MRVAE_OPERATOR_FINDING.md`, `tbiom/RESUME_MAP.md`.
