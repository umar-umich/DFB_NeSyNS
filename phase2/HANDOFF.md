# Phase-2 (v2 brief) — code phase COMPLETE, 2026-08-20

Spec: `docs/DiCoME_eval/discern-v2-phase2-v2.md`. Branch `discern-v2-phase2`.
All nine stages are implemented and tested. **Nothing has been trained or evaluated.** Every number
that appears in a smoke-test log so far is from subsampled synthetic or partial data and is NOT a
result.

**Nothing is committed.** All work is in the working tree.

Execution plan: **`phase2/RUN_COMMANDS.md`** — ordered, 🔴 Umar's to launch, method lists generated
from the sealed split so they cannot drift from it.

---

## Two decisions still needed (🟡 ASK-UMAR)

1. **Diverse-real corpus (blocks Stage 1).** The only real-face corpus on this machine that is not
   itself an evaluation source is `/data/umar/Datasets/ffhq256_subset` (8,750 images, 927 MB). No
   VGGFace2, no CelebA-HQ, no LAION-face; DF40's `real/` is FF++ and Celeb-DF-v2, i.e. corpora we
   test on. Options: FF++ reals ∪ re-cropped FFHQ (`--balance`), or fetch ~40 GB.
2. **Calibration protocol.** This brief's ground rule 2 says FF++-only. The shipped V1 Stage-D/E
   artifact used `diverse` (FF++ ∪ Celeb-DF-v2) on your explicit instruction. Phase-2 defaults to
   `ffpp` per the brief; `diverse` stays selectable and every artifact records which was used, with
   the lost zero-shot status computed rather than described.

---

## Stage by stage

| stage | status | code | result file it writes |
|---|---|---|---|
| −1 discovery | ✅ done | — | `phase2/REPO_MAP.md` |
| 0.1 DF40 split | ✅ **sealed** | `configs/discern_v2/df40_split.json` | `phase2/DF40_SPLIT.md` |
| 0.2 DF40-Dev eval | coded | `training/eval_v1.py --df40` | — |
| 0.3/0.4 complementarity | coded | `analysis/discern_v2/phase2/complementarity.py` | `STAGE0_DF40DEV_{protocol,insample}.md` |
| 1.1 corpus + overlap | coded | `prepare_diverse_reals.py`, `identity_overlap.py` | `phase2/stage1/` |
| 1.2 refit `P_R` | coded | `cache_image_features.py`, `fit_diverse_reference.py` | `configs/discern_v2/reference_diverse/` |
| 1.3 domain audit | reuses V1 | `analysis/discern_v2/domain_audit.py` | already audits `p_ref`/`u_ref` |
| 2.1 MR-VAE branch | coded | `fit_rate_operator.py`, `networks/discern_v2/rate_branch.py` | `configs/discern_v2/rate/` |
| 2.3 rate structure | coded | `rate_response_analysis.py` | `STAGE2_MRVAE_rate_response.md` |
| 3 HARD GATE | coded | `stage3_gate.py` | `phase2/STAGE3_GATE.md` |
| 4 paired training | coded | `train_v1.py --sampling`, `dataset/paired_sampler.py` | `metrics.jsonl` + `SELECTION.md` |
| 5/6/7 | coded | `training/stage567.py`, `networks/discern_v2/ccf_fusion.py` | `phase2 .../STAGE567.md` |
| 8 final table | coded | `stage8_final.py` | `phase2/STAGE8_FINAL.md` |

## Tests — 15 files, all passing

```
ds_fusion 20   ccf_fusion 11   rate_branch 10   applicability_gate 13   risk_model 13
process_branch 7   reference 13   semantic_branch 9   integration 12   applicability 21
discern_v2 23   fsvfm 11   paired_sampler 12   stage567 10   v1_model 12 (+1 skip: real weights)
```

Four of these validate things that would otherwise fail silently:

* **`test_ccf_fusion`** reproduces **Table I of arXiv:1805.01388** exactly — b(x) 0.629,
  b(x̄) 0.182, u 0.189, P(x) 0.723. The binary specialisation of equation (7) was derived here, so
  it is checked against the paper's published numbers rather than against my own reasoning.
  Measured pairwise-vs-multi-source gap on those inputs: max |Δb| = 0.0063.
* **`test_stage567`** checks Shapley *efficiency* (the parts sum to the coalition gain), reduction
  to the brief's literal two-specialist formula, and — the one that matters — that the Shapley
  share is provably **smaller** than V1's pairwise gain when specialists are redundant. That is
  exactly why the brief replaced the target.
* **`test_paired_sampler`** checks that pair members receive the *identical* augmentation draw and
  non-partners do not, and that pairing does not shrink the training set.
* **`test_rate_branch`** checks the operator stays frozen under `model.train()`, that `hidden_dim`
  survives the artifact round trip (the reference branch shipped with that bug), and that the head
  can learn **either** polarity — the brief forbids hardcoding "larger residual means fake".

---

## Findings worth carrying into the writing

**DF40 path resolution: 60/81 → 77/81.** `heygen` — the brief's designated harm probe — resolved at
exactly **0.0**, as did both `e4e` arms, `styleclip` and `whichisreal`. None of it was missing data.
Four causes, each confirmed by listing the directory: a leading `./` and an absolute cluster path
prefix; renamed and double-nested method directories; `cdf/Celeb-real` → `cdf/Fake_from_Celeb-real`
(with `YouTube` → `Youtube`); and the pre-DF40 `cdfv2` layout. `pixart` genuinely is not staged.
All nine Dev arms and `heygen` now resolve at 1.000.

**DF40 cannot supply a strictly zero-shot Holdout.** Phase 1 produced a per-generator AUROC table
over 70 generators. `phase2/DF40_SPLIT.md` carries the exact wording that survives scrutiny —
"sealed before Phase 2 and never read by any Phase-2 decision", not "never seen" — plus the one
contamination that bears on Stage 2: **P1d's privileged status was itself selected on a DF40
number** (`_base.yaml`), so a DF40-Dev result confirming it is a re-test, not independent evidence.

**The MR-VAE cannot be attached as the brief describes.** It is a feature-space operator; both
existing checkpoints live in feature spaces V1 does not have, and V1's Branch-A space does not exist
until Stage 4 trains its LoRA. Hosted on the frozen FS-VFM embedding and re-fit instead — recorded
as a deviation in `REPO_MAP.md` item 5, with the two alternatives and why they are worse.

**Stage 7 is a rename, not a recomputation** — correcting the earlier note in this file. V1's
`reliability()` already computes `w_b = q_b(1-u_b)`, `C = Σ w_i w_j JS / (Σ w_i w_j + ε)`, and over
two specialists `1 − mean_b w_b` **is** the brief's `1 − 0.5 Σ_b w_b`. So V1 and Phase-2 reliability
numbers **are** directly comparable; `A` → `U_sup` only removes the collision with aleatoric
uncertainty. `test_stage567` asserts the identity.

**The brief's three fusion arms confound the operator comparison.** Applicability-CCF vs
Applicability-DS differs in both the operator and the gate, because Stage 5 learns `q` under the
downstream operator. Equal-CCF vs Applicability-CCF stays clean, and the brief correctly names it
as the decisive one. A fifth arm, `applicability_ds_shared_q`, fuses with DS over the CCF arm's `q`
so the operator effect is isolated — one extra fusion, no extra training. Both comparisons are
reported and labelled `ccf_vs_ds_confounded` / `ccf_vs_ds_isolated`.

**Zero method-level headroom is a result, not a missing value.** On the V1 export the anchor is
already the best single branch on 5 of 6 sources, which makes the recovered fraction *undefined*
rather than small. Reporting it as `nan` would let a reader mistake "no headroom exists" for "not
computed", so `complementarity.py` says `none to recover` and counts those sources explicitly.
