# Stage 0 — Resume-state map (T-BIOM, post-FPAD)

Inventoried on 2026-08-25 on branch `phase3/discern-v2-tbiom`. Everything below was listed, loaded
or byte-compared; nothing is carried over from the sprint's assumptions.

Env: `/data/umar/miniconda3/envs/dfb_nesy/bin/python`.

---

## Two things block Stage 1 as written, and one candidate is already out

| | status |
|---|---|
| **proc / LDM branch** | **OUT.** The diffusion-specific eval was never run (§6). The brief's rule is explicit — "If it never ran or stayed near chance, proc is out and is not a Stage 1 candidate." |
| **MR-VAE rate expert** | **Cannot be evaluated yet.** The operator was never fit (`configs/discern_v2/rate/` does not exist) and the rate-response analysis was never run. Both are prerequisites before it is even a candidate. |
| **FS-VFM direct expert** | **Ready.** The FPAD students are LoRA-adapted FS-VFM with trainable direct EDL heads — exactly what this brief specifies. Reusable as-is. |

So the Stage 1 candidate set is **FS-VFM direct**, plus **MR-VAE rate only if its two prerequisites
are run first**. That is one certain candidate and one conditional, not three.

---

## 1. Checkpoints on disk

| what | path | state |
|---|---|---|
| CLIP-LoRA semantic anchor | `logs/v1/stage_b_seed42/` | 20 epochs; **§11-selected epoch 7**, VAL_select video AUROC 0.9986 |
| FPAD ordinary student (λ=0) | `logs/fpad/studentA_ordinary_seed42/` | 10 epochs, `epoch_009.pth` |
| FPAD preservation student (λ=1) | `logs/fpad/studentA_preserve_seed42/` | 10 epochs, `epoch_009.pth` |
| frozen SDXL-VAE process stats | `configs/discern_v2/process/process_stats.pt` | fit — but proc is out (above) |
| reals-only reference `P_R` | `configs/discern_v2/reference/reference_C3_ae_cosine.pt` | fit — **archived, framing falsified** |
| MR-VAE rate operator | — | **NOT FIT** |

**Which FPAD student is the FS-VFM candidate is a real choice, 🟡 ASK-UMAR.** The brief says "reuse
the FPAD B3 student", and B3's student is the *preservation* one. But `L_preserve` was a
trajectory-motivated constraint, and this brief wants a *plain discriminative expert* — for which
the ordinary student is the cleaner object. Measured direct-readout detection at epoch 8–9,
1,920 frames per source:

| | FF++ val | Celeb-DF-v2 | DFDCP |
|---|---:|---:|---:|
| preservation, direct | 0.9942 | **0.8346** | **0.8302** |
| ordinary, direct | 0.9907 | 0.8319 | 0.8193 |

Preservation is marginally better OOD (+0.003 / +0.011), both inside or near the 0.01 band. Either
is defensible; preservation has the evidence edge, ordinary has the cleaner story.

## 2. The two queued FPAD diagnostics — **neither landed**

* **Epoch-9 uncapped mechanism validation: NOT RUN.** The only artifact is `wacv/stage2_gate.json`
  at **epoch 1, capped**, verdict BORDERLINE. Both students have `epoch_009.pth`, so it is
  runnable immediately.
* **FS-VFM official linear probe: FAILED, launcher now fixed.** It died on
  `unrecognized arguments: --local-rank=1` — torch ≥ 2.0 passes `--local-rank` (hyphen) while the
  2024-era FSFM code declares `--local_rank`. Irrelevant either way, because
  `FSFM-CVPR25/fsvfm/util/misc.py:233` reads `LOCAL_RANK` from the environment.
  `analysis/fpad/queue_fsvfm_lp.sh` now uses `torchrun`. **No linear-probe OOD numbers exist.**

The brief names the linear-probe numbers as "an input to Stage 1". They are not available. Stage 1
can proceed without them — the membership decision rests on CLIP-vs-expert complementarity, not on
the external baseline — but the external comparison row stays `TODO(run)` until it is re-run
(`EPOCHS=10 GPUS=1,2 bash analysis/fpad/queue_fsvfm_lp.sh`, ~3–4 h).

## 3. DF40 sealed split — reuse verbatim

`configs/discern_v2/df40_split.json`, sealed 2026-08-20. **38 Dev / 35 Holdout**, resolution
measured per method. `phase2/DF40_SPLIT.md` carries the exact zero-shot wording this supports —
Holdout is *sealed before Phase 2 and unread by any Phase-2 decision*, **not** never seen.

Dev (38): `CollabDiff DiT_cdf DiT_ff MidJourney SiT_cdf SiT_ff StyleGAN2_cdf StyleGAN2_ff
VQGAN_cdf VQGAN_ff blendface_cdf blendface_ff danet_cdf danet_ff deepfacelab faceswap_cdf
faceswap_ff facevid2vid_cdf facevid2vid_ff heygen mcnet_cdf mcnet_ff rddm_cdf rddm_ff
sadtalker_cdf sadtalker_ff simswap simswap_cdf simswap_ff stargan starganv2 styleclip tpsm_cdf
tpsm_ff uniface uniface_cdf uniface_ff uniface_ori`

## 4. VALmix — exists, verified clean, and does **not** need the HDF5

Found at `/data/umar/Repos/DiCoME/eval_adaptation/data/h5/VALmix.h5` (5.0 GB, built 2026-08-13).
Contents match the documentation exactly:

| domain | videos | frames |
|---|---:|---:|
| CDFv2val (Celeb-real, Celeb-synthesis, YouTube-real) | 450 | 14,395 |
| DFDCPval (method_A, method_B, real) | 450 | 12,725 |
| DFEval24val (fake, real) | 450 | 13,022 |
| **total** | **1,350** | **40,142** |

Balanced 450/450/450, which matters because the brief selects on the **macro-average** across
domains so no large domain dominates.

**Test-overlap check (the brief's requirement): CLEAN.** Zero VALmix videos appear in any of the
three sources' test splits — 0/450 against Celeb-DF-v2's 518, 0/450 against DFDCP's 652, 0/450
against Deepfake-Eval-2024's 814.

**The HDF5 is not needed.** VALmix frames are **byte-identical** to our own preprocessed PNGs
(8 frames spot-checked across 4 videos, mean absolute difference 0.000 in RGB order), and all
**1,350/1,350 videos are reachable** in `/data/umar/Datasets/preprocessed/`:

```
CDFv2val     Celeb-DF-v2/{Celeb-real,Celeb-synthesis,YouTube-real}/frames
DFDCPval     DFDCP/{method_A,method_B,original_videos}/frames
DFEval24val  Deepfake-Eval-2024/frames
```

So VALmix should be rebuilt as a **video-id manifest over our own tree**, not read from the h5.
That keeps selection reading byte-identical pixels to training and test, removes a 5 GB
cross-repo dependency, and makes the split inspectable. Note `Deepfake-Eval-2024`'s finetuning
frames are on disk (2,034 video dirs) even though the dataset JSON indexes only the 814 test
videos — so a manifest can reach them, but `NeSyDeFakeDataset` cannot without one.

### The zero-shot partition this forces

| | status after adopting VALmix |
|---|---|
| Celeb-DF-v2, DFDCP, Deepfake-Eval-2024 | **domain-seen** — not strict zero-shot |
| CDFv1, CDFv3, DFD, DFDC, UADFV | strict zero-shot |
| DF40-Holdout | strict zero-shot (in its precise sealed-not-unseen form) |
| DF40-Dev | never zero-shot — drove design |

One prior cost is now **subsumed rather than additive**: FPAD's Stage 2 read 178 Celeb-DF-v2 *real*
videos / 2,841 frames as its domain axis (recorded in `wacv/stage2_gate.json` under
`provenance_cost`). Celeb-DF-v2 loses strict zero-shot via VALmix anyway, so that debt no longer
costs anything extra. It should still be stated, because it is a different kind of exposure.

## 5. Reuse inventory — everything exists except the MR-VAE fit

| component | path | state |
|---|---|---|
| §20 domain audit | `analysis/discern_v2/domain_audit.py` | reuse verbatim |
| EDL loss | `training/train_v1.py::edl_loss` | reuse |
| evidence → Dirichlet → opinion | `networks/discern_v2/{dirichlet,ds_fusion}.py` | reuse |
| DS fusion + V/C/A reliability | `networks/discern_v2/ds_fusion.py` | reuse |
| **multi-source CCF** | `networks/discern_v2/ccf_fusion.py` | reuse — **11/11 synthetic tests pass**, reproduces Table I of the source paper |
| 5-fold gate cross-fitting | `networks/discern_v2/applicability_gate.py` | reuse |
| Shapley marginal utility | `training/stage567.py::marginal_utility` | reuse (exact for ≤3 experts) |
| risk model + frozen DeferPolicy | `networks/discern_v2/risk_model.py` | reuse |
| source-paired sampler + matched augmentation | `training/dataset/paired_sampler.py` | reuse, 12 tests |
| MR-VAE projector + frozen rate branch | `networks/discern_v2/{projectors,rate_branch}.py` | code exists, **operator unfit** |

**Stage 4's required CCF unit tests already pass**: vacuous inputs, perfect agreement, strong
disagreement, an inapplicable expert, all-inapplicable, and permutation invariance — plus the
paper's own Table I. The brief asks for these before trusting CCF on data; they are green.

### What is missing and must be written

* **VALmix manifest + loader** and macro-AUROC selection wired into training (nothing reads VALmix
  today; `stage_de.py`'s `diverse` protocol is FF++ ∪ Celeb-DF-v2 *val*, which is **not** VALmix
  and is test-contaminated — see the firewall note below).
* **`VAL_guard` / `VAL_meta` split.** The brief wants FF++ val split ~60/40 with `VAL_guard` as the
  in-domain guardrail. `configs/discern_v2/meta_split.json` already provides exactly this geometry
  under the names `VAL_select` (60%, 13,436 frames) and `VAL_meta` (40%, 8,960 frames),
  identity-disjoint by union-find. **Rename in reporting, reuse the file.**
* Stage 1's three complementarity numbers in the brief's corrected form — in particular the
  ceiling as `1 − P(both wrong)` rather than a label-aware oracle AUROC.

## 6. proc / LDM branch — **never evaluated, therefore out**

The eight DF40-Dev diffusion methods where AEROBLADE's premise actually holds were identified
(`CollabDiff DiT_cdf DiT_ff MidJourney SiT_cdf SiT_ff rddm_cdf rddm_ff`) but never scored. Under
V1 it was the weakest branch — standalone mean AUROC 0.6225, contribution 0/7 sources outside the
noise band, and `DATASET DETECTOR` on 5 of 6 in the §20 audit. Per the brief's rule it is out.

---

## Firewall check against the existing code

The brief's firewall: diverse-validation labels select checkpoints and **nothing else**; gates, the
risk model and thresholds use FF++ `VAL_meta` only.

`training/stage567.py` defaults to `--val-protocol ffpp` and records which protocol ran, so it
complies **as written**. But its `diverse` option resolves to `["FaceForensics++:val",
"Celeb-DF-v2:val"]`, and every non-FF++ `val` split in this repo is an **alias of test**
(Celeb-DF-v2 val ∩ test = 518/518; DFDCP 652/652). Selecting that option would calibrate gates on
test videos. It is not VALmix and must not be confused with it — 🟡 recommend renaming it
`diverse_LEGACY_test_contaminated` or removing it before Stage 3.
