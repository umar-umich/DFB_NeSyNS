# CEC — Decision & Gate Log
Causal Evidence Certification. Timestamps UTC. Append-only; newest entry at the bottom.

Governing docs: `docs/implementation/CURRENT_STATE.md` (wins over everything) ·
`docs/implementation/CLAUDE_CODE_VEG_IMPLEMENTATION_v2.md` (the task list) ·
`docs/pilots/PLANNING.md` + `docs/pilots/CODEBOOK.md` (vocabulary, controls, claim scope).

Invariants: **freeze before you measure · training only after the audit freeze · every
component earns a number · no invented numbers.**

---

## 2026-07-16 17:33 — Naming decided (closes ledger item 5)
Umar picked **CEC (Causal Evidence Certification)**. `veg/` is retired as the working
package name. Layout agreed this session:

- docs → `docs/cec/` (this log lives at `docs/cec/LOG.md`)
- code → `cec/` at repo root, mirroring the Implementation-v2 layout with `cec/` in place
  of `veg/`: `cec/registration/`, `cec/data/`, `cec/masks/`, `cec/instruments/`,
  `cec/repair/`, `cec/proposer/`, `cec/gate/`, `cec/records/`, `cec/assembly/`,
  `cec/dpo/`, `cec/eval/`, `cec/pilots/`, `cec/scripts/`.

**Superseded in part by the 18:05 entry below** — detector classes follow DeepfakeBench
house style in `training/`, not `cec/`. See "File organization convention".

No "graph" language anywhere in code, comments, or outputs — the graph was removed as a
component; claims live in per-image Certified Evidence Records.

---

## 2026-07-16 17:33 — TASK 0 (recon) complete. Reported; no code written.
Repo state at recon: branch `main`, HEAD `ba8ccce`, working tree has uncommitted doc
moves (`docs/` → `docs/pilots/`, `docs/implementation/` untracked).

### 0. Repo confusion resolved (session-level)
The first session was launched in `/data/umar/Repos/DeepfakeJudge` by mistake. That repo
is **not ours**: it is a third-party clone (`KjAeRsTuIsK/DeepfakeJudge`, CVPR-2026,
MBZUAI/Monash) holding only `pointwise/inference.py`, `pairwise/inference.py`, prompts,
and two shell scripts. It has no `training/detectors/`. Per CURRENT_STATE it is only an
asset for the **T5 correlation table** (DeepfakeJudge-7B scores vs our certification
rate). The real project repo — DeepfakeBench checkout + DISCERN, docs, pilot harness,
pilot results — is **`/data/umar/Repos/DFB_NeSyNS`** (this repo). Work continues here.

### 1. Detector inventory — NOT in the DeepfakeBench registry
All four frozen detectors live in a **third repo**, `/data/umar/Repos/GenD_NeSy`, with
weights on disk. They are loaded through the registry at `scripts/pilot_detectors.py`,
which `sys.path`-inserts and `chdir`s into GenD (checkpoints are resolved by RELATIVE
path; FSFM string-matches its exact checkpoint path, so the chdir is load-bearing).

| Detector | Checkpoint (relative to `GenD_NeSy/`) | Notes | In DFB `training/detectors/`? |
|---|---|---|---|
| effort | `weights/Effort/effort_clip_L14_trainOn_FaceForensic.pth` | `model_type="effort"` | yes (`effort_detector.py`, `module_name='effort'`) |
| forada | `weights/ForAda/ForAdackpt_best.pth` | | **no** |
| fsfm | `weights/FS-VFM/FS-VFM-ViT-L.pth` | `zoom=1.3` via `CustomPreprocessing` | **no** |
| gend | `yermandy/GenD_CLIP_L_14` (HF hub) | | **no** |

DeepfakeBench's own registry contains only `effort` and `clip`. Interface is uniform:
`model(tensor) -> HeadOutput.logits_labels` ([real, fake]); `model.get_preprocessing()`
returns a PIL→tensor transform; `p(fake) = softmax(logits)[:, 1]`.

**Extra find (Task 5 relevant):** FreqNet is already implemented at
`GenD_NeSy/src/model/FreqNet.py` with weights at `weights/FreqNet/4-classes-freqnet-v2.pth`.
A spectral-instrument candidate that needs **no download**. Also present:
`weights/FS-VFM/FS-VFM-ViT-L-Adapter.pth`, `weights/forensics_adapter/ViT-L-14.pt`.

**CONFLICT 1 (open, 🟡).** Implementation v2 Task 2 says to wrap each detector "through
DeepfakeBench's own preprocessing/normalization — a mismatched transform silently changes
every number." For three of the four detectors that is impossible (not in the DFB
registry), and for all four it is *wrong*: every frozen anchor number in the ledger was
produced through **GenD's** `model.get_preprocessing()` via `pilot_detectors.py`.
Following the sentence literally would cause exactly the silent renumbering it warns
against. **Recommendation:** `cec/instruments/spatial.py` delegates to the GenD registry
(port/wrap `pilot_detectors.py`); amend the Task 2 sentence. Awaiting Umar.

**Open question (🟡):** Umar asked that "our detector main class" be placed in
`training/detectors/` alongside the previous detectors. That collides with the
Implementation-v2 layout, which puts the wrapper at `instruments/spatial.py`. Options:
(a) wrapper in `cec/instruments/spatial.py` only; (b) class in `training/detectors/`,
imported by `cec/instruments/spatial.py`. Awaiting Umar.

### 2. Phase-1 harness located (PORT, never rewrite)
| File | Contents |
|---|---|
| `scripts/pilot_detectors.py` | frozen-detector registry; `load_detector`, `pfake_batch` |
| `scripts/intervention_pilot.py` (388 ln) | **v1 harness.** paths, `read_raw_frame`, `revert` (**hard paste**), `shift_mask`, `make_control_region` (equal-area region outside the mask, hugging its boundary), `gather_samples`, `build_sample`, QC constants |
| `scripts/pilot1_rev3.py` (360 ln) | **rev3 harness.** `poisson_paste` (cv2 `seamlessClone`, hard-paste fallback <20 px), `blur_region`, `shift_region`, `Lpips`, `match_corruption`, `build_sample_rev3`. Imports v1 for sampling/alignment |
| `scripts/pilot1_spectral.py` (208 ln) | **Study-1B ops.** `notch_hf(r_lo=.55, r_hi=.85)`, `checkerboard_supp(r_min=.35, pct=99.5)`, `residual_renorm` |

Harness constants (v1, inherited by rev3): `METHODS=["Deepfakes","FaceSwap"]`, `N_PER=50`,
`MAX_PER_VID=2`, `QC_MSE=60.0`, `FG_MIN/MAX=0.02/0.60`, `CTRL2_OFFSET=1`, `SEED=0`;
rev3 `LPIPS_TOL=0.15`. Alignment: `preprocessing/preprocess.py:align_face` (raw frame +
5-pt landmarks → 256 crop). Run env: `/data/umar/miniconda3/envs/GenD/bin/python`, with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

rev3 variants per sample: `gt_repair` (Poisson, GT mask), `wrong_region` (Poisson, ctrl
region), `real_offset` (Poisson on the paired real using a ±1 frame), plus LPIPS-matched
`match_blur` / `match_shift`.

**Frozen anchors re-verified** against `results/pilot_rev3/ALL_DETECTORS.md` (100 FF++ c23
frames, 50 DF + 50 FS, test split, seed 0) — median GT-repair drops match CURRENT_STATE
item 6 exactly: effort 0.479 · fsfm 0.713 · gend 0.625 · forada 0.839. Controls inert
(wrong-region median |Δ| ≤ 0.003; real-offset median ≤ 0.033).

**CONFLICT 2 (open, 🟡) — the Task 4 acceptance number.** Task 4 says: port Poisson repair
+ the three rev3 controls; **acceptance = effort `257_420/762`, Δtrue=+0.659, Δctrl=−0.001**.
That pair of numbers is from **`results/pilot/` (v1)**, whose repair is a **hard paste**,
not Poisson. Verified from disk:

| Source | repair op | Δ true/gt | Δ control |
|---|---|---|---|
| `results/pilot/effort/deltas.json` | hard paste (`revert`) | `drop_true` **0.6588** | `drop_ctrl` **−0.0010** |
| `results/pilot_rev3/effort/deltas.json` | Poisson (`seamlessClone`) | `drop_gt` **0.6480** | `drop_blur` **+0.0013**, `drop_wrong` −0.0004, `drop_shift` −0.0034 |

(Same sample, both: `p_orig` 0.9875, `p_real_orig` 0.3943.) So the port **as specified can
never reproduce 0.659**; the doc mixes the v1 acceptance target with the rev3 port spec.
**Recommendation:** retarget acceptance to the rev3 numbers — Δgt = 0.648, Δblur = +0.001
(within tolerance). Alternative: also port v1's hard paste to reproduce 0.659 exactly.
This is a pre-committed number, so Umar decides. **No downstream work until Task 4 passes.**

### 3. Preprocessed data layouts
**FF++** — `/data/umar/Datasets/preprocessed/FaceForensics++`, raw at
`/data/umar/Datasets/FaceForensics++`.
```
manipulated_sequences/<method>/c23/{frames,landmarks,masks}/<id1_id2>/<nnn>.{png,npy}
original_sequences/{youtube,actors}/c23/...
raw paired real: original_sequences/youtube/c23/videos/<id1>.mp4  (read frame <nnn>, align_face w/ fake's landmarks)
```
Methods present: Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter,
DeepFakeDetection. **NeuralTextures has frames + landmarks + masks (1000 mask dirs)** —
sufficient to carry Pilot 1L on its own. Extra per-method dirs from prior work:
`forensic_features/`, `fast_semantic/`, `facebench_semantic/`.
**Only c23 is preprocessed** — no c40, no raw. (Compression-level work would need
re-preprocessing; Study 1B's compression arm presumably ran from raw video.)

**DF40** — `/data/umar/Datasets/df40` (159 GB; path supplied by Umar this session).
```
{train,test}/<family>/{ff,cdf}/{frames,landmarks}/<clip_id>/<nnn>.png     # 256x256
```
40 test families (swaps: simswap, inswap, blendface, facedancer, fsgan, faceswap,
mobileswap, uniface, e4s, deepfacelab; reenactment/localized: wav2lip, sadtalker, fomm,
MRAA, tpsm, mcnet, hyperreenact, danet, pirender, lia, facevid2vid, one_shot_free;
generative: StyleGAN2/3/XL, DiT, SiT, VQGAN, RDDM, ddim, sd2.1, pixart, CollabDiff,
MidJourney, stargan(v2), styleclip, e4e, whichfaceisreal, heygen). `ff` subsets use
FF++-compatible `<id1_id2>` naming; `cdf` uses Celeb-DF ids.

**BLOCKER 3 (open, 🟡) — DF40 has no masks.** `find` for any `masks/` dir under
`/data/umar/Datasets/df40` returns **nothing**: frames + landmarks only. The spatial gate
needs (a) a GT manipulated-region mask and (b) a paired real. DF40 ships neither. Its
crops also come from DF40's own cropper rather than DFB's `align_face`, so reconstructing
paired reals from FF++ originals risks geometry mismatch — the harness `QC_MSE ≤ 60` check
would reject misaligned samples. Consequences:
- DF40 is usable **as-is** for detector inference and the **spectral** branch (no paired
  real needed).
- Task 10 / **Pilot 1L localized spatial slices** on DF40 would require mask generation +
  re-cropping. **FF++ NeuralTextures alone can carry Pilot 1L** (masks present).
**Recommendation:** treat DF40 as spectral-only for v1; keep 1L on FF++ NT. Awaiting Umar.

### 4. Environment
2× H200 (148 GB each). Conda env `GenD` at `/data/umar/miniconda3/envs/GenD/bin/python`.
HF cache holds a ready **Task 1 proposer candidate: `OpenGVLab/InternVL3-8B`** (+ `-AWQ`).
Also cached and possibly useful later: `facebook/PE-Core-L14-336`, `jonathandinu/face-parsing`
(a BiSeNet-class alternative should Task 3's landmark regions prove too coarse),
`runwayml/stable-diffusion-inpainting`, DINOv2/v3, SigLIP. No Qwen-VL cached.
`veg/` (or `cec/`) package does **not** exist yet — nothing built.

### Status → next
TASK 0 **done**, reported, stopped before code as instructed. Open 🟡 decisions at the time
of this entry: (1) detector-wrapper placement — **RESOLVED, see below**; (2) Task 2
amendment — wrap via GenD preprocessing, not DeepfakeBench; (3) Task 4 acceptance retarget
— rev3 0.648/+0.001 vs porting v1's hard paste for 0.659; (4) DF40 spectral-only vs
re-preprocess with masks.
Task 1 otherwise ready: `params.yaml` + `frozen_prompts/` + proposer pin (InternVL3-8B is
on disk as candidate 1; candidate 2 TBD).

---

## 2026-07-16 18:05 — File organization convention (closes open question 1)
Umar's instruction: **organize CEC to the existing DFB_NeSyNS structure. All new detector
main-class code goes in `training/detectors/`**, following the previous detector at
`training/detectors/nesy_defake_detector.py` (DISCERN / NeSy-DeFake). Implementation code
must be **modular and well organized** — not one long file.

### The house style, as read from `nesy_defake_detector.py` (546 ln)
The DISCERN detector is the reference for how a detector is added to this repo:

| Concern | Location | Convention |
|---|---|---|
| Detector main class | `training/detectors/<name>_detector.py` | subclasses `AbstractDetector` (`.base_detector`); decorated `@DETECTOR.register_module(module_name='<name>')`; docstring header naming the file + forward paths |
| Registration | `training/detectors/__init__.py` | one `from .<name>_detector import <ClassName>` line; `DETECTOR` comes from `metrics/registry.py` (a plain name→cls `Registry`) |
| Heavy submodules | `training/networks/<name>/` | DISCERN splits into `foundation_models/`, `classifiers/`, `fusion/`, `losses/`, `causal/`, `concept_branch.py`, `causal_branch_factory.py` — the detector file only wires them together |
| Config | `training/config/detector/<name>.yaml` | one YAML per detector/variant (e.g. `nesy_defake_ablation1.yaml`, `effort.yaml`) |

`AbstractDetector` requires: `features`, `classifier`, `forward`, `build_backbone`,
`build_loss`, `get_losses`, `get_train_metrics`.

### How CEC maps onto it
**Rule of thumb: anything that is a _model_ follows DFB house style under `training/`;
anything that is _pipeline / experiment / artifact_ lives under `cec/`.**

```
training/detectors/cec_instrument_detector.py   # MAIN CLASS(ES). frozen instruments,
                                                #   registered 'cec_spatial' / 'cec_spectral'
training/networks/cec/                          # modular guts (GenD registry adapter,
                                                #   preprocessing, spectral ops)
training/config/detector/cec_*.yaml             # one YAML per instrument
cec/                                            # pipeline only: registration/, data/,
                                                #   masks/, repair/, proposer/, gate/,
                                                #   records/, assembly/, dpo/, eval/,
                                                #   pilots/, scripts/
```
`cec/instruments/spatial.py` and `cec/instruments/spectral.py` therefore become **thin
adapters** exposing the Implementation-v2 API (`SpatialInstrument.p_fake(img) -> float`
+ batched variant) on top of the registered detector classes. The Implementation-v2 layout
line "instruments/spatial.py" is satisfied by that adapter; the model code itself lives in
`training/`, per this repo's convention. No duplicate implementations.

### Two design notes for whoever writes Task 2
1. **`AbstractDetector` is a training abstraction; CEC instruments are frozen.** Its
   contract demands `build_loss`, `get_losses`, `get_train_metrics` — meaningless for a
   frozen instrument that is never trained. **Decision:** implement `features`,
   `classifier`, `forward` for real, and have the three training-only methods raise
   `NotImplementedError` with a message pointing at CURRENT_STATE rule 4 (detectors and
   instruments stay frozen; only the proposer is tuned). The abstract class then *enforces*
   the freeze invariant instead of fighting it. 🟡 flag if Umar prefers silent no-ops.
2. **The `os.chdir` hazard.** `scripts/pilot_detectors.py` does a module-level
   `os.chdir("/data/umar/Repos/GenD_NeSy")` at import, because GenD resolves checkpoints by
   relative path (FSFM matches its exact checkpoint string). That is safe in a standalone
   pilot script but **not** safe inside `training/detectors/`, where DFB resolves its own
   config/weight paths relatively — a global chdir at import time would silently break
   them. **Decision:** wrap the chdir in a scoped context manager used only around GenD
   model construction, never at import. Absolute paths everywhere else.

### Status
Open question 1 **closed**. Three 🟡 remain and still block a clean Task 1 start: the Task 2
preprocessing amendment, the Task 4 acceptance retarget, and the DF40 mask decision.

---

## 2026-07-16 — Three 🟡 closed by Umar; Implementation v2 amended
All three open conflicts from the recon entry are decided. Each amends the task list;
the amendments are recorded here because Implementation v2 is not edited in place.

**CONFLICT 1 + Task 2 preprocessing → GenD registry delegation.** `cec/instruments/spatial.py`
wraps `scripts/pilot_detectors.py` (via `training/networks/cec/`) and uses GenD's own
`model.get_preprocessing()`. The Task 2 sentence "through DeepfakeBench's own
preprocessing/normalization" is **amended** — it is unsatisfiable for forada/fsfm/gend
(not in the DFB registry) and wrong for all four, since every frozen anchor in ledger
item 6 was produced through GenD's transform. Following it literally would have caused
exactly the silent renumbering it warns against. Rationale recorded in
`cec/registration/detectors.yaml`.

**CONFLICT 2 + Task 4 acceptance → retargeted to rev3.** The pre-committed pair
(`257_420/762`, Δtrue=+0.659, Δctrl=−0.001) is a **v1 hard-paste** number and cannot be
reproduced by the rev3 Poisson port Task 4 specifies. Acceptance is **retargeted** to the
rev3 numbers on the same sample: **Δgt = 0.648, Δblur = +0.001** (from
`results/pilot_rev3/effort/deltas.json`). v1's hard paste is **not** ported. Task 4 still
gates all downstream work.

**BLOCKER 3 + DF40 → spectral-only for v1.** DF40 has no masks and is cropped by its own
cropper, not `align_face`; the spatial gate needs a GT mask + paired real and DF40 has
neither. DF40 carries detector inference + the spectral branch. **Pilot 1L runs on FF++
NeuralTextures alone** (frames + landmarks + 1000 mask dirs present). No re-preprocessing
of the 159 GB tree. Recorded as `data.df40.scope: spectral_only` in `params.yaml`.

**Proposer candidate 2 → Qwen2.5-VL-32B-Instruct** (Umar's pick over the recommended 7B).
Stronger proposer and a wider contrast against InternVL3-8B; cost is a materially more
expensive Task 9 timing probe and audit run. 🔴 download outstanding.

---

## 2026-07-16 — Layout re-confirmed (no change)
Umar queried why `cec/` sits at the repo root. Clarified that **no detector code exists
yet** — Task 1 is registration config only, and Task 2's detector classes are already
slated for `training/detectors/cec_instrument_detector.py` per the 18:05 entry. The 18:05
split is **re-confirmed unchanged**: models under `training/` in DFB house style;
pipeline/config under `cec/`, alongside the repo's existing root-level `scripts/`,
`analysis/`, `preprocessing/`, `results/`. Nothing moved.

---

## 2026-07-16 — TASK 1 (registration + pins) built. 🟡 thresholds await confirmation.
Written (🟢, all verified to load and self-check):

| File | Contents |
|---|---|
| `cec/registration/params.yaml` | K=4, order-of-listing, T=0/seed 17, gate thresholds, control battery, QC constants, NM reporting rules, data scope |
| `cec/registration/detectors.yaml` | four frozen detectors + checkpoints, GenD-delegation rationale, chdir hazard, frozen anchors, FreqNet spectral candidate |
| `cec/registration/pins.yaml` | proposer pins + VRAM estimates, spectral candidates, measured environment |
| `cec/registration/frozen_prompts/proposer_type_b.txt` | the ONE proposer prompt: Type-B content + embedded codebook vocab + strict JSON schema. hash `3950a62e3b6d8ea8` |
| `cec/registration/frozen_prompts/verdict_requery.txt` | verdict re-query prompt. hash `8fb8eba28192730f` |
| `cec/registration/loader.py` | typed readers; no defaulting — a missing key is an error, not an invented number |
| `cec/registration/vocab.py` | codebook rev-3 vocabulary as code (9 artifacts, 14 regions) + `check_prompt_vocab()` drift guard + spectral routing rule |

### The two threshold values (🟡 — confirm once, then FROZEN)
Both are derived from `results/pilot_rev3/*/deltas.json`; no invented numbers.

**1. Per-detector certify margin.** Rule: `margin = max(0.10, ceil_0.05(p99 of the pooled
control drop distribution))`, pooling {matched blur, matched shift, wrong region} over the
100 rev3 samples — i.e. the margin sits above the 99th percentile of what each detector
does when nothing causal changed. The 0.10 floor stops a very inert detector certifying on
noise.

| detector | pooled control p99 | margin |
|---|---|---|
| effort | 0.026 | **0.10** (floor binds) |
| fsfm | 0.087 | **0.10** |
| gend | 0.124 | **0.15** |
| forada | 0.080 | **0.10** |

**2. Control inertness.** `wrong_region_inert_max = 0.05` (pilot median |Δ| ≤ 0.003, so
0.05 is far above the measured null); `real_offset_max = 0.15` (reusing Task 9's
model-level offset gate per-claim; pilot medians 0.013–0.033).

**Correction to a first attempt at this rule.** The LPIPS-matched corruption was initially
treated as a fourth inertness control (|Δ| ≤ 0.05) — wrong, and it discarded 17–25% of
samples. The matched corruption is a **contrast**, not a control that must sit at zero:
the pilot uses it as the GAP column, and Task 9's gate reads "true-region ≥ corruption+20pp".
Fixed to `matched_corruption_gap = 0.20`. Certify rule is now:

```
NM >= certify_margin[det]  AND  NM - max(NM_blur, NM_shift) >= 0.20
AND |NM_wrong| <= 0.05     AND  real_offset <= 0.15
```

Rates these thresholds produce on the rev3 GT-mask repairs (the ceiling case that *should*
mostly certify): **effort 78% · fsfm 80% · gend 71% · forada 82%.** Binding constraint is
margin+gap; the wrong-region control fires on 0–5%, as expected for a control the pilot
measured as inert. Recorded in `params.yaml` as `pilot_certification_rate` so later drift
is visible.

### Environment corrections found while pinning
- The box has **4× H200 NVL @ 140 GB**, not the 2× 148 GB the docs assume.
- **`trl` is not installed** in env `GenD` (torch 2.8.0+cu128, transformers 4.56.2,
  peft 0.14.0). Task 11's DPO needs it — 🔴 install before Task 11, not a Task 1 blocker.
- InternVL3-8B pinned at revision `853e3a797a661694b1b8ece0cb72dc2b23e3dac9` (on disk).
  Qwen2.5-VL-32B-Instruct revision is `null` until 🔴 download; pin the snapshot hash then.

### Status → next
TASK 1 **built, not frozen** — the two threshold values need Umar's one-time confirmation
(Implementation v2 Task 1: "confirm the two threshold values once, then frozen"). Also
open: 🔴 Qwen2.5-VL-32B download. On confirmation → **TASK 2** (detector wrapper:
`training/detectors/cec_instrument_detector.py` + `training/networks/cec/` + thin adapter
at `cec/instruments/spatial.py`), then TASK 3, then TASK 4's acceptance gate at
Δgt = 0.648 / Δblur = +0.001.

---

## 2026-07-16 — Qwen2.5-VL-32B-Instruct downloaded and pinned
🔴 done by Umar. 64 GB in the HF cache; snapshot
`7cfb30d71a1f4f49a57592323337a4a4727301da`, recorded in `pins.yaml` (`on_disk: true`).
Both proposer candidates are now on disk. Task 1's remaining 🟡 (threshold confirmation)
is unaffected.

---

## 2026-07-16 — TASK 2 (detector wrapper) built. Smoke 3/4; effort under investigation.
Written, following the DFB house style fixed by the 18:05 convention entry:

| File | Role |
|---|---|
| `training/detectors/cec_instrument_detector.py` | MAIN CLASS. `CECInstrumentDetector` + four registered subclasses `cec_effort` / `cec_fsfm` / `cec_gend` / `cec_forada`. Reads `cec/registration/detectors.yaml` (plain YAML, no cec import) so checkpoints are stated once |
| `training/networks/cec/gend_registry.py` | GenD registry adapter: scoped chdir, model construction, BGR→PIL |
| `training/networks/cec/inference.py` | batched `pfake_batch`, ported from the pilot |
| `training/config/detector/cec_spatial.yaml` | DFB-house config; names the instrument, restates no checkpoint |
| `cec/instruments/spatial.py` | thin adapter: `SpatialInstrument.p_fake` / `p_fake_batch` / `necessity_margin`. Lazy-loads torch |
| `cec/scripts/smoke_instruments.py` | the Task 2 smoke test |

**The freeze is enforced by the type.** `build_loss` / `get_losses` / `get_train_metrics`
raise `NotImplementedError` pointing at CURRENT_STATE rule 4, per the 18:05 design note;
`train(mode=True)` also refuses, since a frozen instrument in train mode would drift
BN/dropout. `AbstractDetector` now enforces the freeze rather than fighting it.

**Two deviations from `scripts/pilot_detectors.py`, both documented in the module
docstring, both intended to be behaviour-preserving:**
1. `os.chdir(GEND)` at import → a scoped `gend_cwd()` around construction only. A global
   chdir inside `training/` would break DFB's own relative path resolution.
2. `sys.path.insert(0, GEND)` → `sys.path.append`. Prepending lets GenD's top-level
   `datasets/`, `config/`, `scripts/`, `results/` dirs shadow same-named modules — via
   PEP-420 namespace packages that includes the HuggingFace `datasets` library. `src` and
   `run` are unique to GenD, so appending resolves them identically.

### Smoke test: strict by design
The weak test is "fake scores above real". The strict test — what `smoke_instruments.py`
actually runs — re-scores the exact frame the rev3 pilot scored (Deepfakes/485_425/465,
read straight off the preprocessed PNG with no re-alignment, exactly as the harness does)
and demands the pilot's own `p_orig` back. If either deviation above had changed a
transform or a checkpoint resolution, `p_orig` moves and every anchor in ledger item 6 is
invalid.

**Result: fsfm, gend, forada reproduce `p_orig` exactly.** forada 0.9995 = 0.9995.
**effort returned 0.990071 vs the pilot's 0.989948 — off by 1.2e-4**, just past the 1e-4
tolerance. 🟡 **UNRESOLVED.** Hypothesis: float nondeterminism from batch composition (the
pilot scores in batches of 64; the smoke test used a batch of 2, and CLIP attention
kernels are not batch-invariant), which would make the tolerance wrong rather than the
wrapper. **This is an untested hypothesis, not a finding.** A probe scoring the same frame
at batch sizes 1/2/8/64 and measuring the spread will settle it. Nothing downstream should
rely on effort until it does — effort is the detector Task 4's acceptance gate is written
against.

### Environment: the eager-import wall (shared code changed — flagged)
`import detectors` pulls in all ~30 DFB detectors, so reaching one frozen CLIP wrapper
demanded slowfast/fvcore, tensorboard, EfficientNet, and more. The `GenD` env holds only
what the four instruments need, by design.

**Rejected:** installing DeepfakeBench's full requirements into `GenD`. That env produced
every frozen anchor; a transitive numpy/torch pin (dlib, imgaug, albumentations) could
silently renumber them. Exactly what "freeze before you measure" forbids.

**Done instead:** `training/detectors/__init__.py` and `training/networks/__init__.py` now
import each detector/backbone independently, skipping any whose third-party dependencies
are absent and logging what was skipped. **With all dependencies installed, behaviour is
identical** — same names, same registry. A missing detector is logged, and asking
`DETECTOR` for it still raises `KeyError`. These are **shared files** that DISCERN
training also imports; the change is only to the failure mode. Flagged for Umar.

Small deps added to `GenD` (all `--no-deps` where transitive risk existed):
`simplejson`, `fvcore`, `iopath`, `yacs`, `portalocker`, then `tensorboard`, `loralib`.
**Verified unchanged before/after: torch 2.8.0+cu128 · transformers 4.56.2 · numpy 2.2.6.**

### Effort 1.2e-4 question — RESOLVED (batch-composition float noise)
Probe scored the same effort frame at several batch compositions:

| composition | p_fake | Δ vs pilot |
|---|---|---|
| padded to 64 | 0.98994815 | **0.00e+00** |
| repeated ×8 | 0.98994827 | 1.2e-07 |
| alone (bs=1) | 0.99007100 | +1.23e-04 |
| with real (bs=2) | 0.99007100 | +1.23e-04 |

**At batch size 64 — the pilot's own batch size — effort reproduces `p_orig`
bit-for-bit.** The 1.23e-4 gap was purely the smoke test scoring in a batch of 2; CLIP
attention kernels are not batch-invariant. The wrapper changed nothing; the 1e-4 tolerance
was tighter than the frozen pipeline's own reproducibility floor. Smoke tolerance loosened
to 5e-4 with the reason recorded in `smoke_instruments.py`.

**Consequence carried forward to Task 7:** NM = p_fake(orig) − p_fake(intervened). Score
original and intervened in the SAME batch so this ~1e-4 offset cancels in the difference.

### Smoke test — PASSED 4/4
```
instrument   p_fake  expected   p_real  anchor  result
effort       0.9901    0.9899   0.1892   0.479  OK
fsfm         0.8442    0.8442   0.2292   0.713  OK
gend         0.9924    0.9923   0.2086   0.625  OK
forada       0.9995    0.9995   0.1103   0.839  OK
```
All four reproduce pilot `p_orig` within 5e-4 and score the fake far above the real.
TASK 2 **complete and verified** — GenD delegation preserves every number; the scoped
chdir and appended sys.path are proven behaviour-neutral.

### Status → next
TASK 2 **done**. Next: TASK 3 (pairing + masks) — `cec/data/pairing.py` (FF++ fake →
paired real, port the pilot re-alignment, `verify_pair()`) + `cec/masks/regions.py`
(landmark → binary region masks). Then TASK 4's acceptance gate (effort Δgt = 0.648 /
Δblur = +0.001).
Still open: 🟡 the two Task 1 threshold values await Umar's confirmation (blocks Task 7,
not Task 3/4).

---

## 2026-07-16 — Gate thresholds FROZEN (closes the last Task 1 🟡)
Umar confirmed both threshold values at the pilot-derived defaults — no values changed,
they were already in `params.yaml`; this just freezes them. A `FROZEN` note is now stamped
on the `gate:` block. Changing any gate value from here invalidates the Task 9 audit.

**1. Certify margin** — per-detector, `max(0.10, ceil_0.05(pooled control p99))`:
effort 0.10 · fsfm 0.10 · gend 0.15 · forada 0.10. GT-mask certify rates 78/80/71/82%.

**2. Control battery** — wrong-region |Δ| ≤ 0.05, real-offset ≤ 0.15, matched-corruption
gap ≥ 0.20. Wrong-region fires on 0–5% of pilot samples (genuinely inert).

Task 1 is now fully closed. Nothing blocks Task 7 on thresholds anymore.

---

## 2026-07-16 — TASK 3 (pairing + masks) built and verified.
Written:

| File | Role |
|---|---|
| `cec/data/pairing.py` | FF++ fake → pixel-registered paired real. Ports `read_raw_frame` + the re-alignment + QC from `intervention_pilot.py`/`pilot1_rev3.py`. `build_pair()` → `PairSample`; `verify_pair()` → (ok, bg_mse) |
| `cec/masks/regions.py` | codebook location vocab → deterministic binary masks from the 5-pt landmarks. `region_mask(region_id, landmarks)`; `FaceGeometry`; `align_matrix`; per-region `FIDELITY` tags |
| `cec/scripts/smoke_pairing_masks.py` | the Task 3 smoke test |

**Ported, not reinvented.** `align_face` is imported from `preprocessing/preprocess.py`;
`intervention_pilot` is deliberately NOT imported (it imports `pilot_detectors`, which
does a module-level `os.chdir` into GenD). `pairing.py` stays chdir-free.

**The re-alignment subtlety, preserved.** The pre-aligned youtube crops on disk are
aligned independently of the fake crops (bg-MSE outside the mask ~800). The paired real is
instead the RAW youtube frame re-aligned with the FAKE's landmarks, giving bg-MSE ~2–9
(compression noise only). `build_pair()` enforces the pilot's QC unchanged: FG ∈ [0.02,
0.60], alignment succeeds, bg_mse ≤ 60.

**Landmarks are 5-point, in raw coords** (left_eye, right_eye, nose, left_mouth,
right_mouth), verified on Deepfakes/FaceSwap/NeuralTextures. `align_face` maps them to a
fixed template, so the aligned-crop positions are deterministic. Region masks are built in
the aligned 256 frame from those five anchors, scaled by inter-ocular distance.

**The 5-point limit is recorded, not hidden.** Six regions are anchored directly
(both eyes, inter_ocular, nose, mouth, nasolabial); seven are geometric extrapolations
(cheeks, jawline, chin, forehead, hairline, face_boundary) — tagged `extrapolated` in
`FIDELITY`. This is exactly the coarseness Implementation v2 Task 3 flags for the BiSeNet
upgrade "only if Pilot 1L shows landmark regions too coarse". `jonathandinu/face-parsing`
is cached and ready if 1L calls for it (LOG recon §4). `whole_face` is not a spatial
region: `region_mask` returns None, callers route it to the spectral instrument.

### Smoke test — PASSED (Deepfakes/000_003/000, fg 0.218, bg_mse 2.21)
- **`align_matrix` reproduces `align_face` to mean |Δ| = 0.0000 px** — the reproduced
  transform matrix is exact, so regions sit where alignment actually puts the face.
- Every spatial region builds non-empty and in-bounds; `whole_face` → None; unknown region
  → KeyError.
- GT-overlap is anatomically sane: eyes/nose/mouth/nasolabial 100% inside the GT swap,
  cheeks 84–90%, inter_ocular 86%, forehead/hairline low (a Deepfakes swap leaves them
  alone). Inner regions cover 65% of the GT mask.
- Visual overlay (fake | re-aligned real | regions) confirms the real is pose/frame
  registered and every colour lands on the right facial part.
- One informational note: `face_boundary` reaches the frame edge when the face fills the
  crop. Harmless here (repair is inert at bg_mse≈2) and inherent to landmark-only geometry;
  downgraded from a hard failure.

### Status → next
TASK 3 **done**. Next: **TASK 4** — port Poisson repair + the three rev3 controls into
`cec/repair/`, then the acceptance gate: effort **Δgt = 0.648 / Δblur = +0.001** (retargeted
from the v1 hard-paste number; see the 🟡-closing entry above). 🔴 Umar runs the acceptance;
no downstream work until it passes. `pairing.py` + `regions.py` are the inputs it builds on.

---

## 2026-07-16 — TASK 4 (repair + controls) built. ACCEPTANCE GATE PASSED.
Written:

| File | Role |
|---|---|
| `cec/repair/repair.py` | `poisson_paste` (production; ported verbatim from `pilot1_rev3.poisson_paste`) + `hard_paste` (v1 `revert`, kept ONLY as the tiny-region <20px fallback) |
| `cec/repair/controls.py` | `blur_region`, `shift_region`, `Lpips`, `match_corruption`, `make_control_region` (ported), and `build_variants(pair, lp)` assembling the full battery for a `PairSample` |
| `cec/scripts/acceptance_task4.py` | the mandatory acceptance gate |

**v1 hard paste NOT the production op** (CONFLICT 2 resolution): Poisson is production;
hard paste survives only as the seamlessClone fallback for regions < 20 px.

### Acceptance gate — PASSED
Sample **FF++ Deepfakes / 257_420 / 762** (the frame the v1 acceptance named; retargeted
to the rev3 Poisson numbers). Orig + every variant scored in ONE batch so the ~1e-4 batch
offset cancels in each drop.

```
quantity       measured     target      |Δ|     tol  result
p_orig           0.9875     0.9875   0.0000   0.010  OK
p_real           0.3943     0.3943   0.0000   0.010  OK
drop_gt          0.6482     0.6480   0.0002   0.020  OK
drop_blur        0.0013     0.0013   0.0000   0.020  OK
controls: drop_shift -0.0034 · drop_wrong -0.0004 · real_offset |Δ| 0.0122
```
`p_orig`/`p_real` exact; `drop_gt` off by 2e-4 (discrete LPIPS blur-kernel sweep + batch
noise). The full spatial chain — pairing → masks → Poisson repair → control battery — is
proven byte-faithful to the frozen rev3 pilot. **This was the one hard checkpoint; it
clears, and downstream work is unblocked.**

### Status → next
TASKS 0–4 **done and verified**. Next: **TASK 5** — spectral instrument + PILOT S. Needs a
🟡 from Umar: two public-weight freq-detector candidates. Candidate 1 is FreqNet (already
on disk, `GenD_NeSy/weights/FreqNet/4-classes-freqnet-v2.pth`, no download); candidate 2 is
open (NPR-class), plus the pre-approved fallback (linear probe on the DISCERN spectral
bank). Then port the Study-1B interventions (notch / checkerboard-suppress / residual
renorm) and run Pilot S: does a freq detector (i) separate on our crops (AUC) and (ii)
respond to spectral interventions where CLIP detectors ≈ 0. DF40 is spectral-only, so it
feeds this arm.

---

## 2026-07-16 — TASK 5 scaffold built (NPR-independent parts). Pilot S blocked on 🔴 NPR.
Candidate 2 decided by Umar: **NPR** (Neighboring Pixel Relationships, CVPR 2024).

**NPR is not on disk** — must be downloaded (🔴). What's on disk is only adjacent:
`NSG-VD/models/npr.py` is the canonical NPR *architecture* (from chuangchuangtan) but
NSG-VD ships only *video* weights (SEINE/Pika); the `image_cnpr/*.pth` files
(978 MB each, in `model_weights_backup/` and `ProbeTruthInference/`) are a CLIP-scale
"CNPR" variant, not canonical NPR (~44 MB, ResNet-based). Canonical weights live on the
`github.com/chuangchuangtan/NPR-DeepfakeDetection` Google Drive. Handed to Umar.
**Caveat recorded:** NPR is ProGAN/ForenSynths-trained, so it may not separate on FF++
face-swaps — which is what Pilot S measures, not a defect.

Built (all NPR-independent, verified):

| File | Role |
|---|---|
| `training/networks/cec/spectral_ops.py` | notch_hf / checkerboard_supp / residual_renorm, ported verbatim from `pilot1_spectral.py`. Model-free. `INTERVENTIONS` maps codebook intervention_type → op |
| `cec/registration/detectors.yaml` | `freqnet` added under `detectors:` (instrument: spectral). GenD routes it by the "weights/FreqNet" checkpoint string — same wrapper, on disk, no download |
| `training/detectors/cec_instrument_detector.py` | `cec_freqnet` registered; class docstring generalized (spatial + spectral). The wrapper is instrument-agnostic |
| `cec/instruments/spectral.py` | `SpectralInstrument` adapter: `intervene()`, `p_fake`, `necessity_margin(s)`. Lazy-load; raises a clear message if `cec_npr` is asked for before the download |

### FreqNet smoke — PASSED (wiring), with a substantive observation
`cec/scripts/smoke_spectral.py` on Deepfakes/000_003/000:
```
FreqNet loaded. p_fake=0.0062  p_real=0.0000  (fake>real, but both ≈ real)
spectral_notch          NM=+0.0062
checkerboard_suppress   NM=+0.0062
residual_renormalize    NM=+0.0000
```
Wiring is correct — FreqNet wraps, scores, and all three interventions build well-formed
256³ uint8 crops with finite margins. **But FreqNet calls this FF++ fake essentially real
(p_fake≈0.006), so the intervention margins are ≈0.** This is the GAN-training caveat made
concrete: FreqNet has little to grab on an FF++ face-swap. Early signal that the spectral
arm will lean on **DF40's generative families** (StyleGAN/diffusion, strong frequency
fingerprints), not FF++ swaps — consistent with the spectral-only DF40 scope decision.
Pilot S will quantify this per-detector on ~100 fakes + 100 reals.

### Status → next
TASK 5 scaffold **done**; PILOT S blocked on: (1) 🔴 NPR download, then wire NPR arch from
`NSG-VD/models/npr.py` + pin it; (2) the Pilot S data run itself (🔴; AUC + intervention
response vs CLIP on ~100+100, on FF++ and DF40). Gate: pass → dual-instrument; fail →
pre-committed fallback (single instrument + spectral characterization, or the DISCERN
spectral-bank linear probe). Everything through Task 4 is done and verified.

**Pilot S decisions (Umar, 2026-07-16):** (1) run on **both FF++ and DF40** — FF++ for the
CLIP-separates/spectral-may-not contrast, DF40's generative families for where a frequency
detector should separate at high AUC. (2) **I run it, Umar checks** the AUC + intervention
numbers (same arrangement as the Task 4 acceptance). Both still gated on the 🔴 NPR download.

---

## 2026-07-16 — NPR wired + PILOT S run. Gate looks like a FAIL — Umar's call.
NPR: arch ported to `training/networks/cec/npr.py` (truncated ResNet-50, Bottleneck [3,4],
fc1 512→1; image 4-D forward). Loads with **zero key mismatch** (build_npr_model raises on
any mismatch — it didn't). Registered `cec_npr` (loader: npr), pinned in `pins.yaml`.
Built `cec/data/df40.py` (generative-family loader, resize→256 with a native-resolution
audit) + `cec/pilots/pilot_s/run_pilot_s.py`. Results in `results/pilot_s/pilot_s.json`.

### DF40 resolution confound — CAUGHT (would have faked a 1.0 spectral AUC)
DF40 generative families ship reals at a DIFFERENT native resolution than fakes:
stargan/starganv2 256↔256 (matched); **CollabDiff 512↔218×178, MidJourney 1024↔256
(confounded)**. A uniform resize then injects a different interpolation fingerprint into
each class, and a frequency detector separates on THAT, not the generator. The loader
records native resolutions; only matched families feed the `df40:matched` headline.

### Arm (i) — AUC (fake vs real)
```
dataset          effort(CLIP)  freqnet   npr
ffpp                 0.980      0.510    0.511
df40:stargan         1.000      0.653    0.497
df40:starganv2       0.812      0.569    0.410
df40:matched         0.875      0.609    0.454     <- clean headline
df40:CollabDiff ⚠     0.840      1.000    1.000     <- RES-CONFOUND, not real
df40:MidJourney ⚠     0.528      1.000    1.000     <- RES-CONFOUND, not real
```
**Frequency detectors do NOT separate on clean face crops** (freqnet 0.61, npr 0.45 =
below chance). FreqNet/NPR are ProGAN/ForenSynths-object-trained and don't transfer to
tight c23 face crops. CLIP (effort) separates everywhere clean (0.88–0.98). The two 1.000s
are the resize confound, not signal.

### Arm (ii) — intervention response (mean Δp on fakes; CLIP should be ≈0)
```
             effort            freqnet           npr
ffpp   notch/chk  -0.02/-0.03   +0.01/+0.14   -0.26/-0.46
df40   notch/chk  -0.04/-0.02   +0.03/-0.63   -0.87/-0.60
```
Freq detectors react to spectral ops while CLIP stays flat — BUT the sign is **negative**
(interventions RAISE p_fake), i.e. the ops inject detectable artifacts rather than remove a
cited cue. For certification you need removing the cue to LOWER p_fake. Residual arm reads
+0.000 (self-reference simplification; uninformative this run).

### Verdict: dual-instrument gate FAILS as instantiated with FreqNet + NPR
Off-the-shelf GAN-trained frequency detectors don't work on our face distribution.
Pre-committed fallbacks (CURRENT_STATE ledger 3): (a) single instrument + spectral as
characterization; (b) DISCERN spectral-bank linear probe (trained on our data — the most
likely rescue, since domain-matched). **🟡 Umar decides.** All spatial work (Tasks 0–4)
stands and is unaffected — CLIP is the strong instrument and owns the spatial gate.

---

## 2026-07-16 — Fallback chosen (Umar): DISCERN spectral-bank probe. FF++ first read POSITIVE.
The DISCERN forensic/spectral bank is precomputed on our FF++ crops
(`.../c23/forensic_features/*.pt`, 83-d/frame; both fakes AND reals — 2000 fake videos,
9640 real). The DCT/spectral subset is 6 dims (indices 20–25: `ff_dct_hf_{skin,eye,mouth,
nose}`, `ff_dct_ratio_{eye,mouth}_skin`). Built `cec/pilots/pilot_s/probe_discern_spectral.py`:
logistic regression, per-video mean features, **identity-grouped** train/test split (no
source id in both — else it separates on identity, not artifact).

### Result (FF++, identity-grouped test, 2020 test videos)
```
full 83-d forensic bank : AUC 0.968      (mixes grad/blur/sym/color — NOT purely spectral)
DCT-only 6-d spectral   : AUC 0.765      (the fair spectral number)
  vs off-the-shelf: FreqNet 0.510 · NPR 0.511 · vs CLIP reference 0.980
```
**The domain-matched probe works where the transferred GAN detectors did not.** Pure
spectral DCT separates at 0.765 vs ~0.51 for FreqNet/NPR — the spectral signal on our face
data is real and learnable, just not captured by ProGAN-trained off-the-shelf detectors.

### What this does and does NOT establish
- DOES: a domain-matched spectral instrument is viable in principle (FF++, arm i).
- Does NOT: (1) the DF40 generative-family test (the real spectral use case) — the bank is
  NOT precomputed for DF40, and the extractor needs `mediapipe`+`insightface` (both MISSING
  from `GenD`; installing risks the frozen-anchor env → a 🔴 decision, not done blindly).
  (2) intervention response (arm ii) — needs the extractor callable on intervened crops.

### Status → next (🟡 Umar)
Decision open: how far to push the probe. Options — (A) install mediapipe+insightface in an
ISOLATED env, extract the bank on res-matched DF40 families (stargan/starganv2 + source a
res-matched diffusion family), re-run arms (i)+(ii) → complete the gate properly; (B) adopt
"domain-matched spectral probe" on the FF++ evidence and defer DF40 to the audit; (C) treat
the FF++ DCT 0.765 as sufficient characterization and run spectral as characterization-only
(single-instrument certification). Spatial pipeline (Tasks 0–4) unaffected regardless.

---

## 2026-07-16 — Umar chose (A): extract DISCERN bank on DF40 in isolated env.
**No new env needed.** The `dfb_nesy` env (py3.10, torch 2.9.1+cu128, CUDA) already has
mediapipe + insightface + the DISCERN extractor deps, and is fully isolated from `GenD`
(torch 2.8.0) — so the frozen anchors are untouched. Checked all 8 envs; `dfb_nesy`,
`dfb_nesy_facebench`, `gpu_env`, `ide_net` all have the deps.

**Res-matched DF40 families for the clean test:** stargan + starganv2 (fake+real both 256,
same DF40 cropper). GAN families. The diffusion families are deferred — CollabDiff/MidJourney
are res-confounded, and the ff/cdf diffusion subsets (ddim etc.) have an inconsistent
real-pool layout (ddim/cdf holds only `*-real` dirs, no co-located fakes) that needs
careful pairing before it yields a clean number. Flagged, not forced.

**Built (this is the CEC use case — train on available, certify on unseen families):**
| File | Role |
|---|---|
| `cec/pilots/pilot_s/extract_df40_forensic.py` | runs DISCERN `ForensicPrecomputer` (SegFormer parse + DCT/forensic) on stargan/starganv2 fake+real. dfb_nesy env |
| `cec/pilots/pilot_s/probe_transfer_df40.py` | trains the spectral probe on FF++, tests AUC on DF40 GAN families (full-83d + DCT-6d) |

Extraction running (background). Next: run the transfer probe → arm (i) DF40 AUC; then
build arm (ii) intervention response (re-extract bank on notch/checkerboard'd fakes, check
the probe score DROPS). Gate call is Umar's on the DF40 numbers.

### RESULT — the spectral probe does NOT transfer. Gate fails at arm (i).
Extracted stargan/starganv2 banks (150 fake + 150 real each, 83-d). FF++-trained probe,
tested on DF40 GAN families:
```
family        full-83d AUC   DCT-6d AUC
stargan          0.538          0.490
starganv2        0.499          0.523
```
**All at chance.** The FF++ separation (0.765 DCT / 0.968 full) was in-distribution
overfitting to FF++ face-swap artifacts — it does NOT generalize to unseen generative
families, which IS the CEC use case (certify on unseen families within known families).
Arm (ii) not run: a probe at chance cannot certify via intervention, so the gate already
fails at arm (i).

### The whole spectral investigation, together
| instrument | FF++ AUC | DF40 GAN AUC | verdict |
|---|---|---|---|
| FreqNet (off-shelf) | 0.510 | 0.61 | doesn't separate on faces |
| NPR (off-shelf) | 0.511 | 0.45–0.65 | doesn't separate on faces |
| DISCERN probe (DCT) | 0.765 | 0.49–0.52 | FF++-overfit, no transfer |
| DISCERN probe (full) | 0.968 | 0.50–0.54 | FF++-overfit, no transfer |
| **effort / CLIP (contrast)** | **0.980** | **stargan 1.00 · starganv2 0.81** | **transfers robustly** |

The spectral instrument fails to generalize in every instantiation; the CLIP-spatial
instrument transfers well. **Recommendation: option (C) — single-instrument certification
(CLIP-spatial), spectral reported as characterization only** (the pre-committed
single-instrument fallback, CURRENT_STATE ledger 3). Whole_face frequency/noise claims
become UNTESTABLE (honest abstention). 🟡 Umar's call. Spatial pipeline (Tasks 0–4)
unaffected and remains the certification engine.

---

## 2026-07-16 — "One more angle" (Umar): in-domain DF40 probe. Confirms spectral is dead.
Tested whether the spectral signal exists in DF40 AT ALL, independent of the FF++→DF40 gap
(`probe_indomain_df40.py`, on the stargan/starganv2 banks):
```
test                      full-83d   DCT-6d(spectral)
within stargan               0.756       0.525
within starganv2             0.706       0.487
cross stargan->starganv2     0.571       0.502
cross starganv2->stargan     0.642       0.504
```
**Pure spectral (DCT) is at chance even IN-DISTRIBUTION** (0.49–0.53) → hypothesis H1: the
generative fingerprint is simply not in the DISCERN DCT bank for these faces. The full
83-d bank has modest within-family signal (0.71–0.76) but that is NON-spectral (blend/
gradient/symmetry/color forensics), family-specific, and decays cross-family (0.57–0.64).
Neither rescues a spectral instrument.

### Conclusive across every angle tried
off-the-shelf freq detectors (fail on faces) · domain-matched probe FF++→DF40 (chance) ·
domain-matched probe in-domain DF40 (DCT chance; full-bank weak/non-spectral/family-specific).
CLIP-spatial transfers robustly throughout (DF40 stargan 1.00). **The spectral instrument
cannot be salvaged for the CEC use case.** Recommendation stands and is now evidence-backed:
**option (C) single-instrument (CLIP-spatial) certification + spectral as characterization**
(pre-committed fallback, CURRENT_STATE ledger 3). Awaiting Umar's confirmation to adopt and
proceed to Task 6 (proposer).

---

## 2026-07-16 — Generalization pilot added (Umar): CelebDF-v2 + DFDC AUC matrix.
Umar's call — and a sound one: FF++/DF40 alone don't make the cross-dataset generalization
case, and the framework anchors certification to ONE frozen detector, so that choice must
be made on the field-standard benchmarks (Celeb-DF-v2, DFDC), not FF++.

**On-disk recon:** both preprocessed (`preprocessed/{Celeb-DF-v2,DFDC}`), frames + landmarks
+ forensic_features, **no masks** (same as DF40). So the region-repair certification gate
cannot run there; **detector AUC (fake vs real) is the feasible + decisive test** — it
picks the certification instrument, and the gate itself is already byte-verified on FF++.
Labels: CelebDF `List_of_testing_videos.txt` (0=fake, 1=real; 517 clips); DFDC
`metadata.json` `is_fake` (balanced 2500/2500, 4704 with frames).

Built `cec/pilots/pilot_gen/run_generalization.py`: AUC for all four spatial detectors
(effort/fsfm/gend/forada) on ffpp (in-domain ref) + celebdf_v2 + dfdc, random balanced
samples, middle frame per clip, seed 0. Running (background). Result → the instrument
choice, decided on cross-dataset evidence.

### RESULT — fsfm is the best certification instrument
```
dataset       effort   fsfm    gend   forada
ffpp (ref)    0.996   0.996   0.999   0.989
celebdf_v2    0.881   0.894   0.857   0.850
dfdc          0.784   0.825   0.796   0.759
```
All near-ceiling in-domain (uninformative). Cross-dataset, **fsfm wins BOTH** (CelebDF
0.894, DFDC 0.825); effort close 2nd; gend/forada trail. Notably **forada has the largest
GT-repair necessity margin (0.839) but the WORST generalization** — leading certification
with it would have overfit FF++ and failed on unseen data. fsfm combines best generalization
+ a strong margin (0.713, 2nd). **Decision: fsfm is the primary certification instrument**;
effort the secondary. All four remain available as verdict-producers per the design; fsfm
leads the headline. Cross-dataset AUC 0.83–0.89 is solid for frozen detectors.

### Components now SETTLED (the "solid combination" Umar asked for)
- **Verdict + accuracy:** frozen CLIP-family panel; **fsfm primary** (best generalizer),
  effort secondary. gend/forada available.
- **Certification instrument:** spatial region-repair gate (Task 4, byte-verified). Single
  instrument — spectral dropped (doesn't generalize, three ways).
- **Spectral:** characterization only; whole_face frequency/noise claims → UNTESTABLE.
- **Proposer:** InternVL3-8B + Qwen2.5-VL-32B (pinned, on disk). → TASK 6 next.

---

## 2026-07-16 — TASK 6 (proposer + validator) built. Pipeline verified 5/5.
Built:
| File | Role |
|---|---|
| `cec/proposer/validate.py` | ~30-line validator. Parses JSON, coerces to codebook: artifact∉vocab→OTHER, location∉regions→None, routes each claim spatial/spectral/untestable. schema_valid iff parses + decision∈{real,fake} |
| `cec/proposer/infer.py` | `Proposer.propose(image)` — two backends (InternVL3 `.chat()` trust_remote_code; Qwen2.5-VL processor+generate), T=0/seed 17, disk cache keyed (image, model, prompt-hash) |

**Validator unit-tested** (no model): clean/fenced/garbage/missing-decision all correct;
routing verified (jawline→spatial, whole_face+frequency→spectral, null/OTHER→untestable).

**Live probe — InternVL3-8B, 5 FF++ images (3 fake, 2 real): 5/5 schema-valid.**
Two design-confirming observations:
1. **MLLM verdict near chance** — called a real image "fake" with claims identical to a
   fake's. This is CURRENT_STATE's "MLLM-as-verdict near chance (hence proposer-not-judge)"
   made concrete. Harmless: the frozen detector owns the verdict; the MLLM only proposes;
   the gate certifies only causally-load-bearing claims.
2. **Proposer over-uses `texture_anomaly/whole_face`** → routes to UNTESTABLE (texture isn't
   spectral, whole_face isn't spatial). Exactly the proposal-quality pattern the DPO arm
   (Task 11) is designed to fix — teach the proposer to cite testable evidence.

### Status → next
TASK 6 pipeline verified at small scale. Remaining for the Task 6 gate: the **100-image
schema-validity probe (≥95%, one prompt revision allowed)** — 🔴-ish run; the 32B is slow.
🟡 scope: both proposers or InternVL first. Then TASK 7 (certification gate → records),
now single-instrument spatial (fsfm primary), spectral claims → UNTESTABLE.

### TASK 6 GATE — PASSED, both proposers (100 FF++ images, seed 0)
```
proposer            schema-valid   claims   routes
internvl3-8b        99/100 (99%)   150      spatial 79, untestable 71
qwen2.5-vl-32b      100/100 (100%) 52       spatial 32, spectral 7, untestable 13
```
Both clear ≥95% — no prompt revision used. Route distribution previews the DPO pool:
**InternVL verbose+noisy** (1.5 claims/img, ~47% untestable — over-uses whole_face/no-locus);
**Qwen-32B conservative+targeted** (0.52 claims/img, ~62% spatial/testable). With spectral
dropped, Qwen's 7 spectral claims also → untestable. InternVL = bigger candidate pool, Qwen
= cleaner fraction; Task 9 audits both, decide the headline proposer then. This is exactly
the proposal-quality signal the DPO arm (Task 11) will optimise.

### Status → next
**Tasks 0–6 done and verified.** Component combination settled (fsfm-primary single-instrument
spatial certification; spectral characterization-only; both proposers pass validity). Next:
**TASK 7** — `cec/gate/certify.py`: per (image, claim) route → repair cited region vs the
instrument → NM → controls → CERTIFIED/REJECTED/UNTESTABLE → per-image Certified Evidence
Record (JSONL). Single-instrument (spatial, fsfm primary); whole_face/no-locus → UNTESTABLE.
Then Task 8 (assembly + abstention), Task 9 (audit + FREEZE).

---

## 2026-07-16 — TASKS 7, 8, 10 built + verified. Pilot 1L PASSES.
**Task 7 — certification gate** (`cec/gate/certify.py` + `cec/records/store.py`): per
(image, claim) route → repair cited region vs instrument → NM → controls → CERTIFIED/
REJECTED/UNTESTABLE → per-image Certified Evidence Record (resumable JSONL). Single spatial
instrument (fsfm); spectral/no-locus → UNTESTABLE. Orig+variants scored in one batch.

**Task 8 — assembly** (`cec/assembly/output.py`): verdict passthrough · rationale = CERTIFIED
claims only · empty → `rationale_ungrounded` abstention · untestable surfaced as a count.
Unit-tested: certified→shown, rejected→excluded, none→abstain. The hallucination-free
guarantee holds by construction.

**Gate end-to-end (Deepfakes, fsfm):** all single-region claims REJECTED (NM≈0) → abstention.
Correct, not a bug: repairing one landmark region of a full-face swap leaves the rest fake.
This IS CURRENT_STATE's "14/376 single-region necessity on full-face swaps".

**Task 10 — PILOT 1L** (`cec/pilots/pilot_1l/`, fsfm, 40 fakes/method):
```
method            ceiling NM   best-region NM   DPO pool   dominant regions
Deepfakes           0.776         0.009            5
FaceSwap            0.741         0.013            6
NeuralTextures      0.701         0.151           47        mouth 24, jawline 9, cheeks 11
```
- Ceiling NM (GT-mask repair) 0.70–0.78 = matches the frozen fsfm pilot (0.713) → gate scoring correct, certification achievable.
- Full-face swaps: recovery ~1–2%, pool 5–6 → single regions not necessary (documented).
- **NeuralTextures: pool 47 from 40 images, dominated by MOUTH (24)** — exactly where NT
  manipulates. Causally sensible, load-bearing. **DPO positive pool is viable** (a full NT
  run yields hundreds). Recovery ≥70% only 13.9% (landmark regions < full manipulation), but
  the operative metric — certified-claim count — passes. **Pilot 1L GATE: PASS.**

Design validated: certification works, abstains honestly on full-face swaps, and the DPO
pool comes from localized manipulations as predicted. Next: Task 9 (audit harness built,
`cec/scripts/run_audit.py`) + Task 11 (DPO) + Task 12 (eval).

---

## 2026-07-16 — TASKS 9, 11, 12 built + run end-to-end. DPO hits the pre-committed 🟡.
All remaining components built and composed end-to-end:
| Task | File | Status |
|---|---|---|
| 9 audit + freeze | `cec/scripts/run_audit.py` | RAN: reval PASS (offset 0.013), 60 NT images, freeze hash `49ffffc4f486bd8b` |
| 11 DPO prefs | `cec/dpo/build_prefs.py` | ran on records |
| 11 DPO train | `cec/dpo/train_dpo.py` | built; import-guarded (trl NOT installed anywhere — isolated-env install flagged, never GenD) |
| 12 eval tables | `cec/eval/tables.py` | ran on records |

Bug found+fixed: audit passed `pair.fake` (ndarray) to the path-based proposer → added
`PairSample.fake_path`.

### Audit result (InternVL3-8B, fsfm, 60 NeuralTextures) — THE finding
```
eval table:  60 imgs · CERTIFIED 0% · REJECTED 52% · UNTESTABLE 48% · ABSTENTION 100%
prefs:       0 preference pairs  → 🟡 scarce pool
```
Cited locations: **whole_face 32, face_boundary 26, inter_ocular 7, nose 1, mouth 1.**
NeuralTextures manipulates the MOUTH (Pilot 1L certified mouth 24/47), but the untuned
InternVL cites the mouth **once in 67 claims**. It proposes plausible-but-inert regions
(whole_face→untestable, face_boundary→rejected); the gate correctly rejects/abstains on all.

**This is the framework's thesis confirmed AND a DPO bootstrap obstacle:**
- Thesis: MLLM proposes causally-wrong evidence; gate refuses to certify → 100% honest
  abstention, zero hallucinated rationales.
- Obstacle: the pre-committed pairing (certified≻rejected from the SAME proposer output)
  needs certified claims the untuned proposer doesn't produce → 0 pairs. The pre-committed
  **scarce-pool 🟡 (Implementation v2 Task 11) fires exactly as designed.**

### 🟡 DECISION (Umar) — how to construct DPO pairs given the inert-region bias
- (A) **Oracle-positive pairs**: chosen = a gate-CERTIFIED region for that image (the gate
  found the mouth in Pilot 1L), rejected = the proposer's inert cited region (face_boundary).
  Directly teaches "cite the mouth, not the boundary." Arguably MORE faithful to the stated
  novelty ("the gate's labels train the proposer") — the gate's labels are the supervision
  whether or not the proposer already found them. Needs a build_prefs mode change.
- (B) Try Qwen-32B (more targeted: 62% spatial in the validity gate) — may cite the mouth more.
- (C) Pre-committed fallback: report scarce pool, DPO does not run; publish the audit +
  abstention result (which is itself a strong hallucination-free finding).

**Everything is built and verified; this is the one genuine open decision.** Tasks 0–12
code complete; spatial certification + audit + assembly + eval all run end-to-end.

---

## 2026-07-23 — T13–T18 implementation (addendum CLAUDE_CODE_CEC_T13_T18.md)
Executing the addendum that resolves the region-gate blocker + completes the training arm.

### T13 — scope routing + region-gate calibration  [code done; calibration RAN]
- **Scope routing** (`vocab.claim_scope`, `certify.py`): each claim routes by location →
  **region** (landmark → `gate_region:`), **composite** (whole_face + non-spectral →
  repair the inner-face union → frozen `gate:`), **spectral** (whole_face + freq/noise →
  UNTESTABLE), **untestable** (no locus). This reclassifies the 32/67 whole_face claims from
  the old audit as testable (composite). `masks.composite_mask` = inner-face ellipse.
- **Drift-proofing:** extracted `CertificationGate.measure_region()` — the single source of
  the counterfactual measurement, shared by the gate AND the calibrator (plan's key risk).
- **Failure instrumentation:** each claim record now carries `scope`, `checks{margin,gap,
  wrong,offset}`, `fail_reasons[]`, `thresholds{}` (additive; readers unaffected).
- **`loader.py`** accessors are scope-aware (`gate` vs `gate_region` block; KeyError on missing).
- **Calibration RAN** (`calibrate_region_gate.py`, fsfm, train split, 780 region samples):
  pooled-control p99 = 0.0485 → margin floor 0.10 binds; control-contrast p99 = 0.0078 →
  **gap_region = 0.05** (was the impossible 0.20); wrong 0.05; offset 0.20. Written to a NEW
  `gate_region:` block; **frozen `gate:` byte-unchanged.**
- **Finding (important):** FF++ GT masks do NOT localize reenactment manipulations — region
  IoU with the GT mask maxes at **0.21**, so the IoU≥0.30 "true region" label found ZERO
  true regions. The manipulation is localized by the **NM signal**, not the mask: on
  NeuralTextures the **mouth median NM = +0.085**, an order of magnitude above every other
  region (cheeks 0.015, rest ≈0) — causally exactly right (NT edits the mouth). Because the
  0.10 margin floor sits just above that median, region certification is **partial**.
  🟡 OPEN: keep the 0.10 floor (doc T13b spec) vs drop to the control-p99 value 0.05 (same
  p99-of-controls rule the frozen gate used). Not lowering unilaterally; Pilot 1L rates inform it.

### T14b — DF40 composite-pairing probe: GATE **FAIL** (pre-committed outcome)
DF40 `ff` landmarks are 81-pt in CROP space, so the raw re-crop trick can't apply; the
crop-to-crop affine variant gives median bg-MSE **~11,000** (QC ≤ 60) on simswap/inswap/
faceswap — DF40 frames were re-sampled independently, so same-index FF++ frames don't
register. **DF40 stays out of the gate; no re-preprocessing of the 159 GB tree** (exactly
as originally recorded). Documented limitation. Blocks nothing.

### Built (import-verified), pending 🔴 full runs
- T14: `pilot_1l` rewritten to call the REAL gate (kills Error-1 overcount), video-level
  sampling (`qc.frames_per_video: 8`), region vs composite regimes, fail-reason breakdown.
- T15: prompt revision — abstention form `{"verdict_support":"no_certified_evidence",
  "examined":[...]}` (new hash `126b045a663e1340`; old `3950a62e3b6d8ea8`); validator parses
  it (`Claim.is_abstention`, `ValidatedResponse.abstained/manipulation_claims`);
  `build_real_pair` for the real-image path; three-case `build_prefs` (fake-certified /
  real-abstention / weakfake-abstention; tiers region≻composite≻rejected; never empty chosen);
  `eval/metrics.py` FP-claim rate + coverage (raw, pre-suppression).
- T16: `run_audit.py` — `--group localized|fullface|all`, `--split`, video sampling, real
  images 1:1 (`split_label`, `n_manipulation_claims`), `family` provenance, `--oracle`
  proposer→certifiable-region hit-rate.

### T14 — Pilot 1L (fixed gate, test split, 8 frames/video): the blocker is RESOLVED
```
method            regime      frame-cert   video-cert   dominant fails
NeuralTextures    region        0.342        0.867       margin+gap (non-mouth regions)
Face2Face         region        0.146        0.400       margin+gap
Deepfakes         composite     0.667        0.933       offset/gap/margin (mixed)
FaceSwap          composite     0.725        1.000       mixed
DeepFakeDetection composite     0.000        0.000       (ids not in FF++ test.json split)
```
Old audit = 0 CERTIFIED / 100% abstention. Now: region claims certify on localized methods
(NT 87% video), composite claims certify on full-face swaps (DF/FS 93-100% video). Fail
breakdown is margin+gap dominant on region regime — i.e. the non-load-bearing regions
correctly fail; only genuinely-necessary ones pass. **Both scopes work.** (DeepFakeDetection
= 0: its video ids are outside the FF++ train/test split json; DF/FS carry the composite arm.
DFD would need its own split list — noted, not blocking.)

### Audit-chain smoke (T15/T16): composite scope reclaims the whole_face claims
First smoke (internvl3-8b, localized, tiny): 11 CERTIFIED / 26 claims — ALL 11 are
**composite** (whole_face texture, NM 0.17-0.85), the exact claims the OLD gate discarded as
UNTESTABLE. T13a's "reclassify the 32/67 whole_face claims as testable" is realized.
Bug found+fixed: the real-image loop read `pair.fake_path` (a bogus manipulated_sequences
path for reals) → FileNotFoundError. Added `PairSample.image_path` (youtube frame for reals,
manipulated frame for fakes); `run_audit` uses it. Re-running to verify the real path +
three-case prefs + FP-claim metric.

### T15/T16/T18 — full chain verified (smoke, internvl3-8b, localized, 28 records)
Real-image path fixed (`image_path`); audit reached FREEZE (smoke hash `a4cdbd0b63dbf21e`),
14 fake + 14 real records.
- **build_prefs (three-case):** 20 pairs — fake_certified 11 (composite), real_abstention 7,
  weakfake_abstention 2. All three cases populate; tiers + contentful abstention target work.
- **metrics:** FP-claim **50%** (7/14 reals asserted a manipulation) paired with coverage
  **78.6%** (11/14 fakes certified). The FP↔coverage pair the DPO arm must improve (push FP
  down, hold coverage). Reported as a pair (abstention-only is not the objective).

**All T13–T18 code is built and verified at smoke scale.** Remaining = 🔴 UMAR-RUNS full
runs: (a) region-gate calibration for effort/gend/forada (only fsfm calibrated); (b) full
audit both proposers `--group all --split test --videos 40 --oracle`; (c) re-freeze +
commit records; then T18 pool count vs the ≥500 trigger.

### 🟡 OPEN decisions for Umar
1. **Region margin floor 0.10 vs 0.05.** Pilot 1L: NT mouth median NM 0.085 sits just under
   the 0.10 floor, so region cert is partial (NT 34% frame / 87% video). Dropping to the
   control-p99 value (0.05) — the same p99-of-controls rule the frozen gate uses — would
   certify the mouth directly. Doc T13b says keep 0.10. Composite scope is unaffected (it
   carries most certifications anyway). Decision pending.
2. **DeepFakeDetection** ids are outside the FF++ train/test split json → 0 samples. Add a
   DFD-specific split list, or drop DFD from the composite set (DF/FS suffice)?
3. **T17** (`trl` in isolated env `cec_dpo`) + T18 training: only after the full re-freeze,
   and only on Umar's go if the pool clears ≥500.

---

## 2026-07-23 — T19–T22 (addendum CLAUDE_CODE_CEC_T19_T22.md)

### Decisions applied (carried in, not re-asked)
- **Region margin floor -> 0.05** (control-p99 rule, the same rule the frozen `gate:` uses;
  pooled-control p99 = 0.0485 -> ceil_0.05 = 0.05). `params.yaml` now carries
  `margin_floor: 0.05` and `sensitivity_margins: [0.05, 0.10]` — **every region table must
  report BOTH**; the T13c separation plot is the justification artifact. Calibrator floor
  updated to match.
- **DeepFakeDetection dropped** from the composite set (ids outside the FF++ split json) in
  `run_pilot_1l.py` and `run_audit.py`. Reason recorded inline.
- **effort calibrated** as secondary instrument; gend/forada skipped.

### 🔴 Blocking runs
- **effort region calibration — DONE:** 780 region samples (train). pooled-control p99
  **0.0303**, control-contrast p99 **0.0211** -> margin **0.05**, gap **0.05**, wrong 0.05,
  offset 0.20 (matches what params.yaml already carries for effort).
  Note: effort's region NMs are uniformly tiny (top region 0.018) — it barely responds to
  region repair, consistent with its smaller full-mask NM (0.479 vs fsfm 0.713). **fsfm
  remains the primary instrument for region-scope reporting.**
- **Full audit (internvl3-8b, --group all --split test --videos 40 --oracle):** 300 fake +
  300 real frames, RUNNING. Qwen-32B to follow.

### Bug found + fixed (would have corrupted the headline figure)
`calibrate_region_gate.py` wrote to a single `region_gate_calib.json` regardless of
instrument, so the **effort run silently overwrote the fsfm rows**. The first generated
Figure-1 was therefore plotting effort's profile while asserting a hardcoded "NM spikes at
the mouth" caption that effort's data does not support. Fixed: (a) per-instrument output
`region_gate_calib_<instrument>.json`; (b) Figure 1 caption is now **derived from the data**
(reports the top region and whether NM concentrates, or explicitly says it does NOT);
(c) separation plot is instrument-aware. fsfm calibration re-running to restore its rows.

### Built (import-verified)
- **T19 Disclosure Policy** (`cec/assembly/output_policy.py`, class `DisclosurePolicy`;
  performs NO certification). Three-tier: region > composite > abstain. Modes
  `certified_mode` ("certified evidence") / `screening_mode` ("causally supervised
  evidence", never "certified"). Disagreement (detector REAL + proposer asserts) flagged
  always, suppressed only when ON (default OFF for audit so FP-claim measures raw
  behaviour). **All four outcomes + both modes unit-tested**, incl. that screening_mode
  never emits the word "certified".
- **T20** `cec/registration/dpo.yaml` (frozen: LoRA r=16/β=0.1/bf16/seed 17) + final
  `train_dpo.py`. **Splits pre-committed in config**: thresholds calibrated on TRAIN, pairs
  from TEST, Pilot D evaluates on VAL — all disjoint, so the eval split cannot be chosen
  after seeing results. Refuses to train below the 500-pair trigger unless explicitly
  overridden (override is recorded). `Proposer(adapter=...)` loads LoRA; cache key includes
  the adapter so base/tuned never collide; base weights untouched.
- **T21 Pilot D** (`cec/pilots/pilot_d/run_pilot_d.py`): base vs tuned over the SAME
  held-out images through the SAME untouched gate; reports FP-claim, coverage, region vs
  composite (never pooled) + ratio, oracle hit-rate, abstention (real/weak-fake/all), claim
  diversity. **Pre-committed verdict function** unit-tested: correctly separates a genuine
  win from the "went mute" failure (coverage collapse / diversity collapse / over-abstention).
- **T22** `cec/eval/figures.py` (fig1 NM-localization, fig3 FP-vs-coverage) +
  instrument-aware separation plot. Nothing hand-entered; all regenerated from results.

### fsfm recalibration + Figure 1 (headline) — regenerated after the clobber fix
fsfm recalibration reproduced the original numbers EXACTLY (pooled-control p99 0.0485,
ctrl-contrast p99 0.0078 -> margin 0.05, gap 0.05) — the calibration is deterministic.
`results/region_gate_calib/region_gate_calib_fsfm.json` restored.

**Figure 1 (`results/figures/fig1_nm_localization.png`), caption derived from data:**
on NeuralTextures the necessity margin **concentrates at the mouth (0.085, 5.7x the next
region)** while **GT-mask IoU is diffuse and peaks at the FOREHEAD (max 0.15)** — a region
with ~zero causal necessity. The causal signal localizes the manipulation; the dataset's own
annotation points elsewhere. This is the paper's strongest single figure.
Separation plot written to `region_separation_fsfm.png`.

### Full audit (internvl3-8b) — INTERIM at 157/600 records (fakes only so far)
- **Oracle hit-rate 0 / 52.** The gate independently finds >=1 certifiable region on
  **52 of 157 images (33%)**; the untuned proposer cited such a region **0 times**.
- Claims: 70 CERTIFIED (composite) / 154 REJECTED.
This is the DPO motivation quantified: the certifiable evidence exists on a third of images
and the untuned proposer never points at it. INTERIM — confirm on completion.

### T22 tables verified (smoke records), incl. the mandated sensitivity row
```
family            scope      claims   cert@0.05  cert@0.10
Face2Face         composite       5       1.00       1.00
Face2Face         region          5       0.00       0.00
NeuralTextures    composite       8       0.75       0.75
NeuralTextures    region          8       0.00       0.00
```
Region/composite never pooled; region:composite ratio emitted. Sensitivity row is currently
flat because the binding failures are gap/wrong, not margin — watch on the full audit.

### 🔴 FULL AUDIT COMPLETE — internvl3-8b · fsfm · all 4 families · test split
**594 records (295 fake / 299 real). FREEZE hash `18a835f8bbd746d0`.**
(Prior hashes kept: original 0-certified audit `49ffffc4f486bd8b`; smoke `a4cdbd0b63dbf21e`.)

```
FP-claim rate (reals, raw)  58.2%  (174/299)
Coverage (fakes)            49.1%  (145/295)
Oracle hit-rate              0.0%  (0/62 eligible; 62/295 = 21% of images HAVE a
                                    certifiable region — the proposer never cites one)
abstention 74.7% · certified-claim NM median 0.763

certification by family x scope (SENSITIVITY margin 0.05 vs 0.10)
family            scope       claims   @0.05    @0.10
Deepfakes         composite       69   0.638    0.638
Deepfakes         region          75   0.000    0.000
Face2Face         composite       38   0.711    0.711
Face2Face         region          53   0.000    0.000
FaceSwap          composite       46   0.826    0.826
FaceSwap          region          61   0.000    0.000
NeuralTextures    composite       46   0.783    0.783
NeuralTextures    region          45   0.000    0.000
region:composite ratio = 0.0 for every family
```
**Three findings.**
1. **Region certification from proposer claims is 0.000 in every family**, while the gate
   independently finds a certifiable region on 21% of images. The certifiable evidence
   exists; the untuned proposer never points at it. This is THE motivation for the DPO arm
   and the number Pilot D must move.
2. **Composite carries all 145 certifications** (0.64-0.83 by family) — the whole-face claims
   the pre-T13 gate discarded as UNTESTABLE.
3. **The 0.05 vs 0.10 sensitivity row is IDENTICAL everywhere.** The margin choice changes
   no reported number, because proposer region claims fail on gap/wrong, not margin. This
   defuses the post-hoc-threshold critique outright — worth stating in the paper.

### T18 — DPO POOL COUNT: **389 pairs — BELOW the >=500 trigger. STOPPING.**
```
by case:     fake_certified 148 · real_abstention 169 · weakfake_abstention 72
by scope:    composite 147 · abstention 241 · region 1
by proposer: internvl3-8b 389
```
Per the pre-committed T18 rule: **do not train.** 🟡 ASK-UMAR with the escalation ladder.
Note the Qwen-32B audit has NOT yet run — it is ladder rungs (i)+(ii) combined and would
roughly double the pool while giving the per-proposer hit-rate comparison T14d asks for.

### Bug found by the Qwen audit: gate did not handle the abstention scope
The first Qwen-32B audit crashed in `certify_claim`: it handled `spectral`/`untestable`
scopes but NOT `abstention`, so an abstention-object claim (route `abstention`) fell through
to `_certify` -> `region_mask(None)` -> KeyError. Why InternVL passed and Qwen didn't:
InternVL abstained via EMPTY claim lists (no claim reaches the gate); Qwen follows the
revised T15 prompt well and emits the abstention OBJECT form, which does reach the gate.
Fix: `certify_claim` returns label `ABSTENTION` (reason=abstention, examined=[...]) for
abstention-scope claims — never touches region_mask. Verified: downstream is safe
(_manip_claims excludes reason=abstention; metrics use vr counts; tables skip non-region/
composite scopes; assembly counts only CERTIFIED/UNTESTABLE). InternVL's 594 records are
unaffected (no abstention-object claims). Qwen audit re-running with the fix.

### 🔴 QWEN-32B AUDIT COMPLETE (after abstention fix) — the opposite personality
**600 records (300/300). FREEZE hash `775d1dde3d6cbdc6`.**
```
                    InternVL-8B     Qwen-32B
coverage (fakes)      49.1%           5.0%
FP-claim (reals)      58.2%           1.3%
oracle hit-rate        0.0%           4.8%  (3/63)
region certs             0              3
```
Qwen ABSTAINS explicitly on almost everything (283 ABSTENTION claims) → honest on reals
(1.3% FP) but mute on fakes (5% coverage); it is the ONLY proposer that ever cites a
certifiable region (hit-rate 4.8%). Clean two-axis contrast: InternVL over-asserts, Qwen
over-abstains. Qwen certs: 20 composite + 3 region.

### T18 pool (pooled InternVL+Qwen) = 410 — still < 500
by case: fake_certified 163 · real_abstention 173 · weakfake_abstention 74
by scope: composite 159 · abstention 247 · region 4 · by proposer: internvl 389 / qwen 21
Qwen adds only 21 (its no-claim abstentions leave nothing to contrast). Rung (2) done, short.

### Rung (1): WIDEN — re-running InternVL audit at 65 videos/method (resumable)
Per Umar's pre-authorized "2 then 1". InternVL yields ~2.4 pairs/video; 40->65 videos scales
~389 -> ~630 pairs, clearing 500 on InternVL alone. Same test split, resumable store.

### T18 TRIGGER CLEARED — pooled pool = 648 (>= 500)
Widened InternVL audit (65 videos/method) -> store 949 records, FREEZE hash `7fc3614d1ac96745`.
Pooled InternVL+Qwen preference pool:
```
total 648   fake_certified 245 (composite 240 / region 5) · real_abstention 283 · weakfake_abstention 120
by proposer: internvl 627 / qwen 21   (InternVL alone = 627, clears 500)
```
Per the pre-committed T18 rule, >=500 -> proceed to DPO ON UMAR'S GO.
Caveat for T20/T21: pool is abstention-heavy (403/648) and fake-certified pairs are almost
all COMPOSITE (240 vs 5 region). DPO will mainly teach composite-certified >- rejected and
abstention >- hallucination (on-spec). The region-specificity SECONDARY goal has thin data
(5 region pairs) — read Pilot D's region:composite-shift metric with that in mind.

### Frozen record hashes (provenance trail)
- `49ffffc4f486bd8b`  original 0-certified audit (pre-T13; reportable history)
- `a4cdbd0b63dbf21e`  T15/T16 smoke
- `18a835f8bbd746d0`  InternVL audit, 40 videos/method
- `775d1dde3d6cbdc6`  Qwen-32B audit, 40 videos/method
- `7fc3614d1ac96745`  InternVL audit widened to 65 videos/method (the T20 training source)

### T17 — cec_dpo env (isolated; GenD MUST stay unchanged)
Decisions (Umar): (1) I create cec_dpo, verify GenD after; (2) train BOTH proposers.
GenD baseline (to verify unchanged): torch 2.8.0+cu128 · transformers 4.56.2 · numpy 2.2.6
· peft 0.14.0 · trl ABSENT. Approach: `conda create --clone GenD -n cec_dpo` (independent
copy — does NOT touch GenD) then `pip install trl` into cec_dpo only. Clone already runs
both VLMs + has peft; only trl is added.

### T17 DONE — cec_dpo created, GenD verified unchanged
`conda create --clone GenD -n cec_dpo` succeeded, then `pip install trl` into cec_dpo only.
- cec_dpo: **trl 1.9.0** + datasets 5.0.0 (pulled by trl) + peft 0.14.0 + torch 2.8.0+cu128.
- **GenD VERIFIED UNCHANGED**: torch 2.8.0+cu128 · transformers 4.56.2 · numpy 2.2.6 · peft
  0.14.0, trl still ABSENT. Isolation holds — the frozen anchors are safe.
Env python: `/data/umar/miniconda3/envs/cec_dpo/bin/python`.

### T20 — MULTIMODAL DPO (Umar: multimodal, both proposers)
Decision: image-conditioned DPO (not text-only), so the model learns "given THIS image,
prefer the certified claim/abstention." `train_dpo.py` rewritten for TRL 1.9.0 vision DPO:
`processing_class` = the VLM processor, dataset carries an `images` column + conversational
prompt/chosen/rejected. Both proposers loaded via their NATIVE transformers classes
(InternVLForConditionalGeneration / Qwen2_5_VLForConditionalGeneration — both present in tf
4.56), avoiding InternVL's custom-chat incompatibility with TRL. `resolve_image_path` maps a
record 'image' label to the frame PNG. Each proposer trains ONLY on its own pairs (a pair
reflects that proposer's output distribution). LoRA r=16/β=0.1/bf16/seed17 + gradient
checkpointing. Running a 1-step InternVL smoke to prove the vision DPO path accepts the arch
before the full run.

### T20 multimodal DPO — feasibility resolved (smoke tests)
- **Qwen-32B: multimodal DPO WORKS.** Native `Qwen2_5_VLForConditionalGeneration` + processor
  loads, the vision dataset builds, TRL DPOTrainer initialises and starts. Two trivial fixes:
  TRL 1.9.0 dropped `max_prompt_length` (removed), and wandb prompted in no-tty (set
  `report_to="none"`). Both applied.
- **InternVL-8B: multimodal DPO BLOCKED.** The pinned `OpenGVLab/InternVL3-8B` is the
  `internvl_chat` custom checkpoint; native `InternVLForConditionalGeneration` fails with an
  embedding size mismatch (ckpt 151674x3584 vs native 151936x4096), and its custom
  image-tiling processor is not compatible with TRL's vision-DPO collator.
- **THE BIND (🟡 for Umar):** the data-rich proposer (InternVL, 627 pool pairs) can't train
  multimodally via TRL; the multimodal-capable one (Qwen) has only 21 pool pairs. Options:
  (1) InternVL text-only + Qwen multimodal; (2) custom InternVL collator (hours);
  (3) grow Qwen's pool via more Qwen audit; (4) find/convert an HF-native InternVL3 ckpt.
  Awaiting Umar's direction. All state durable on /data; resume-safe.
