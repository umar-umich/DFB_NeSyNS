# CEC — Findings & Build Status
*Causal Evidence Certification · DFB_NeSyNS · as of 2026-07-17*

> Companion to `docs/cec/LOG.md` (the append-only decision log). This document is the
> readable summary of what was built, what was learned, and the current open blocker.

## 1. What CEC is
A causal certification framework for deepfake detection, built inside the DeepfakeBench
checkout. A **frozen detector** owns the verdict and the accuracy numbers. An **MLLM
proposer** reads a face and emits candidate *claims* in JSON (`{artifact, location}`). A
**certification gate** tests each claim counterfactually — repair the cited region with
paired-real pixels, re-query the detector, measure the necessity margin
**NM = p_fake(orig) − p_fake(repaired)**. Only claims that causally hold are shown;
otherwise the system abstains (`rationale_ungrounded`). Gate labels later train the
proposer via DPO. Target venue: IEEE TIFS.

**Spine:** detector decides · MLLM proposes · gate certifies counterfactually · assembly
shows only certified evidence and abstains otherwise · DPO teaches the proposer to cite
load-bearing evidence · nothing trained ever touches a verdict.

## 2. Build status (Tasks 0–12)

| Task | Component | Status |
|---|---|---|
| 0 | Recon | ✅ done |
| 1 | Registration (params, pins, prompts, vocab) | ✅ frozen |
| 2 | Detector wrapper (4 instruments) | ✅ reproduces pilot to 5e-4 |
| 3 | Pairing + region masks | ✅ alignment exact to 0.0 px |
| 4 | Repair + controls | ✅ acceptance passed (Δgt 0.648) |
| 5 | Spectral + Pilot S | ✅ dropped on evidence → characterization |
| — | Cross-dataset generalization | ✅ fsfm chosen as instrument |
| 6 | Proposer + validator | ✅ both proposers pass ≥95% |
| 7 | Certification gate → records | ✅ runs; **bug found (§5)** |
| 8 | Assembly + abstention guarantee | ✅ unit-tested |
| 9 | Audit + freeze | ✅ ran (hash `49ffffc4f486bd8b`) |
| 10 | Pilot 1L | ⚠️ result invalid — used wrong criterion (§5) |
| 11 | DPO build_prefs + train | ✅ built (`trl` install pending, isolated env) |
| 12 | Eval tables | ✅ ran |

## 3. Verified positive findings

**Spatial certification chain is byte-faithful to the frozen pilot.** Pairing → region
masks → Poisson repair → control battery reproduce the rev3 pilot exactly. Task 4
acceptance: effort **Δgt = 0.648 / Δblur = +0.001** on Deepfakes/257_420/762 (matched to 4
decimals). The reproduced alignment matrix matches `align_face` to **0.0000 px**.

**Cross-dataset generalization → fsfm is the certification instrument.** AUC (fake vs real),
200 samples/dataset:

| dataset | effort | fsfm | gend | forada |
|---|---|---|---|---|
| FF++ (in-domain) | 0.996 | 0.996 | 0.999 | 0.989 |
| **CelebDF-v2** | 0.881 | **0.894** | 0.857 | 0.850 |
| **DFDC** | 0.784 | **0.825** | 0.796 | 0.759 |

fsfm wins on both benchmarks. Notably **forada has the largest necessity margin (0.839) but
the worst generalization** — leading with it would have overfit FF++.

**Proposer validity gate passed (both models).** 100-image schema-validity: InternVL3-8B
**99%**, Qwen2.5-VL-32B **100%** (≥95% required, no prompt revision used). Route mix:
InternVL verbose/noisy (1.5 claims/img, ~47% untestable); Qwen conservative/targeted
(0.52 claims/img, ~62% spatial).

## 4. The spectral instrument was dropped (Pilot S) — a real result

The dual-instrument (spectral) design failed generalization **three independent ways**:

| instrument | FF++ AUC | DF40 GAN AUC | verdict |
|---|---|---|---|
| FreqNet (off-the-shelf) | 0.510 | 0.61 | doesn't separate on faces |
| NPR (off-the-shelf) | 0.511 | 0.45–0.65 | doesn't separate on faces |
| DISCERN probe (DCT-only) | 0.765 | 0.49–0.52 | FF++-overfit, no transfer |
| DISCERN probe (full 83-d) | 0.968 | 0.50–0.54 | FF++-overfit, no transfer |
| **fsfm / CLIP (contrast)** | **0.98** | **stargan 1.00** | **transfers** |

Off-the-shelf frequency detectors (ProGAN-trained) don't work on face crops; a
domain-matched probe separates on FF++ but is at chance on unseen DF40 families (and at
chance even *in-domain* on DF40 DCT features). **Decision: single-instrument certification;
spectral reported as characterization; whole-face frequency claims → UNTESTABLE (honest
abstention).**

A methodological catch worth recording: DF40's diffusion families ship mismatched-resolution
reals (CollabDiff 512↔178, MidJourney 1024↔256) — resizing blind would have produced a
spurious 1.0 spectral AUC; only resolution-matched families (stargan, starganv2) were
trusted.

## 5. ⚠️ Two errors in the certification gate (the current blocker)

An audit of 60 NeuralTextures images returned **0 CERTIFIED / 100% abstention / 0 DPO
pairs**, contradicting Pilot 1L's "47 certifiable regions." Investigation found two real
errors.

### Error 1 — Inconsistent definitions of "certified"
`cec/pilots/pilot_1l/run_pilot_1l.py` (~line 101) counts a region using **margin only**:
```python
if nm >= margin:      # 0.10
    pool += 1
```
The real gate `cec/gate/certify.py` (lines 94–98) requires the **full battery**:
```python
certified = (nm >= self.margin                         # 0.10
    and (nm - max(nm_blur, nm_shift)) >= self.gap      # 0.20
    and abs(nm_wrong) <= self.wrong_inert              # 0.05
    and real_offset <= self.offset_max)                # 0.15
```
So "47" (margin-only) and "0" (full gate) were never comparable. Pilot 1L's result is
invalid as reported.

### Error 2 — Thresholds calibrated for the wrong operation (dominant cause)
All thresholds in `cec/registration/params.yaml` (`gate:` block) were derived from
`results/pilot_rev3/` — which repaired the **entire manipulation mask** (drops 0.65–0.84).
But the gate repairs a **single small landmark region** (drops ~0.15 max):

| method | full-mask NM (thresholds set on this) | best single-region NM (gate actually gets this) |
|---|---|---|
| NeuralTextures | 0.701 | **0.151** |
| Deepfakes | 0.776 | 0.009 |
| FaceSwap | 0.741 | 0.013 |

The gap threshold is **0.20**. A 0.151 drop **cannot satisfy `NM − max(blur,shift) ≥ 0.20`**
— region claims fail *by arithmetic*, regardless of the proposer. (Secondary: untuned
InternVL cited the mouth only 1/67 times on NeuralTextures — a proposal-quality issue, but
moot while the bar is mathematically unreachable.)

**Root cause:** we judge small region-repairs (~0.15) with a bar built for whole-face
repairs (~0.65).

### Silver lining
Both errors *validate* the framework's premise: the MLLM proposed plausible-but-inert
regions (whole_face, face_boundary), and the gate correctly refused to certify any of them →
**100% honest abstention, zero hallucinated rationales.**

## 6. Agreed fix direction (locked 2026-07-17)
1. **Re-calibrate region-level thresholds** (keep the counterfactual design; don't change
   what's repaired). Derive the bar from region-level control noise on the **train split**
   (disjoint from the test split the rev3 pilot used — codebook rule). Expected
   `gap_region ≈ 0.05–0.10 ≪ 0.20`. Add a new `gate_region:` block; leave the frozen `gate:`
   block byte-unchanged.
2. **Fix Pilot 1L** to call the same `CertificationGate` (removes Error 1).
3. **Broaden** to mask-available FF++ methods: localized (NeuralTextures, Face2Face) vs
   full-face (Deepfakes, FaceSwap, DeepFakeDetection). *FaceShifter and DF40 have no masks →
   cannot run the gate.*
4. **Run both proposers** (InternVL3-8B + Qwen2.5-VL-32B); add a proposer→oracle hit-rate.

Files to touch: `cec/gate/certify.py`, `cec/registration/loader.py`,
`cec/registration/params.yaml` (add block), `cec/pilots/pilot_1l/run_pilot_1l.py`,
`cec/scripts/run_audit.py`, new `cec/scripts/calibrate_region_gate.py`.
Full plan: the approved plan file (region-gate recalibration) + `docs/cec/LOG.md`.

## 7. Environment & data notes
- **Detectors** load via the GenD registry (`/data/umar/Repos/GenD_NeSy`); env `GenD`
  (torch 2.8.0 — **frozen, protects anchors**). Forensic extraction uses the isolated
  `dfb_nesy` env (mediapipe/insightface).
- **Proposers pinned & on disk:** InternVL3-8B (`853e3a79…`), Qwen2.5-VL-32B (`7cfb30d7…`).
- **Datasets with masks (gate-capable):** FF++ Deepfakes / FaceSwap / NeuralTextures /
  Face2Face / DeepFakeDetection. **No masks:** FaceShifter, all DF40, CelebDF, DFDC
  (AUC-only).
- **Splits:** FF++ train 360 / test 70 / val 70. `trl` not installed anywhere (DPO training
  prerequisite; install in an isolated env, never GenD).
- Full decision log: `docs/cec/LOG.md`.

## 8. Open questions
1. Is a region-calibrated gap of ~0.05 a *meaningful* causal claim, or too permissive?
   (Report true-vs-control separation, not just the threshold.)
2. Keep single-region granularity, or add a coarser "manipulation-region" claim type for
   full-face swaps (which otherwise always abstain)?
3. If the DPO pool stays small after the fix: oracle-positive pairs, or publish the
   100%-abstention hallucination-free result as-is?
