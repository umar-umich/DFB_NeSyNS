# Evaluation protocol v3 (declared 2026-09-14, before any VALmix-v2 score was computed)

Supersedes v2 §1 (checkpoint selection). Thresholds (τ = EER on FF++ val for the selected checkpoint), run
naming (R0-historical kept separate), provenance and reporting rules from v2 are unchanged.

## 1. Why

VALmix (CDFv2val, DFDCPval, DFEval24val) contains no reenactment, talking-face or synthesis content, so selection
could not see arm C's failure: C-v2 had the highest VALmix score of every run while scoring below chance on
reenactment. VALmix-v2 covers all forgery types available on this server.

## 2. VALmix-v2 composition

| source (key prefix) | origin | genuine | fake subsets | forgery types |
|---|---|---|---|---|
| CDFv2val, DFDCPval, DFEval24val | existing VALmix, unchanged | as before | as before | swap, swap, in-the-wild |
| `CDFv3valG` | GenD val `CDFv3-x1.3-th0.5-all` | Celeb-real, YouTube-real | 22 generators | swap, reenactment, talking face |
| `DS11val` | GenD val `DeepSpeak-1.1` (train-* split) | real | facefusion, facefusion_gan, facefusion_live, retalking, wav2lip | swap, talking face |
| `DS20val` | GenD val `DeepSpeak-2.0` (train-* split) | real | facefusion, liveportrait, hellomeme, diff2lip, latentsync, memo | swap, reenactment, talking face |
| `FFIWval` | GenD val `FFIW` | real | fake | in-the-wild |
| `DF40train` | DF40 train split (`/data/saad/datasets/video/df40/train`) | **FF++ val genuine videos (140)** from each checkpoint's FF++ val export | 10 reenactment, 2 talking face, 10 entire-face-synthesis methods | reenactment, talking face, synthesis |

Face editing has no source outside our test sets and is **not represented**.

## 3. Forgery-type assignment

| type | cells (source: fake subsets) |
|---|---|
| **swap** | CDFv2val; DFDCPval; CDFv3valG: BlendFace, Celeb-DF-v2, GHOST, HifiFace, InSwapper, MobileFaceSwap, SimSwap, UniFace; DS11val: facefusion, facefusion_gan, facefusion_live; DS20val: facefusion |
| **reenactment** | CDFv3valG: DaGAN, FSRT, HyperReenact, LIA, LivePortrait, MCNET, TPSMM; DS20val: liveportrait, hellomeme; DF40train: danet, facevid2vid, fomm, hyperreenact, lia, mcnet, MRAA, one_shot_free, pirender, tpsm |
| **talking face** | CDFv3valG: AniTalker, EchoMimic, EDTalk, FLOAT, IP_LAP, Real3DPortrait, SadTalker; DS11val: retalking, wav2lip; DS20val: diff2lip, latentsync, memo; DF40train: sadtalker, wav2lip |
| **entire-face synthesis** | DF40train: ddim, DiT, pixart, RDDM, sd2.1, SiT, StyleGAN2, StyleGAN3, StyleGANXL, VQGAN |
| **in-the-wild** | DFEval24val; FFIWval |

## 4. Selection score

- **Cell** = one (source, type): AUROC of all that source's fake subsets of that type against that source's genuine
  videos (DF40train cells: against FF++ val genuine videos). Video-level DS replay, unrounded.
- **Type score** = mean of its cells. **Selection score** = mean of the five type scores (equal per forgery type).
- Candidates, epoch-from-file, ties within 1e-6 → earliest epoch, FF++ val AUROC as diagnostic only: as in v2.
- Script: `analysis/tbiom/select_checkpoint_v3.py` → `logs/tbiom/stage23/<arm>_selection_v3.json`.

## 5. Sampling (numpy `default_rng(0)` per subset)

- fake subsets: at most **64 videos**; genuine subsets: at most **128** per genuine folder (CDFv3valG Celeb-real and
  YouTube-real each), **256** for DeepSpeak `real`; FFIWval kept whole (250 / 250).
- **16 frames per video**, evenly spaced over the released frames. Existing VALmix exports keep their frames.
- Label rule of the loader (genuine iff subset name contains "real"): CDFv3valG `Real3DPortrait` is stored as
  **`R3DPortrait`**, otherwise it would be silently labelled genuine.

## 6. Leakage handling and exposure

- CDFv3valG videos whose name appears in our CDFv2 or CDFv3 test splits are **removed** (20 found in the audit).
- VALmix-v2 shares distribution with **CDFv2, CDFv3 (incl. CDFv3 generators), FFIW, DF40 generators and DeepSpeak**.
  Those are selection-exposed; DeepSpeak can no longer be an untouched final-test set.
- DF40train fakes are generated from our 720 FF++ **training** identities; FF++ val genuine videos use different
  identities from the same source.
- Test sets not touched by VALmix-v2: DFD, DFDC, DFDCP test, DFEval24 test, KoDF, IDForge, DFDM, PolyGlotFake.
