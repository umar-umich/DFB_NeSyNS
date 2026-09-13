# Face / video deepfake datasets available on this server (2026-09-13)

Read-only survey of `/data`. Audio-only and image-only forgery collections excluded as requested.
Counts are from directory listings (folder counts approximate videos). `/data/Shared_Datasets`
exists but is **permission-denied** for this account.

Other users' directories are listed as found; confirm with their owners (kutub, saad, awais)
before relying on their copies, and check each dataset's licence terms.

## Already in our pipeline (duplicates elsewhere, no new evidence)

FF++, Celeb-DF v1/v2/v3, DFD, DFDC, DFDCP, Deepfake-Eval-2024, DF40 test, UADFV — copies also
exist in `/data/kutub/datasets/{DFD,GEND}`, `/data/saad/datasets/video`, `/data/awais/datasets`.

## New, ready to use — `/data/kutub/datasets/GEND/extracted/` (GenD benchmark release)

256×256 face crops, one folder per video, label in folder name. Crop convention is GenD's
(`x1.3`, detector threshold 0.5), **not** ours, so pixel parity with our h5 cannot be assumed.

| dataset | split(s) | manipulation types | ≈ video folders | notes |
|---|---|---|---:|---|
| **FFIW** | val, test | face swap, in-the-wild multi-face | 3,470 | new |
| **KoDF** | test | DFL, FaceSwap(dffs), audio-driven, adversarial | 2,239 | Korean identities; new |
| **IDForge-v1** | test | InfoSwap, Roop, SimSwap (+TTS) | 21,821 | new |
| **DeepSpeak-2.0** | val, test | LatentSync, LivePortrait, MEMO, HelloMeme, Diff2Lip, FaceFusion | 3,279 | **talking-face / reenactment**; new |
| **DeepSpeak-1.1** | val, test | lip-sync, face swap | — | new |
| **PolyGlotFake** | test | VideoRetalking, Wav2Lip (lip-sync translation) | 14,367 | **lip-sync**; new |
| **FakeAVCeleb** | test | FaceSwap, FSGAN, Wav2Lip (visual-fake subsets FV-*) | 21,066 | use FV-* only |
| **FaceShifter (FSh)** | test | FaceShifter swap | 280 | small |
| **DFDM** | test | DFL, Dfaker, IAE, LightWeight | 1,718 | built on Celeb-DF identities |
| **FaceFusion** | test | HifiFace, InSwapper | 666 | built on Celeb-DF identities |
| FF, CDFv2, CDFv3, DFD, DFDC-pre, UADFD | — | — | — | duplicates of ours |
| ~~RedFace~~ | test | — | — | **image-level** (single 161×161 images) — excluded |

## New, needs preparation

| dataset | location | state | relevance |
|---|---|---|---|
| **LAV-DF** | `/data/saad/datasets/audiovideo/LAV-DF` | 136,304 raw mp4; 66,703 with visual manipulation | lip-sync content edits; temporal localisation labels |
| **LipSyncTIMIT** | `/data/saad/datasets/audiovideo/lipsynctimit` | 9,696 raw mp4, c23 and c40 | lip-sync + compression levels |
| **DeepSpeak v2 (raw)** | `/data/saad/datasets/audiovideo/deepspeakv2` | 16,640 mp4 + annotations | same as GEND DeepSpeak-2.0, uncropped |
| **FakeAVCeleb (raw)** | `/data/saad/datasets/audiovideo/FakeAVCeleb_v1.2` | 21,544 mp4 (500 real-video) | same as GEND copy |
| **WildDeepfake** | `/data/awais/datasets/DFW/deepfake_in_the_wild` | tar.gz sequences: real 371/42, fake 592/115 (train/test) | in-the-wild swaps; needs extraction |
| **DeepfakeTIMIT** | `/data/kutub/datasets/DFD/FACES/DTIM` | 224×224 faces, 32 subjects | low-quality swaps, small |
| **President deepfakes** | `/data/kutub/datasets/DFD/FACES/PDD` | 224×224 faces, Biden/Trump | tiny, narrow |
| **DF40 train split** | `/data/saad/datasets/video/df40/train` | 31 subsets | only test split used so far |
| **FaceVid-Forensics** | `/data/kutub/datasets/argus-datasets` | `videos.zip` not extracted; train/test/OOD splits with text explanations | includes fully generated video; contents unverified |

## Genuine clips with natural motion (for the temporal probe)

| dataset | location | notes |
|---|---|---|
| **VoxCeleb2** | `/data/saad/datasets/audiovideo/voxceleb2` | raw talking-head mp4, very large; unfamiliar identities and natural motion |
| real subsets of DeepSpeak, KoDF, FFIW, IDForge, FakeAVCeleb | as above | genuine clips outside FF++/Celeb-DF |

## Not relevant here

`/data/kutub/datasets/{AIGI,AIGV,MFD}` (AI-generated images/video, image forgery), audio-only
collections, GenVideo copies in several project folders.

## Recommendation

1. **Reserve untouched final-test sets now, before looking at any results on them.** Candidates
   with no overlap with FF++ training or already-exposed sets: **FFIW, KoDF, IDForge-v1**, and one
   talking-face set (**DeepSpeak-2.0** test or **PolyGlotFake**). Record the reservation in the
   decision log so nothing downstream selects on them.
2. **Temporal-probe development set**: DeepSpeak-2.0 val (LivePortrait, LatentSync, MEMO,
   HelloMeme — exactly the reenactment/talking-face failure) plus VoxCeleb2 genuine clips for
   natural motion. LipSyncTIMIT c23/c40 adds controlled compression.
3. **Dependencies to report**: DFDM and FaceFusion reuse Celeb-DF identities, so they share
   structure with CDFv2/CDFv3; FakeAVCeleb real videos number only 500.
