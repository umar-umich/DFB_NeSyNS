# Phase 2 pre-registration, written before any phase-2 result exists

Status. Committed before Pilot 1.5, Study 1B, and Pilot 2R runs, per PILOTS_2.md hard
rules. Records the reference corpus and retrieval feature definition, the Option A
generator to family mapping and held-out generators, and the DISCERN feature to codebook
attribute mapping table. Numbers are filled by the runs, never here.

---

## 1. Retrieval reference corpus (Pilot 1.5)

- Corpus. FFHQ at 256, real human faces, out-of-domain relative to FaceForensics++ so a
  retrieval can never return the paired-real source of a test fake. Source, HuggingFace
  dataset bitmind/ffhq-256, shards train-00000 and train-00001 of 16, decoded to PNG at
  256x256. Size 8750 faces. Path /data/umar/Datasets/ffhq256_subset.
- Calibration reals. The paired-real FF++ frames used in phase 1 remain the calibration
  reals for the real-repair offset, they are NOT added to the retrieval corpus, to keep
  the corpus out-of-domain.
- The index is frozen once built. It is not trained. Frozen features only.

### 1.1 Retrieval feature definition (frozen, pre-registered)

Each corpus face is treated as canonically aligned (FFHQ alignment approximates the GenD
5-point template used for the query crops). For each codebook region_id box on the 256
crop we store:
- a frozen CLIP ViT-L/14 visual embedding of the region crop (primary descriptor),
- region landmark geometry, the region box in normalized crop coordinates,
- a head-pose proxy, horizontal and vertical offset of the inter-ocular midpoint from crop
  center,
- a lighting descriptor, mean luminance and the horizontal and vertical luminance gradient
  sign over the region.
Retrieval distance is cosine on the CLIP region embedding. Geometry, pose, and lighting
descriptors are logged and used only as tie-breakers and for the identity-mismatch probe,
not trained. FAISS inner-product index on L2-normalized embeddings, one index per
region_id.

### 1.2 Retrieval repair operator

For a cited region on a query fake, retrieve the nearest real region of the same
region_id, take the retrieved real face's pixels inside the query region mask, Poisson
blend into the query masked to the face side of any seam. Real pixels, not synthesized.

---

## 2. Option A generator to family mapping (Pilot 2R)

Change from phase-1 Option B. Edits are removed from entire_face_synthesis. EFS becomes
clean synthesis only. The four DF40 edit generators are all test-only, so they cannot be a
trained target, they become the designated ABSTENTION family, matching PLANNING section 7.

Three trained families, each with a held-out generator for the transfer test.

### face_swap
Seen: blendface, faceswap, fsgan, simswap, inswap, uniface, mobileswap, facedancer, e4s
Held out: deepfacelab

### face_reenactment
Seen: fomm, facevid2vid, MRAA, one_shot_free, pirender, tpsm, lia, danet, sadtalker,
mcnet, hyperreenact, wav2lip
Held out: heygen

### entire_face_synthesis (clean, no edits)
Seen: StyleGAN2, StyleGAN3, StyleGANXL, VQGAN, sd2.1, ddim, pixart, DiT, SiT, RDDM
Held out: CollabDiff, MidJourney, whichfaceisreal

### abstention family, NOT a trained target, evaluated only for abstention behavior
e4e, stargan, starganv2, styleclip

Chance level is recomputed under three trained families and reported in the run.

---

## 3. DISCERN forensic feature to codebook attribute mapping (Pilot 2R step 3)

The enriched predicate vector concatenates the DISCERN forensic bank in
preprocessing/forensic_helpers.py with InternVL semantic predicates. The deterministic
forensic features map to codebook attributes as below. New attributes added to the
codebook now, before the run, are marked NEW.

| DISCERN feature group | codebook attribute |
|---|---|
| radial FFT slope | frequency_anomaly.radial_spectrum_slope |
| radial FFT high-frequency ratio | frequency_anomaly.hf_energy_ratio |
| FFT peak band, concentration | frequency_anomaly.dominant_band, band_concentration |
| checkerboard / upsampling peak energy | frequency_anomaly.checkerboard_score |
| SRM residual std | noise_inconsistency.residual_std |
| PPNC paired-patch noise consistency | noise_inconsistency.cross_region_noise_gap |
| CCNC cross-channel noise correlation | noise_inconsistency.cross_channel_noise_corr NEW |
| multiscale LoG noise residual | noise_inconsistency.noise_periodicity |
| region boundary gradient (SegFormer parse) | boundary_artifact.gradient_sharpness |
| regional blur | texture_anomaly.pore_detail_loss |
| left-right symmetry | geometry_inconsistency.symmetry_deviation |
| color consistency across regions | lighting_inconsistency.color_temperature_gap |
| regional DCT statistics | frequency_anomaly (regional) NEW regional tag |

SegFormer face parsing is a frozen measurement dependency only. Intervention regions still
come exclusively from deterministic landmarks, never from the parser.

### 3.1 InternVL serving for DF40-scale extraction

InternVL3-8B served via LMDeploy AWQ 4-bit. Before any 2R result, re-run phase-1 step 10,
proposal recall and precision against masks on the same 100 FF++ frames, under the
quantized model, and report the delta against the fp16 numbers, frame-recall 1.000,
region-precision 0.947, region-recall 0.325. If recall or precision degrades materially,
fall back to fp16 and report the throughput cost.

---

## 4. Seeds and reporting

All sampling and every fitted model use three seeds, mean and spread reported. p(fake)
values reuse phase-1 detectors. Temperature calibration, PILOTS_1 step 0, was not run in
phase 1, raw softmax was used, monotone so gate signs are unaffected. It is established
before any phase-2 headline confidence number.
