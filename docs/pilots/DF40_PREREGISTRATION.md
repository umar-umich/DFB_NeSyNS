# DF40 generator to family pre-registration, DRAFT awaiting confirmation

Status. DRAFT. This file must be committed before any Pilot 2 result is produced
(PLANNING.md section 8.2, CLAUDE_CODE_PILOTS.md hard rules, CODEBOOK.md section 5).
It is written before any attribution number exists. One open decision, marked below,
needs a human call before it is frozen.

> **2026-08-07 — DF40 train split NO LONGER ON DISK.** `/data/umar/Datasets/df40/train`
> (74 GB, 31 generators) was deleted to make room for the 109 GB DF40 test-split H5 build
> for the generation-process pilots (`docs/DiCoME_eval/README-generation-process-pilot.md`).
> This was a deliberate call made with the conflict known. **Pilot 2 cannot run as
> pre-registered until train is re-downloaded** from DF40's Google Drive (link in
> `/data/umar/Datasets/df40/dataset_download_details.txt`, "DF40 (training data)", ~93 GB).
> Nothing else about this pre-registration changes: the seen-vs-held-out generator
> assignment below is a property of DF40's published split membership, not of local files,
> so it remains valid and re-downloading restores reproducibility exactly. The test split
> is untouched and fully on disk.

Data source. `/data/umar/Datasets/df40`, the 40 generator DF40 benchmark, train and
test splits already on disk. Nine generators appear only in the test split and are the
benchmark's designated cross method evaluation set. They are used here as the held out
transfer generators, which is exactly the claim scope, unseen generators within known
families.

Assignment principle, from CODEBOOK.md section 5. Family is defined by manipulation
scope, not generative backbone. A diffusion based face swap is face_swap. A GAN based
whole face sample is entire_face_synthesis. Rationale is logged per boundary case.

---

## 1. Family mapping

Four scopes. face_swap, face_reenactment, entire_face_synthesis are the attribution
targets with both seen and held out generators. face_edit is treated as the novel
scope for the abstention test, see the open decision in section 3.

### face_swap, identity of a target face replaced, inner face scope
Seen, in train and test
- blendface
- faceswap
- fsgan
- simswap
- inswap
- uniface
- mobileswap
- facedancer
- e4s          (boundary case, see 2.1)
Held out, test only
- deepfacelab

### face_reenactment, motion or expression driven, identity preserved
Seen, in train and test
- fomm
- facevid2vid
- MRAA
- one_shot_free
- pirender
- tpsm
- lia
- danet
- sadtalker
- mcnet
- hyperreenact
- wav2lip       (boundary case, see 2.2)
Held out, test only
- heygen

### entire_face_synthesis, whole face generated from noise or latent, no source face region
Seen, in train and test
- StyleGAN2
- StyleGAN3
- StyleGANXL
- VQGAN
- sd2.1
- ddim
- pixart
- DiT
- SiT
- RDDM          (boundary case, see 2.3)
Held out, test only
- CollabDiff
- MidJourney
- whichfaceisreal   (boundary case, see 2.4)

### face_edit methods, FOLDED into entire_face_synthesis per resolved decision (Option B)
All four DF40 editing methods are test only. Per the 2026-07-02 decision they are folded
into entire_face_synthesis as additional held out generators, so the fourth codebook
family diffusion_edit is not instantiated on DF40. Held out under this fold
- e4e
- stargan
- starganv2
- styleclip

Rationale for the fold. A whole face GAN attribute edit changes global appearance and no
paired real face region is repaired, which is closer to synthesis than to a localized
region repair. The scope definition is stretched, identity is more preserved than in pure
synthesis, and this is logged as the known cost of Option B. Option C, importing a
trainable diffusion edit set to instantiate a genuine fourth family, is deferred to the
journal version.

---

## 2. Boundary case rationale, logged per CODEBOOK.md section 5

2.1 e4s. Backbone is fine grained face editing, but the manipulation scope is identity
transfer onto a target, so by the scope principle it is face_swap, not face_edit.

2.2 wav2lip. Edits only the mouth region for lip sync. Scope is a localized region, but
the operation is motion and audio driven with identity preserved, which is reenactment.
Assigned face_reenactment. Logged because it is the only region local member of that
family and may behave differently under a spatially local verification gate.

2.3 RDDM. Residual denoising diffusion. Diffusion backbone, but it generates the entire
face rather than editing a region, so by scope it is entire_face_synthesis, not a
diffusion edit.

2.4 whichfaceisreal. StyleGAN2 samples presented as a real versus fake set. Backbone is
GAN, scope is whole face synthesis, assigned entire_face_synthesis. Held out.

---

## 3. Open decision, needs a human call before freezing

CODEBOOK.md section 5 names the fourth target family diffusion_edit. DF40 does not
contain localized diffusion edits. Its four editing methods, e4e, stargan, starganv2,
styleclip, are GAN based attribute edits and all four are test only. This creates a
mismatch that changes the headline experiment structure. Three options.

Option A, recommended. Rename the fourth family from diffusion_edit to face_edit and use
it as the novel scope abstention set, not a seen attribution target. Attribution is then
evaluated over three known families each with seen and held out generators, face_swap,
face_reenactment, entire_face_synthesis, and the edit methods measure abstention on a
scope absent from training. This matches PLANNING.md section 7, which already asks for an
abstention rate on a truly novel family, and it needs no data the benchmark lacks.

Option B. Fold the edit methods into entire_face_synthesis, on the argument that a whole
face attribute edit changes global appearance. This keeps four scopes collapsed to three
and gives more held out EFS generators, but it blurs the scope definition since editing
preserves identity and layout while synthesis does not.

Option C. Keep a fourth attribution target but populate it from a different dataset that
has trainable diffusion edits, for example an inpainting or instruct edit set. This adds
a dataset and a domain shift confound and is the heaviest option.

Recommendation is Option A. It is honest about what DF40 contains, it strengthens rather
than weakens the paper by turning the all held out edit set into the designed abstention
probe, and it leaves the three transfer families intact.

---

## 4. Held out summary

Primary transfer test, unseen generators within known families
- face_swap, held out deepfacelab
- face_reenactment, held out heygen
- entire_face_synthesis, held out CollabDiff, MidJourney, whichfaceisreal

Novel scope abstention test, under Option A
- face_edit, e4e, stargan, starganv2, styleclip

Note on transfer test thinness. face_swap and face_reenactment each have a single test
only held out generator. If a stronger within family transfer test is wanted, one seen
capable generator per family can additionally be moved to held out, at the cost of train
diversity. Not done by default, flagged here so the choice is explicit rather than
silent. Any such move is recorded in this file before results exist.

---

## 5. Counts on disk, train split, for reference

Face swap and reenactment generators carry roughly 1400 to 2000 sample folders each.
Synthesis generators carry roughly 677 each. uniface is the smallest at 374. Exact per
generator counts are logged by the Pilot 2 data loader at run time and appended here
before any accuracy number is reported.

Seeds. All sampling seeded and logged. Results averaged over at least three seeds per
CLAUDE_CODE_PILOTS.md.
