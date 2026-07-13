# Verified Evidence Graph, pilot findings brief, 2026-07-07

This is a self-contained summary for planning discussion. It assumes no access to the
codebase. All numbers are real measured pilot results. Nothing here is a placeholder.

---

## 1. What the project is

A deepfake forensics framework. Binary real vs fake detection is saturated, so the project
targets the two things detectors still cannot do.
- Certify that the forensic evidence a detector relies on is actually load bearing, using a
  physical counterfactual test, before any reasoning uses it.
- Attribute a fake to its generator family, including on generators never seen in training.

The design freezes all perception (a contrastive detector, a grounding VLM, a diffusion
inpainter, a CLIP encoder). It certifies each piece of evidence with an intervention test,
assembles the survivors into a Verified Evidence Graph, and runs a symbolic program over
that graph to attribute the generator family with an editable, checkable proof. Only
causally verified evidence is allowed to enter reasoning.

Two research questions.
- RQ-certify. Can forensic evidence be certified as necessary for the detector verdict,
  correctly localized, and specific, before it is used.
- RQ-attribute. Does reasoning over verified evidence attribute generators better under
  distribution shift than reasoning over unverified evidence or than a same capacity neural
  baseline.

Frozen detectors used, all third party, not tuned. effort (CLIP ViT-L/14 + SVD residual),
fsfm (self-supervised face ViT), forada (Forensics-Adapter on CLIP), gend (CLIP linear
probe). Proposer VLM, InternVL3-8B. Inpainters, SD1.5 and LaMa. Encoder for attribution
features, CLIP ViT-L/14. Datasets, FaceForensics++ c23 for verification, DF40 (40
generators) for attribution, FFHQ as a real-face corpus.

---

## 2. Phase 1, verification and a first attribution look

### 2.1 The core gate. STRONG PASS

Test. On a fake, repair the ground-truth manipulated region by pasting the paired real
pixels back with Poisson blending. Compare the drop in the detector p(fake) against an
inert corruption of the same region matched to the repair in perceptual distance (LPIPS),
so the control has equal perceptual magnitude. Add a wrong-region control and a
real-frame-offset control. 100 FF++ frames, 50 Deepfakes and 50 FaceSwap.

| detector | gt-repair drop (median) | LPIPS-matched blur | matched shift | wrong-region (med) | real-offset (med) | verdict flip |
|---|---|---|---|---|---|---|
| effort | 0.479 | -0.034 | -0.027 | -0.0003 | 0.033 | 0.52 |
| fsfm | 0.713 | -0.020 | +0.012 | 0.0001 | 0.013 | 0.74 |
| gend | 0.625 | -0.038 | -0.037 | -0.0004 | 0.032 | 0.60 |
| forada | 0.839 | -0.049 | -0.039 | 0.0000 | 0.023 | 0.80 |

Reading. On all four detectors, restoring the real region drops p(fake) by 0.45 to 0.84
while a perceptually equal inert corruption moves it by about zero. The cited region is
necessary for the verdict and correctly localized, and the effect is not detector
fragility. This is the anchor result of the whole project.

### 2.2 Deployable repair by inpainting. DEAD

The gate uses paired real pixels, which are not available at deployment, so the deployable
version tries a diffusion inpainter. On the same frames, SD1.5 and LaMa both recover about
zero percent of the gt-repair effect, and both push an inpainted real frame up to about
0.57 p(fake). A better inpainter does not help, LaMa is non-generative and fails
identically. Inpainting a face region produces synthesized content the detector flags.

### 2.3 InternVL proposal recall. GOOD, but localization unmeasurable on FF++

InternVL, told a frame is flagged, localizes the manipulation with frame-level recall 1.000
and region precision 0.947. The per-region necessity localization was unmeasurable on FF++
because full-face swaps cover nearly every region box, so mask overlap is saturated near 0.9
with no variance to correlate against.

### 2.4 Whole-face spectral verification. NULL on FF++

Notch, checkerboard suppression, and residual renormalization move p(fake) by about zero on
FF++ fakes. At the time this was ambiguous between compression and manipulation type. Phase
2 Study 1B resolves it.

### 2.5 First attribution look on DF40. SOFT

Family attribution over frozen features vs predicate features, seen vs held-out. The
predicate model was above chance and had the smallest seen-to-held gap, but trailed CLIP
and DCT on seen accuracy. Went to phase 2 for a proper re-run.

---

## 3. Phase 2, deployable verifier, scoping studies, attribution re-run

### 3.1 Pilot 1.5, retrieval-based repair. STOP

Idea. If synthesized pixels are flagged, retrieve real pixels instead. Built a FAISS index
over 8750 real FFHQ faces, retrieved the nearest real face per query, Poisson-blended its
real pixels into the manipulated region.

| operator | repair drop (median) | fraction of gt recovered | real-repair offset (median) |
|---|---|---|---|
| ground-truth Poisson | 0.479 | 1.00 | small |
| SD1.5 inpainting | -0.002 | ~0 | 0.574 |
| LaMa inpainting | +0.001 | ~0 | 0.568 |
| retrieval, real FFHQ pixels | -0.002 | -0.004 | 0.619 |

Identity-mismatch vs offset correlation was -0.187, so the failure is not identity
mismatch. Conclusion, the detector flags ANY replacement of the inner-face region, real or
synthetic, because inserting foreign content recreates the composited-face artifact that is
itself the manipulation signal. Only the true original pixels lower p(fake). No deployable
pixel-space verifier exists. Ground-truth paired repair remains a valid in-lab
certification verifier. This is a scope limit, not a failure of the core test.

### 3.2 Study 1A, inpainting offset vs region area

Inpainting a real region raises p(fake) monotonically with area. Median offset 0.024 at 1
percent of the crop, 0.044 at 3 percent, 0.204 at 8 percent, 0.489 at 15 percent, 0.59 at
full inner-face mask. Small predicate-targeted inpainting is admissible only at or below
about 3 percent of the crop, above that the operator artifact swamps the signal.

### 3.3 Study 1B, compression ladder on DF40. Resolves the spectral question

Two facts phase 1 had conflated.
1. The spectral fingerprint is real and strong. DISCERN radial-FFT spectral features
   separate real from fake at AUC 0.94 to 0.99 across GAN, diffusion, and swap groups, and
   it survives JPEG compression to q50 (GAN spectral AUC 0.989 native, 0.986 at q50). Cross-
   channel noise correlation is the most compression-sensitive, 0.86 down to 0.72.
2. The CLIP-family detectors do not use it. Spectral interventions move p(fake) by about
   zero on every family at every compression level, including GAN and diffusion at native.

So the phase-1 spectral null was detector-limited, not compression and not only
manipulation type. Consequence, spectral evidence carries strong attribution signal but
cannot be verified against a CLIP-family detector. Verifying it needs a frequency-based
detector. This is a detector-choice point.

### 3.4 Pilot 2R, attribution re-run under a cleaner family grouping. SOFT, toward STOP

Three families, face_swap, face_reenactment, entire_face_synthesis. Edit generators moved
to an abstention set. Held-out is the five test-only non-edit generators. Chance 0.400.

First cut, deterministic enrichment, phase-1 attributes plus the DISCERN forensic bank.

| baseline | seen | held-out | gap |
|---|---|---|---|
| CLIP linear probe | 0.761 | 0.548 | +0.213 |
| raw DCT | 0.689 | 0.512 | +0.177 |
| predicate count | 0.500 | 0.276 | +0.224 |
| enriched predicate MLP | 0.690 | 0.496 | +0.194 |
| retrieval kNN | 0.623 | 0.316 | +0.307 |

VLM ceiling test, adding InternVL semantic predicates (fp16, smaller sample).

| vector | seen | held-out | gap |
|---|---|---|---|
| CLIP | 0.755 | 0.629 | +0.126 |
| VLM-only, 21-d semantic | 0.464 | 0.349 | +0.114 |
| deterministic, phase1 + DISCERN | 0.630 | 0.624 | +0.006 |
| full, + VLM semantic | 0.628 | 0.599 | +0.029 |

Findings.
- Enrichment lifted seen accuracy from the phase-1 0.598 to about 0.63 to 0.69, so the
  forensic features add real signal.
- The predicate model never beats a frozen CLIP linear probe, on seen or reliably on
  held-out.
- The VLM semantic layer, which was supposed to lift the ceiling, adds nothing. InternVL
  proposes a stereotyped set of regions across generators, so it does not discriminate
  families. VLM-only is barely above chance.
- Transfer estimates are noisy. The deterministic model scored 0.496 held-out in one cut
  and 0.624 in another on the same five held-out generators. Five held-out generators is too
  few for a reliable transfer claim.

Gate reading. Not a clean GO. Matches the pre-registered soft-to-stop boundary. Predicates
carry signal and transfer competitively but do not dominate raw features, and the intended
ceiling lift failed.

---

## 4. Synthesis, what is proven, dead, and open

Proven.
- The counterfactual certification test is sound and strong. Restoring the true region is
  necessary for the verdict, correctly localized, specific under a perceptually matched
  control, on four independent detectors. This is the paper's anchor.
- The spectral forensic fingerprint is real, strong, and compression-robust as a feature.

Dead or scoped down.
- Deployable pixel-space verification by content replacement. Both inpainting and real-pixel
  retrieval fail, the detector flags any inner-face replacement. Certification is in-lab,
  using paired real repair, or scoped to sub-3-percent small-region interventions.
- Whole-face spectral verification against CLIP-family detectors. Detector-limited, would
  need a frequency-based detector.
- The attribution headline. The symbolic-over-predicates story does not beat a CLIP probe,
  and the VLM semantic layer does not help. The transfer advantage is not robustly shown.

Open.
- Whether a more careful attribution study, more held-out generators and more seeds, would
  firm up or kill the transfer claim.
- Localization on localized manipulations. The per-region necessity check was unmeasurable
  on full-face swaps. On lip-sync or edit generators, where region overlap varies, it could
  still work. Not yet run.
- Whether verifying spectral evidence against a frequency-based detector, instead of a CLIP
  detector, would let spectral predicates enter the graph.

---

## 5. Candidate directions to discuss

1. Pivot the paper to certification of evidence. Lead with the phase-1 gate, the 1.5 and 1A
   scoping, and 1B. Report attribution honestly as a negative or secondary result. This is
   the cleanest defensible paper given the evidence.
2. Rescue attribution. Re-run 2R with many more held-out generators and multiple seeds to
   settle the noisy transfer number. Consider a frequency-based verification detector so the
   strong spectral fingerprint can actually be certified and then reasoned over. Only worth
   it if the transfer number firms up.
3. Reframe as a study of what detectors actually use. The recurring result is that
   CLIP-family detectors respond to composited-face artifacts and ignore global spectral
   cues. That is itself a publishable characterization with the intervention test as the
   instrument.
4. Complete Pilot 1L, localization on localized manipulations, regardless, since it is cheap
   and independent and would either restore or bound the localization claim.

---

## 6. Caveats and infra notes for whoever plans next

- Temperature calibration of the detector, planned as step 0, was never run. All p(fake)
  are raw softmax. Monotone, so it cannot change the sign of any gate result, but it must be
  established before any headline confidence or ECE number.
- InternVL AWQ serving via LMDeploy 0.14 fails under transformers 4.56, a version
  incompatibility, so the VLM ceiling test used fp16 on a smaller sample. A clean AWQ setup
  needs a matched LMDeploy and transformers pair in a separate environment.
- Reals for Study 1B and the retrieval corpus are FFHQ, out-of-domain, since DF40 ships no
  reals. Stated as a caveat in those results.
- Attribution transfer is measured on only five held-out generators, high variance.
- Artifacts, results/pilot_rev3 (phase-1 gate, spectral, VLM, diffusion), results/pilot1_5
  (retrieval, 1A), results/study1b, results/pilot2 and results/pilot2r (attribution). Full
  write-ups, results/PILOT_RESULTS.md and results/PHASE2_RESULTS.md. Instruction specs,
  docs/PILOTS_1.md and docs/PILOTS_2.md. Pre-registration, docs/DF40_PREREGISTRATION.md and
  docs/PHASE2_PREREGISTRATION.md.
