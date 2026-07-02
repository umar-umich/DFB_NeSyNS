# NeSy-DeFake — Causal Sub-Graphs, What's In Them, Why They Work
### State of the art as of 2026-04-28

This document is a deep-dive companion to `README_april_22.md`. The
April-22 doc explains the **interpretability pipeline as a whole** —
which file produces which figure, where the artefacts land on disk, how
each one maps to the seven-level explainability taxonomy from the
April-20 write-up. It gets you from *"the model finished evaluating"*
to *"here is the gallery of plots."*

What it does **not** do is open the four causal sub-graphs and explain
what is actually inside each one. That is the job of this document. By
the end of it you should be able to:

1. Name every node group inside every sub-graph and say in one sentence
   what each group is supposed to capture.
2. Explain why the **forensic_noise** sub-graph dominates on FF++ and
   why **identity** looks weak there — without falling for the trap
   that "weaker = useless".
3. Decide, based on numbers we already have, whether to drop, keep, or
   re-weight any sub-graph.
4. Read the new visuals (`*_graph_separated.png`, the cropped /
   math-labelled side-by-side heatmaps, the discriminative-gap plot)
   without a key.

The figure-production layer was upgraded on 2026-04-28; if you opened
an old `interpretability/` folder and figures look different from the
April-22 examples, that is expected — see Section 6.

---

## 1. One-line TL;DR

The causal branch is a small zoo of four **structural causal models**
that each operate on a different *kind* of evidence — semantic-identity
attributes, macro-structural forensics, micro-noise residuals, and
frequency-domain anomalies. They run in parallel; each produces a
`r_diff` signal saying *"how much does this sample need an SCM trained
on fakes rather than one trained on reals to be reconstructed
faithfully?"*. The branch then sums those four signals into the causal
evidence channel of the EDL fusion.

On FF++ the noise sub-graph is doing 85 % of the work. On Celeb-DF the
mix usually flips. Don't tune one ablation around one dataset.

---

## 2. The four sub-graphs at a glance

| # | Sub-graph              | Nodes                              | What it should learn on REAL faces                                                  | Code in `improved_scm_branch.py`         |
| - | ---------------------- | ---------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------- |
| 1 | **identity**           | 32 latent + 26 curated + 12 rules  | Expression ↔ AU couplings, head-pose ↔ gaze, geometric ratios, attribute coherence  | `self.identity_pair`                     |
| 2 | **forensic_structural**| 32 latent + 30 macro forensic       | Edges should be sharp; blur, colour and DCT energy uniform across regions           | `self.forensic_pairs[0]`                 |
| 3 | **forensic_noise**     | 32 latent + 43 micro-noise         | Residual noise consistent across patches and Y/Cr/Cb channels                       | `self.forensic_pairs[1]`                 |
| 4 | **forensic_spectral**  | 32 latent + 10 FFT                 | 1/f spectral fall-off, balanced radial bins                                         | `self.forensic_pairs[2]`                 |

Each row is one independent SCM **pair** — i.e. **two** small
nonlinear MLPs, `SCM_real(x) ≈ x` and `SCM_fake(x) ≈ x`, each trying
to reconstruct its own class' samples. At inference time both are run
on the same input; the difference between their reconstruction errors
is the per-sub-graph signal `r_diff`.

---

## 3. Inside each sub-graph

### 3.1 Identity (70 nodes)

The identity SCM is the only path through which **semantic** signal
reaches the causal branch. Its node-set splits into three groups:

- **32 latent** `z_0 … z_31`. A linear compression of the 1024-d CLIP
  spatial embedding of the face. The compressor is detached from CLIP
  during training, so these are "what the visual stream thinks the face
  looks like, in 32 slots."
- **26 curated facial attributes** `fs_*`, sourced from the
  fast-semantic vector and listed in
  `networks/nesy_defake/semantic/refined_attributes.py`:
  - **Demographics (2)**: `fs_gender_score`, `fs_age_score`
  - **Geometry (6)**: jaw-width / nose-width / face-width-height /
    interpupillary ratios, chin angle, brow-height ratio
  - **Expression-AU coherence (10)**: `fs_expr_happy/sad/angry/surprise`
    paired with `fs_AU1, fs_AU4, fs_AU6, fs_AU9, fs_AU12, fs_AU15`
  - **Pose-gaze (5)**: `fs_pose_yaw, fs_pose_pitch,
    fs_gaze_left_x, fs_gaze_right_x, fs_gaze_left_y`
  - **Symmetry + quality (3)**: `fs_eye_lr_symmetry, fs_jaw_symmetry,
    fs_det_confidence`
- **12 consistency rules** `cr_*`, from
  `consistency_rules.py` / `consistency_rules_v7.py`. Each rule is a
  hand-coded check that should be *near zero on a real face* — for
  example
  - `cr_happy_au6 = |fs_expr_happy − fs_AU6|` (smiling should fire AU6)
  - `cr_neutral_any_au = fs_expr_neutral · max(fs_AU{1,4,6,12,15,9})`
    (neutral faces shouldn't fire AUs strongly)
  - `cr_gaze_lr_divergence = |fs_gaze_left_x − fs_gaze_right_x|`
  - `cr_eye_asymmetry`, `cr_jaw_asymmetry`, `cr_image_quality`, …

What the identity SCM is *supposed* to learn:

> "On a real face, expression labels co-fire with the right AUs; gaze-L
> tracks gaze-R; head yaw bends interpupillary skew; jaw width and
> nose width co-vary with gender; symmetry features are tightly
> coupled. The SCM's job is to memorise those couplings on reals."

A faceswap pastes a generated face onto a host's head pose — so on
fakes those couplings are *rewired*. The fake-trained SCM has to
reconstruct samples where smiling no longer drives AU6, where gaze-L
disagrees with gaze-R because the texture came from a different frame,
where jaw geometry no longer tracks the gender label. That rewiring
shows up as `|A_real − A_fake|` on the identity adjacency matrix and
as elevated `r_diff_identity` on individual fakes.

**Why it can look weak on FF++.** FF++ fakes (Deepfakes, Face2Face,
FaceSwap, NeuralTextures) ship through compression and pixel-level
blending; their semantic coherence is often roughly preserved. The big
artefact is *noise*, not *broken AUs*. So `r_diff_identity` shrinks
even though the sub-graph is doing its job. Don't read the FF++
divergence in isolation — see Section 4.

### 3.2 Forensic-structural (62 nodes)

The "macro-forensics" sub-graph: gradients, blur, colour balance,
symmetry, low-frequency DCT energy. 30 nodes, all from
`forensic_features.py` indices 0..29:

- **Boundary gradients (6)** — Sobel `|∇I|` integrated along the
  face↔background, eye↔skin, lip↔skin, nose↔skin, brow↔skin
  contours, plus a mean. Faceswaps blend at exactly these boundaries
  and tend to leave residual edges or, conversely, to over-smooth the
  seam. Either failure mode shows up here.
- **Regional blur (6)** — Laplacian variance `σ²_Lap` per parsed
  region (skin, eye, mouth, nose), plus eye/skin and mouth/skin
  ratios. GAN outputs typically over-smooth flat skin while keeping
  texture in the eyes; the *ratio* features are designed to catch that
  imbalance.
- **Symmetry (4)** — left-right symmetry of eye, mouth, cheek and
  jawline. Reconstruction-based fakes break L-R symmetry when the
  generator hallucinates one side from a partial view.
- **Colour χ² (4)** — chi-squared histogram distance between
  region pairs (eye-skin, mouth-skin, nose-skin, L-R cheek). Picks up
  illumination mismatches at blending boundaries.
- **DCT high-frequency (6)** — energy of the high-frequency DCT
  coefficients per region, plus eye/skin and mouth/skin ratios. Many
  GAN architectures introduce a global high-frequency stripe that this
  catches.
- **Quality (4)** — antispoof score, detector score, blendshape
  symmetry, landmark jitter. Less interpretable, more of a "is the
  face well-formed at all?" sanity bus.

What it should learn on reals: most boundary, blur, colour and DCT
features should be tightly coupled to each other and to the latent z's
— a sharp face is sharp everywhere and lit consistently. On fakes
those couplings break; the SCM trained on fakes ends up with a
qualitatively different adjacency.

**Why it works (sometimes).** Excellent on classical face-swap fakes
(Deepfakes, Face2Face) where regional inconsistencies are visible.
Less effective on style-GAN-style end-to-end synthesis where the whole
face is generated in one shot — those samples look uniform and the
sub-graph has nothing to flag.

### 3.3 Forensic-noise (75 nodes) — *the dominant signal on FF++*

The "micro-forensics" sub-graph: PPNC patch noise, CCNC channel
noise, SRM filters, multi-scale wavelet noise. 43 nodes from indices
30..72:

- **PPNC pairwise patch noise consistency (8)** — μ and σ of the
  noise residual on outer-eye, inner-eye, cheek and jaw patches.
  Reformulates the "is the camera sensor noise consistent across
  the face?" intuition.
- **CCNC cross-channel noise consistency (8)** — energies in Y, Cr,
  Cb, the three pairwise correlations, and Y/Cr and Y/Cb energy
  ratios. Genuine sensor noise has a known spectral shape across
  channels; GAN noise doesn't.
- **SRM noise residual filters (15)** — five fixed *Steganalysis Rich
  Model* high-pass kernels (`Hedge, Vedge, HLap, Sq, Diag`), each
  applied to the face crop, with the response summarised by
  `(μ, σ, κ)`. SRM is the workhorse of forgery detection and the
  reason this sub-graph is so strong on FF++.
- **Multi-scale noise (12)** — wavelet-style decomposition at scales
  s=1, 2, 4 with `(μ, σ, κ)` per scale plus cross-scale correlations
  (`xcorr_s1s2`, `xcorr_s2s4`) and a fine/coarse energy ratio.

What the SCM should learn on reals: the noise field is structured —
its mean is near zero, its variance is region-stable, kurtosis is
pinned by sensor physics, the cross-channel correlations follow Bayer
demosaicing, and the multi-scale energy decays smoothly. On fakes
*every one* of those constraints can break.

**Why it works (and where).**

- *Works strongly* on FF++ — every fake there is encoded through MPEG
  compression; even subtle GAN tweaks to noise statistics leave a
  detectable trace, especially in the SRM and CCNC features.
- *Less reliable* on heavily denoised / photo-touched fakes where the
  generator post-processes the output (DDIM sampling, GFPGAN
  enhancement). Those wash out the residual signal, and `r_diff_noise`
  drops substantially.

### 3.4 Forensic-spectral (42 nodes)

The smallest sub-graph: 10 FFT features from indices 73..82 of the
forensic vector:

- **8 radial bins** — mean magnitude in 8 concentric rings of the
  log-magnitude FFT.
- **FFT slope `β`** — least-squares slope of the radial spectrum on
  log-log axes. Real photos sit close to a `1/f` law; GAN images
  consistently deviate.
- **HF/LF ratio** — total energy in the outer rings divided by the
  inner rings.

Captures the "spectral fingerprint" anomaly first reported by Wang et
al. (CVPR 2020). Smaller than the noise sub-graph but a useful
*independent* signal: classifiers can fool the noise stream and still
get caught here.

**Why it works.** It's the most architecture-specific signal. Different
GAN families (DCGAN, StyleGAN, diffusion) leave very different
spectra; the per-method radar (`scm/r_diff_radar_per_method.png`)
should show clearly different spectral footprints across FF-DF / FF-FS
/ FF-NT.

**Why it sometimes fails.** Strong JPEG re-encoding flattens the
spectrum and hides the artefact. On Celeb-DF where the videos are
stronger compressed, spectral usually drops below structural.

---

## 4. What the empirical numbers say (FF++, ablation4_causal, epoch 10)

From the existing run
`logs/train/nesy_defake_ablation4_causal_2026-04-24-01-59-25/interpretability/epoch_10/FaceForensics++/report.txt`:

| Sub-graph              | L1 divergence  | Dominant in fakes |
| ---------------------- | -------------- | ----------------- |
| identity               | **1.92**       | 1.7 %             |
| forensic_structural    | 7.87           | 2.5 %             |
| forensic_noise         | **20.11**      | **84.6 %**        |
| forensic_spectral      | 7.68           | 11.3 %            |

L1 divergence is the sum of `|A_real − A_fake|` over the adjacency
matrix; "dominant in fakes" is the fraction of fake samples whose
argmax `r_diff` lands on this sub-graph.

Two things to read from this:

1. **Noise is doing most of the work on FF++** — both in the static
   adjacency divergence and in per-sample dominance. That's expected
   given Section 3.3 and consistent with the SRM literature.
2. **Identity looks tiny** (1.92 vs 20.11 noise, 1.7 % dominance).
   That is **not** evidence the sub-graph is broken. It is evidence
   that FF++ fakes do not break semantic couplings strongly enough to
   compete with their noise residuals.

Before deciding what to do with identity, run the same interp on
**Celeb-DF-v2**. CDF videos have weaker noise traces (cleaner
post-processing) and stronger semantic break-down (the source-target
identity gap is wider). The expectation is that identity divergence
moves up and noise dominance falls — possibly to an even split.

---

## 5. Should we drop identity?

Short answer: **no, but down-weight it adaptively.**

Long answer:

- The identity sub-graph is the **only** path through which curated
  facial attributes (`fs_*`) and consistency rules (`cr_*`) reach the
  causal evidence stream. Drop it and the causal branch becomes pure
  DSP forensics. The "neuro-symbolic" claim of the paper rests on
  having semantics in the loop.
- The single-dataset numbers in Section 4 are a *bias of FF++*, not a
  property of the sub-graph. We have not yet quantified identity on
  CDF / DFDC.
- The fix is to let the **data** weight the four sub-graphs instead of
  hard-coding equal contribution. Two ways to do that:
  - **Attention-over-summaries.** Inside `ImprovedCausalBranch.forward`,
    replace the `torch.cat([identity_summary, …])` with an attention
    block that produces a per-sample weight over the four 8-d
    summaries before the evidence MLP. ~10 lines.
  - **Per-sub-graph gates.** Add four scalar `nn.Parameter` gates,
    one per sub-graph, initialised at 0.25 and softmax-normalised
    before the evidence MLP. Cheaper, less expressive.
- The right move for the rules **is not** to add more rules of the
  same kind. The current `rule_firing_rates.png` shows real and fake
  fire the existing 12 rules at near-identical rates — adding more
  emotion-AU pairs will not help. Instead, target known *faceswap
  failure modes that aren't already covered*:
  - `cr_yaw_gaze_misalign`: head-yaw vs mean(gaze_x) — a swap keeps
    the host's pose but transplants the source face's gaze, breaking
    this constraint.
  - `cr_pose_facewidth`: face width-to-height ratio should depend
    monotonically on yaw; faceswaps flatten the dependency.
  - `cr_lip_geometry_au12`: `fs_AU12` (smile) should track mouth-corner
    height geometric ratio; texture-based swaps decouple them.
  - `cr_brow_eye_couple`: `fs_brow_height_ratio` should track AU1
    (inner-brow raise) and AU4 (brow lowerer).

These four are concrete enough to add as new entries in
`consistency_rules_v7.py` and they target axes that *aren't* already
flat in the current firing-rates plot.

---

## 6. New visuals on 2026-04-28

Three additions to the figure pipeline. None of them require
re-training; they only change how `interpretability.engine.run(...)`
renders results.

### 6.1 Math-labelled, cropped side-by-side heatmaps

`scm/{sub}_side_by_side.png` used to be 4000-px-wide canvases with the
actual adjacency content stuffed into one corner and a colour bar
shared between the data and the divergence panel — making the
divergence panel almost invisible.

After the patch:

- Inactive rows / columns are **auto-cropped**; only the active block
  is rendered.
- The divergence panel gets its **own** colour bar with its own range
  (it is no longer washed out by `A_real`'s much wider scale).
- Tick labels are pretty mathtext: `ff_srm_hedge_kurt` becomes
  `κ_SRM^Hedge`, `ff_dct_hf_eye` becomes `E_DCT^eye`, `cr_happy_au6`
  becomes `R_happy-AU6`, `z_5` becomes `z_5`.
- Tick labels are tinted by group; a colour-key legend sits below the
  figure.

Implementation: `interpretability/visualization.plot_side_by_side_heatmap`,
new module `interpretability/pretty_names.py` for the name mapping.

### 6.2 Per-class separated SCM graphs

`scm/{sub}_graph.png` is the existing overlaid view (blue arrows for
`A_real`, red for `A_fake`, parallel curved). It is good for "what
edge weights diverge", less good for "what edges are *missing* in the
fake SCM."

The new `scm/{sub}_graph_separated.png` solves that:

- Two panels, one per class, with a **shared node layout** (computed
  on the union graph so positions are directly comparable).
- Strong edges drawn in the class colour; **edges absent in this
  class are drawn as faint grey ghosts** at the same position. Your
  eye traces "real had this edge; the fake SCM dropped it" without
  reading widths.
- Off-node label tags in white-bordered bubbles — labels never overlap
  edges or node circles.
- Node fill colour codes the semantic group (latent / curated / rule
  / forensic-sub-type).
- Layout is `Graphviz dot` if pygraphviz / pydot is installed, falling
  back to a pinned-seed spring layout otherwise.

Implementation: `interpretability/visualization.plot_separated_class_graphs`,
wired into `SCMAnalyzer.visualize`.

### 6.3 Discriminative-gap rule plot

`rule_firing_rates.png` (the side-by-side red-vs-grey bar plot) is
kept for completeness but is **misleading on its own** — real and fake
fire most rules at nearly identical rates. The new
`rule_discriminative_gap.png` plots `mean_fake − mean_real` per rule,
ranked by `|gap|`, top-12 only. Bars are red if the rule fires more on
fakes (the expected behaviour) and blue if more on reals (often a
clue that the rule is pathological).

Implementation: `interpretability/visualization.plot_discriminative_gap`,
wired into `ConsistencyRuleAnalyzer.visualize`.

---

## 7. Reading the figures (paper-ready captions)

Captions you can copy-paste into the paper, assuming numbers from the
FF++ epoch_10 reference run:

- **Figure (uncertainty histogram)**: *"EDL epistemic uncertainty
  `u = K / Σα` on FF++. Fake samples concentrate at `u ≈ 0.06`; reals
  spread broader at `u ≈ 0.14`. The detector is not just confident —
  it knows when to be confident."*
- **Figure (branch evidence, proportional)**: *"Composition of
  evidence on FF++. Spatial drives 60 % of evidence on real samples;
  on fakes the share shifts decisively to causal (≈ 70 %), with
  spatial dropping to ≈ 26 %. The architecture decomposes its
  decision; no single branch is doing all the work on either class."*
- **Figure (`scm/r_diff_radar.png`)**: *"Per-sample causal residual by
  sub-graph, real vs fake. The forensic-noise arm dominates the fake
  signal on FF++; identity and structural contribute on samples where
  the noise residual is suppressed by post-processing."*
- **Figure (`scm/forensic_noise_side_by_side.png`)**: *"Adjacency
  matrices `A_real`, `A_fake`, and their absolute difference for the
  forensic-noise sub-graph. The fake-trained SCM rewires couplings
  among SRM kurtosis nodes (κ_SRM^*) and the multi-scale noise
  cross-correlations (ρ_noise^{1,2}, ρ_noise^{2,4}), exactly the
  features sensitive to GAN-induced micro-statistics."*
- **Figure (`scm/forensic_noise_graph_separated.png`)**: *"The
  forensic-noise SCM, learned on real (left) and fake (right)
  samples, drawn over the same node layout. Faint grey edges are
  present in one class only; their absence in the other is the
  detection signal. Node colours encode the semantic group."*
- **Figure (`rule_discriminative_gap.png`)**: *"Top-12 consistency
  rules ranked by `|mean_fake − mean_real|`. Red bars fire more on
  fakes; blue bars fire more on reals. The four AU-driven rules
  carry the bulk of the gap; bone-structure rules contribute
  negligibly on FF++."*
- **Figure (reliability + risk-coverage)**: *"Calibration and
  selective prediction on FF++. ECE = 0.017 without temperature
  scaling. EDL uncertainty matches the softmax-margin baseline as a
  selector and reaches near-oracle accuracy by 50 % coverage."*

---

## 8. Quick decision tree

1. **Is identity divergence < 0.1 × max(other sub-graphs) on the
   second-most-important test dataset (Celeb-DF-v2)?**
   - Yes → consider dropping the identity SCM in a future ablation
     (call it ablation4-noid) and report side-by-side. Keep the
     baseline that has it.
   - No → keep all four. Identity is doing dataset-dependent work.
2. **Does adding the proposed pose-gaze + structural rules
   (Section 5) *change* the discriminative-gap plot?**
   - Yes (new red bars appear in the top-12) → ship them.
   - No → leave the rule set at 12 and stop adding.
3. **Does the attention-over-summaries variant (Section 5) improve
   E-AURC on Celeb-DF without hurting it on FF++?**
   - Yes → make it the default for the camera-ready ablation.
   - No → keep the equal-weight concat as the simpler baseline.

---

## 9. Files touched on 2026-04-28

```
training/interpretability/pretty_names.py                 (new)
training/interpretability/visualization.py                 (heatmap, graph, evidence, gap)
training/interpretability/analyzers/scm_analysis.py        (separated graph, mathtext keys)
training/interpretability/analyzers/consistency_rules.py   (discriminative-gap plot)
docs/README_april_28.md                                    (this file)
```

No model weights, no training-time code paths, no config schemas were
touched. Re-running `interpretability.engine.run(...)` on any existing
checkpoint emits the new figures — no retraining required.
