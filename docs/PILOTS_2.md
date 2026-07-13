# Claude Code instructions, pilot phase 2

Purpose. Phase 1 pilots (revision 3) produced a GO on the core verification gate, a PARTIAL on the deployable repair arm, and a soft outcome on attribution that went to the human. The human decision is to proceed with targeted follow ups before any full build. This file is the complete instruction set for phase 2. Each pilot is a gate. Report real numbers and stop at each gate for a human decision.

Read PLANNING.md and CODEBOOK.md first. Phase 1 results live in results/pilot_rev3 and results/pilot2. Reuse those artifacts, frames, seeds, and calibration wherever possible so phase 2 numbers are directly comparable to phase 1.

State inherited from phase 1.
- PROVEN. Ground truth Poisson repair drops calibrated p(fake) 0.45 to 0.69 median across effort, fsfm, gend, forada while LPIPS matched corruption is inert, wrong region repair is inert, real repair offset is small. The counterfactual test is sound.
- DEAD. Whole region inpainting as the deployable verifier. SD1.5 and LaMa both recover roughly zero percent of the GT effect and push inpainted real frames to about 0.57 p(fake). Detectors flag synthesized inner face content per se. Do not re run whole region inpainting as a verifier. It remains only as a known failed reference arm.
- FINDING. On full inner face swaps no single small region is individually necessary, 14 of 376 single region repairs cleared the 0.10 bar. Necessity exists at the granularity of the manipulation. This is a result, report it, do not fight it.
- OPEN. Verified refit of attribution baselines, spectral interventions on DF40 content, localization check on localized manipulations, and whether predicates can approach frozen feature accuracy on seen generators once enriched.

Hard rules, carried over and extended.
- Do not train or fine tune any perception model. Temperature scaling remains the single permitted fit. The retrieval index is built from frozen features and is not trained.
- Regions come only from deterministic facial landmarks bound to codebook region ids.
- Real measured numbers only. No placeholders. Incomplete runs are reported as incomplete.
- Log all seeds. Three seeds minimum where anything is fit or sampled.
- Pre register before results exist. For this phase that means, the reference real corpus and its retrieval feature definition, the Option A generator to family mapping, and the held out generators. Commit these files before the relevant runs.
- Infrastructure change. Serve InternVL through LMDeploy with AWQ 4 bit for all DF40 scale extraction. Before using it for any result, re run phase 1 step 10 (proposal recall and precision against masks on the same 100 FF++ frames) under the quantized model and report the delta against the fp16 numbers. If recall or precision degrades materially, fall back to fp16 and report the throughput cost.

Environment. Two GPUs, roughly 148 GB. Batch heavily.

Execution order. Pilot 1.5 first, it decides whether a deployable verifier exists. Studies 1A and 1B run alongside it, both are cheap. Pilot 2R after the LMDeploy validation and after the 2R pre registration files exist, including the DISCERN feature to codebook mapping table. Pilot 1L can run in parallel with 2R. Pilot 2V runs only on a 2R GO. Stop for a human decision at every gate.

---

## Pilot 1.5, retrieval based repair, the replacement deployable verifier

Motivation. Phase 1 proved only real pixels lower p(fake), synthesized pixels are flagged regardless of inpainter type. Therefore the deployable counterfactual should retrieve real pixels, not generate them.

Setup.
1. Pre register and build the reference corpus. Real face images from FFHQ or CelebA-HQ plus the calibration reals. Log the exact corpus and size.
2. Build a retrieval index over real face regions. For each corpus face, run deterministic landmarking, extract per region descriptors, landmark geometry of the region, head pose, and a lighting descriptor, plus a frozen visual embedding of the region crop. Store region crops indexed by region_id. Index with FAISS or equivalent. The index is frozen once built.
3. Retrieval repair operator. For a cited region on a query frame, retrieve the nearest real region of the same region_id by the pre registered descriptor distance, warp it to the query landmark frame, Poisson blend it in, masked to the face side of any seam.

Runs, on the same 100 FF++ frames, same calibrated detectors, same seeds as phase 1.
4. Retrieval repair on the true manipulated region. Record p(fake) drop per detector. Report as a fraction of the phase 1 GT repair effect per frame.
5. Real repair offset for retrieval. Apply retrieval repair to the corresponding region of the paired real frame. Record the change. This measures the identity mismatch and blending penalty. Do not require it to be zero, record the distribution, it defines the normalization for retrieval verified confidence.
6. Wrong region control for retrieval. Retrieval repair on a non manipulated region of the same fake frame. Record the drop.
7. Identity mismatch probe. For a subsample, also record the id_embedding_distance between retrieved region and query face, and correlate it with the real repair offset. This tells us whether offset is driven by identity mismatch, and whether descriptor weighting should prefer identity similarity.
8. Reference arm. Include the phase 1 inpainting numbers in the same table as the known failed reference. Do not re run them.

Report.
- Retrieval repair drop distribution per detector, fraction of GT effect recovered, median and spread.
- Real repair offset distribution for retrieval versus the roughly 0.57 inpainting offset from phase 1.
- Wrong region drop.
- Identity mismatch versus offset correlation.

Gate.
- GO if retrieval repair recovers a substantial fraction of the GT repair effect, its real repair offset is clearly below the inpainting offset and small relative to the fake repair effect, and wrong region remains near inert. The deployable verifier exists. Retrieval repair becomes the verification operator for Pilot 2V and the full build.
- PARTIAL if retrieval recovers real signal but the offset is dominated by identity mismatch per step 7. Report and stop, the human decides on descriptor re weighting toward identity similarity before one retry. One retry maximum.
- STOP if retrieval repair behaves like inpainting, near zero recovery or large offset not explained by identity. Then no deployable pixel space verifier exists, and the deployable claim scopes to spectral interventions plus small region inpainting per the offset area study, with GT paired repair reserved for in lab certification. This is a scope decision for the human, not a framework failure, phase 1 already proved the test itself is sound.

---

## Study 1A, inpainting offset versus region area, cheap, runs alongside Pilot 1.5

Purpose. Define the operating regime where inpainting based verification remains valid for small, predicate targeted interventions.

Steps. On 50 real frames, inpaint regions of increasing area with SD1.5, from small single landmark boxes up to the full inner face mask, at least five area levels, log areas as fraction of face area. Record p(fake) change per detector at each level. Produce the offset versus area curve. Repeat the smallest two levels with LaMa to confirm operator independence. Optionally produce the same curve for retrieval repair.

Report. Offset versus area curves per detector, and the largest area at which the median offset stays below a nuisance threshold relative to phase 1 GT effect sizes, stated per detector.

No gate. This study parameterizes decisions, small region inpainting is admissible below the measured area threshold, inadmissible above it. Feed the threshold into Pilot 1L and Pilot 2V.

---

## Study 1B, compression sensitivity of spectral evidence on DF40, cheap, runs alongside Pilot 1.5

Purpose. Phase 1 section 1.3 found near zero spectral intervention drops on FF++ c23. Two explanations are confounded, compression destroyed the spectral fingerprint before the notch could, or blending based FF++ manipulations carry no global spectral fingerprint at all. FF++ raw is not available, so the disambiguation runs on DF40 with a self controlled compression ladder, which is the better testbed anyway since DF40 GAN and diffusion families are where global spectral fingerprints are expected to exist.

Steps.
1. Data. A subsample of DF40 fakes, at least 50 per group across three groups, a GAN synthesis group, a diffusion group, and a blending based face_swap group for the family contrast, plus matched reals, at native quality. Log the native format and any known prior compression.
2. Build the compression ladder. Re encode each frame at three levels, native (no re encoding), c23 equivalent (H.264 CRF 23 or JPEG quality about 90), and c40 equivalent (H.264 CRF 40 or JPEG quality about 50). Log the exact codec and settings.
3. At each ladder level, run the spectral interventions, notch, checkerboard suppression, residual renormalization, same protocol as phase 1 section 1.3, with real offsets at the same level.
4. At each ladder level, compute the DISCERN spectral and noise feature groups (SRM, CCNC, PPNC, multiscale, radial FFT) on fakes and reals, and report per feature real versus fake separability AUC.

Report. Spectral intervention drops per family at each compression level. Per feature separability AUC across the ladder, highlighting which feature groups survive re encoding.

Caveat to state in the report. DF40 native frames carry whatever compression they shipped with, so the ladder measures additional compression sensitivity with native as the baseline. Absolute raw sensitivity is not recoverable without original uncompressed sources.

No gate. Outcomes and their consequences.
- Family contrast, the sharpest single result available. If the DF40 blending based face_swap group shows near zero drops at native quality while the GAN and diffusion groups show real drops on the same detectors at the same quality, the manipulation type explanation for the phase 1 FF++ null is confirmed directly, no raw FF++ needed. Report this contrast as its own row.
- Drops exist at native and vanish by c23 equivalent. Compression explanation supported. Spectral verification is scoped as compression sensitive in the paper, a deployment relevant finding, and the FF++ c23 null from phase 1 is at least partly explained by compression.
- Drops exist at native and survive c23 equivalent. Then the FF++ null was manipulation type, blending swaps carry no global spectral fingerprint, and the phase 1 reading stands with this as its evidence.
- No drops even at native on GAN and diffusion content. Then these detectors do not rely on global spectral cues anywhere, whole_face spectral verification is detector limited, and the finding feeds the detector choice discussion.
This study shares its DF40 spectral runs with Pilot 2R step 6, coordinate so the native level work is done once.

---

## Pilot 2R, attribution gate re run, Option A plus enriched predicates

Motivation. Phase 1 part one was soft. D was above chance with the smallest seen to held out gap, +0.023, but trailed CLIP and DCT on seen. Two named depressors, the Option B family grouping averaged away spectral separations inside EFS, and the predicate vector contained only 13 deterministic attributes with zero VLM semantic predicates. This pilot tests the ceiling.

Steps.
1. Pre register the Option A mapping before results. Edits become a separate family or a designated abstention family, EFS becomes clean synthesis only. Log the mapping file and held out generators, at least one per family.
2. Extract InternVL semantic predicates over landmark regions for all DF40 samples via LMDeploy AWQ, after the quantization validation described in the hard rules. Whole_face frequency predicates included.
3. Assemble the enriched predicate vector. Deterministic probes are the DISCERN 83 dimensional forensic bank from forensic_helpers.py, region based forensics over SegFormer parsing (boundary gradients, regional blur, left right symmetry, color consistency, regional DCT), PPNC paired patch noise consistency, CCNC cross channel noise correlations, the SRM filter bank statistics, multiscale LoG noise residuals, and the radial FFT profile with slope and hf ratio. Concatenate the InternVL semantic predicate attributes. Pre register the exact feature list and commit a feature to codebook attribute mapping table before any 2R result exists, for example ff_fft_slope maps to frequency_anomaly.radial_spectrum_slope, PPNC maps to noise_inconsistency.cross_region_noise_gap, CCNC correlations enter as new noise_inconsistency attributes added to the codebook now, before the run. SegFormer is a frozen measurement dependency only, intervention regions still come exclusively from deterministic landmarks. Dimensions logged.
4. Re run Baselines A, B, C, D under Option A on the same splits, three seeds. A and B need refitting only because the label space changed.
5. Baseline E, retrieval attribution. k nearest neighbors over the enriched predicate vectors against the seen generator library, family by majority vote, k pre registered. Report seen and held out.
6. Spectral verification probe on DF40. On a subsample of GAN and diffusion family fakes, run the whole_face spectral interventions, notch, checkerboard suppression, residual renormalization, with real offsets, exactly as phase 1 section 1.3. This answers whether spectral predicates verify on the content they were designed for, FF++ already showed they correctly fail on blending swaps.
7. Clustering supplementary. On the enriched predicate vectors, k means at k equal to family count and HDBSCAN, report ARI and NMI against family labels, computed without using labels in the fit. Additionally embed the held out generators and report whether any forms a separated cluster, as empirical support for abstention on novelty. Also run NMF or PCA on the predicate matrix and report the top factors for the attribute pruning discussion.

Report.
- A, B, C, D, E accuracy, seen and held out, chance stated, three seeds, mean and spread, plus the seen to held out gap per baseline, side by side with the Option B phase 1 numbers.
- Spectral drops on DF40 fakes versus real offsets, per family.
- ARI, NMI, held out cluster separation, factor summary.

Gate, decided on the enriched D and E rows.
- GO if enriched D on seen closes most of the gap to the best raw feature baseline, within a few points, and enriched D held out beats both raw feature baselines held out while keeping a clearly smaller seen to held out gap. The predicate ceiling is high enough, proceed to 2V.
- SOFT GO if enriched D held out beats raw baselines held out with the smallest gap but seen remains clearly below raw features. Report and stop, the human decides whether the transfer advantage alone carries the thesis with reframed claims.
- STOP if enrichment does not move D materially. The predicate signal has a real ceiling, the paper pivots toward certification of evidence, where the phase 1 gate result is already the anchor experiment.

---

## Pilot 1L, localization on localized manipulations

Motivation. Phase 1 step 11 was unmeasurable on full face swaps, mask overlap saturated near 0.9 with no variance. The per region necessity machinery needs manipulations where region overlap varies.

Steps.
1. Data. FF++ NeuralTextures frames with masks, plus a DF40 localized slice, lip sync or edit generators, at least 100 frames total with ground truth or reliably derivable manipulation regions. Log the exact selection.
2. Interventions, chosen by validity. GT paired repair where pairs exist. Retrieval repair if Pilot 1.5 passed. Small region inpainting only below the Study 1A area threshold. Spectral for whole_face predicates.
3. Re run the phase 1 localization protocol, per region repair drop versus mask overlap, Pearson and Spearman, pass versus fail proposal overlap, wrong region control, co verification logged.
4. Also report the fraction of proposals clearing the 0.10 necessity bar, against the 14 of 376 full face swap number, to substantiate the granularity finding.

Report. Correlations, pass versus fail overlap, necessity clearance rates by manipulation scope.

Gate.
- GO if, on localized manipulations, necessity tracks mask overlap and confirmed predicates clearly beat rejected on overlap. The region level machinery works where regions are meaningful, and the paper states the granularity scoping with both numbers.
- STOP if even on localized manipulations necessity does not track overlap. Then region level verification does not localize and the claim must retreat to manipulation level necessity only. Human decision.

---

## Pilot 2V, verified refit, runs only after a 2R GO

Steps. On a subsample sized to fit within a day of GPU time and logged, run the verification gate using the operators validated above, retrieval repair if 1.5 passed, otherwise spectral plus small region inpainting within the Study 1A regime, with wrong region and real offset controls. Refit Baselines C, D, E on verified only predicates, same subsample for verified and unverified so the comparison is clean. Report per predicate family verification rates by generator family.

Gate.
- GO if the verified refit does not collapse the enriched predicate signal. Proceed to the full build, the symbolic layer targets matching D verified on seen and beating it on held out.
- STOP if verification destroys the signal. Report which predicate families were pruned and where. Human decides between fixing the gate and the certification only pivot.

---

## Out of scope for this phase

The MLLM explanation audit study, auditing DeepfakeJudge, Skyra, and FAQ style rationales with the counterfactual gate, is planned but is a separate instruction set. Do not start it. The full Verified Evidence Graph build, the Scallop or DeepProbLog reasoner, and the rule editing study remain gated behind 2V.

## After all gates

Stop. Report a single consolidated summary table across 1.5, 1A, 2R, 1L, 2V with each gate outcome, and wait for the human go before any full build work.

1608ec0d-772f-430a-8b82-39d00bd54ed9
