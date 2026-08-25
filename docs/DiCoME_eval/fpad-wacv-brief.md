# FPAD — Face-Prior Adaptation Dynamics — WACV Standalone Build Brief

This brief builds a standalone deepfake-detection method for a WACV submission, reusing DFB_NeSyNS components wherever they exist. The method is a frozen FS-VFM teacher paired with a LoRA-adapted student, read as a depth-resolved adaptation profile. The contribution is the prior-preserving teacher-student adaptation trajectory as a forensic and localization signal, not another cross-dataset AUC number.

Timeline is seven days with paper writeup running in parallel, so stages are ordered by how early they produce a paper-critical result, not by architectural tidiness. The mechanism-validation gate (Stage 2) runs first so the paper's thesis is confirmed or pivoted on day one or two.

Partition constraint that must hold. This method is this paper's headline. Inside T-BIOM it is at most a supporting expert. The same mechanism cannot be the novelty of both papers. Nothing in this brief may be presented as T-BIOM's contribution.

## Ground rules

- Reuse first. Every component below almost certainly exists in DFB_NeSyNS in some form (FS-VFM loading, LoRA, FF++ loaders, source-paired sampling, DF40 eval, the §20 domain audit, the EDL head). Discover and reuse. Write new code only where discovery finds nothing.
- Resolve real names before running. Paths, scripts and config keys are described by role, not literal path, because the tree is not confirmed here. Discover the real name, then use it. Never invent a path.
- FF++ only for anything learned or selected. All training, LoRA adaptation, projection fitting, thresholds, checkpoint selection and risk calibration use FF++ c23 only. No CDF, DFDC, DFD, Deepfake-Eval or DF40 label may enter training or selection. DF40-Dev is read for analysis only, DF40-Holdout stays sealed until the final table.
- Numbers come from reading result files programmatically, never from memory. Unrun cells are `TODO(run)`, never filled.
- Write a short markdown result file at every stage boundary.

## Notation

Frozen teacher `E_T` and LoRA student `E_S`, both FS-VFM ViT-L/16, same initialization. For a face `x`, layer-`l` representations `h_T^(l)(x)`, `h_S^(l)(x)` (CLS) and patch tokens `z_T^(l,j)`, `z_S^(l,j)`.

Core signal, the teacher-student deviation at depth `l`:
```
d_l^TS(x) = 1 - cos(h_T^(l)(x), h_S^(l)(x))
```
The depth-resolved adaptation profile is the curve `D(x) = [d_1^TS, ..., d_L^TS]`. This curve is the branch. Patch form `d^TS_{l,j}(x) = 1 - cos(z_T^(l,j), z_S^(l,j))` is the localization map.

Two readouts, kept distinct throughout. A *direct* readout classifies from the student's own adapted features. A *trajectory* readout classifies from `D(x)` through `H_traj`. The whole scientific question is whether the trajectory readout carries forensic signal beyond the direct one, so the two are never conflated.

Diagnostic-only terms, computed for the audit and ablations, not shipped as the forensic feature: frozen-teacher trajectory `d_l^T, m_l^T, c_l^T` and student trajectory `d_l^S`. The frozen-teacher terms are OOD-driven by construction and are expected to fail the domain audit. The student-only terms are close to redundant with the classifier. Report both to justify the design, do not depend on them.

Two-stage training (governs Stages 1 through 3). The student is never driven through the cosine-distance term, because at initialization `h_T = h_S`, so `d_l^TS = 0` and the gradient of cosine distance is zero there. Driving LoRA only through `D(x) -> H_traj -> L_cls` starts in a flat region. So training splits.

- **Stage A, adapt.** Train the LoRA student and a direct student head with `L_adapt = L_cls_direct + lambda_preserve * L_preserve`. LoRA gets an ordinary classification gradient, not a degenerate one. Then freeze the student.
- **Stage B, interpret.** With the student frozen, compute `D(x)` and train `D(x) -> H_traj -> e_face`.

This also makes the causal claim honest. The adaptation is created by an independent objective and the trajectory only measures where it departed from the prior, rather than the trajectory both creating and classifying itself.

---

## Stage 0 — Repository discovery (DFB_NeSyNS)

Produce `wacv/REPO_MAP.md`. Resolve and record:

1. FS-VFM loading and the exact checkpoint used by the existing reference branch. Record whether the checkpoint is the online encoder only, and whether intermediate layers are already exposed or need forward hooks. All ViT-L blocks share hidden dim, so hooks on selected blocks are the fallback.
2. LoRA infrastructure. The CLIP branch already uses a LoRA recipe. Record it and whether it is reusable on an FS-VFM ViT-L backbone with layer selection.
3. FF++ c23 loaders and the source-paired Real/Fake sampler built in Phase 2. Record how pairing is toggled, since Stage 6's ranking loss needs it.
4. DF40 eval harness and the DF40-Dev / DF40-Holdout split file from the Phase-2 work. Reuse the same sealed split. Confirm the inverted-anchor reenactment rows (danet, mcnet, tpsm, facevid2vid-cdf) are the rescue targets.
5. The §20 domain-audit code that produced the V1 forensic-vs-domain separability tables. This is reused verbatim on the new signal.
6. The EDL evidence head and the evidence-to-opinion conversion. Reuse for `H_traj`.
7. FF++ manipulation masks on disk. The localization evaluation in Stage 5 needs per-frame manipulated-region masks. Record which FF++ manipulation types have masks staged. If none, record it now, because it changes Stage 5 (fallback is source-paired real-minus-fake pseudo-masks, weaker but usable).
8. Compression variants. Record whether FF++ c40 (and raw, if present) frames are staged, for the Stage 6 robustness check.

Pass condition. `wacv/REPO_MAP.md` answers 1 through 8. Items 1, 7 and 8 are the ones that change downstream scope, so surface them explicitly. If the checkpoint is encoder-only, that is fine, the teacher-student design needs only the encoder.

---

## Stage 1 — Wire teacher-student, lock the ladder, run the external baseline

Build the skeleton and pin the baseline ladder on two axes, student adaptation and prediction readout, so every rung isolates one thing.

- Load `E_T` frozen. Instantiate `E_S` from the same checkpoint with LoRA on the attention and MLP projections. Select a fixed layer set, e.g. `{4, 8, 12, 16, 20, 24}` for ViT-L/24, recorded in config.
- Extract `h_T^(l)`, `h_S^(l)` and patch tokens at the selected layers via hooks. Compute `d_l^TS` and `D(x)`.
- `H_traj` consumes `D(x)` and emits `e_face`, converted by the reused EDL conversion.

The locked ladder.

| rung | student adaptation | prediction readout |
|---|---|---|
| B0 | frozen FS-VFM | direct FS-VFM head |
| B1 | ordinary LoRA | direct student feature |
| B2 | ordinary LoRA | trajectory `D(x)` |
| B3 | LoRA + real-prior preservation | trajectory `D(x)` |
| B4 | B3 + paired objective | trajectory `D(x)` |

The two contrasts that carry the paper. `B2 - B1` asks whether the trajectory holds anything useful beyond the ordinary adapted student, holding the student fixed and changing only the readout. `B3 - B2` asks whether authentic-prior preservation improves that trajectory. Because B1 and B2 share the *same* ordinary-LoRA student under two-stage training, `B2 - B1` is a clean readout-only comparison.

External baseline, not optional. `B0` as "our frozen head" is our own use of FS-VFM and a reviewer will call it weak. In Stage 0 identify the authors' official FS-VFM downstream protocol in the FSFM / FS-VFM extension repo (linear probing, and the FS-Adapter if practical) and run it on the same FF++ c23 train and OOD test split. Report it as a distinct row from B0 and B1, so the comparison is against the authors' intended use of the model, not only ours.

Pass condition. B0, B1 and the official FS-VFM linear-probe baseline train on FF++ c23, select on FF++ val only, and produce in-domain AUROC. Record selected epochs. Treat small val differences as noise, since in-domain val saturates.

---

## Stage 2 — MECHANISM GATE — ordinary-LoRA delta vs preservation delta

This is the derisking stage and it runs before the full method is trained. It answers the one question argument cannot: is the teacher-student delta manipulation-specific, or is it just the decision boundary in a new coordinate.

Compute the identical teacher-student delta from two students.

- **Ordinary-LoRA delta**, the delta B2 reads. Deviation between frozen `E_T` and the ordinary-LoRA student.
- **Preservation delta**, the delta B3 reads. Deviation between frozen `E_T` and the preservation student.

Matched probes, or the comparison is confounded by training maturity rather than preservation. Both students share the same initialization, LoRA targets, seed, batch order, augmentations, optimizer and number of optimizer steps. The only difference is whether `L_preserve` is on.

Report per layer, not just aggregate. Give the full `D(x) = [d_4, d_8, d_12, d_16, d_20, d_24]` for each, because preservation may clean early and mid layers while later task-specialized layers stay different, and the aggregate would hide that.

Selection data, strict. This decision uses **only** FF++ validation, FF++ benign domain shifts (c40 or raw if staged), and DF40-Dev. The domain audit needs OOD reals to define domain separability, so use DF40-Dev reals and the FF++ compression shift as the OOD side. Do **not** touch CDFv2/v3, DFDC, DFDCP, DFD or Deepfake-Eval for this decision. Using any of them here reclassifies them as development data and forfeits their zero-shot status in the final table.

Decision.

- If the preservation delta is meaningfully cleaner on the audit than the ordinary-LoRA delta (higher forensic-minus-domain gap on a majority of the permitted sources), and especially if it cleans the early and mid layers, preservation is buying manipulation-specificity. The thesis holds. Proceed to Stage 3 as a manipulation-targeted trajectory method.
- If the two deltas separate real from fake about equally and both fail the audit similarly, `L_preserve` is doing little and the signal is mostly the boundary. Do not proceed as an anomaly method. Pivot to localization-first, where the claim is a faithful region-resolved adaptation map rather than a domain-robust detector, and lean Stage 5 harder. Flag this to the author immediately, day two at the latest, since it changes the writeup.

Write `wacv/STAGE2_GATE.md` with the per-layer deltas, the audit on permitted sources only, the real-fake separations, and the decision. This file gates the paper framing.

---

## Stage 3 — Two-stage training of B2 and B3, then cross-dataset eval

Train on FF++ c23 with source-paired sampling and matched benign augmentation on both members of a pair. Both rungs use the two-stage scheme from the notation.

Stage A, adapt the student.
```
L_cls_direct = L_EDL(direct_head(h_S(x_R)), y_R) + L_EDL(direct_head(h_S(x_F)), y_F)
L_preserve   = sum_l d( h_T^(l)(x_R), h_S^(l)(x_R) )        # authentic faces only, anchored to teacher
L_adapt      = L_cls_direct + lambda_preserve * L_preserve
```
`L_preserve` on authentic faces only is what forces adaptation to concentrate on manipulation rather than domain. `lambda_preserve` is a config value, start near 1.0 with a small recorded sweep. For B2 set `lambda_preserve = 0`, for B3 turn it on. Then freeze the student.

Stage B, interpret the frozen adaptation.
```
D(x) -> H_traj -> e_face,   trained with L_EDL against real/fake
```
Only `H_traj` trains here. The student is frozen, so the trajectory measures a fixed adaptation rather than co-creating it.

- Select checkpoints on FF++ val only.
- Evaluate B0, B1, B2, B3 and the official FS-VFM linear probe on the OOD suite, video-level AUROC, train FF++ c23 and test CDFv2, CDFv3, DFDC, DFDCP, DFD, UADFV, Deepfake-Eval-2024. This is the paper's main table. Sources spent on the Stage 2 decision (DF40-Dev, FF++ c40) are not part of the zero-shot claim.

Pass condition. B3 completes and the main table is filled for B0 through B3 plus the external baseline. The headline comparison is `B2 - B1` and `B3 - B2`, not B3 against chance. B3 must beat or meaningfully complement the ordinary fine-tuned FS-VFM path, which prior analysis showed is already strong on this task.

---

## Stage 4 — DF40-Dev rescue and the depth profile

Two analyses that carry the method's forensic and interpretability claims.

- **Rescue.** Evaluate B3 on DF40-Dev, especially the inverted-anchor reenactment rows where the semantic anchor sits below chance. Report where B3 rescues the anchor and where it harms already-correct rows. This is the specialization evidence and the bar the earlier frozen ideas missed.
- **Depth profile.** Plot the mean `d_l^TS` curve over depth for reals against fakes, and per DF40 family. The forensic question is where in the hierarchy manipulation supervision forces departure from the prior. Early-versus-late departure is itself the interpretable, manipulation-anchored signal. Report whether the profile shape differs between real and fake and across families.

Write `wacv/STAGE4_RESCUE_PROFILE.md`.

Pass condition. Both analyses complete. Rescue on the reenactment rows is the result that most strengthens the paper. Its absence is reportable, not fatal, if Stage 5 localization carries the contribution.

---

## Stage 5 — Patch-level localization and faithfulness

This is the axis that makes it a standalone paper rather than an AUC wrapper, because FS-VFM claims detection but not faithful region-localization.

- Compute the patch map `d^TS_{l,j}(x)` and aggregate across selected layers into a per-frame spatial adaptation map.
- Localization ground truth, three tiers, do not blur them. With genuine FF++ manipulation masks (Stage 0 item 7), report quantitative localization, patch-level IoU or AUPRC against the mask. With no masks, quantitative *faithfulness* is still allowed since it needs no mask. Source-paired real-minus-fake difference maps are for qualitative and sanity checking only, explicitly labeled as pseudo-masks, and are never reported as localization accuracy, because pixel differences carry alignment, compression, color and rendering changes unrelated to the manipulated region.
- If the Stage 2 pivot makes localization the headline, genuine masks become necessary. FF++ ships per-manipulation masks for its four types, so source those specifically rather than falling back to pseudo-masks.
- Faithfulness. Report a deletion or insertion check, whether suppressing the highest-adaptation patches drops the fake evidence more than suppressing random patches. This shows the map explains the decision rather than decorating it, and it needs no mask.

Write `wacv/STAGE5_LOCALIZATION.md` with the metric, its ground-truth tier, a few qualitative overlays, and the mask provenance.

Pass condition. Faithfulness numbers exist regardless of masks. Quantitative localization is reported only where genuine masks back it. Under a Stage 2 pivot this stage is the paper's center, so genuine masks must be secured, not pseudo-masks.

---

## Stage 6 — Robustness and the paired ranking loss (conditional)

- **Compression robustness.** If c40 is staged, evaluate B3 trained on c23 at c40 and report the AUROC drop, and re-run the §20 audit at c40. This directly supports the manipulation-not-compression claim, so it is high value for the thesis.
- **B4, source-paired ranking.** Only if B3 showed life in Stages 2 through 4. `D(x)` is a vector, so rank on the trajectory classifier's scalar fake score `s_traj(x)`, the fake evidence or logit from `H_traj(D(x))`, not on `D(x)` directly.
```
L_pair = max(0, m - ( s_traj(x_F) - s_traj(x_R) ))
```
over source-paired members. This isolates manipulation because the pair shares nuisance factors, and it avoids the unvalidated assumption that every fake simply has a larger mean deviation. Train B4 = B3 + `L_pair` and compare to B3. This is the rung most likely to convert "works" into "works for the right reason."

Pass condition. Robustness table exists. B4 is reported if trained, or explicitly deferred with reason if the sprint is tight.

---

## Stage 7 — Final tables and figures

Freeze everything. Assemble for the paper: the main cross-dataset table (B0 through B4 plus the official FS-VFM linear probe), the DF40-Dev rescue table, the DF40-Holdout table (read for the first time here, the only zero-shot claim), the domain-audit contrast (frozen-teacher trajectory vs preservation delta, and ordinary-LoRA delta vs preservation delta, showing the design choices), the depth-profile figure, and the localization overlays with the faithfulness number. Keep any baseline column that was never run as `TODO(run)` rather than filling it.

Provenance rules into the paper. DF40-Dev drove design and cannot be presented as zero-shot. Only DF40-Holdout is zero-shot. Any source whose val split was used for selection is not OOD.

---

## Seven-day mapping

- Day 1. Stage 0 discovery, Stage 1 B0/B1 and the official FS-VFM linear probe running, train the two matched Stage-A students (ordinary-LoRA and preservation).
- Day 2. Stage 2 gate decision on the two deltas. Thesis confirmed or paper pivoted. This is the hinge.
- Day 3. Stage 3 B2/B3 training and the main cross-dataset table.
- Day 4. Stage 4 rescue and depth profile.
- Day 5. Stage 5 localization and faithfulness.
- Day 6. Stage 6 robustness, B4 if warranted.
- Day 7. Stage 7 final tables and figures, writeup consumes stage files as they land.

Writeup runs in parallel from day 1. Methods and related work can be drafted before results, since the mechanism is fixed. Results and the abstract's headline claim wait on the Stage 2 decision.

## Novelty framing for the writeup

Do not claim novelty for FS-VFM, LoRA, ViT trajectories or EDL individually. The contribution is the prior-preserving teacher-student adaptation dynamics read as a depth-resolved, region-resolved forensic signal, with authentic-face preservation as the mechanism that makes the adaptation manipulation-targeted rather than domain-driven. The evaluation axes that distinguish it from a fine-tuned detector are the domain audit, DF40 rescue, the depth profile, and faithful localization, not AUC alone. The honest baseline the method is measured against is B1, ordinary fine-tuned FS-VFM, which is already strong on this task.