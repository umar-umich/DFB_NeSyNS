# DISCERN-v2 / T-BIOM — Resume Build Brief (post-FPAD)

This resumes the Phase-2 architecture against DFB_NeSyNS. The design is unchanged in shape. What changed is that the FPAD investigation resolved the identity of the FS-VFM branch and demoted it from an assumption to a candidate. So the first stage of resumed work is branch membership decided on complementarity, not fusion training. Everything after that is the fusion, applicability, and reliability build already specified.

The spine that has survived every experiment in this project is the reliability decomposition. V/C/U_sup gave error-detection AUROC around 0.81 with correct coefficient signs and a real selective-risk reduction, and it held while four branch ideas did not. That is the paper's center of gravity. The specialists are additive to it, not load-bearing under it.

## What the FPAD post-mortem settled, carried in as constraints

- The FS-VFM branch, if it enters, is a direct adapted-FS-VFM discriminative expert. The reference-anomaly (P_R) and depth-trajectory framings are both falsified and archived. No H_traj, no P_R, no localization claim from this branch.
- Detection and localization select different depths in this family and the detection objective wins. Any future localization work must be mask-supervised and decoupled from the detection objective. Not in scope here, recorded so it is not re-learned.
- Rescue is computed fresh against the actual anchor's actual errors. The project was bitten twice by inheriting a prior baseline's failure rows. Never inherit them.

## Ground rules

- Reuse first. The gate cross-fitting, the §20 audit, the EDL head and conversion, the DF40 split, the source-paired sampler, the fusion and reliability code all exist from Phase-2. Discover and reuse. Write new code only where discovery finds nothing.
- Resolve real names before running. Paths and scripts are described by role. Discover the real name, then use it. Never invent a path.
- FF++ c23 is the only gradient-training data. No external frame or label ever enters gradient training. Everything learned by backprop sees FF++ c23 only.
- Validation partitions, kept disjoint by source video. FF++ val splits into `VAL_guard` (about 60 percent, the in-domain guardrail readout) and `VAL_meta` (about 40 percent, applicability gates, risk model and defer thresholds only, never touched for checkpoint selection). A separate diverse-validation set is used for generalization-aware checkpoint selection, protocol below. The final test sets are never used for any selection.
- Diverse validation for checkpoint selection (GenD-style), and what it costs. In-domain FF++ val saturates in epoch 0 in this project, so selecting on it can pick a checkpoint after cross-domain performance has already started to decline. To avoid that, discover and reuse the previously built diverse-validation set (this is VALmix, from Celeb-DF-v2 non-test plus DFDCP official-train plus Deepfake-Eval-2024 finetuning-train), confirming no test video or frame overlaps it and that splits are fixed in advance. Select checkpoints on the macro-average video AUROC across its domains, one AUROC per domain then averaged so no large domain dominates, with `VAL_guard` AUROC reported alongside as the in-domain guardrail. This is not sample leakage, since no external data enters gradient training and no test sample is used. But it is not free either. Any domain that appears in diverse validation is domain-seen during development and cannot be reported as strict zero-shot. Per VALmix that means Celeb-DF-v2, DFDCP and Deepfake-Eval-2024 lose the zero-shot label. The strict zero-shot claim therefore rests only on the sealed sets, DF40-Holdout and any OOD dataset not present in VALmix. Record this partition explicitly in the results.
- The firewall. Diverse-validation labels select checkpoints and nothing else. Applicability gates, the risk model, defer thresholds and any calibration use FF++ `VAL_meta` only. If the reliability system ever sees diverse-validation labels, the applicability and reliability contributions become target-domain-supervised and the novelty is gone.
- Comparability. If any baseline in the paper was selected on FF++ val while this model is selected on diverse validation, the comparison is not apples-to-apples. Either select all compared models the same way or state the mismatch.
- DF40-Dev is architecture-development data. It may be used for branch-membership and complementarity analysis in Stage 1. It may never enter gradient training, checkpoint selection, gate fitting, thresholds or risk calibration, and it can never be claimed as zero-shot afterward. DF40-Holdout stays sealed until the final table.
- No other OOD label enters training or selection. CDF, DFDC, DFD, and any dataset not designated above stay untouched until final evaluation.
- Numbers come from reading result files programmatically. Unrun cells are `TODO(run)`.
- Within-band effects need a second seed. Treat `|ΔAUC| < 0.01` or inconsistent family-level effects as requiring a confirming seed before any promotion or removal.
- Provenance bookkeeping. Any source whose val split calibrated a gate or the defer policy is labeled calibrated, not zero-shot. V1's Celeb-DF-v2 gated rows were calibrated. DF40-Dev drove design and is never a zero-shot claim, only DF40-Holdout and non-VALmix OOD are.
- Write a short markdown result file at every stage boundary.

## Architecture recap (operator-agnostic novelty)

Three candidate heterogeneous priors, not three assumed branches.

- **Semantic anchor.** CLIP ViT-L/14 plus LoRA, trained on FF++ real/fake. Always on, `q_sem = 1`, undiscounted. This is the generalist and the baseline every specialist must complement.
- **FS-VFM direct expert** (candidate). FS-VFM ViT-L/16 base weights frozen, trainable LoRA adapters and a trainable EDL head, direct readout. Membership decided in Stage 1.
- **MR-VAE rate expert** (candidate). Frozen mechanism-valid MR-VAE producing the multi-rate response, with an EDL head that interprets the response. Membership decided in Stage 1, and separately conditional on the rate-response analysis if it has not been run.

Central EDL conversion to subjective opinions, learned applicability gates `q_b`, applicability discounting, fusion, reliability decomposition, Real/Fake/Defer. The claimed novelty is the combination, heterogeneous experts plus conditional applicability plus applicability-discounted conflict reasoning plus explicit reliability and defer under heterogeneous forgery shift. Not CLIP, LoRA, FS-VFM, MR-VAE, EDL or CCF individually.

---

## Stage 0 — Resume-state discovery (DFB_NeSyNS)

Produce `tbiom/RESUME_MAP.md`. Inventory rather than assume, since a sprint just ran.

1. Checkpoints on disk. The CLIP-LoRA anchor if trained, the FPAD B3 preserve student (the direct adapted-FS-VFM expert this brief may reuse), the frozen MR-VAE, any proc/LDM artifact. Record paths and which are usable as-is.
2. The two queued FPAD diagnostics. Confirm the epoch-9 uncapped validation and the FS-VFM official linear probe landed, and record the linear-probe OOD numbers, because they are an input to Stage 1.
3. The sealed DF40-Dev / DF40-Holdout split file. Reuse verbatim. Record the Dev method list.
4. The diverse-validation set (VALmix). Confirm it is on disk, record its path and its constituent domains (Celeb-DF-v2 non-test, DFDCP official-train, Deepfake-Eval-2024 finetuning-train), and verify no overlap with any final test video or frame. Record which OOD test datasets are therefore domain-seen and which remain strict zero-shot.
5. Reuse inventory. The §20 audit, the EDL head and conversion, the 5-fold gate cross-fitting, the CCF and DS fusion code, the reliability and risk code, the source-paired sampler. Record what exists and what is missing.
6. The proc/LDM branch status. Record whether the diffusion-specific eval that was queued (proc on the eight DF40-Dev diffusion methods) ever ran, and its result. If it never ran or stayed near chance, proc is out and is not a Stage 1 candidate.

Pass condition. `RESUME_MAP.md` answers 1 through 6. The candidate expert set for Stage 1 is fixed here, most likely FS-VFM direct and MR-VAE rate, proc only if the diffusion eval revived it.

---

## Stage 1 — BRANCH MEMBERSHIP GATE (the decision)

This decides which experts enter T-BIOM at all, on complementarity with CLIP, before any fusion training. It is the hard gate. If nothing passes, the architecture collapses to the reliability-centered fallback below, which is a real paper.

### 1.1 Fresh both-sides computation

For CLIP and for each candidate expert, compute per-row DF40-Dev results in the same harness, freshly. Do not inherit CLIP's or any expert's failure rows from a P0-DS, B1 or V1 table. The rescue claim is only as valid as the baseline it is measured against.

### 1.2 Complementarity, three numbers

For each candidate expert, restricted to DF40-Dev, compute all three and never report the first without the others.

- **Rescue conditional probabilities.** `P(expert correct | CLIP wrong)` against `P(expert wrong | CLIP correct)`, per family and pooled. The first should clearly exceed the second, or the expert harms as much as it helps.
- **Complementarity ceiling.** Do not report a label-aware per-sample selector as an oracle video-AUROC, which is an artificial number. Report the oracle as error or accuracy, or as `1 - P(CLIP wrong AND expert wrong)`, the fraction of samples at least one of them gets right. This is the honest upper bound.
- **Realizable-gate recovery.** A cheap cross-fit logistic gate on each expert's own evidence and confidence only, reported as the actual realizable fused AUC and as the fraction of the ceiling-minus-anchor gain it recovers. This fraction, not the ceiling, is the go signal, because a large ceiling with near-zero realizable recovery means the applicability signal is not visible in the features.

### 1.3 Decision

- An expert enters T-BIOM if it shows meaningful rescue over CLIP and a realizable gate recovers a useful portion of its oracle gain. The FS-VFM direct expert is judged here as a plain discriminative expert, not as an anomaly or trajectory. The MR-VAE expert additionally requires that its rate-response carries conditional structure, so run the rate-response and UMAP analysis here if it was never run, and drop the branch if the response is a monotone magnitude rather than family-structured.
- An expert that shows neither rescue nor recoverable complementarity is dropped from the architecture entirely. Three branches for symmetry is not a requirement.
- If no expert passes, go to the reliability-centered fallback. Do not build applicability machinery over experts that carry no recoverable signal.

Write `tbiom/STAGE1_MEMBERSHIP.md` with the fresh CLIP and expert rows, the three complementarity numbers per candidate, the rate-response finding, and the membership decision with justification. Everything downstream is conditional on this.

---

## Stage 2 — Train the anchor and the surviving expert heads

Only for experts that passed Stage 1. Train on FF++ c23 with source-paired sampling and matched benign augmentation on both members of a pair.

- **CLIP-LoRA anchor.** CLIP base frozen, LoRA and the semantic EDL head trained on FF++ real/fake. Always on, undiscounted.
- **Surviving frozen-backbone experts.** FS-VFM and MR-VAE base weights stay frozen, LoRA adapters and the EDL head are trainable. Reuse the FPAD B3 student directly if Stage 1 validated it, otherwise train the LoRA plus EDL head. No trajectory head, direct readout only.
- Central EDL conversion. Each branch emits two-dimensional non-negative evidence `e_b`, converted to `alpha_b = e_b + 1`, `S_b = sum_k alpha_{b,k}`, `b_{b,k} = e_{b,k}/S_b`, `u_b = 2/S_b`, base rate `a = [0.5, 0.5]`. `u_b` is vacuity from lack of evidence, not generic uncertainty.
- Select checkpoints on the diverse-validation macro video AUROC (VALmix, per the ground rules), reporting `VAL_guard` FF++ AUROC alongside as the in-domain guardrail. Never inspect final test results for selection. Log both AUROCs per epoch and watch for the overfitting signature, FF++ AUROC rising while the diverse macro AUROC falls, which is exactly what near-saturated in-domain selection would miss. Record selected epochs.

Pass condition. The anchor and each surviving expert produce opinions, with checkpoints selected on the diverse-validation macro AUROC and the FF++ guardrail logged. `VAL_meta` is untouched and reserved for Stage 3.

---

## Stage 3 — Cross-fitted applicability gates

Learn `q_b(x)` in `[0,1]` for each surviving specialist on FF++ `VAL_meta` only. Applicability answers whether the specialist improves the collective decision for this sample. It is not confidence, so `q_b` is not `1 - u_b`.

- Target via counterfactual marginal utility, `U(S) = -CE(p_fusion(S), y)`, with the form matched to how many specialists survived Stage 1. One specialist, `phi_b = U(A+b) - U(A)`. Two specialists, the anchor-conditioned Shapley value `phi_b = 0.5[U(A+b) - U(A)] + 0.5[U(A+b+other) - U(A+other)]`. More than two, exact Shapley over the optional experts if cheap, otherwise a predetermined Monte-Carlo Shapley approximation with a fixed sample count. Target `t_b = 1[phi_b > 0]`.
- Per-operator gates, or the fusion comparison is rigged. Train `q_b^CCF` from CCF marginal utility and `q_b^DS` from DS marginal utility. Using a CCF-trained `q` inside DS, or the reverse, biases the Stage 4 comparison toward whichever operator the gate was trained on.
- Gate inputs may contain branch probabilities, vacuity and compact branch diagnostics. Never label, dataset, manipulation or generator identity.
- Cross-fit with 5 folds so no sample scores its own gate. Produce the out-of-fold `q` the risk model consumes.

Pass condition. Gates train and produce out-of-fold `q`. Judge them on three things, `AUROC(q_b, t_b) > 0.5`, accuracy or BCE against the target-prevalence constant baseline, and most importantly the actual downstream fused gain from discounting. A gate that beats prevalence but yields no downstream gain has not earned its place.

---

## Stage 4 — Fusion comparison

Apply applicability discounting before fusion. `b'_{b,k} = q_b b_{b,k}`, `u'_b = (1 - q_b) + q_b u_b`, so an inapplicable specialist contributes ignorance, not evidence for Real. The anchor is undiscounted.

Four arms, using the per-operator gates from Stage 3.

- **Undiscounted CCF.** CCF with all `q_b = 1`. Note this is not equal-importance weighting. It removes applicability discounting only. CCF still treats opinions differently because their belief and vacuity states differ, so do not describe it as equal-weight.
- **Applicability-CCF** (`q^CCF`), primary candidate. Multi-source Consensus and Compromise Fusion of van der Heijden, Kopp, Kargl, "Multi-Source Fusion Operations in Subjective Logic", FUSION 2018, DOI 10.23919/ICIF.2018.8455615. Implement the genuine multi-source operator, not sequential pairwise fusion, since it is non-associative.
- **Undiscounted DS** and **Applicability-DS** (`q^DS`). Dempster-Shafer with `q_b = 1` and with DS-trained applicability. The DS pair is the fusion-operator comparator against the CCF pair.

The two comparisons that matter. Undiscounted-CCF against Applicability-CCF isolates whether learned applicability helps under CCF. The CCF pair against the DS pair isolates whether the operator matters, each with its own correctly trained gate. If a genuine equal-importance baseline is wanted, add simple equal probability or evidence averaging as a separate row rather than misreading undiscounted CCF as that.

Only `V = u_f` and the fused margin depend on the operator, so recalibrate those two per arm. C and U_sup are pre-fusion branch quantities and carry over unchanged.

Synthetic unit tests for the multi-source CCF implementation, run before trusting it on data. Vacuous opinions, perfect agreement, strong disagreement, one inapplicable expert, all specialists inapplicable, and permutation or order invariance. A multi-source operator that is not order-invariant on these is implemented as pairwise chaining and must be fixed.

Pass condition. All four arms plus the equal-average control produce opinions, the synthetic tests pass, and V and the fused margin are recalibrated per arm.

---

## Stage 5 — Reliability, risk and Real/Fake/Defer

Compute from the fused opinion and branch states.

- Fused vacuity `V = u_f`.
- Informative conflict `C = [sum_{i<j} w_i w_j D_JS(p_i, p_j)] / [sum_{i<j} w_i w_j + eps]`, `w_b = q_b (1 - u_b)`.
- Support deficit `U_sup = 1 - (1/|E|) sum_{b in experts} q_b (1 - u_b)`. Name it `U_sup`, not A.

Fit the logistic risk model `R(x) = g(V, C, U_sup, M_f)`, `M_f = |p_f(Fake) - 0.5|`, target being whether the fused prediction is wrong, on out-of-fold FF++ `VAL_meta` predictions only. Keep it around five parameters so coefficients stay readable. Emit Real/Fake/Defer and the risk-coverage curve. Report error-detection AUROC and selective risk at the 10 percent abstention budget, mirroring the V1 reliability report for direct comparison.

Pass condition. Coefficient signs match the reading, more vacuity, more informative conflict and more support deficit each raise risk, a more decided margin lowers it. Selective risk falls against full coverage. This must survive, since it is the paper's spine.

---

## Stage 6 — Final untouched evaluation

Freeze everything. Evaluate on the full OOD suite (CDFv1/v2/v3, DFD, DFDCP, DFDC, UADFV, Deepfake-Eval-2024) and on DF40-Holdout, read for the first time here. Produce the family and method breakdown.

Keep four provenance-distinct comparison columns distinct, this system, plain DS ungated, anchor-only, and the reproduced baseline. Any unrun cell is `TODO(run)`. Apply the calibrated-vs-zero-shot and Dev-vs-Holdout rules from the ground rules.

---

## Reliability-centered fallback (if Stage 1 passes nothing)

This is a legitimate strong paper, not a failure, but the reliability model changes shape and the current V/C/U_sup decomposition cannot be kept as written. With only the CLIP anchor there is no inter-branch conflict `C` and no specialists to form `U_sup`, so those two terms do not exist. Manufacturing them from one branch would be wrong.

Use the single-branch risk model `R_fallback(x) = g(V_sem, M_sem)`, where `V_sem` is the anchor's vacuity and `M_sem = |p_sem,F - 0.5|` is its margin. A genuine extra uncertainty signal can be added later if one is available, but do not fabricate `C` or `U_sup` when there are no specialists to compute them from. Train the anchor, fit this two-term risk model on FF++ `VAL_meta`, and center the contribution on selective prediction and defer under forgery shift. V1 already demonstrated the reliability-and-defer story works.

Note what this implies. The full V/C/U_sup decomposition, which is the paper's strongest reliability result, itself requires at least one complementary specialist to exist. That is a further reason Stage 1 is the decisive stage and not a formality. The paper claim in the fallback becomes reliability and abstention on a single strong anchor rather than multi-expert rescue, and the evaluation axis is the risk-coverage curve rather than a rescue table.

---

## Seven candidate T-BIOM backlog items (record, do not build now)

- FPAD as a detection-only expert, kept only if Stage 1 complementarity passes, dropped otherwise.
- I-JEPA as a spatial-predictive dual-prior expert, and a temporal face-motion prior for the reenactment family. Both enter only through the Stage 1 membership gate, and any localization use must be mask-supervised and decoupled from the detection objective, per the FPAD finding.

## Novelty framing

Do not claim novelty for the parts. The contribution is conditional specialist applicability learned by counterfactual marginal utility, applicability-discounted conflict reasoning, and explicit reliability with a Real/Fake/Defer output, under heterogeneous forgery shift. CCF is a fusion finding reported if Applicability-CCF beats Applicability-DS and Equal-CCF, not a load-bearing claim. The question every experiment answers is whether each specialist provides unique rescue where the anchor fails, and whether applicability-aware fusion exploits that rescue without harming cases where the anchor is already strong.