# DISCERN-v2 Phase-2 — Claude Code Implementation Brief

This brief is for a coding agent operating inside the live DISCERN/DFB_NeSyNS repository. It is staged. Every stage ends with a pass condition. Do not begin a stage until the prior stage's pass condition is met and its artifacts are written to disk. A hard gate sits after Stage 3 and can terminate the multi-specialist build entirely.

Ground rules for the whole brief.

- Resolve real names before running anything. Paths, script names, config keys and dataset locations in this brief are described by role, not by literal path, because the tree has not been confirmed. Discover the real name, then use it. Never invent a path such as `/data/VGGFace2/...`.
- No calibration or selection may see any non-FF++ label. Applicability targets, gate cross-fitting, risk-model fitting, thresholds and checkpoint selection use FF++ only. CDF, DFDC, DFD, Deepfake-Eval and DF40 labels never enter calibration.
- Every promotion, removal or architecture change needs a second seed when the effect sits inside the noise band. Treat `|ΔAUC| < 0.01` or inconsistent family-level effects as requiring a confirming seed before acting.
- Write a short markdown result file at the end of every stage. Numbers in those files come from reading result files programmatically, never from memory or estimation. Leave any unrun cell as `TODO(run)` rather than filling it.

---

## Stage -1 — Repository discovery (run once, before Stage 0)

Goal is to resolve the real tree so later commands carry no placeholders. Produce a file `phase2/REPO_MAP.md` recording what you found.

Discover and record:

1. The V1 evaluation entry point. Find the script that produced the V1 video-AUROC tables (the one that emits per-source AUROC and per-branch AUROC for `sem`, `ref`, `proc`). Record its path, its config file, and how source lists are passed.
2. The per-sample dump path. V1 wrote a per-sample parquet with per-branch `p`/`u`, residual diagnostics, process statistics and V/C/A. Find it and record the column schema. Complementarity and oracle analysis read this, not raw logits.
3. DF40 staging. Locate the DF40 test data on disk and record which methods are present as directories. Confirm specifically whether `danet`, `mcnet`, `tpsm`, `facevid2vid` (Celeb-DF source) and `heygen` are staged. Record whether the V1 eval harness already has a code path that loads DF40 methods, or whether adding them needs a new loader or manifest.
4. The FS-VFM reference branch code. Find where `P_R` is fit and where its residual normalization statistics are stored. Record which real set it is currently fit on (V1 used FF++ reals only) and where those statistics live.
5. The MR-VAE / P1d operator. Find the Phase-1 mechanism-valid MR-VAE implementation and its frozen checkpoint. Record the rate schedule `{beta_1..beta_M}` it was trained with.
6. Candidate diverse-real corpora already on the machine. List every real-face dataset directory present (FFHQ, CelebA-HQ, VGGFace2, LAION-face crops, anything). Do not pick one yet. Record path, approximate size, and any identity metadata available. Flag any dataset whose identities are known to overlap Celeb-DF or DFDC subjects.
7. The training entry point and the checkpoint-selection code, including how source-paired sampling would be toggled if it exists, and the LoRA config used for the CLIP branch.

Pass condition. `phase2/REPO_MAP.md` exists and answers items 1 through 7. If any item cannot be resolved, stop and report the specific gap rather than guessing. Items 3 and 6 are the two that most change downstream effort, so surface them explicitly.

---

## Stage 0 — Current V1 on DF40-Dev (rescue, harm, complementarity)

This stage tests whether the *current* `ref`/`proc` specialists carry conditional information on the families where Phase-1 saw rescue. It does not yet decide the architecture.

### 0.1 Define the Dev/Holdout split first

Before evaluating anything, write `phase2/DF40_SPLIT.md` fixing two disjoint sets.

- **DF40-Dev** — every DF40 method already inspected during Phase 1 or touched in this stage. At minimum `danet`, `mcnet`, `tpsm`, `facevid2vid` (cdf), `heygen`. Any method whose numbers inform an architecture decision belongs here permanently.
- **DF40-Holdout** — untouched families reserved for final evaluation only. Choose these now and record them as sealed. No configuration, gate, threshold or promotion decision may ever read a Holdout number before Stage 8.

Once a method is in Dev it cannot move to Holdout. This split is the guard that lets Stage 8 present Holdout as genuine zero-shot evidence.

### 0.2 Evaluate current V1 on DF40-Dev

Using the V1 entry point resolved in Stage -1, evaluate the frozen V1 checkpoint on DF40-Dev. Emit, per method, video-level AUROC for the fused system and for each branch `sem`, `ref`, `proc` separately. If the harness has no DF40 code path, build the minimal loader needed and record in `REPO_MAP.md` that you added it.

### 0.3 Complementarity — two numbers, not one

Read the per-sample parquet. Restrict to DF40-Dev. Compute both of the following and never report the first without the second.

- **Oracle upper bound.** For each sample allow an ideal selector to pick the single best available expert among `{sem, ref, proc}` (and later P1d). Report the video-AUROC of that oracle per method and pooled. This is an unrealizable ceiling.
- **Realizable-gate recovery.** Fit a cheap logistic gate per specialist on that specialist's own evidence and confidence only (its `p`, its `u`, compact branch diagnostics), with the applicability target defined as in Stage 5. Cross-fit it so no sample scores its own gate. Report the video-AUROC of the realizable-gated system, and report the **fraction of the oracle-minus-anchor gain that the realizable gate recovers**.

The recovered fraction, not the oracle, is the decision signal. A large oracle with near-zero realizable recovery means the applicability signal is not visible in the features and the reasoner is speculative.

### 0.4 Rescue and harm

Per DF40-Dev method, tabulate rescue (specialist AUROC minus anchor AUROC where anchor is weak, especially where anchor sits below chance) and harm (specialist minus anchor where anchor is already strong, `heygen` being the key harm probe).

Write `phase2/STAGE0_DF40DEV.md` with the branch AUROC table, the oracle and realizable-recovery numbers, and the rescue/harm table.

Pass condition. Stage 0 always completes and is never itself a stop. It produces the numbers the Stage 3 gate consumes. Record, but do not yet act on, whether current `ref`/`proc` show rescue on the inverted-anchor rows.

---

## Stage 1 — Diverse-real FS-VFM reference and full domain audit

The V1 `ref` branch is a dataset detector because `P_R` was fit on FF++ reals only, so its residual measured "unlike FF++ authentic" rather than "unlike authentic". Refit on diverse reals and re-audit.

### 1.1 Choose the corpus deliberately

From the candidates listed in Stage -1, choose the diverse-real set. Do not default to VGGFace2 merely because FS-VFM was pretrained on it. Two constraints govern the choice.

- Identity disjointness from test reals. The corpus must not share identities with Celeb-DF or DFDC subjects. If the only large corpus available has such overlap, document the overlap and quantify its likely effect rather than proceeding silently, because a frozen `P_R` fit on familiar identities turns partly into an identity-familiarity detector on those sources.
- Distributional breadth. Prefer coverage across capture conditions and demographics over raw count.

Record the choice and the identity-overlap analysis in `phase2/STAGE1_REFERENCE.md`.

### 1.2 Refit and freeze

Refit `P_R` on the chosen diverse-real set. Start with the deterministic AE variant. Fit residual normalization statistics on the same diverse-real population, not on FF++. Freeze `P_R` and its statistics permanently before any detector training.

### 1.3 Re-run the complete §20 audit

Re-run the provenance/domain audit on the refit `ref` branch across all seven conventional and wild sources. Audit not only `ref_residual_norm` and `ref_angle` but also the head outputs `p_ref` and `u_ref`. This matters because V1 showed the head partly laundering the provenance signal — raw residual failed the audit while `p_ref`/`u_ref` passed on several sources — so a clean raw residual after the refit does not by itself prove the head is clean. Report the forensic-vs-domain gap per quantity per source, same format as the V1 audit.

Pass condition. The refit `ref` branch clears the domain audit on a majority of the seven sources on both the raw residual quantities and the head outputs, or you document specifically which sources still fail and why. If the branch still reads provenance after a diverse-real refit, the deterministic AE is not the problem to escalate yet — record it for the Stage 3 decision rather than swapping to a VAE variant now.

---

## Stage 2 — Mechanism-valid MR-VAE (P1d) on DF40-Dev

P1d is a separate specialist candidate from `ref`/`proc` and must be judged on its own, because the current pair failing does not condemn conditional specialist reasoning.

### 2.1 Wire the frozen MR-VAE as a branch

Attach the Phase-1 mechanism-valid MR-VAE as Branch C. Compute the multi-rate response `R(x) = [r_{beta_1}, ..., r_{beta_M}]` using the exact rate schedule the operator was trained with. The evidence head interprets the response vector. Do not hardcode "larger residual means fake". MR-VAE stays frozen during detector training.

### 2.2 Validate on DF40-Dev

Evaluate the P1d branch on DF40-Dev, especially the reenactment families where Phase-1 reported rescue. Add P1d into the oracle and realizable-recovery computation from Stage 0.3 so the complementarity numbers now cover `{sem, ref, proc, P1d}`.

### 2.3 Rate-response structure

Produce the rate-response and embedding analysis that was flagged twice before and, per the handoff, was never run. For DF40-Dev samples, plot `R(x)` component behavior by family and run UMAP or t-SNE on the response vectors. The question is whether `R(x)` carries family-specific structure or is a monotone magnitude. P1d enters the final architecture only if this analysis shows conditional structure, not on faith.

Write `phase2/STAGE2_MRVAE.md` with the DF40-Dev branch AUROC, the updated complementarity numbers, and the rate-response finding.

Pass condition. Stage 2 always completes. It records whether P1d shows conditional information and rescue. No stop here.

---

## Stage 3 — HARD GATE: choose specialists on conditional information

This is the decision point. Read the recovered-fraction numbers from Stages 0 and 2 across all candidate specialists `{ref (refit), proc, P1d}`.

Gate logic.

- If at least one candidate shows meaningful oracle complementarity on DF40-Dev **and** a realizable gate recovers a useful portion of it, proceed to Stage 4 with the specialists that passed. Drop any specialist that shows neither conditional information nor audit compliance, per the §5 gate. In particular, drop `proc` if it clears neither bar after Stage 1, and keep `ref` only if the refit fixed its audit and it carries rescue.
- If no candidate — including the validated P1d branch — has meaningful complementarity and no realizable gate recovers a useful portion, stop the multi-specialist build. Center the paper on the reliability decomposition and defer, which V1 already showed working. Do not spend a week building a reasoner over specialists that carry no recoverable signal.

Write `phase2/STAGE3_GATE.md` recording which branch reached the gate, the recovered-fraction each produced, and the go or no-go decision with its justification. Everything after this stage is conditional on a go.

---

## Stage 4 — Source-paired FF++ training

Train Branch A and the three evidence heads. Reference and MR-VAE mechanisms remain frozen throughout.

- Train the CLIP ViT-L/14 branch with the validated LoRA recipe, CLIP base frozen, adapters and the semantic evidence head trained on FF++ Real/Fake. The LoRA recipe is a CLIP-adaptation choice only, not an adoption of any external fusion architecture.
- Each branch emits two-dimensional non-negative evidence `e_b = [e_{b,R}, e_{b,F}]`. Convert centrally to a subjective opinion with `alpha_b = e_b + 1`, `S_b = sum_k alpha_{b,k}`, `b_{b,k} = e_{b,k}/S_b`, `u_b = 2/S_b`, base rate `a = [0.5, 0.5]`. `u_b` is vacuity from lack of evidence, not generic epistemic uncertainty.
- Sampling is source-paired Real/Fake, matched benign augmentation applied to both members of a pair where possible. Keep random sampling as a recorded control run. Source-paired is the primary, because the V1 audit showed a provenance shortcut and paired sampling attacks that shortcut at the sampler.
- Do not add SBI. It stays an isolated later ablation with its own confirming seed, because it manufactures swap-like blending boundaries and could sharpen the swap side while doing nothing for the reenactment rescue that is the whole specialization thesis.
- Select the checkpoint on FF++ validation only. Record the selected epoch. Note in the result file that V1 diagnostics showed in-domain validation saturating in epoch 0 with a between-model to within-model signal ratio near 2, so treat checkpoint selection as close to arbitrary and do not over-read small validation differences.

Pass condition. Training completes, the checkpoint is selected on FF++ only, and both the source-paired run and the random-sampling control exist for comparison.

---

## Stage 5 — Cross-fitted applicability gates

Learn `q_ref(x)`, `q_rate(x)` in `[0,1]`. Applicability answers whether a specialist improves the collective decision for this sample. It is not confidence, so `q_b` is not `1 - u_b`.

- Target via counterfactual marginal utility on FF++ validation only. With semantic anchor A, `phi_ref = 0.5[U(A+ref) - U(A)] + 0.5[U(A+ref+rate) - U(A+rate)]`, and symmetrically for rate, where `U(S) = -CE(p_fusion(S), y)`. Target `t_b = 1[phi_b > 0]`. Compute the utility under the fusion operator that will be used downstream, so applicability is learned against the real fusion, not a proxy.
- Gate inputs may contain branch probabilities, vacuity and compact branch diagnostics. They must never contain label, dataset, manipulation or generator identity.
- Cross-fit with 5 folds so no sample scores its own gate, and produce the out-of-fold `q` that the risk model in Stage 7 consumes.

Pass condition. Gates train, out-of-fold `q` is produced, and the gate AUROC beats its always-admit baseline. Record mean `q` per gate.

---

## Stage 6 — Fusion comparison (the central control)

Apply applicability discounting before fusion. `b'_{b,k} = q_b b_{b,k}`, `u'_b = (1 - q_b) + q_b u_b`, so an inapplicable specialist contributes ignorance rather than evidence for Real. The semantic anchor is undiscounted.

Run three fusion arms on identical discounted opinions.

- **Applicability-CCF** — primary candidate. Implement the proper multi-source Consensus and Compromise Fusion of van der Heijden, Kopp and Kargl, "Multi-Source Fusion Operations in Subjective Logic", FUSION 2018, DOI 10.23919/ICIF.2018.8455615. Implement the genuine multi-source operator, not sequential pairwise fusion, because the operator is non-associative and pairwise chaining changes the result.
- **Applicability-DS** — strong baseline and fallback. Dempster-Shafer over the same discounted opinions.
- **Equal-CCF** — the critical control. CCF with all branches at equal standing, `q_b = 1`. This isolates whether learned applicability, not the fusion operator, is what helps.

Because only `V = u_f` and the fused margin depend on the operator, recalibrate those two under each arm. C and U_sup are pre-fusion branch quantities and carry over unchanged.

Pass condition. All three arms produce fused opinions on FF++ validation and the OOD sources, with V and fused margin recalibrated per arm. The comparison Equal-CCF vs Applicability-CCF is the number that tells you whether applicability earns its place.

---

## Stage 7 — Reliability, risk and Real/Fake/Defer

Compute from the fused opinion and branch states.

- Fused vacuity `V = u_f`.
- Informative conflict `C = [sum_{i<j} w_i w_j D_JS(p_i, p_j)] / [sum_{i<j} w_i w_j + eps]` with `w_b = q_b (1 - u_b)`.
- Support deficit `U_sup = 1 - 0.5 sum_{b in {ref,rate}} q_b (1 - u_b)`. Name it `U_sup`, not `A`, to avoid collision with aleatoric uncertainty.

Fit the logistic risk model `R(x) = g(V, C, U_sup, M_f)` with `M_f = |p_f(Fake) - 0.5|`, target being whether the fused prediction is wrong, fit on out-of-fold FF++ meta predictions only. Keep it five-ish parameters so coefficients stay readable. Produce the operational Real/Fake/Defer output and the risk-coverage curve. Report error-detection AUROC and selective risk at the 10 percent abstention budget, mirroring the V1 reliability report so the two are directly comparable.

Pass condition. Coefficient signs match the reliability reading (more vacuity, more informative conflict and more support deficit each raise risk; a more decided margin lowers it), and selective risk falls against full coverage. This is DISCERN's core identity and it must survive the architecture change.

---

## Stage 8 — Final untouched evaluation

Freeze everything. Evaluate on the full OOD suite (CDFv1/v2/v3, DFD, DFDCP, DFDC, UADFV, Deepfake-Eval-2024) and on **DF40-Holdout**, which no earlier stage has read. Produce the family and method breakdown.

Two bookkeeping rules that must survive into the paper.

- Any source whose validation split was used to calibrate gates or the defer policy is labeled calibrated, not zero-shot. V1's Celeb-DF-v2 gated rows were calibrated for exactly this reason. If Stage 5 or 7 touched a source's val split, that source is not OOD.
- DF40-Dev methods can never be presented as zero-shot evidence for the final model, because they drove architecture decisions. Only DF40-Holdout carries the zero-shot claim.

Write `phase2/STAGE8_FINAL.md` with all four provenance-distinct comparison columns kept distinct (this system, plain DS ungated, anchor-only, and the reproduced baseline) and any unrun cell left as `TODO(run)` rather than filled.

---

## Novelty framing (carry into writing, operator-agnostic)

Do not claim novelty for CLIP, LoRA, FS-VFM, MR-VAE, EDL or CCF individually. The contribution is the combination: heterogeneous forensic experts, conditional specialist applicability learned by counterfactual marginal utility, applicability-discounted conflict reasoning, and explicit reliability with a Real/Fake/Defer output. CCF is a fusion finding reported if Applicability-CCF beats Applicability-DS and Equal-CCF, not a load-bearing claim. The Multi-Rate Forensic Response becomes an additional contribution only if its Stage 2 DF40-Dev analysis supports it.

The question every experiment answers. Does each specialist provide unique rescue where the semantic anchor fails, and can applicability-aware fusion exploit that rescue without harming cases where the anchor is already strong.