# DISCERN-v2 / T-BIOM — Recovery Build Brief (locked order)

For a coding agent in DFB_NeSyNS. This replaces the earlier build order after the FF++ ⊕ DF40 collapse. The strategy is frozen. Do not add branches, fusion operators, or uncertainty methods, and do not start the corpus rebuild, until the gates below resolve. The experiments are ordered so the cheapest, most decisive ones run first and each gates the next.

Two things are settled by measurement and are not reopened here. The DF40 training arm is dead, the cause was real-support asymmetry and source-label confounding (a narrow real support against many fake-derived distributions taught "outside the narrow real support implies fake"), not batch imbalance. And AUROC alone hid that collapse, so ranking is never judged without an operational metric beside it.

## Ground rules

- Reuse first, resolve real names before running, never invent a path. Discover the real script, config, and checkpoint names in DFB_NeSyNS, then act.
- Health dashboard, not AUROC alone, on every anchor and every OOD evaluation. Report AUROC (ranking), EER (threshold-independent operating quality), FPR on OOD reals at a threshold frozen once on development data (the operational-collapse headline), and probability separation `Δ_RF = E[p_F | F] - E[p_F | R]`. Continue reporting mean p(fake) on reals as a calibration diagnostic. The collapse signature is FPR_real and mean-p(fake)-on-reals spiking on unfamiliar reals while AUROC barely moves.
- FF++ c23 is the only gradient-training data until Step 9 is explicitly triggered. No external real enters training before then.
- Firewall. Diverse validation (VALmix) selects checkpoints only. Applicability gates, risk model, defer thresholds and calibration use FF++ `VAL_meta` only.
- Zero-shot bookkeeping. VALmix domains (Celeb-DF-v2, DFDCP, Deepfake-Eval-2024) are domain-seen, not zero-shot. Celeb-DF-v1 and v3, DFD, DFDC, UADFV stay strictly zero-shot. Anything the gates touched is calibrated, not zero-shot. DF40-Dev is architecture-development, DF40-Holdout stays sealed.
- Terminology for the writeup. The gate is label-free-at-inference, not label-free (its target uses labels). Cause statements say real-support asymmetry and source-label confounding, not "real diversity." A negative gate result is "current test-time-observable applicability features cannot realize the complementarity," not "gating cannot work."
- Numbers read from result files, unrun cells `TODO(run)`, within-band effects need a confirming seed, write a short markdown result file at every step boundary.

---

## Step 0 — Repo discovery

Produce `tbiom/RECOVERY_MAP.md`. Resolve: the DiCoME reproduction code and released checkpoint path, and specifically whether an accessible DiCoME-direct or CLIP-only readout exists inside it. Whether the earlier successful DiCoME-retrained-on-FF++ path, code, and checkpoint still exist in the tree. The current CLIP-port anchor config (LoRA targets, rank, alpha, dropout, optimizer, WD, LR, augmentation, head, selected epoch). VALmix manifest and its verified disjointness. The FS-VFM preserve and ordinary students and the MR-VAE branch outputs from Stage 1. The §20 domain-audit code. The health-dashboard metrics, adding any missing metric (EER, FPR@τ, Δ_RF).

Pass condition. The map answers all of the above, and in particular says clearly whether a retrainable DiCoME recipe exists, because that changes Step 2.

---

## Step 1 — Like-for-like anchor comparison (cheapest, run first)

Before any recipe search, determine whether the roughly five-point CDFv2 gap belongs to the CLIP branch or to DiCoME's extra machinery. Evaluate three configurations on the same pipeline and the health dashboard, DiCoME-full, DiCoME-direct or CLIP-only if accessible, and the current CLIP port.

- If DiCoME-CLIP-only ≈ the port (near 0.92 CDFv2), there is no CLIP implementation gap. DiCoME's decomposition and internal DS fusion create the difference, and the "gap" is architectural by definition.
- If DiCoME-CLIP-only ≈ 0.97 while the port stays near 0.92, there is a genuine recipe gap. Then and only then investigate LoRA targets, rank/alpha/dropout, optimizer, WD, augmentation, LR, checkpoint epoch, head, preprocessing.

Write `tbiom/STEP1_ANCHOR_DIAG.md`. This resolves whether Step 2 is a recipe fix or a chassis choice.

---

## Step 2 — Recover the strongest FF++-only anchor

Recover the best FF++-only anchor, retrainable. The candidate set is the recovered CLIP port with the recipe fix if Step 1 found one, and the retrainable DiCoME recipe if Step 0 confirmed it exists. Prefer a strong and retrainable anchor over a frozen one, since retrainability is required for the corpus ablation and for any later integration. Frozen released DiCoME is the last resort, taken only if no retrainable path reaches competitive numbers, and taken knowing it forecloses Step 9 on the anchor and forces the fusion-in-fusion discussion.

Train on FF++ c23 only. Checkpoint selection on VALmix macro video AUROC with the FF++ guardrail logged, but real-side FPR@τ overrides AUROC as the health gate. No FFHQ, no SBI, no DF40 in training here.

Pass condition. A retrainable FF++-only anchor is selected and its checkpoint recorded, chosen on the dashboard with real-side health explicitly checked.

---

## Step 3 — Anchor health dashboard

Evaluate the chosen anchor across VALmix and the untouched diagnostic domains on the full dashboard, AUROC, EER, FPR_real@τ, Δ_RF, mean p(fake) on reals. The decisive question is whether the FF++-only anchor shows real-side collapse on unfamiliar reals. Write `tbiom/STEP3_ANCHOR_HEALTH.md`.

Pass condition and branch. If real-side health is intact (FPR_real controlled, Δ_RF preserved on domains whose reals were unseen), the corpus is fine and Step 9 is skipped. If real-side collapse persists, Step 9 is triggered later. Record which.

---

## Step 4 — Single-anchor reliability gate (early, decisive)

Do not assume the reliability spine transfers. The validated `R = g(V, C, U_sup, M)` used multiple branches. With one anchor, C and U_sup do not exist, so test the actual fallback `R = g(V_sem, M_sem)` on FF++ `VAL_meta`, out-of-fold. Ask directly whether anchor vacuity and margin alone predict errors under OOD shift, reported as error-detection AUROC and selective risk at the 10 percent budget, comparable to V1's 0.81.

Write `tbiom/STEP4_RELIABILITY.md`. If `(V, M)` alone predicts errors well, single-anchor reliability is a viable spine. If not, the reliability paper needs another principled uncertainty source and must not fabricate C or U_sup from one branch. This result gates whether reliability can carry the paper.

---

## Step 5 — Re-test expert complementarity against the chosen anchor

Complementarity was only ever measured against the weaker CLIP port. Re-run the membership test for the FS-VFM preserve/ordinary experts and the MR-VAE branch against the Step 2 anchor, on DF40-Dev and VALmix, freshly scored both sides. The criterion is conditional information, not standalone AUC. An expert may sit below the anchor's AUC and still qualify if `P(expert correct | anchor wrong)` is substantial, harm is low enough, and a realizable gate can locate those cases. Report rescue, harm, the honest ceiling `1 - P(both wrong)`, and the realizable-gate recovery fraction.

Expect the stronger anchor to make this harder, not easier, since rescue is measured against fewer anchor errors. Write `tbiom/STEP5_COMPLEMENTARITY.md`. This decides whether any fusion path is live at all.

---

## Step 6 — Realizability ladder with the domain-audit clamp

This is the highest-novelty experiment. Reframe the question as, at what level of test-time-observable information does expert complementarity become predictable under shift. Train the applicability gate on progressively richer inputs, all label-free at inference.

- G0, confidence only `[p_b, u_b]`.
- G1, cross-branch state `[p_A, p_b, u_A, u_b]`.
- G2, relational `[p_A, p_b, u_A, u_b, |p_A - p_b|, D_JS(p_A, p_b)]`.
- G3, compact branch diagnostics (MR-VAE rate statistics, FS-VFM representation diagnostics).
- G4, a small learned probe on frozen `[h_A ‖ h_b]`, detectors not fine-tuned.

Report at each rung the realizability ratio `rho_realize = (Err(A) - Err(gated)) / (Err(A) - Err(oracle))`.

Audit clamp, mandatory, especially on G3 and G4. Run the §20 forensic-versus-domain separability audit on the gate itself. A high-capacity gate can appear to realize complementarity by recognizing dataset provenance rather than genuine applicability, which is the project's recurring failure mode. A rung that only passes because it reads domain does not count as realizing complementarity and must be reported as failing the audit.

Also test the target, since `t_b = 1[phi_b > 0]` gives opposite labels to near-zero phi. Compare it against a margin-filtered target (ignore `|phi_b| <= delta`) and against continuous regression of phi_b. This may be why gate AUROC sits near chance.

Write `tbiom/STEP6_REALIZABILITY.md` with per-rung rho, the audit verdict per rung, and the target-variant comparison.

---

## Step 7 — Applicability / fusion framework (conditional)

Only if useful experts survived Step 5 and a ladder rung realized complementarity while passing the audit. Build the applicability-discounted fusion and reliability decomposition on the surviving experts, with the fusion comparison (undiscounted CCF, applicability-CCF with per-operator gates, DS comparator) and the full V/C/U_sup risk model. Otherwise skip.

---

## Step 8 — DiCoME fork (conditional)

Only if DiCoME is the strongest anchor and experts survive complementarity against DiCoME specifically. Fork a clean copy, keep the original DiCoME reproduction untouched as baseline, and in the fork use DiCoME as the anchor, add FS-VFM/MR-VAE only if they pass the Step 5 criterion (rescue plus low harm plus recoverable applicability) against DiCoME, and port the applicability gates, discounting, and reliability decomposition. Do not fuse on top of DiCoME's internal DS fusion blindly. First compare treating DiCoME's fused output as one anchor opinion against exposing its internal branch opinions for one unified outer fusion, and prefer the unified formulation if it performs similarly or better. Do not port the framework before complementarity against DiCoME is demonstrated.

---

## Step 9 — FFHQ / SBI corpus ablation (conditional)

Only if Step 3 showed the FF++-only anchor still has real-side collapse. Then attack real-support asymmetry as an ablation so the cause of any fix is legible, FF++, FF++ +FFHQ, FF++ +SBI, FF++ +FFHQ+SBI. Rules. FFHQ-256 broadens real support (stills, so appearance not motion). Every added real domain gets matched fake or synthetic counterparts where possible, so generate SBI counterparts from FFHQ as well, or FFHQ-implies-real becomes a new label shortcut. SBI provides paired manipulation diversity on video-derived frames but does not itself provide a temporal-motion prior, do not claim it fixes motion diversity. Celeb-DF reals stay out to protect CDFv1/v2 zero-shot. Judge every arm on real-side FPR@τ first, AUROC second.

---

## The paper follows the evidence

- Ladder realizes complementarity while passing the audit, plus experts survive, gives an applicability and realizability framework with reliability.
- Ladder fails the audit but single-anchor `(V, M)` reliability is strong, gives a reliability and defer paper, with the realizability gap reported as a characterized negative.
- Experts complement DiCoME and are gateable, build the DiCoME-based multi-prior framework.
- Both single-anchor reliability and realizability fail, stop forcing the T-BIOM thesis and redesign rather than manufacture a result.

Do not add I-JEPA, V-JEPA, another fusion operator, or another uncertainty method until these steps answer the questions already in front of the project.