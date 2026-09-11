# DISCERN-v2 — Branch-Combination Phase (continues the recovery brief)

For the coding agent in the live tree (DFB_NeSyNS, with DiCoME and DISCERN_Ext). This continues `discern-v2-recovery-brief.md` after the anchor work. The goal is one complete three-branch framework that retains P0-DS across its strong datasets while capturing P1d's DFDCP and real-side benefit and P2a's complementarity. The deliverable is one fused framework number per dataset on the health dashboard, not per-branch tables.

Hard constraints, fixed for this phase.

- Three evidence branches maximum. No fourth branch. The MR-VAE rate response is evidence inside Branch 2, not a branch of its own.
- Do not change the backbone during this phase. Backbone swaps and branch/fusion changes cannot run together or attribution is lost.
- Health dashboard is primary, AUROC (ranking), EER, real-side FPR at a threshold frozen once per model on its own FF++ val, and real-versus-fake separation. AUROC never decides alone, since it hid the earlier collapse.
- One variable at a time. Each stage changes one thing against a fixed comparison point.
- FF++ c23 only for gradient training. VALmix macro AUROC for checkpoint selection with FF++ guardrail logged. DF40-Dev is analysis-only, DF40-Holdout sealed. Numbers read from files. Within-band effects (`|ΔAUC| < 0.01`) need a confirming seed.

## Target framework (three information sources)

- **Branch 1, semantic.** The strongest retrainable DiCoME-style CLIP representation, keeping the auxiliary alignment/VAE gradient that was shown to help the shared CLIP features. Internally the semantic branch. It is not a privileged anchor and does not receive special fusion weight, and "anchor" is not claimed as a contribution.
- **Branch 2, manifold/rate.** The β-VAE artifact path preserved, augmented internally with the MR-VAE rate response. One EDL head over `[f_a^β ‖ normalized rate curve]`. This is the updated β-VAE, see the Branch-2 decision rule below.
- **Branch 3, generative-process.** P2a, a frozen generative reconstruction residual (SDXL-VAE), mechanistically distinct from the CLIP/manifold path.

P1b (β-TCVAE) is dropped from the active architecture. Do not add its TC objective unless later evidence specifically justifies it.

---

## Stage 0 — Discovery

Produce `tbiom/BRANCH_MAP.md`. Resolve from the tree, do not assume: the β-VAE projector and its alignment/EDL losses, the MR-VAE projector and rate_branch and the FiLM/BETA_GRID rate-response code, the model file with view construction and DS fusion, the config schema and LoRA injection, and the corrected video-grouping evaluation script with the health-dashboard metrics. Two paths the earlier docs did not give and that block Branch 3, find them explicitly: the P2a/AEROBLADE SDXL-VAE residual module, and the model file where P2a's third view attaches and fuses. If either is missing, report it before proceeding.

Pass condition. The map resolves all paths above, P2a module and wiring included.

---

## Stage 1 — Full seven-dataset MR-VAE health re-score (load-bearing gap)

MR-VAE's operational advantage is established on only three of seven datasets, and its AUROC losses on CDFv3 (−0.066) and DFEval24 (−0.027) were never re-scored on real-side metrics. Re-score MR-VAE on all seven datasets on AUROC, EER, real-side FPR at frozen τ, and separation, with CDFv3 and DFEval24 as the decisive rows.

Pass condition and branch. If MR-VAE's real-side advantage holds broadly and CDFv3/DFEval24 do not carry a large FPR penalty, its rate response is a live Branch-2 ingredient. If CDFv3 or DFEval24 collapse on real-side FPR the way their AUROC did, the rate response cannot enter as fused evidence on those axes and must be gated per the Branch-2 rule. Record which.

---

## Stage 2 — Experiment A, MR-VAE projector only

Disentangle the two changes P1d conflated. Run the MR-VAE projector with no rate concatenation, producing a 64-d artifact exactly like β-VAE's. This isolates whether the projector itself carries DFDCP's +0.038, separate from the rate curve. Evaluate on the dashboard.

Pass condition. Stage 2 always completes and reports whether the projector alone moves DFDCP and CDFv3. It informs, it does not by itself decide Branch 2.

---

## Stage 3 — Experiment B, β-VAE artifact + MR-VAE rate curve

The preferred Branch 2. Keep the β-VAE projector unchanged, compute the rate response `r_rate = [d_0.1, d_0.32, d_1, d_3.16, d_10]` from the MR-VAE operator, normalize it (record the normalization, per-dimension standardization on FF++ train statistics is the default), and concatenate to the β-VAE artifact feature into one EDL head. The design intent is to keep β-VAE's CDFv3 strength while importing MR-VAE's DFDCP and real-side gain. Evaluate on the dashboard, and evaluate it inside the fused framework, not as a standalone branch, since Branch 2 is judged by complete-framework performance.

Pass condition. Stage 3 completes with dashboard and in-framework numbers for the β-VAE-plus-rate Branch 2.

---

## Stage 4 — Branch-2 decision rule (with the standing instruction)

Choose the Branch-2 formulation from complete-framework performance and health, not standalone AUROC. The standing decision is that the updated β-VAE (β-VAE artifact + MR-VAE rate response) is the default Branch 2 and ships even if its gain is not significant, for differentiation and possible real-side robustness. This is subject to one guardrail.

- If the rate response is neutral or positive on the strong axes (CDFv3 AUROC and real-side FPR within noise or better), ship it as Branch 2. Report its actual contribution honestly, including "comparable AUROC with a real-side or interpretability benefit" if that is what the numbers show. Do not oversell it as a driver of gains it did not produce.
- If the rate response degrades CDFv3 AUROC or real-side FPR beyond the noise band, do not ship the regression. Instead gate or down-weight the rate features (a learnable scalar gate on the rate block, or demote the rate curve to an auxiliary explanatory output that is not fused into `e_art`). Branch 2 then remains β-VAE with the rate response present but non-degrading.
- If Experiment A's projector beats Experiment B's concat form on the full framework and does not hurt CDFv3, that projector may be used instead, still as one branch.

The principle is that the updated β-VAE is retained for differentiation as long as it does not cost the CDFv3 strength that is the reason to keep β-VAE at all. Neutral is acceptable, regression is not.

---

## Stage 5 — Branch 3, wire P2a

Attach the P2a SDXL-VAE process residual as the third evidence branch using the paths from Stage 0. P2a is mechanistically independent, so it is the genuine third view. Watch that its weak datasets do not drag the strong ones, and if fusion lets it, hold that concern for the fusion stage rather than hand-weighting it here.

Pass condition. The three branches produce evidence opinions on FF++ val and the OOD suite, checkpoints selected on VALmix macro AUROC with the FF++ guardrail, real-side FPR checked.

---

## Stage 6 — Assemble and compare simple fusion against P0-DS

Fuse the three branches with simple baselines first, equal-weight evidence averaging and standard DS, before anything sophisticated. Report one complete-framework number per dataset on the full dashboard against P0-DS.

Target. Retain P0-DS on CDFv2, CDFv3, DFDC, and DFEval24 within noise, capture P1d's DFDCP benefit, and capture useful P2a complementarity, with real-side FPR at least as good as P0-DS. Write `tbiom/FRAMEWORK_RESULTS.md`.

Pass condition. The three-branch framework matches or beats P0-DS on its strong datasets and gains on DFDCP, on complete-framework numbers. If a simple-fusion arm suppresses a strong single view (for example the fused output loses to the best branch on CDFv2, the defect already seen with DS), record it, since that is what motivates Stage 7.

---

## Stage 7 — Conflict-aware fusion (only if Stage 6 shows suppression)

Only run if Stage 6 measured a strong view being suppressed by simple fusion. The measured problem is how to fuse heterogeneous evidence without suppressing a strong view, which is what a conflict-aware operator targets. Compare conflict-aware fusion against the best simple-fusion arm on the full dashboard. Adopt it only if it fixes the suppression without harming the strong datasets.

Pass condition. Conflict-aware fusion beats simple fusion on the suppressed cases while holding the strong datasets, or it is not adopted.

---

## Stage 8 — Applicability (only if it demonstrably helps)

Only after the branch combination and fusion are healthy. Test whether a per-sample applicability gate adds anything beyond conflict-aware fusion. Do not force it in because it was in the original plan. It enters only if it improves complete-framework results over the Stage 7 operator, judged on the dashboard, and passes the domain-audit guard so it is not routing on dataset provenance.

Pass condition. Applicability improves the framework over conflict-aware fusion and survives the audit, or it stays out and the paper's mechanism is the branch combination plus conflict-aware fusion.

---

## Sequencing and attribution

Run in order. Stage 1 and the two experiments (2, 3) settle Branch 2 before wiring. Then Branch 3, then simple fusion, then conflict-aware only on measured need, then applicability only on measured benefit. Never change the backbone during any of this. Every within-band decision gets a confirming seed. The paper's novelty is decided by what the measurements support in this order, not asserted in advance, and the reported result is one framework number per dataset, not per-branch wins.