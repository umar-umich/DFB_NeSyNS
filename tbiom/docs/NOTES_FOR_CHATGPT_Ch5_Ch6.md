# Revision notes — Chapters 5 and 6, and the front matter

For ChatGPT. Source: `Toward_Generalizable_Deepfake_Video_Detection_Oct_15.pdf` (57 pp).
Chapter 5 begins p. 28 (§5.1 p. 53 of the internal numbering), Chapter 6 p. 52 (§6.1 p. 77).

Goal: turn Chapters 5–6 from "a plan with preliminary numbers" into **a near-complete paper**.
Almost everything Chapter 5 proposes to measure has now been measured. The chapter must stop
asking questions it already answers.

All numbers below are final and come from
`tbiom/reports/branch_fusion/PROPOSAL_TABLES_3.md` (paper tables) with the fuller grids in
`PROPOSAL_TABLES_2.md`. **Do not invent numbers. If a number is not in this file, leave a
placeholder and flag it.**

---

## 1. THE FRAMEWORK COMPOSITION CHANGED — this is the single biggest edit

The proposal defines CROSS-EXAM as **APP + LFS + TD**. It is now **VD + LFS + STT**.

| proposal | revision | what it is |
|---|---|---|
| TD | **VD — Video Dynamics** | V-JEPA 2 ViT-L, LoRA on encoder q/v, 16-frame clip. Rename only. |
| LFS | **LFS — Latent Facial Structure** | CLIP + MR-VAE orthogonal residual. Unchanged. |
| APP *(removed)* | **STT — Spatio-Temporal Texture** | PE-Core-L block-16 patch mean (not the final layer) → 256-d projection → temporal mixer with **real relative timestamps**, masked mean over frames. |

Use the full names once at first mention in the ablation table — *Video Dynamics (VD)*, *Latent
Facial Structure (LFS)*, *Spatio-Temporal Texture (STT)* — and the abbreviations thereafter.
In every comparison against other methods the system is called **CROSS-EXAM**, not a list of
its branches.

### 1.1 The APP hypothesis is now answered, and the answer is negative

§5.1 states Hypothesis 1 as "the separate APP path contributes useful information beyond LFS
and TD". **Rewrite this as a finding, not a hypothesis.** The evidence:

- APP is the most redundant of the four branches. Measured score correlations: **APP–LFS
  +0.745, APP–STT +0.840** — the two highest pairings; VD–LFS is +0.399.
- Over twelve test sets, four branches (App+VD+LFS+STT) reach **0.9482** against CROSS-EXAM's
  **0.9468** — a 0.0014 gap against **0.0126** measured seed noise on this objective.
- Over DF40's 41 generator sets the four-branch set reaches 0.9588 and CROSS-EXAM 0.9576;
  again inside noise.
- On DF40 **reenactment** specifically, three branches beat four on **19 of 22** sets.

So: the separate appearance path does **not** earn its parameters once a mid-depth
spatio-temporal view is present. Frame this as a *result that shaped the architecture* — it is
more interesting than a positive result, and it is the kind of negative finding a reviewer
trusts. State it in §5.1, carry it into the ablation, and remove APP from §5.2.2 entirely.

### 1.2 Consequences elsewhere

- **§5.2.2 "Appearance Evidence and Nuisance Suppression"** — delete or repurpose. If the
  nuisance-suppression idea survives, it must be re-attached to STT and re-justified; if it was
  only ever an APP component, cut it and cut the matching work package §6.2.2.
- **§6.2.2** is titled "Representation Complementarity and Processing Robustness" and is built
  around "compares the separate PE-Core branch with shared-CLIP appearance". That comparison is
  done and negative. Rewrite the package as *completed*, reporting the outcome.
- Anywhere the text says the views are "APP, LFS, TD" or indexes them `v ∈ {A, L, T}`, change to
  `v ∈ {V, L, S}` or similar, consistently, including Eq. (5.9)–(5.14) and Figures 5.1–5.2.

---

## 2. THE FUSION CITATION — cite ICLR 2021, not DiCoME

**What is wrong now.** The evidential multi-view construction — a Dirichlet opinion per view,
combined across views — is presented as following the DiCoME line. It does not originate there.

**What to cite.** *Han, Zhang, Fu, Zhou — "Trusted Multi-View Classification", ICLR 2021*
(TMC). TMC is the source of: per-view Dirichlet evidence, the vacuity `u = K/S` convention, and
the `DS_Combin` multi-view Dempster fold. Cite it as the framework CROSS-EXAM builds on.

**Why this is correct and not merely a preference.** DiCoME's Eq. (7) is the textbook Dempster
rule of combination; it is the same expression as TMC's `DS_Combin`, which DiCoME does not
cite. We verified in code that our Dempster path is **bit-identical to `DS_Combin` at 2, 3, 4
and 5 views**. So the lineage is TMC → us, with DiCoME a contemporaneous application of the
same rule.

**How to position DiCoME.** As a **baseline we reproduce and compare against**, nothing more.
We trained it in its own repository and reproduce its reported CDFv2 number (0.9644 on our
census), and it appears in every comparison table. Remove any phrasing that makes CROSS-EXAM
sound derived from it.

**What is ours, stated precisely.** Not the evidential framework and not PCR6+ itself (cite
Smarandache & Dezert, already ref. [19]). Ours is:
1. the **choice of three complementary representations**, with the redundancy of a fourth
   demonstrated rather than assumed;
2. **replacing Dempster's rule with direct multisource PCR6+** inside the TMC-style pipeline —
   evaluated jointly over sources, not as a sequential pairwise fold, since proportional
   redistribution is not associative (§5.2.7 already says this well; keep it);
3. **one-sided IBDC** (§5.3.2), which removes a certainty incentive present under shared
   ignorance — the derivation there is sound and is a genuine contribution;
4. retaining **conflict K** alongside vacuity as a separate output.

---

## 3. THE METHOD IS TWO-STAGE — §5.3 is already right, but must be made the backbone

§5.3 already describes Stage 1 (separate view training with EDL) and Stage 2 (joint head
training with one-sided IBDC, representations frozen). **This is the method. Present it as
settled and deliberate, not as one option among experiments.**

Edits:
- Update the branch names (VD/LFS/STT) in Eq. (5.15)–(5.20) and Figure 5.2.
- Stage 1: VD updates V-JEPA 2 LoRA + head; LFS updates CLIP LoRA + MR-VAE + head; STT updates
  PE-Core LoRA/LayerNorm + temporal mixer + head. All other backbone weights frozen.
- Stage 2: **the three evidential heads only**. Reliability coefficients are fitted after
  Stage 1 on source-grouped development partitions and held fixed through Stage 2 and inference.
- Keep the honest paragraph at the end of §5.3.2 ("The loss removes a specific certainty
  incentive; it does not guarantee calibrated probabilities"). It is good writing and reviewers
  reward it.

### 3.1 Add the end-to-end negative control — this converts "random experiments" into design

We trained the whole stack end to end (all branches unfrozen, joint fusion loss) as an
alternative to the two-stage scheme. **It is worse.** On a matched video census the end-to-end
model reached **0.9276** against the two-stage system's **0.9531** — a deficit of **0.0255**,
twice the seed noise. Arm ordering behaved sensibly (removing the video branch hurt most), so
this is a real result and not a broken run.

Report this as a **designed control that justifies the two-stage choice**: joint optimisation
of representations and fusion lets the branches collapse toward each other, which is precisely
the redundancy the architecture exists to avoid. One short subsection, three sentences and one
number. Do not present the end-to-end attempt as an exploratory detour.

---

## 4. RESULTS TO INSERT — these are final

Twelve test sets, video level, video-level AUROC unless stated. CDFv3 and KoDF use the widened
census (5,418 and 2,239 videos; lip-sync/audio-driven and adversarial subsets admitted).

### 4.1 Ablation (Table 1) — every pair beats every single, the triple beats every pair

| branches | 12-set AUROC |
|---|---:|
| Video Dynamics (VD) | 0.8477 |
| Latent Facial Structure (LFS) | 0.9087 |
| Spatio-Temporal Texture (STT) | 0.9290 |
| VD+LFS | 0.9302 |
| VD+STT | 0.9356 |
| LFS+STT | 0.9379 |
| **VD+LFS+STT (CROSS-EXAM)** | **0.9468** |

Say explicitly that monotonicity **did not have to hold** — on DF40's reenactment half a
four-branch set is beaten by three — so it is a result rather than an artefact of adding
parameters. Note also that weights are refitted for each row, so each row is the best that
subset can do, not the full model with parts switched off. That is the stricter reading.

### 4.2 Against the state of the art (Table 3), twelve sets

| method | AUROC | FPR@TPR90 | ECE-15 | Brier | E-AURC |
|---|---:|---:|---:|---:|---:|
| **CROSS-EXAM** | **0.9468** | **0.1406** | **0.1230** | **0.1048** | **0.0447** |
| DiCoME | 0.9039 | 0.3210 | 0.1357 | 0.1275 | 0.0619 |
| GenD-PE | 0.9008 | 0.3154 | 0.1259 | 0.1311 | 0.0542 |

CROSS-EXAM leads every column. The **FPR result is the one to foreground**: at matched 90%
sensitivity it flags **14.1%** of genuine videos where the best baseline flags **31.5%** —
less than half the false-alarm rate. For a forensic framing that matters more than AUROC.

### 4.3 DF40 cross-manipulation (Tables 5 and 6), 21 generators, source halves pooled

| method | FaceSwap (10) | ExpressionSwap (11) |
|---|---:|---:|
| **CROSS-EXAM** | **0.9880** | **0.9300** |
| DiCoME (their repo) | 0.9749 | 0.7240 |
| GenD-PE | 0.9722 | 0.8843 |
| DiCoME (our training) | 0.9685 | 0.7896 |

**ExpressionSwap is the headline of the chapter.** DiCoME's published cross-manipulation table
reports face swap only; reenactment is half of DF40 and the half where their method is weakest.
The baselines lose 0.09–0.16 between the two families; CROSS-EXAM loses 0.058. That spread is
the complementarity claim *measured*, not asserted, and it is the strongest argument the
chapter has. Per-generator: DiCoME scores **0.5283 on mcnet** and **0.6264 on facevid2vid** —
at or near chance — so "weak on reenactment" understates it; on specific generators it is not
detecting at all.

### 4.4 Fusion operator (Table 2) — report the split honestly

| rule | AUROC | FPR@TPR90 |
|---|---:|---:|
| Mean (unweighted) | 0.9453 | 0.1473 |
| Mean (fitted weights) | 0.9461 | 0.1455 |
| Dempster (DS-C) | **0.9484** | 0.1433 |
| **PCR6+-R (ours)** | 0.9468 | **0.1406** |

**Do not claim PCR6+ wins outright.** Dempster leads AUROC by 0.0016; PCR6+ leads FPR at
matched sensitivity by 0.0027. The same split appeared on a four-branch set (0.9501 vs 0.9482)
and in a standalone operator sweep — three independent measurements, so it is not noise.

Two defensible framings; pick one and commit:
- **(a) Lead on FPR.** PCR6+ is adopted because it reduces false alarms at matched sensitivity,
  which is the operating characteristic that matters forensically, at a cost of 0.0016 AUROC.
  State the cost in the text, not only in the table.
- **(b) Adopt Dempster** and drop the PCR6+ claim, keeping PCR6+ as a reported alternative.

(a) is well supported and consistent with the dissertation's stated courtroom framing, but it
must be argued, not implied by bolding.

---

## 5. CHAPTER 6 — convert plan into completed work

Most work packages are done. Rewrite each as *completed* with its outcome, and keep as future
work only what genuinely remains.

| package | status |
|---|---|
| 6.2.1 Reference reproduction | **Done.** DiCoME reproduced in its own repo (CDFv2 0.9644 on our census); all comparators scored on identical videos. |
| 6.2.2 Representation complementarity | **Done, and the APP arm is negative.** Rewrite around that outcome. Drop processing-robustness if it was APP-only. |
| 6.2.3 Joint evidential-head training | **Partly done.** The one-sided IBDC derivation stands; the matched original/one-sided/no-IBDC comparison still needs to be reported if it has been run — otherwise keep as the one genuinely remaining experiment. |
| 6.2.4 Difficult-dataset evaluation | **Largely done.** Twelve test sets including DFDC, DeepSpeak-1.1/2.0, and 41 DF40 generator sets. State what remains (e.g. DeepfakeEval-2024). |
| 6.2.5 Manuscript integration | Keep. |

§6.1 currently says "Appearance adds relatively little when it shares the CLIP feature path"
and uses that to motivate a *separate* PE-Core appearance encoder. **That motivation is now
falsified** — the separate encoder was built and still adds little. Update the paragraph.

---

## 6. ABSTRACT, INTRODUCTION, BACKGROUND — the specific weaknesses

**Abstract.** Currently describes intentions. It should state the result in numbers: three
complementary facial representations fused by direct multisource PCR6+ under evidential
uncertainty; twelve cross-dataset benchmarks and 21 DF40 generators; AUROC 0.9468 against 0.9039
for the strongest baseline; **false-alarm rate at matched 90% sensitivity less than half** the
best baseline's; and the largest margin on reenactment, the manipulation family prior
cross-manipulation tables omit.

**Introduction.** Three gaps to fix:
1. **No quantified problem statement.** Give the generalisation gap a number early — detectors
   that score ~0.99 in-domain fall to 0.83–0.90 cross-dataset.
2. **The contribution list does not distinguish adopted from novel.** State plainly: evidential
   multi-view fusion is adopted from TMC (ICLR 2021); PCR6+ is adopted from Smarandache &
   Dezert; *ours* are the representation set, the substitution of PCR6+ for Dempster inside the
   evidential pipeline, one-sided IBDC, and retaining conflict as an output.
3. **No statement of what would falsify the claim.** Add one sentence: if the three branches
   were redundant, dropping one would not cost accuracy and their errors would coincide. The
   ablation and the family split are the tests.

**Background (Ch. 2).** §2.6 "Evidential Learning and Multi-View Fusion" is the section that
must carry TMC. It should trace: subjective logic / Dirichlet evidence → EDL (Sensoy et al.)
→ **TMC (Han et al., ICLR 2021)** for multi-view evidential fusion → the known limitation that
Dempster's rule mishandles high conflict → PCR6/PCR6+ as the response. That chain makes the
contribution legible; at present PCR6+ appears without the problem it solves. Also add one
paragraph on **why reenactment is harder than face swap** (identity preserved, motion altered,
so appearance-only cues weaken) — the results depend on this distinction and the background
never sets it up.

---

## 7. THINGS NOT TO CLAIM

Reviewers will check these.

1. **Do not call the branches statistically independent.** §5.1 already says this correctly —
   "independent denotes separate parameterized feature paths and separate first-stage
   supervision". Keep that sentence.
2. **Do not claim PCR6+ beats Dempster overall.** See §4.4.
3. **Do not claim CROSS-EXAM beats DiCoME's *published* DF40 numbers.** Their published census
   cannot be inspected. The like-for-like comparison is our trained DiCoME on identical videos;
   published numbers are context only, and must be labelled as such.
4. **Do not quote the 12-set baseline numbers from the earlier draft.** Baselines were re-run
   on the widened CDFv3/KoDF census and several moved — DiCoME's mean was 0.9196 on a 5-set
   matched support and is 0.9039 over twelve.
5. **Do not describe reliability coefficients as learned per video.** They are fitted once on
   development and held fixed — §5.2.6 says this; keep it.
6. **Vacuity is not a complete description of uncertainty.** §5.2.8 says so and reports conflict
   K separately. Keep that.

---

## 8. WHAT STILL NEEDS A NUMBER

Flag these rather than inventing them:
- The matched original / one-sided / no-IBDC comparison (§6.2.3), if not yet run.
- DeepfakeEval-2024 as the difficult-dataset stress test, if still intended.
- Inference cost per video for the three-branch configuration.
- Seed variability for CROSS-EXAM itself (we have 0.0126 for the fitting objective; three
  complete adaptation runs were planned in §6.2.4).
