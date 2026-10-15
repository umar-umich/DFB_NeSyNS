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

## 1. THE FRAMEWORK COMPOSITION — what the chapter must describe

The proposal defines CROSS-EXAM as **APP + LFS + TD**. The method is **VD + LFS + STT**.

| in the draft | in the method | what it is |
|---|---|---|
| TD | **VD — Video Dynamics** | V-JEPA 2 ViT-L, LoRA on encoder q/v, 16-frame clip. Rename only. |
| LFS | **LFS — Latent Facial Structure** | CLIP + MR-VAE orthogonal residual. Unchanged. |
| — | **STT — Spatio-Temporal Texture** | PE-Core-L block-16 patch mean (not the final layer) → 256-d projection → temporal mixer, masked mean over frames. |
| APP | *not in the method* | Delete. See below. |

Use the full names once at first mention in the ablation — *Video Dynamics (VD)*, *Latent Facial
Structure (LFS)*, *Spatio-Temporal Texture (STT)* — and abbreviations after. In every comparison
against other methods the system is called **CROSS-EXAM**, not a list of its branches.

### 1.1 Deletions — mechanical, not argued

The paper describes the method it has. It does not justify a component that is not in it, so
**none of the following should be explained, defended, or reported as a finding — just removed.**

- **§5.1, Hypothesis 1** ("the separate APP path contributes useful information beyond LFS and
  TD"). Delete the hypothesis. §5.1 should state the three retained views and the one hypothesis
  that survives: that a one-sided disagreement objective improves uncertainty without rewarding
  certainty during shared ignorance.
- **§5.2.2 "Appearance Evidence and Nuisance Suppression"** — delete the section. Replace it
  with the STT subsection (material in the companion notes, Part E3).
- **§6.2.2** — the work package is built on comparing a separate PE-Core appearance branch
  against shared-CLIP appearance. Delete it; see §5 below for the replacement package list.
- **§6.1**, the sentence "Appearance adds relatively little when it shares the CLIP feature
  path", and the clause that uses it to motivate a separate PE-Core appearance encoder.
- Every index `v ∈ {A, L, T}` becomes `v ∈ {V, L, S}`, throughout Eq. (5.9)–(5.20) and
  Figures 5.1–5.2. Four-view notation anywhere must become three.

**Do not write a paragraph explaining why appearance was dropped.** A reader of the paper has
never been told it was in. The three-branch ablation (§4.1) is the complete argument for the
composition that exists.

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
1. the **choice of three complementary representations**, shown by ablation and by the
   per-family split to carry information the others miss rather than assumed to;
2. **replacing Dempster's rule with direct multisource PCR6+** inside the TMC-style pipeline —
   evaluated jointly over sources, not as a sequential pairwise fold, since proportional
   redistribution is not associative (§5.2.7 already says this well; keep it);
3. **one-sided IBDC** (§5.3.2), which removes a certainty incentive present under shared
   ignorance — the derivation there is sound and is a genuine contribution;
4. a cross-manipulation evaluation that reports **reenactment**, which prior tables omit.

One-sided IBDC is NOT in this list: it is proposed in Chapter 6 as future work, with the
measurement that justifies deferring it (§4.5).

---

## 3. THE REPORTED METHOD IS STAGE 1 PLUS ES-PCR6+ AT INFERENCE

UMAR, 2026-10-15: **the second training stage is not part of the reported method.** The chapter
reports each branch trained separately, then combined at inference by ES-PCR6+. §5.3.2's joint
head training and the one-sided IBDC objective move to Chapter 6 as future work.

### 3.1 What §5.3 becomes

- **Keep §5.3.1 (Stage 1)** essentially as written, with the branch renames. It is the method:
  VD updates V-JEPA 2 LoRA + head; LFS updates CLIP LoRA + MR-VAE + head; STT updates PE-Core
  LoRA/LayerNorm + temporal mixer + head. All other backbone weights frozen. Each branch learns
  without any loss that matches another branch's class probability — that sentence is the
  independence claim and it should stay.
- **Rename the section.** "Two-Stage Evidential Training" becomes something like "Branch-wise
  Evidential Training", because there is now one training stage and a fusion rule.
- **Move §5.3.2 to Chapter 6** intact. The derivation is sound and worth keeping; it becomes a
  proposed extension with a measured reason for not having adopted it (§4.5 below).

### 3.2 Why separate training is the right design — argue it, with two controls

The chapter should not merely assert that branch-wise training is correct. Two measurements
support it, and both are negative results about the alternatives:

1. **End-to-end from scratch is worse.** Training all branches jointly on FF++ reaches **0.9209**
   against **0.9558** for separately-trained branches fused, on five matched test sets — and the
   loss concentrates where generalisation is hardest (KoDF −0.098, CDFv3 −0.046) while the
   easiest set ties (UADFV +0.005).
2. **Joint head training does not recover it.** A second stage over frozen representations gains
   only +0.0145 on development, and the three objective variants land within 0.001 of each other.

One mechanism explains both: the branches' value comes from their **separate** pretraining and
adaptation, and re-fitting them together on one source erodes the diversity the fusion exploits.
State that as the design rationale, with the two numbers.

---

## 4. RESULTS TO INSERT — these are final

Twelve test sets, video level, video-level AUROC unless stated. CDFv3 and KoDF use the widened
census (5,418 and 2,239 videos; lip-sync/audio-driven and adversarial subsets admitted).

### 4.1 Ablation (Table 1) — every pair beats the singles it contains, the triple beats every pair

| branches | 12-set AUROC |
|---|---:|
| Video Dynamics (VD) | 0.8450 |
| Latent Facial Structure (LFS) | 0.9041 |
| Spatio-Temporal Texture (STT) | 0.9277 |
| VD+LFS | 0.9273 |
| VD+STT | 0.9345 |
| LFS+STT | 0.9361 |
| **VD+LFS+STT (CROSS-EXAM)** | **0.9452** |

Say it as containment, not as a blanket ordering: every pair beats both singles it is built
from, and the triple beats every pair. VD+LFS (0.9273) sits just under STT alone (0.9277), so
the sentence "every pair beats every single" is false and a reviewer reading the table will
see it.

Note that weights are **refitted for each row**, so a row is the best that subset can do, not
the full model with parts switched off. That is the stricter reading of an ablation and worth
one sentence, because it rules out the objection that the gains come from the removed branch's
weight being stranded.

### 4.2 Against the state of the art (Table 3), twelve sets

| method | AUROC | FPR@TPR90 | ECE-15 | Brier | E-AURC |
|---|---:|---:|---:|---:|---:|
| **CROSS-EXAM** | **0.9452** | **0.1507** | **0.1263** | **0.1088** | **0.0455** |
| GenD-CLIP | 0.9022 | 0.3084 | 0.1271 | 0.1199 | 0.0539 |
| GenD-PE | 0.8995 | 0.3127 | 0.1281 | 0.1331 | 0.0559 |
| DiCoME | 0.8986 | 0.3261 | 0.1349 | 0.1301 | 0.0642 |

CROSS-EXAM leads every column, over all eight baselines, not only the three shown. The **FPR
result is the one to foreground**: at matched 90% sensitivity it flags **15.1%** of genuine
videos where the best baseline flags **30.8%** — less than half the false-alarm rate. For a
forensic framing that matters more than AUROC. On ECE the lead is **0.0008** over GenD-CLIP;
that is a tie, and the text should say the fused probabilities are *no worse* calibrated
rather than better.

### 4.3 DF40 cross-manipulation (Tables 5 and 6), 21 generators, source halves pooled

| method | FaceSwap (10) | ExpressionSwap (11) |
|---|---:|---:|
| **CROSS-EXAM** | **0.9880** | **0.9300** |
| DiCoME (their repo) | 0.9749 | 0.7240 |
| GenD-PE | 0.9722 | 0.8843 |
| DiCoME (our training) | 0.9685 | 0.7896 |

**ExpressionSwap is the headline of the chapter.** DiCoME's published cross-manipulation table
reports face swap only; reenactment is half of DF40 and the half where their method is weakest.
The baselines lose 0.08–0.25 between the two families; CROSS-EXAM loses 0.058. That spread is
the complementarity claim *measured*, not asserted, and it is the strongest argument the
chapter has. Per-generator: DiCoME scores **0.5283 on mcnet** and **0.6264 on facevid2vid** —
at or near chance — so "weak on reenactment" understates it; on specific generators it is not
detecting at all.

### 4.4 Fusion operator (Table 2) — seven rules, three metrics

Reported metrics are **AUROC, FPR at TPR 90%, and mean vacuity**. All seven rules read the same
three branches and the same reliability-discounted masses, so the only thing that varies is
where conflicting mass goes.

| fusion rule | AUROC | FPR@TPR90 | mean vacuity |
|---|---:|---:|---:|
| Mean (unweighted) † | 0.9437 | 0.1573 | — |
| Mean (fitted weights) † | 0.9446 | 0.1550 | — |
| Dempster | **0.9469** | 0.1521 | 0.0445 |
| Yager | 0.9434 | 0.1532 | 0.2393 |
| WBF (Jøsang) | 0.9440 | 0.1634 | 0.1940 |
| Cautious (Denœux) | 0.9416 | 0.1693 | 0.1368 |
| **ES-PCR6+ (ours)** | 0.9452 | **0.1507** | **0.0330** |

† The two probability means emit a probability, not an opinion, so vacuity is undefined for
them — not zero. They are the comparator that shows what the opinion-level machinery buys.

**ES-PCR6+ is the adopted rule.** The argument to make, in this order:
1. It gives the **lowest false-alarm rate at matched 90% sensitivity** (0.1507). For evidentiary
   use this is the operating point a result is read at, and it is the metric the dissertation
   has framed from the start.
2. It is the **most committed rule** by mean vacuity (0.0330) — it resolves conflict onto the
   classes that caused it rather than leaving mass uncommitted, which is what PCR6 is for.
3. On AUROC it is **second, behind Dempster by 0.0017**. State this plainly in the text. Do not
   let bolding carry the claim; a 0.0017 gap is not a win for anyone and saying so is cheap.

On a binary frame Yager coincides with Dubois–Prade and PCR5 collapses onto PCR6, so several
rules the literature separates are one row here — worth one sentence so a reviewer does not
ask where they went.

### 4.5 Why the disagreement objective is deferred — the measurement to give

Chapter 6 should propose one-sided IBDC with the reason it is not yet adopted, which is
empirical and specific:

- Three matched arms, differing ONLY in the disagreement term, over frozen representations.
  Validation AUROC: **no IBDC 0.8588, one-sided 0.8596, original 0.8598.**
- One-sided and original are **0.0002 apart, and that is structural, not noise**: the two
  objectives are identical wherever committed conflict `κ ≥ u_f`, and across the twelve test
  sets fused vacuity exceeds conflict in only **3.50% of videos (1,158 of 33,076)**.

So the shared-ignorance regime the one-sided form corrects for is almost absent once the
branches are trained. The derivation stands; the condition it addresses is rare here. The sharp
version of the experiment — does one-sided win **on the 3.5% where the objectives differ** — has
not been run, and that is the right thing for Chapter 6 to propose.

---

## 5. CHAPTER 6 — convert plan into completed work

Most work packages are done. Rewrite each as *completed* with its outcome, and keep as future
work only what genuinely remains.

| package | status |
|---|---|
| 6.2.1 Reference reproduction | **Done.** DiCoME reproduced in its own repo (CDFv2 0.9644 on our census); all comparators scored on identical videos. |
| 6.2.2 Representation complementarity | **Replace.** The package compared an appearance branch that is not in the method. The retained content is the three-branch ablation and the per-family split; write the package around those. |
| 6.2.3 Joint evidential-head training | **Partly done.** The one-sided IBDC derivation stands; the matched original/one-sided/no-IBDC comparison still needs to be reported if it has been run — otherwise keep as the one genuinely remaining experiment. |
| 6.2.4 Difficult-dataset evaluation | **Largely done.** Twelve test sets including DFDC, DeepSpeak-1.1/2.0, and 41 DF40 generator sets. State what remains (e.g. DeepfakeEval-2024). |
| 6.2.5 Manuscript integration | Keep. |

§6.1 motivates a separate PE-Core appearance encoder. That encoder is not in the method; cut
the motivation with it, and let the paragraph motivate the three views that are.

---

## 6. ABSTRACT, INTRODUCTION, BACKGROUND — the specific weaknesses

**Abstract.** Currently describes intentions. It should state the result in numbers: three
complementary facial representations fused by direct multisource PCR6+ under evidential
uncertainty; twelve cross-dataset benchmarks and 21 DF40 generators; AUROC 0.9452 against 0.9022
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
