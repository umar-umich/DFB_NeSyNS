# Modification notes — uncertainty-aware fusion at inference, and how to defend it

Third companion file, with `NOTES_FOR_CHATGPT_Ch5_Ch6.md` and
`..._FrontMatter_and_Method_Depth.md`. For ChatGPT.

This file does one thing: state exactly what CROSS-EXAM does at inference time, and give the
chapter the framing and the citations that make each step defensible rather than asserted.

**Every number here is measured. Do not invent any. Where a number is marked `[VERIFY]` the
table it comes from must be regenerated first — see §7.**

---

## 1. WHAT THE SYSTEM DOES AT INFERENCE — state it in this order

For a video, each branch `v ∈ {VD, LFS, STT}` independently produces:

1. **Evidence.** A head emits non-negative `e_v ∈ R²` through a softplus. Nothing is normalised
   across branches; each reports what it found.
2. **A Dirichlet.** `α_v = e_v + 1`, strength `S_v = Σ α_v`.
3. **A subjective-logic opinion.** `b_v,k = (α_v,k − 1)/S_v` for `k ∈ {genuine, fake}`, and
   **vacuity** `u_v = K/S_v` with `K = 2`. By construction `b_v,G + b_v,F + u_v = 1`.

The three opinions are then combined by **direct multisource PCR6+**, and the system outputs
three quantities, not one:

- **`p_f = b_f,F + u_f/2`** — the authenticity score, the projected probability.
- **`u_f = Π_v u_v`** — fused vacuity: how little evidence there is in total.
- **`K`** — retained conflict: how much of the mass was incompatible across branches.

A single decision threshold is fixed once at the **EER on FaceForensics++ validation** and never
re-tuned per benchmark.

### 1.1 Three properties to state explicitly, because a reviewer will derive them anyway

- **The fusion has no fitted parameters.** As of 2026-10-15 the reliability discount is gone (§4).
  Inference is: heads → opinions → PCR6+ → score. Nothing at fusion time was tuned on any
  development set. This is a *strength* and the chapter should say it plainly.
- **`u_f` is a product, so it is small when ANY branch is confident.** Say so. It follows that
  fused vacuity alone cannot express "the branches disagree" — which is exactly why `K` is
  reported beside it rather than folded into it. §5.2.8 already makes this point; keep it.
- **PCR6+ is not associative**, so it is evaluated jointly over the three sources rather than as
  a sequential pairwise fold. This is a correctness requirement, not an implementation choice,
  and §5.2.7 states it correctly.

---

## 2. WHAT IS ADOPTED AND WHAT IS OURS — the paragraph that protects every later claim

Write this early and without hedging. A reviewer who sees the borrowed parts named will trust
the novel ones; a reviewer who finds an uncited standard component will distrust everything.

> **Adopted.** Evidential deep learning with Dirichlet outputs and vacuity `u = K/S`
> (Sensoy et al., NeurIPS 2018). Subjective logic's belief/uncertainty decomposition (Jøsang).
> Evidential **multi-view** fusion — per-view Dirichlets combined across views — from Trusted
> Multi-View Classification (Han et al., **ICLR 2021**). Proportional conflict redistribution
> PCR6, and its PCR6+ variant, from Dezert–Smarandache theory (Smarandache & Dezert; Dezert,
> Dezert & Smarandache, *JAIF* 16(1), 2021).
>
> **Ours.** (i) The three-representation set and the evidence that they are jointly necessary.
> (ii) **Substituting PCR6+ for Dempster's rule inside the TMC-style evidential pipeline**, and
> evaluating it jointly over sources. (iii) Reporting retained conflict `K` as an output
> distinct from vacuity. (iv) A cross-manipulation evaluation that includes reenactment.

**Do not claim the evidential framework, the vacuity definition, or PCR6+ itself.** The
contribution is the *composition* and its empirical justification in this domain.

---

## 3. WHY A CONFLICT-REDISTRIBUTION RULE BELONGS HERE — the defensible argument

This is the argument the chapter currently does not make, and it is the one that makes the
method look necessary rather than decorative.

**Step 1 — Dempster's rule has a documented failure mode under high conflict.** It normalises by
`1 − C`, so as the conflict coefficient `C → 1` the result is dominated by whatever little mass
survives; Zadeh's counterexample is the canonical statement. This is textbook and uncontested —
cite it as such.

**Step 2 — cross-manipulation detection is a high-conflict regime, and we measured it.** The
branches do not merely differ in accuracy; **they fail on opposite manipulation families**. On
DF40's 41 generator sets: VD scores 0.8933 on face swap and 0.9452 on reenactment; LFS 0.9684
and 0.7855; STT 0.9829 and 0.7963. VD's sign is opposite to the other two. So on a reenactment
video a face-swap specialist and a motion specialist *should* disagree — the conflict is
informative, not noise.

**Step 3 — therefore the routing of conflicting mass is a design decision with consequences,**
and PCR6+ returns it to the sources that committed to it instead of normalising it away.

That is a three-sentence argument from a documented limitation, through a measurement, to a
design choice. It is far stronger than "we use PCR6+ because it handles conflict".

---

## 4. THE RELIABILITY DISCOUNT HAS BEEN REMOVED — and the reason is worth a paragraph

Earlier drafts fuse reliability-discounted evidence `e'_v = r_v e_v` with `r` fitted by grid
search on development (`PCR6+-R`). **This is dropped.** Two reasons, and give both:

1. **It is subsumed by the evidential heads.** Reliability discounting is evidence scaling, and
   the heads are what produce the evidence. A head can learn to emit smaller evidence; a fixed
   external `r` is capacity the model already has. (Be precise: it is not an exact
   reparameterisation, since `r·softplus(z) ≠ softplus(r·z)`, but the head's freedom strictly
   contains the scalar's.)
2. **It removes the last fitted quantity from inference.** Without it, nothing in the fusion was
   tuned on held-out data, so "how were the weights chosen and does the gain survive refitting"
   stops being a question the paper has to answer.

**Report what it cost, honestly.** On twelve test sets at the widened census, VD+LFS+STT:

| | AUROC | FPR@TPR90 | mean vacuity |
|---|---:|---:|---:|
| PCR6+ with fitted `r = [0.25, 1.00, 0.25]` | 0.9452 | 0.1507 | 0.0330 |
| **PCR6+, no discount (adopted)** | 0.9420 | 0.1581 | 0.0082 |

So dropping it costs 0.0032 AUROC and 0.0074 FPR. Say that, and say why it is worth paying:
a parameter-free fusion is a stronger claim than a marginally better fitted one.

**Note the vacuity column.** Discounting removes evidence, so it inflates fused vacuity four-fold
(0.0082 → 0.0330). Without `r`, the vacuity the paper reports is a property of the *rule*, not
of a tuning choice — which is what makes it interpretable as a mechanism.

---

## 5. CROSS-DOMAIN GROUNDING — where this behaviour is already established

UMAR's point, and it is a good defensive move: evidential fusion with conflict handling is not
something this work invented for deepfakes. Place it in a lineage.

- **Sensor and target fusion.** DST and the PCR family originate in multi-sensor data fusion and
  automatic target recognition, where sources are heterogeneous, individually unreliable, and
  routinely conflicting. That is the problem shape here: three encoders with different
  sensitivities and no shared failure mode.
- **Multi-view and multi-modal classification.** TMC (ICLR 2021) applies per-view Dirichlet
  evidence and combination to multi-view benchmarks and medical data, explicitly to get
  *trustworthy* predictions rather than only accurate ones.
- **Medical imaging.** Evidential fusion across modalities is used where an abstention or a
  flagged-uncertain case has real value — the same operating logic as evidentiary forensics,
  where a confident wrong answer is worse than an uncertain one.
- **Classifier combination in remote sensing.** A long literature on combining heterogeneous
  classifiers under conflicting evidence, where Dempster's normalisation is a known hazard.

Write one short paragraph covering these, with the point being: **the machinery is established
and its failure modes are documented; the contribution is identifying that deepfake
cross-manipulation detection sits squarely in the regime those tools were built for, and showing
it with a measurement.**

Do not overstate the parallel. Say "analogous problem structure", not "the same problem".

---

## 6. CLAIMS TO MAKE, AND CLAIMS TO REFUSE

**Make these — each has a measurement behind it:**

- The three branches are complementary: they fail on opposite manipulation families (§3, step 2).
- The fusion has no parameters fitted at inference (§1.1).
- Vacuity and conflict are reported separately because vacuity alone cannot express disagreement.
- The decision threshold is fixed once on FF++ validation and never re-tuned per benchmark.

**Refuse these — the data does not support them:**

- ✗ *"PCR6+ outperforms Dempster."* It leads FPR at TPR 90% and trails on AUROC. State both.
- ✗ *"Our uncertainty is calibrated."* We report ECE and Brier; removing a certainty incentive
  does not guarantee calibration, and §5.3.2 already says so. Keep that sentence.
- ✗ *"The one-sided disagreement objective improves uncertainty."* See §6.1 — it does not, here.
- ✗ *"Fused vacuity captures the model's total uncertainty."* It is a product and it is small
  whenever any branch is confident.

### 6.1 The one-sided IBDC result — report it as a measurement, not a disappointment

Stage 2 was run in three matched arms differing only in the disagreement term. Validation AUROC
on VALmix: **no IBDC 0.8588, one-sided 0.8596, original 0.8598.**

One-sided and original are **0.0002 apart, and that is not noise — it is structural.** The two
objectives are identical wherever committed conflict `κ ≥ u_f`, and across the twelve test sets
**fused vacuity exceeds conflict in only 3.50% of videos (1,158 of 33,076)**. The shared-ignorance
regime the one-sided form was derived to fix is almost absent once the branches are trained.

This is a publishable negative with a mechanism, and it is more credible than a small positive.
Write it as: the derivation is correct, the pathology it removes is rare in this setting, and the
measured difference is correspondingly nil. If the chapter wants the claim alive, the test is
whether the one-sided arm wins **on the 3.5% where the objectives differ** — that is the sharp
version, and it has not been run.

---

## 7. ONE THING THAT MUST BE RESOLVED BEFORE THESE TABLES ARE QUOTED

**The reported tables still use `PCR6+-R`.** `PROPOSAL_TABLES_3.md` was generated with the
reliability discount in place; §4 removes it. Those tables must be regenerated with plain PCR6+
before the chapter quotes them, or the text and the numbers will disagree. Mark every fusion
number `[VERIFY]` until that is done.

**A second, larger discrepancy is open.** The same Stage-1 weights score **0.8897** on VALmix
through the cached-opinion path and **0.8453** through the DISCERN_Ext inference path — a
**0.044 gap on identical weights**, three times anything the training stage changes. All reported
tables come from the first path. Until it is attributed, the chapter should not claim that the
reported numbers are reproducible from the deployed inference code. The likely locus is the
V-JEPA preprocessing route — VD's per-video opinions correlate at 0.874 between the two paths
where STT's correlate at 0.985.

Do not paper over either of these. The first is a regeneration; the second is a real open
question and should be stated as one until measured.
