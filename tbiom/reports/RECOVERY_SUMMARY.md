# Recovery brief — outcome

Steps 0-6 complete. Steps 7, 8 and 9 are conditional and every condition resolved to SKIP.

---

## What each step answered

| step | question | answer |
|---|---|---|
| 0 | does a retrainable DiCoME recipe exist? | **yes** — `P0-DS`, FF++ c23, checkpoints on disk |
| 1 | is the anchor gap CLIP recipe or DiCoME machinery? | **recipe**: semantic-only −0.0451 vs port, fusion adds only +0.0055 |
| 2 | can checkpoint re-selection fix real-side health? | **no lever** — all P0-DS epochs cluster (FPR 0.196–0.246) |
| 2b | is the VAE/alignment objective the cause? | **partly** — removing its gradient costs −0.0211 macro AUROC, ≈ half the gap |
| 3 | does the FF++-only anchor collapse on unfamiliar reals? | **no** — FPR_real 0.169 zero-shot, 51% of separation retained |
| 4 | does single-anchor `R = g(V, M)` predict errors? | **partially** — OOD risk-AUROC 0.748, +14.0% selective-risk reduction |
| 5 | do experts complement the chosen anchor? | **no** — best recovery 5.7% against a 25% bar |
| 6 | at what information level does complementarity become realizable? | **none tested** — every rung ≤ 0, and the richest fails the audit |

---

## The conditional steps

**Step 7 (applicability / fusion framework) — SKIPPED.** It requires experts surviving Step 5
*and* a ladder rung realizing complementarity while passing the audit. Neither holds: no expert
cleared the bar against P0-DS, and G0–G3 produced rho of +0.000, −0.010, −0.181 and −0.210, with
G3 failing the domain audit at 0.844.

**Step 8 (DiCoME fork) — SKIPPED.** It requires experts surviving complementarity *against
DiCoME specifically*. Step 5 measured exactly that pairing and none survived. The brief's
instruction — "do not port the framework before complementarity against DiCoME is demonstrated" —
resolves to not porting it.

**Step 9 (FFHQ / SBI corpus ablation) — SKIPPED.** It triggers only if Step 3 shows real-side
collapse persisting on the FF++-only anchor. It does not: FPR_real runs 0.014 in-domain to 0.169
zero-shot, against the collapse signature of 1.000. Real-support asymmetry was a property of the
FF++ ⊕ DF40 manifest, not of FF++ training. The SBI infrastructure built earlier (81-point
landmarks at 99.83% coverage) stays available but is not needed.

---

## Which paper the evidence supports

The brief lists four outcomes. The evidence selects the second:

> *Ladder fails the audit but single-anchor `(V, M)` reliability is strong → a reliability and
> defer paper, with the realizability gap reported as a characterised negative.*

with one qualification the brief did not anticipate: the reliability result is **moderate, not
strong**. Mean OOD risk-AUROC 0.748 and +14.0% selective-risk reduction at a 10% budget is real
but modest, and V1's 0.8118 is not a comparable target — it was calibrated on FF++ val **plus
Celeb-DF-v2 val** (the protocol since renamed `ffpp_cdf2_TESTCONTAMINATED`) at a 15.24% error
rate against this anchor's 3.9%.

### The negative result is the stronger half, and it is precise

Not "gating did not work" but:

1. **Complementarity is real and stable.** Rescue margin +0.222/+0.233 against P0-DS, essentially
   unchanged from +0.202/+0.206 against the weaker CLIP port — even though a stronger anchor is
   wrong less often and should make rescue harder to demonstrate.
2. **The oracle ceiling is real.** 1 − P(both wrong) sits well above anchor accuracy throughout.
3. **No test-time-observable signal locates it.** Every rung from raw confidence to per-layer
   adaptation and rate statistics fails, and the richest one fails the audit.
4. **The mechanism is identified.** Dropping the `phi = 0` mass collapses gate AUROC from 0.784
   to 0.519. The gate can predict whether the branches will **agree**; it cannot predict **who is
   right when they disagree**, which is exactly what applicability discounting needs.

That fourth point is the paper's contribution as a negative: applicability gating fails here for
a structural reason that can be stated and measured, not for want of capacity or tuning.

---

## Honest limits

- **Single seed** throughout. The spec's §22 rule needs a confirming seed for |ΔAUROC| < 0.01,
  which covers the Step-2b ablation (−0.0211, only just clear) and several Step-3 readout gaps.
- **G4 unrun.** A learned probe on frozen `[h_A ‖ h_b]` needs embedding exports nothing currently
  writes. It is the one rung that could still change Step 6's verdict, and it is recorded as
  `TODO(run)` rather than argued away.
- **Step 5 ran on VALmix only.** The DF40-Dev half is scoring now; VALmix is the cleaner basis
  (100% join coverage) and the prior runs showed VALmix rescue *stronger* than DF40's, so DF40 is
  expected to confirm rather than overturn.
- **Recipe work deliberately stopped.** Per instruction, batch size and precision were not tuned
  after Step 3 showed the anchor operationally adequate. Roughly half the Step-1 gap remains
  unexplained and is a known open thread, not a closed one.
- **The artifact view outperforms DiCoME's own fused output** on Celeb-DF-v2, DFD and DFDC
  (Step 1) and on zero-shot real-side FPR (Step 3, 0.169 vs 0.247). Any later integration should
  treat DiCoME's released fused prediction as one candidate among its internal views, not as the
  anchor by default.
