# Track A — SELECTION (Phase 1 deliverable)

Answers the seven selection questions from `docs/DiCoME_eval/claude-code-discern-v2-phase1.md`.
Every cell is either a number from a real run or `TODO(run)`. Nothing here is asserted from
memory or estimated.

## Status of the evidence base

`analysis/discern_v2/common.py` reads the per-sample exports under
`<DiCoME>/experiments/pilot_<P>/analysis/features/<SOURCE>/`.

| Pilot | DF40 | CDFv3 | CDFv2 |
|---|---|---|---|
| P0-DS | present | present | present |
| P2a | present | present | — |
| P3a | present | present | — |
| P4 | present | present | — |
| **P1a / P1b / P1c / P1d** | **MISSING** | **MISSING** | — |

**The P1 exports are the blocker for this document.** Questions 1, 2, 6 and the projector
recommendation are about the P1 suite and cannot be answered without them. 🔴 UMAR-RUNS:

```
cd /data/umar/Repos/DiCoME
python experiments/common/analysis/export_features.py P1a --source DF40 CDFv3
python experiments/common/analysis/export_features.py P1b --source DF40 CDFv3
python experiments/common/analysis/export_features.py P1c --source DF40 CDFv3
python experiments/common/analysis/export_features.py P1d --source DF40 CDFv3   # MR-VAE projector, so rate_response is captured
```

Then re-run all three analyses and refresh this file:

```
python analysis/discern_v2/a3_rate_response.py
python analysis/discern_v2/a1_complementarity.py
python analysis/discern_v2/a2_gate.py
```

### Provenance of the numbers below

Marked **[verification run]** where they come from my run of 2026-08-15 on the four exports
that exist, made to prove the scripts execute correctly. They are real numbers from real
exports, not placeholders — but the run of record is yours, and the P1 rows are absent from
them. Video level, single frozen threshold.

**Threshold caveat, applies throughout.** No FF++ export exists, so no threshold can be
derived from the permitted protocol source. `frozen_threshold()` therefore returns a fixed
0.5 and reports that provenance rather than tuning anything. Per the instructions, the
threshold-free analyses (AUROC, score correlation, error overlap) carry the conclusions, and
every decision-based table is reported beside its threshold-free counterpart. Exporting a
FF++ slice would remove this caveat and is worth doing before the run of record.

---

## 1. Does manifold-residual evidence add information beyond direct CLIP evidence?

`TODO(run)` — requires P1a/P1b/P1c/P1d exports. A1 computes it (rescue/harm vs P0,
error-overlap Jaccard, Spearman evidence correlation); the row is simply absent today.

What the existing branches show, as the frame this question will be answered in
**[verification run, DF40, 2,927 videos]**: error overlap with P0 is high for every operator
currently exported — Jaccard 0.68 (P2a), 0.75 (P3a), 0.65 (P4). Most errors are shared, so
the complementarity that matters is concentrated in specific regimes rather than spread
across the average. Question 1 should be judged on the inversion rows, not the mean.

## 2. Which P1 projector provides the most usable complementary evidence?

`TODO(run)` — requires the P1 exports. **🟡 ASK-UMAR: the D1 projector choice is yours to
sign off; this document will recommend, not commit.**

The selection rule is fixed in advance so the choice cannot drift to whatever wins on
average: the projector is chosen on **A3 incremental signal + A2a Gate-Recovery**, explicitly
**not** on standalone AUC. Best standalone ≠ best specialist for a gated architecture — P1b's
larger FR rescue is a live candidate precisely because a working gate could suppress its
harmful regime.

`mr_vae` remains the wiring placeholder default in every config. Swapping is one string.

## 3. Does P2a contribute errors/rescues sufficiently different from the manifold and visual branches?

**Against the visual branch: yes, and sharply so, in exactly the regime that matters.**
**[verification run, DF40]**

Per-generator AUROC against the shared real pool, on the rows where the visual baseline is
inverted (ranks fakes as *more real* than genuine video). Inversion rows are detected from
the data, not hard-coded:

| generator | P0-DS | P2a | P3a | P4 |
|---|---|---|---|---|
| sadtalker-ff | 0.253 | 0.447 | 0.249 | 0.359 |
| danet-cdf | 0.286 | **0.753** | 0.510 | 0.723 |
| tpsm-ff | 0.290 | **0.683** | 0.287 | 0.555 |
| danet-ff | 0.401 | 0.611 | 0.477 | 0.599 |
| mcnet-cdf | 0.404 | **0.750** | 0.504 | 0.737 |
| mcnet-ff | 0.432 | **0.720** | 0.471 | 0.660 |
| facevid2vid-ff | 0.480 | 0.698 | 0.509 | 0.672 |
| tpsm-cdf | 0.500 | **0.779** | 0.545 | 0.755 |

P2a recovers 0.28–0.47 AUROC on the reenactment rows where P0 is confidently wrong, while
P3a stays inverted alongside P0. That is the complementarity signature the architecture is
built to exploit. It is invisible in the aggregate: overall DF40 AUROC is P0 0.837 vs P2a
0.844.

**Against the manifold branch: `TODO(run)`** — needs the P1 exports.

**The counter-evidence, which must travel with the above.** On `heygen`, P2a *harms*:
accuracy 0.556 → 0.278, harm rate 0.5 on the samples P0 gets right. This is the
rescue/harm tension the applicability gate exists to resolve, and it is the reason D4 is
posed as a question rather than assumed.

These per-row numbers rest on 18 fake videos each and are noisy; the AUROC column (which
uses the 832-video real pool) is the trustworthy read, and the threshold-based rescue/harm
columns in the CSVs should be treated as indicative only.

## 4. What is the oracle headroom?

**[verification run, candidate set {P0-DS, P2a}]**

| source | base BA | oracle BA | headroom | union-of-correct coverage | remaining shared-error rate |
|---|---|---|---|---|---|
| DF40 | 0.7867 | 0.8301 | +0.0434 | 0.7827 | 0.2173 |
| CDFv3 | 0.7959 | 0.8639 | +0.0680 | 0.8065 | 0.1935 |

Oracle-routed AUROC is written to a separate file marked *label-peeking / non-deployable*
and is not a headline number: the oracle consults the label to choose a branch, so the
routed ranking is a function of the label.

Headroom with the P1 branches added is `TODO(run)` and will be larger — this two-branch
figure is a floor, not the ceiling.

## 5. Can a simple observable gate recover a meaningful fraction of it? (A2a)

**[verification run, {P0-DS, P2a}, logistic gate, leave-one-generator-out]**

| source | mean Gate-Recovery |
|---|---|
| DF40 | **0.015** |
| CDFv3 | **0.380** |

Read this as a provisional split, not a verdict. On CDFv3 a plain logistic gate on
label-free evidence recovers ~38% of the oracle headroom, which is a real signal. On DF40 it
recovers essentially nothing — consistent with DF40 spanning far more generator families, so
a gate trained on siblings transfers poorly even within the optimistic A2a protocol.

Three caveats before this is used to decide anything:
- Both numbers are the two-branch set. The instructions' candidate sets are
  `{P0, P1d, P2a}` and `{P0, P1b, P2a}`; neither can run yet.
- A2a is an **optimistic ceiling** — the gate sees sibling generators. A2b (the
  deployment-valid protocol, specified in `A2_gate/A2b_PROTOCOL.md`) is the real test and is
  not run this phase.
- The threshold caveat above applies to BA_base and BA_oracle alike.

**Go/no-go for D4: `TODO(run)` — do not decide on these two numbers.** The honest current
reading is "inconclusive and source-dependent". If the full candidate sets reproduce the
DF40 result, D4 is a written negative result; the CDFv3 result is strong enough that the
question stays open until the P1 branches are in.

## 6. Does R(x) predict forgery, failure, or applicability even without family clusters?

`TODO(run)` — requires the P1d export, which must come from the MR-VAE projector so
`rate_response` is captured. `a3_rate_response.py` is written and guards on this, printing
the exact export command.

The revised gate is already encoded in the script: P1d survives if R(x) adds incremental
signal on **any** of forgery separability (Q1), family structure (Q2), or P0-error
predictiveness (Q3) — clean family clusters are not required. Every probe reports both a
standalone number and a delta over a p_fused-only baseline, so "incremental" is measured
rather than asserted. Prior handoff records family separation as already negative; A3
re-confirms and broadens rather than trusting that.

## 7. Therefore: which two branches enter D1–D3, and is D4 justified?

**Branch selection: `TODO(run)`** on the manifold side. The visual branch is fixed (it is the
baseline the phase criterion protects), and P2a has the strongest evidence of any exported
branch for the second slot — but the manifold projector, which is the actual open question,
needs the P1 exports.

**D4: not yet justified, and not yet refuted.** Per the instructions and the integration
README, D4 stays unbuilt until A2a returns on the real candidate sets and you greenlight it.

What can be said now, from the verification run: there is genuine oracle headroom
(+0.043 DF40, +0.068 CDFv3 on two branches alone), the complementarity is concentrated in
the reenactment inversion rows rather than spread across the average, and a cheap gate
recovers a meaningful share of it on CDFv3 but not on DF40. That is exactly the shape of
result the phase was designed to resolve rather than assume.

---

## Open 🟡 ASK-UMAR items

1. **D1 projector choice** — recommendation deferred until the P1 exports land.
2. **FF++ export for the protocol threshold** — worth adding before the run of record so the
   frozen threshold has real provenance instead of a documented 0.5 fallback.
