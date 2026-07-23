# CLAUDE_CODE — CEC Tasks T13–T18 (addendum to Implementation v2)
**Read `docs/cec/LOG.md` and the Findings & Build Status doc first. This addendum supersedes nothing; it adds the tasks that resolve the region-gate blocker and complete the training arm.**

**Context in one line:** Tasks 0–12 are built and verified. The audit returned 0 CERTIFIED / 100% abstention because (Error 1) Pilot 1L used a margin-only criterion inconsistent with the real gate, and (Error 2, dominant) region-sized repairs (~0.15) were judged by thresholds calibrated for whole-mask repairs (~0.65). T13–T18 fix this, add the missing real-image path, and gate the DPO decision on a measured pool.

**Contract unchanged:** 🟡 [ASK-UMAR] decisions · 🔴 [UMAR-RUNS] heavy jobs · 🟢 [YOU-RUN] quick work. Status line per task. Append every decision/gate to `docs/cec/LOG.md` (UTC). Invariants: **freeze before you measure · training only after the audit freeze · every component earns a number · no invented numbers.**

**Decisions already made by Umar (do NOT re-ask):**
1. Composite scope: **approved** (see T13).
2. Region calibration rule + failure branch: **approved as stated** (see T13).
3. Prompt revision: **spend the one permitted revision** (see T15).
4. Re-run scope: **approved**, at **video-level sampling — 8–16 frames per video** (matches the 32-frames-per-video convention used in face-swap evaluation; per-video aggregation, not per-frame).
5. Proposers: **both** (InternVL3-8B and Qwen2.5-VL-32B).
6. Real:fake ratio: **1:1** everywhere (audit and DPO pool).
7. DPO pool trigger: **≥500 pairs** (see T18).

**Decided without asking (object if wrong):** abstention JSON schema (T15) · old audit hash `49ffffc4f486bd8b` stays in the log beside the new one · assembly module renamed **Disclosure Policy** (`cec/assembly/output_policy.py`) · FP-claim metrics recorded **before** any disagreement suppression · `trl` installed in a new isolated env `cec_dpo`, never `GenD`.

---

## T13 — Scope routing + region-gate calibration
**Goal:** make region claims judgeable by a bar calibrated for region-sized repairs, and stop discarding composite claims.

🟢 **(a) Scope routing, derived from location — no prompt change.**
In `cec/gate/certify.py`, route each claim by its location, not by a new field:
- location ∈ {`whole_face`, inner-face union} → **scope = composite** → repair the full inner-face region → judge against the **frozen `gate:` block, byte-unchanged**.
- location ∈ specific landmark regions → **scope = region** → judge against the new `gate_region:` block.
- no locus / unmappable → **UNTESTABLE** (unchanged).
This reclassifies the 32/67 `whole_face` claims from the audit as testable. Composite repair = the *same* operation the frozen block was calibrated on, so no threshold is being "lowered" — each operation is judged by its own calibrated bar. Record that sentence in the LOG; it is the reviewer answer.

🟢 **(b) `gate_region:` calibration** — new `cec/scripts/calibrate_region_gate.py`:
- Data: **train split only** (disjoint from the rev3 test split — codebook rule).
- Measure per-detector pooled control drops (matched blur, matched shift, wrong region) on **region-sized** repairs across the landmark region vocabulary.
- `gap_region = p99` of that pooled distribution; keep the `max(0.10, …)` floor convention from the `gate:` block for the certify margin; wrong-region and real-offset inertness thresholds re-measured the same way.
- Write as a **new `gate_region:` block** in `params.yaml`. **Leave `gate:` byte-unchanged.**

🔴 [UMAR-RUNS] the calibration job. Paste back: per-detector `p99`, resulting `gap_region`, and the sample count.

🟢 **(c) Acceptance artifact — the separation plot, not the number.** Produce, per method, overlaid distributions of true-region NM vs pooled control NM, with the chosen threshold marked. This is what goes in the paper, not the threshold value.

**GATE T13 (pre-committed both ways):**
- **PASS** if true-region and control distributions are visibly separated at the chosen bar → region certification proceeds.
- **FAIL** (material overlap) → that method is reported as **region-certification unreliable**; we do **NOT** lower the bar until something passes. Composite scope still stands for that method. Record the failure as a finding.

---

## T14 — Fix Pilot 1L + full re-run, both regimes, both proposers
🟢 **(a) Error 1 fix:** `cec/pilots/pilot_1l/run_pilot_1l.py` must call the same `CertificationGate` used by the audit. Delete the margin-only branch. Any "pool" count it reports is now gate-certified by construction.

🟢 **(b) Video-level sampling.** Per Umar: sample **8–16 frames per video** (fix the number in `params.yaml`, log it), aggregate **per video**, and report both per-frame and per-video certification rates. Do not mix videos across splits.

🟢 **(c) Re-run matrix:**
| Regime | Methods | Expect |
|---|---|---|
| Region | NeuralTextures, Face2Face | region claims certifiable (localized manipulation) |
| Composite | Deepfakes, FaceSwap, DeepFakeDetection | region claims fail; composite claims certify |
Both proposers (InternVL3-8B, Qwen2.5-VL-32B) on the identical frozen image set.

🟢 **(d) New metric — proposer→oracle hit-rate:** fraction of images where the proposer cited a region the gate independently certifies (the gate finds the certifiable region regardless of what was cited). This is the number that predicts DPO viability, and it is reportable on its own.

🔴 [UMAR-RUNS]. Paste back per (method × proposer): certification rate (frame + video), scope mix, hit-rate, abstention rate.

---

## T14b — DF40 feasibility probe (gated; do NOT assume it works)
**Why this is worth probing:** the LOG records DF40 as gate-incapable because it ships no masks. That is correct for **region** scope — but **composite scope needs no mask**: the inner-face region is derivable from landmarks, which DF40 *does* ship. The only remaining requirement is a geometry-consistent **paired real**. For DF40's `ff` subsets (FF++-compatible `<id1_id2>` naming), a paired real may be recoverable by re-cropping the raw FF++ original with **DF40's own landmarks** — the same re-alignment trick already used in `cec/data/pairing.py`.

🟢 Write `cec/pilots/pilot_df40/probe_pairing.py`: for ~50 samples from 2–3 DF40 `ff` swap families, attempt paired-real recovery via `align_face` + DF40 landmarks; report the QC bg-MSE distribution against the existing `QC_MSE ≤ 60` bar.

🔴 [UMAR-RUNS]. Paste back: `{family, n_attempted, n_passing_QC, median_bg_mse}`.

**GATE T14b:**
- **PASS** (≥60% pass QC) → DF40 `ff` swap families join the **composite** arm; this materially widens the pilot and gives cross-generator numbers. Region scope remains FF++-only.
- **FAIL** → DF40 stays out of the gate, as currently recorded. No re-preprocessing of the 159 GB tree. Report the probe as a documented limitation.

---

## T15 — Real-image path + three-case preference builder (the missing half)
**This is the largest gap in the current build:** the audit ran on fakes only, `build_prefs` has no real-image path, and the headline metric (FP-claim rate on reals) cannot currently be computed.

🟢 **(a) Prompt revision — spend the one permitted revision.** Extend the frozen proposer prompt so abstention is an emitable, contentful output:
```json
{"verdict_support": "no_certified_evidence", "examined": ["skin", "blending", "eyes"]}
```
**One abstention form only.** Do **not** create separate tokens for real vs. weak-fake — distinguishing those would require the proposer to make a verdict call, which it must never do. Record old and new prompt hashes in `pins.yaml`.

🔴 [UMAR-RUNS] re-run the **100-image schema-validity gate** (≥95%, both proposers) on the revised prompt. This budget was unspent in Task 6.

🟢 **(b) Real-image audit path.** Run reals (FF++ `original_sequences/youtube`, same splits, **1:1 with fakes**) through the identical gate. Expect every claim REJECTED (NM ≈ 0 — nothing to restore). Reals also anchor the real-repair-offset null.

🟢 **(c) Three-case preference builder** in `cec/dpo/build_prefs.py`:
| Case | chosen | rejected |
|---|---|---|
| Fake, evidence certifies | certified claim | an uncertified claim, same image |
| Real image | contentful abstention (a) | any hallucinated claim |
| Weak fake (nothing certifies) | contentful abstention (a) | any hallucinated claim |
**Preference tiers within the fake case:** region-certified ≻ composite-certified ≻ rejected. This teaches specificity without punishing honest composite claims on overdetermined swaps.
**Never** use an empty list/array as `chosen` — an empty target trains toward silence.

🟢 **(d) Metrics** in `cec/eval/metrics.py`:
- **FP-claim rate**: fraction of real images on which the proposer asserts ≥1 manipulation claim. **Computed on raw proposer output, BEFORE any disagreement suppression.**
- **Coverage**: fraction of certifiable fakes where ≥1 claim certifies.
- Report as a **pair** (and as an FP-vs-coverage curve if restraint weighting is swept). "Maximize abstention" is **not** the objective — it collapses to silence.

---

## T16 — Re-audit and re-freeze
🟢 Update `cec/scripts/run_audit.py` for: scope routing (T13), video-level sampling (T14), real images (T15), both proposers.
🔴 [UMAR-RUNS] 10-image timing probe → hour estimate → full run.
🟢 Hash `records/`, commit, **new freeze hash**. Keep `49ffffc4f486bd8b` in the LOG beside it — the 0-certified run is reportable history, not an embarrassment to delete.
**Nothing in T18 may run before this commit exists.**

---

## T17 — `trl` install (isolated)
🔴 [UMAR-RUNS] create env **`cec_dpo`** with `trl` + `peft` + matching torch. **Never install into `GenD`** — that env produced every frozen anchor. Verify `GenD`'s torch/transformers/numpy versions are unchanged afterward and record them in the LOG.

---

## T18 — Measure the DPO pool, then STOP
🟢 Run `build_prefs` over the re-frozen records. Report:
- total pairs · pairs by case (fake-certified / real-abstention / weak-fake-abstention) · pairs by scope (region / composite) · pairs by proposer.
🟡 **[ASK-UMAR] — STOP HERE. Do not train.** Present the counts against the pre-committed trigger:
- **≥500 pairs** (Umar's threshold) → proceed to DPO on Umar's go.
- **<500** → escalation ladder, in order: (i) widen to more videos/frames; (ii) lean on the proposer with the better hit-rate from T14; (iii) oracle-positive pairs — **note these are off-policy**, so the honest recipe is a small SFT warmup on certified claims *then* DPO on model-proposed pairs; (iv) pre-committed fallback: publish the abstention result as the demonstrated hallucination-free guarantee.

---

## Order and dependencies
T13 → T14 (+T14b in parallel) → T15 → T16 (freeze) → T17 → T18 (stop).
T13's gate can fail without blocking T14's composite arm. T14b can fail without blocking anything. T15(a) blocks T15(b–d). T16 blocks T18.

## Notes carried forward
- Score original and intervened **in the same batch** (the ~1e-4 batch-composition offset cancels in the difference — LOG, Task 2).
- fsfm is the primary certification instrument; effort secondary; spectral is characterization-only, whole-face *frequency/noise* claims remain UNTESTABLE (distinct from whole-face **composite** claims, which T13 makes testable — do not conflate).
- Inference honesty: at deployment nothing certifies live (no paired real). In user-facing text say **"causally supervised evidence"**, reserving "certified" for offline/controlled settings where the gate actually ran.