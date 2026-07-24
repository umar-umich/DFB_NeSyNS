# CLAUDE_CODE — CEC Tasks T19–T22 (complete the framework, run the training arm)
**Addendum to T13–T18. Read `docs/cec/LOG.md` first.**

**State:** T13–T18 built and smoke-verified. Region + composite scopes both certify (NT 87% video, DF/FS 93–100% video). Remaining: the disclosure layer, the DPO run itself, the before/after experiment, and paper outputs.

**Contract unchanged.** 🟡 [ASK-UMAR] · 🔴 [UMAR-RUNS] · 🟢 [YOU-RUN]. Log everything. **No invented numbers.**

**Decisions carried in (do NOT re-ask):**
- Region margin floor → **0.05** (control-p99 rule, same rule the frozen `gate:` used). **Report both 0.05 and 0.10 as a sensitivity row in every region table** — disclosing both is the defense against a post-hoc-threshold critique. Produce the T13c separation plot as the justification artifact; if it is not separated, region certification is reported unreliable for that method (pre-committed).
- **DeepFakeDetection: dropped** from the composite set (ids outside the FF++ split json; DF/FS suffice). Record the reason.
- Calibrate **effort** only as the secondary instrument; skip gend/forada.
- **Region and composite certification rates are reported separately, never pooled.** The region:composite ratio is a headline diagnostic.

---

## 🔴 BLOCKING RUNS (do these before T19–T22 land)
1. Region-gate calibration for **effort** (fsfm already done).
2. **Full audit**: both proposers, `--group all --split test --videos 40 --oracle`.
3. **Re-freeze** → new hash; keep `49ffffc4f486bd8b` and the smoke hash in the LOG.
4. **T18 pool count** → report against the ≥500 trigger, broken down by case / scope / proposer.

Paste back: per-method certification rates (frame + video, region vs composite separately), oracle hit-rate per proposer, FP-claim rate + coverage, pool counts.

---

## T19 — Disclosure Policy (replaces "Assembly")
🟢 Rename `cec/assembly/output.py` → `cec/assembly/output_policy.py`, class `DisclosurePolicy`. It performs **no certification** — it decides what may be shown.

**Three-tier rule (in order):**
1. any **region**-certified claim → show region evidence (most specific).
2. else any **composite**-certified claim → show the composite statement: *"the inner face is a composited replacement; no individual region is independently necessary."*
3. else → abstain with the contentful form.

**Deployment modes (both implemented, mode is a config flag):**
- `certified_mode` — a reference/source image is available, the gate ran live. Wording may say **"certified evidence."**
- `screening_mode` — no paired real. The gate did **not** run. Wording must say **"causally supervised evidence"** (from the CEC-trained proposer), never "certified."

**Disagreement handling:** if detector = REAL and the proposer asserts manipulation claims → suppress claims, emit `detector_explainer_disagreement`, and **log it**. Default **OFF for the audit** (so the FP-claim metric measures raw proposer behavior), **ON for the demo**. Metrics are always computed pre-suppression.

Unit-test all four outcomes. This module stays ~50 lines; it is a display policy, not a model.

## T20 — DPO training
🔴 [UMAR-RUNS] T17 first: env **`cec_dpo`** with trl+peft+matching torch. Never touch `GenD`. Verify `GenD` torch/transformers/numpy unchanged afterward; record.

🟢 `cec/dpo/train_dpo.py` final: TRL `DPOTrainer`, LoRA (r=16, attn+MLP, β=0.1), bf16, one GPU, seed 17. Config frozen in `cec/registration/dpo.yaml`. Train **InternVL3-8B first** (cheaper; Qwen-32B second if the pool and time allow). Checkpoint per epoch; keep the base model untouched on disk.

**Pre-commit before training:** the eval is run by the **untouched gate** on a **held-out split** the DPO data never saw. State the split in the config.

🔴 [UMAR-RUNS] train. Paste back: pairs used, epochs, loss curve, wall-clock, checkpoint path.

## T21 — Pilot D: re-certification (the headline experiment)
🟢 `cec/pilots/pilot_d/run_pilot_d.py`: run base vs tuned proposer over the **same held-out images**, through the **same frozen gate**, and report a single before/after table:

| metric | base | tuned |
|---|---|---|
| **FP-claim rate on reals** (raw, pre-suppression) | | |
| **Coverage** (certifiable fakes with ≥1 certified claim) | | |
| region certification rate | | |
| composite certification rate | | |
| **region : composite ratio** | | |
| oracle hit-rate (cited a certifiable region) | | |
| abstention rate (reals / weak fakes / all) | | |
| claim diversity across artifact×region (detector-mimic check) | | |

**What counts as success — pre-committed, so it is not decided after seeing numbers:**
- **Primary:** FP-claim rate drops **while coverage holds** (within a few points). Both, or it is not a win.
- **Secondary:** region:composite ratio shifts toward region (the tier worked — the model learned specificity, not just to say "whole face").
- **Failure modes to report honestly:** coverage collapses (model went mute) · FP drops only because the model abstains on everything · diversity collapses to one artifact×region (detector mimicry).

Any honest outcome publishes. 🔴 [UMAR-RUNS].

## T22 — Paper outputs
🟢 `cec/eval/`: regenerate everything from records + commit hash, nothing hand-entered.

**Tables:** detector AUC (inherited, disclaimed) · certification rates by method × scope × proposer, with the 0.05/0.10 sensitivity row · Pilot D before/after · pool composition.

**Figures:**
1. **NM localization vs GT mask (NT)** — per-region NM profile (mouth spike ≈0.085 vs ~0 elsewhere) overlaid on the diffuse GT mask that maxes at IoU 0.21. *This is the strongest figure in the paper: the causal signal localizes the manipulation better than the dataset's own annotation.*
2. Separation plot (true-region vs control NM) — the region-gate justification artifact.
3. FP-claim vs coverage, base → tuned.
4. Qualitative: certified region case · composite case · abstention case (real).

---

## Later levers (do NOT start; recorded so they are not lost)
- **Multi-dataset / composite training** (Umar): train the proposer across FF++ methods + any future mask-bearing source, and evaluate baselines under the same multi-source regime. Revisit after Pilot D gives the single-source number — it is the natural next lever if coverage is the bottleneck.
- Identity-matched retrieval for cross-dataset certification (parked, gated on real-repair offset ≤0.15).
- BiSeNet region masks if landmark regions prove too coarse.

## Order
🔴 blocking runs → T19 → T20 (T17 first) → T21 → T22.