# Pilot Spec — Intervention-Based Explanation Verifier

**Status** pre-registered go/no-go pilot. Run this BEFORE writing any method or committing to the workshop paper.
**Goal** decide in about one day whether the intervention test is a real verifier, by checking it against ground-truth manipulation masks on FF++. If it passes, the workshop paper is alive. If it fails, stop and change direction.
**No training. No new data. Reuses your existing frozen detector and FF++.**

---

## 0. The one decision this makes

The workshop paper claims that a deepfake explanation can be verified by intervention. A cited region is real evidence if neutralizing it moves the verdict, and decorative if it does not. That claim only holds if intervening on the ACTUALLY manipulated region moves the verdict much more than intervening on an unmanipulated region. This pilot tests exactly that, against ground truth, with no VLM in the loop yet.

If reverting the true manipulated region does not move the verdict more than reverting a control region, the intervention test is meaningless and the paper is dead. Better to know in a day.

---

## 1. The hypothesis, stated so it can fail

Let `p(fake)` be the frozen detector's output. For a fake frame with ground-truth manipulation mask `M`:

| Operation | Expected effect on p(fake) |
|---|---|
| Revert the true manipulated region `M` to its real pixels | large drop (the fake evidence lived there) |
| Revert a matched-size control region outside `M` | near zero |
| Apply the same paste operation on a real frame | near zero (operation itself is inert) |

The claim is a gap. `drop(true region)` must substantially exceed `drop(control region)`. If they are similar, intervention does not localize evidence, and the verifier is invalid.

---

## 2. Datasets and paths

Use FF++ c23, identity-swap manipulations only, which ship pixel masks.

| Role | Source | Notes |
|---|---|---|
| Fake frames + masks | FF++ `manipulated_sequences/{Deepfakes,FaceSwap,FaceShifter}/c23` | masks under `masks/` per method |
| Paired real source | FF++ `original_sequences/youtube/c23` | same video id, same frame index |
| Real-on-real control | FF++ original frames | for the inert-operation control |

Sampling. 100 fake frames for the day-one gate, balanced across the three methods (about 33 each). Use existing `test.json` indices under your preprocessed FF++ path. Pull the aligned real source frame by matching the original video id and frame number, since FF++ manipulations preserve frame indexing.

Exclude Face2Face and NeuralTextures from the gate. They are reenactment, not identity swap, and their masks behave differently. Keep them as a later diagnostic bucket only.

---

## 3. Frozen detector

Use a detector you already have that outputs a calibrated `p(fake)` per frame, for example your CLIP or PE linear probe. Freeze it. Do not retrain anything. Confirm on a quick sanity batch that it actually separates FF++ real from fake (mean p(fake) clearly higher on fakes), otherwise the deltas below are meaningless.

```python
detector.eval()
with torch.no_grad():
    p = detector(frame)            # scalar in [0,1]
```

---

## 4. The intervention operation (paired-real reversion)

This is the pilot trick. Instead of inpainting, replace the manipulated region with the SAME region from the aligned real source frame. This is a ground-truth counterfactual, pure real pixels, so it cannot introduce synthesis artifacts. It removes the inpainting confound that would otherwise let a reviewer dismiss the result.

```python
# fake: (3,H,W) fake frame
# real: (3,H,W) aligned real source frame (same video, same frame idx)
# mask: (H,W) binary, 1 = manipulated region

def revert(fake, real, region):
    out = fake.clone()
    out[:, region == 1] = real[:, region == 1]
    return out

p_orig   = detector(fake)
p_true   = detector(revert(fake, real, mask))            # revert true region
```

Alignment note. FF++ swaps preserve target pose and frame indexing, so the real source frame is already aligned. If a few samples are visibly misaligned, drop them rather than warping. The pilot does not need all 100 to be perfect.

---

## 5. Controls (these are what make the result credible)

```python
# Control 1: revert a matched-size region OUTSIDE the manipulation
control_region = sample_region(shape_like=mask, disjoint_from=mask)  # e.g. forehead/background
p_ctrl   = detector(revert(fake, real, control_region))

# Control 2: inert-operation check on a REAL frame
# paste a region of real frame B into real frame A (same op, no fake content)
p_real_orig = detector(realA)
p_real_op   = detector(revert(realA, realB_aligned, some_region))
```

Control 1 proves the effect is specific to the manipulated region, not to "any edit anywhere." Control 2 proves the paste operation itself does not move the detector. Report both. Without them the intervention is attackable as "you just added noise."

---

## 6. Metrics and the pre-registered decision

For each fake sample compute:
- `drop_true = p_orig - p_true`
- `drop_ctrl = p_orig - p_ctrl`

Report across the 100 samples:
1. mean `drop_true` and mean `drop_ctrl`
2. verdict-flip rate at threshold 0.5, for true vs control (fraction that cross from fake to real)
3. the gap `mean(drop_true) - mean(drop_ctrl)`
4. mean `|p_real_op - p_real_orig|` (Control 2, must be small)

### Decision table (do not move the goalposts later)

| Result | Verdict | Action |
|---|---|---|
| mean drop_true ≥ 0.30 AND mean drop_ctrl ≤ 0.10 AND Control 2 ≤ 0.05 | PASS | Build the workshop paper. The verifier is real and clean. |
| mean drop_true ≥ 0.30 but drop_ctrl in 0.10–0.20 | WEAK PASS | Verifier works but is leaky. Tighten region selection, then proceed. |
| Control 2 > 0.10 | INVALID OP | The paste operation itself moves the detector. Fix alignment or switch to inpainting before any further conclusion. |
| mean drop_true < 0.20 OR drop_true ≈ drop_ctrl | FAIL | STOP. Reverting the true region does not localize evidence. The intervention test is not a valid verifier. Change direction. |

The single number to watch is the gap `mean(drop_true) - mean(drop_ctrl)`. A large positive gap is the whole paper. A near-zero gap kills it.

---

## 7. Optional, day two only

### Test A2 — validate inpainting against the paired-real gold standard
The paper will need inpainting (real source frames do not exist at test time on arbitrary data). So check that an inpainter approximates the gold-standard reversion. Run LaMa (or similar) on `M`, compute `drop_inpaint`, and correlate with `drop_true` across samples. High correlation means inpainting is a safe stand-in for the method. This is what lets you use the verifier beyond FF++.

### Test B — does the VLM cited region land on the true manipulation
Only if A1 passes. Prompt a frozen VLM (Qwen2.5-VL or InternVL) to name and localize the single most suspicious face region. Then:
- IoU / hit-rate between VLM region and true mask `M`
- among VLM predicates that PASS the intervention test (their cited region, when reverted, drops p(fake)), is mask overlap much higher than among predicates that FAIL?

That last comparison is the workshop paper's headline result in miniature. Intervention-confirmed predicates align with ground truth, rejected ones do not.

---

## 8. Deliverables of the pilot
- `results/pilot/deltas.json` (per-sample drop_true, drop_ctrl, flips, control-2 values)
- `results/pilot/summary.txt` (the four reported numbers and the gap)
- a one-paragraph verdict against the section 6 table

That paragraph decides whether you commit the next 10 days to this paper.

---

## 9. Honesty notes to carry into the writing
- The intervention test proves a cited cue is causally NECESSARY for the verdict, and mask overlap proves it is correctly localized. State the claim as "necessary and correctly localized," not as "the explanation is true."
- The pilot uses paired-real reversion because it is the cleanest validator. The paper uses inpainting for general applicability, justified by Test A2's correlation. Be explicit about this gap so a reviewer cannot frame it as a hidden assumption.
- Detection accuracy is expected to stay flat. The contribution is verified explanation, not better detection. Say so plainly.