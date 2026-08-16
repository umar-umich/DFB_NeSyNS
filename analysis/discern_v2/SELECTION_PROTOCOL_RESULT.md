# Leaked vs leak-free checkpoint selection — measured

The integration README flags a leak-free OOD selection protocol as *a candidate contribution
in its own right*. This measures the thing that claim rests on: how much does selecting the
best epoch on the test sets actually inflate the reported numbers?

Two D0 runs of the **same system** (`nesy_defake_ablation4_ccv`), differing only in which
datasets early stopping was allowed to see. Both frozen checkpoints then evaluated through
the **identical** `test.py` path on all eight sets — comparing against numbers a training
loop printed would have made part of the gap a measurement artefact.

* **leaked selection** — `test_dataset` = all eight; early stop at epoch 24, best epoch 4.
* **leak-free** — `test_dataset` = [FaceForensics++, Celeb-DF-v2]; early stop at epoch 38.
  The other six sets were never seen during training or selection.

## Video-level AUROC

| | FF++ | DFD | CDFv1 | CDFv2 | CDFv3 | DFDC | DFDCP | UADFV | **mean** |
|---|---|---|---|---|---|---|---|---|---|
| leaked selection | 0.9823 | 0.9245 | 0.9614 | 0.9402 | 0.8245 | 0.8335 | 0.8443 | 0.9942 | **0.9131** |
| leak-free | 0.9814 | 0.9246 | 0.9266 | 0.9360 | 0.7750 | 0.8279 | 0.8702 | 0.9929 | **0.9043** |
| **gap** | −0.0009 | 0.0000 | **−0.0348** | −0.0041 | **−0.0495** | −0.0056 | **+0.0259** | −0.0012 | **−0.0088** |

Frame level shows the same shape at about half the magnitude (mean 0.8664 → 0.8618,
gap −0.0045), with FF++ actually *improving* under leak-free selection (+0.0132).

## What this says

**The mean inflation is small — 0.009 video AUROC.** On the headline number, test-set epoch
selection is not buying much, and a paper reporting the leaked figure is not wildly
overstating its system. That is worth saying plainly, because the opposite is often assumed.

**But the gap is not uniform, and that is the real finding.** It is concentrated in
Celeb-DF-v1 (−0.035) and Celeb-DF-v3 (−0.050), near zero on five sets, and *reversed* on
DFDCP (+0.026, where leak-free is better). Leaked selection is not applying a uniform
optimistic bias; it is picking an epoch that happens to suit particular datasets. Per-dataset
numbers can therefore move by up to five points on the strength of the selection rule alone —
five times the mean effect, and larger than most architectural deltas this ladder is trying
to detect.

**Why that matters here specifically.** The D-ladder's rungs are separated by fractions of a
point. A selection rule that moves a single dataset by 0.05 can manufacture or erase a rung's
apparent benefit outright. Whatever protocol is chosen, it has to be the same across
D0–D5 — mixing them would put every delta on a different footing.

**For the paper.** Reporting both is the strongest position and costs nothing now that both
runs exist: the leaked number for comparability with prior work (captioned as
test-set-selected), the leak-free number as the honest one, and this table as the evidence
that the difference is small in aggregate but materially dataset-dependent. That is a more
interesting claim than either number alone, and it is exactly the contribution the
integration README anticipated.

## Caveat

Single seed per protocol. The two leak-free D0 seeds (3407, 1024) differ by 0.0017 on the
validation average, so a 0.0088 mean gap is roughly 5x the seed spread and likely real —
but the *per-dataset* swings of 0.035–0.050 have not been replicated across seeds and could
be partly run-to-run variance. A second leaked-selection seed would settle it.
