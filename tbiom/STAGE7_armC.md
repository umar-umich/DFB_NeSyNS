# Stage 7 — conflict-aware fusion (run1cft)

Replayed over the SAME saved branch opinions; no retraining. Each view's Dirichlet opinion is recovered from its exported `(p, u)` and converted to a subjective-logic opinion, then fused N-ary.

| operator | conflict handling |
|---|---|
| DS | renormalised away — the incumbent |
| Yager | assigned to **vacuity** instead of renormalised |
| Murphy | opinions averaged, then combined N−1 times |
| CCF | multi-source consensus & compromise (FUSION 2018), genuinely N-ary |

## Complete-framework video AUROC

| dataset | DS | Yager | Murphy | CCF | best branch | best simple | P0-DS |
|---|---|---|---|---|---|---|---|
| CDFv2 | 0.9556 | **0.9587** | 0.9567 | 0.9381 | 0.9433 | 0.9660 | 0.9646 |
| CDFv3 | 0.8805 | **0.8886** | 0.8827 | 0.8654 | 0.8947 | 0.9026 | 0.8409 |
| DFD | 0.9372 | 0.9367 | **0.9381** | 0.9248 | 0.9213 | 0.9394 | 0.9421 |
| DFDC | 0.8816 | 0.8810 | **0.8827** | 0.8587 | 0.8623 | 0.8837 | 0.8828 |
| DFDCP | 0.8822 | **0.8848** | 0.8829 | 0.8616 | 0.8729 | 0.8871 | 0.8573 |
| DFEval24 | 0.6801 | **0.6821** | 0.6809 | 0.6648 | 0.6870 | 0.6880 | 0.6922 |
| VALmix | 0.8819 | **0.8850** | 0.8825 | 0.8638 | 0.8712 | 0.8906 | 0.8852 |

| operator | mean AUROC (6 OOD) | mean FPR_real | vs P0-DS |
|---|---:|---:|---:|
| DS | 0.8695 | 0.209 | +0.0062 |
| Yager | 0.8720 | 0.211 | +0.0086 |
| Murphy | 0.8707 | 0.239 | +0.0074 |
| CCF | 0.8522 | 0.290 | -0.0111 |
| *(simple F1)* | 0.8708 | — | +0.0075 |
| *(simple F2)* | 0.8778 | — | +0.0145 |
| *(simple F3)* | 0.8701 | — | +0.0068 |

## Pass condition

From the brief: *beats simple fusion on the suppressed cases while holding the strong datasets.* Suppressed = fused loses to its own best branch under DS.

| suppressed dataset | DS | best branch | best conflict-aware | best simple | fixed? |
|---|---:|---:|---:|---:|---|
| CDFv3 | 0.8805 | 0.8947 | **0.8886** (Yager) | 0.9026 | no |
| DFEval24 | 0.6801 | 0.6870 | **0.6821** (Yager) | 0.6880 | no |

---

## Stage 7 verdict — NOT ADOPTED

The brief's rule: *"Adopt it only if it fixes the suppression without harming the strong
datasets."* No conflict-aware operator fixes it, on either arm.

| arm | suppressed | DS | best branch | best conflict-aware | best **simple** | fixed? |
|---|---|---:|---:|---:|---:|---|
| B | CDFv2 | 0.9439 | 0.9607 | 0.9500 (Murphy) | **0.9597** | no |
| C | CDFv3 | 0.8805 | 0.8947 | 0.8886 (Yager) | **0.9026** | no |
| C | DFEval24 | 0.6801 | 0.6870 | 0.6821 (Yager) | **0.6880** | no |

On every suppressed case the best conflict-aware operator is beaten by **plain simple fusion** —
learned weights (F2) on arm C, and F2 again on arm B's CDFv2. Conflict handling is not the
binding constraint.

Means tell the same story. Arm B: Murphy 0.8787 against simple F1's 0.8778 — **+0.0009**, noise.
Arm C: the best operator of any kind is **simple F2 at 0.8778**, ahead of Yager's 0.8720 by
0.0058. A sophisticated operator that loses to a three-parameter weighting has not earned its
place.

**CCF is the worst performer on both arms** (0.8682 and 0.8522, the latter *below* P0-DS) while
having the best real-side FPR on arm B (0.175). Its consensus step takes `min` belief across all
sources, so a single low-belief view caps the fused belief — protective on the real side,
costly for ranking. Worth recording, since CCF was the phase-2 primary candidate.

## What this actually says

The Stage-7 hypothesis was that DS suppresses a strong view by renormalising conflict away, and
that a conflict-aware operator would recover it. **The first half is confirmed and the second is
refused.** Yager, which is exactly "do not renormalise conflict", recovers only a fraction:
+0.0081 on arm C's CDFv3 against the 0.0142 that DS discards.

So the suppression is not primarily a conflict-renormalisation artifact. The remaining
explanation consistent with everything measured is **weighting**: F2 — one non-negative weight
per view, three parameters, fitted on VALmix — is the best operator on arm C and within noise of
the best on arm B, and it consistently zeroes the semantic view. The branches are not equally
useful, and every operator here except F2 treats them as if they were.

That points the next step at *learned or evidence-scaled per-view weighting inside the model*,
not at a more elaborate combination rule. It is a different fix from the one Stage 7 proposed,
and it is what the numbers support.
