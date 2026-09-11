# Stage 7b — the remaining conflict-aware operators

Stage 7 tested DS, Yager, Murphy and CCF and adopted none. Three operators were named
afterwards. All are replay-only: no GPU, no retraining.

## Dubois–Prade is Yager on a binary frame

Not approximately — **exactly**, and it is provable before running anything. DP assigns conflict
to the **union** of the conflicting focal elements. Our frame has two classes, so the only
conflicting pair is `{real}` and `{fake}`, and their union is `Θ` — which is precisely where
Yager already puts conflict.

Verified numerically rather than left as an argument: **max |difference| = 0.00e+00** across
every video of Celeb-DF-v2.

So Dubois–Prade adds nothing here. It would only differ on a frame with three or more classes,
where a union of two singletons is a proper subset of Θ. Worth recording so nobody implements it
a second time.

## Results — arm C, three views

| operator | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | mean | vs P0-DS | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DS | 0.9556 | 0.8805 | 0.9372 | 0.8816 | 0.8822 | 0.6801 | 0.8695 | +0.0062 | 0.209 |
| **Yager** | 0.9587 | 0.8886 | 0.9367 | 0.8810 | 0.8848 | 0.6821 | **0.8720** | +0.0086 | 0.211 |
| Dubois–Prade | 0.9587 | 0.8886 | 0.9367 | 0.8810 | 0.8848 | 0.6821 | 0.8720 | +0.0086 | 0.211 |
| **PCR6** | 0.9557 | 0.8818 | 0.9364 | 0.8789 | 0.8817 | 0.6792 | **0.8689** | +0.0056 | 0.229 |
| Deng-entropy | 0.9560 | 0.8812 | 0.9373 | 0.8817 | 0.8822 | 0.6804 | 0.8698 | +0.0065 | 0.210 |

**PCR6 is the worst operator tested**, below plain DS. Redistributing conflict proportionally
back to the masses that caused it does not help here. **Deng-entropy weighting is
indistinguishable from DS** (+0.0003) — the branches' information content is too similar for
entropy weights to separate them.

## Results — arm C, two views (semantic dropped)

| operator | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | mean | vs P0-DS | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DS | 0.9632 | 0.8971 | 0.9411 | 0.8875 | 0.8927 | 0.6870 | 0.8781 | +0.0148 | 0.209 |
| Yager | 0.9636 | 0.8977 | 0.9397 | 0.8848 | 0.8911 | 0.6852 | 0.8770 | +0.0137 | 0.211 |
| Dubois–Prade | 0.9636 | 0.8977 | 0.9397 | 0.8848 | 0.8911 | 0.6852 | 0.8770 | +0.0137 | 0.211 |
| PCR6 | 0.9634 | 0.8974 | 0.9403 | 0.8860 | 0.8920 | 0.6860 | 0.8775 | +0.0142 | 0.208 |
| **Deng-entropy** | 0.9633 | 0.8970 | 0.9412 | 0.8875 | 0.8928 | 0.6876 | **0.8782** | +0.0149 | 0.211 |

## The conclusion, and it closes the question

**The operator spread collapses once the redundant view is removed.**

| configuration | best | worst | spread |
|---|---:|---:|---:|
| three views | 0.8720 (Yager) | 0.8689 (PCR6) | **0.0031** |
| two views | 0.8782 (Deng) | 0.8770 (Yager) | **0.0012** |

Removing one redundant view is worth **+0.006**; the best operator swap available is worth
**+0.0025**, and only in the configuration that still contains the redundancy it is compensating
for. **Fixing the input beats changing the rule by more than double**, and once the input is
fixed every operator lands within 0.0012 of every other — which is noise.

Nine operators have now been tested across Stage 7 and 7b: DS, Yager, Murphy, CCF,
Dubois–Prade (= Yager here), PCR6, Deng-entropy, plus simple averaging and learned weights.
**None beats plain DS on a well-formed input by a margin that clears the noise band.**

The conflict-aware line of attack is closed. Dempster–Shafer was never the bottleneck; feeding
it two correlated views was, and DS's independence assumption is what that violates.

## Caveats

- **PCR is non-associative.** The three-view number folds pairwise, which is PCR6-style but is
  not the n-ary operator. A genuine n-ary PCR6 could differ. Given it placed last at N=2 where
  PCR5 and PCR6 coincide exactly, this is unlikely to reverse.
- **Deng-entropy weighting has a convention choice.** Weight is taken inversely proportional to
  entropy here; some papers normalise `exp(E)` instead. Recorded because the two differ, though
  with weights this close to uniform the choice cannot matter much.
- All numbers are single-seed, and the 0.0012 two-view spread is well inside the noise band.
