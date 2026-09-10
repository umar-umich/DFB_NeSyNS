# Phase T — the symbolic coherence term

Arm B seed 42 with `lambda_cons` added and nothing else changed. The term is the differentiable relaxation of overlay rule R4: it charges fused certainty in proportion to how far the certainty-weighted branch votes cancel. It uses no labels and no thresholds, so unlike the gate's predicates it cannot fit the development set.

## 1 — Detection

| arm | mean OOD AUROC | vs arm B | mean FPR_real | vs arm B |
|---|---:|---:|---:|---:|
| arm B (lambda_cons=0) | **0.8768** | — | **0.212** | — |
| lambda_cons=0.1 | 0.8687 | -0.0081 | 0.143 | -0.069 |
| lambda_cons=0.5 | 0.8741 | -0.0027 | 0.177 | -0.035 |

### Per dataset

| dataset | P0-DS | arm B (lambda_cons=0) | lambda_cons=0.1 | lambda_cons=0.5 |
|---|---|---|---|---|
| CDFv2 | 0.9646 | 0.9474 | 0.9360 | 0.9433 |
| CDFv3 | 0.8409 | 0.8871 | 0.8780 | 0.8798 |
| DFD | 0.9421 | 0.9454 | 0.9411 | 0.9470 |
| DFDC | 0.8828 | 0.8906 | 0.8811 | 0.8804 |
| DFDCP | 0.8573 | 0.9038 | 0.8874 | 0.8932 |
| DFEval24 | 0.6922 | 0.6866 | 0.6886 | 0.7010 |

## 2 — Did the term bite?

Measured on the six OOD sets, in the same functional form the loss used. If these do not move, the term changed nothing and the AUROC column above is noise about nothing.

| arm | cross-branch agreement | conflicted-confident mass | mean branch vacuity |
|---|---:|---:|---:|
| arm B (lambda_cons=0) | 0.8631 | 0.1316 | 0.1930 |
| lambda_cons=0.1 | 0.8721 | 0.1209 | 0.2078 |
| lambda_cons=0.5 | 0.8659 | 0.1262 | 0.2204 |

## 3 — In-domain vs out-of-domain: where the coherence went

The penalty is applied on FF++ during training. These columns ask whether the optimiser solved it there, and whether the solution survives off-domain.

| arm | train agreement (e0 → e19) | train penalty (e0 → e19) | **OOD agreement** | train branch vacuity | **OOD branch vacuity** |
|---|---|---|---:|---:|---:|
| arm B (lambda_cons=0) | n/a (term off) | n/a (term off) | **0.8631** | 0.0475 | **0.1930** |
| lambda_cons=0.1 | 0.908 → 0.996 | 0.0853 → 0.0034 | **0.8721** | 0.0483 | **0.2078** |
| lambda_cons=0.5 | 0.930 → 0.996 | 0.0658 → 0.0030 | **0.8659** | 0.0498 | **0.2204** |

## Verdict

**The term was learned in-domain and did not transfer. No detection benefit.**

*It bit, hard, where it was applied.* At lambda=0.1 the penalty fell 0.0853 -> 0.0034 over 20 epochs — a factor of 25 — and training agreement rose 0.908 -> 0.996. The optimiser had no trouble at all satisfying the constraint on FF++.

*Almost none of it survived off-domain.* OOD agreement moved only 0.8631 -> 0.8721 (+0.0090) and conflicted-confident mass 0.1316 -> 0.1209. A constraint driven to near-zero in-domain leaves a ~1% trace on unseen data: the model learned to make its three views agree *on FF++ manipulations*, which is not the same as learning to hold coherent evidence in general.

*Detection got slightly worse, consistently.* Mean OOD AUROC -0.0081 at lambda=0.1 and -0.0027 at lambda=0.5. Neither clears the 0.01 significance band, so neither is individually claimable — but lambda=0.1 is below arm B on 5 of 6 datasets, so the sign is consistent even where the magnitude is not. Nothing here suggests a larger lambda would help: the stronger penalty was *worse* on the OOD coherence it was meant to fix (0.1262 against lambda=0.1's 0.1209), which is non-monotonic and rules out simply turning it up.

*The FPR_real drop is not a win.* Real-side FPR falls 0.212 -> 0.143, but OOD branch vacuity rises over arm B at both lambdas while in-domain vacuity is unchanged (0.0487/0.0512/0.0426 for arm B against 0.0498/0.0524/0.0427 at lambda=0.1). The model is simply less confident on unfamiliar data, which slides the operating point toward Real. AUROC is threshold-free and did not improve, so this is calibration movement, not better separation.

**Read together with the gate, the two failures are the same failure.** The gate's symbolic predicates fitted dev and transferred worse than raw vacuity; the coherence term fitted FF++ and left almost nothing off-domain. In both cases the symbolic structure is learnable on the distribution it was derived from and does not carry. That is a coherent finding about this symbolic layer, and it is the argument for keeping it post-hoc: as an explanation it is faithful and cheap, and every attempt to give it a job in the decision has been measured and has not paid.

**Not claimed:** that symbolic constraints cannot help in training. What is shown is that *this* constraint, at these two strengths, on one seed each, does not. A term built on a predicate that transfers — the gate's diagnostics point at graded vacuity rather than Boolean rules — is untested.
