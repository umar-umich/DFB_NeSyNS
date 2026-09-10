# Referral — does the evidential machinery earn its keep?

Arm D (plain heads, mean-of-logits, cross-entropy) ties the evidential framework on OOD AUROC. If the evidential apparatus is worth carrying, the difference has to show up somewhere other than accuracy. The candidate is **referral**: withholding a decision on the cases the model cannot support, and routing them elsewhere.

Every abstention threshold is a **dev quantile** (FF++ val + VALmix) applied unchanged to the six OOD sets pooled. No per-test-set tuning, no oracle coverage matching. Each model uses its own exported fused probability and its own EER threshold.

## Rejection quality

| model | signal | err-AUROC ↑ | AURC ↓ | E-AURC ↓ |
|---|---|---:|---:|---:|
| arm B — evidential (EDL + DS) | mean branch vacuity | **0.771** | 0.1956 | 0.1624 |
| arm B — evidential (EDL + DS) | mean branch vacuity, stream-normalised | **0.764** | 0.1284 | 0.0951 |
| arm B — evidential (EDL + DS) | fused vacuity | **0.735** | 0.1704 | 0.1371 |
| arm B — evidential (EDL + DS) | fused margin |p-tau| | **0.568** | 0.2596 | 0.2263 |

`err-AUROC` is how well the signal ranks the model's own mistakes — the cleanest read on rejection quality, free of any threshold and of the base error rate. `E-AURC` subtracts the AURC an oracle ranking of the same predictions would reach, which removes the base-accuracy term and is what makes the two arms comparable.

## Risk-coverage, dev-fitted thresholds

Balanced error on the retained set. Coverage is what the dev threshold happens to retain on OOD — it is not matched between arms, and the fact that it differs is itself informative.

**arm B — evidential (EDL + DS)** — pooled OOD AUROC 0.8439, base balanced error 0.2235 at full coverage

| signal | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|
| mean branch vacuity | 0.1913 | 0.1716 | 0.1576 | 0.1276 | 0.0982 |
| mean branch vacuity, stream-normalised | 0.1963 | 0.1747 | 0.1524 | 0.1264 | 0.1044 |
| fused vacuity | 0.1968 | 0.1811 | 0.1513 | 0.1319 | 0.1186 |
| fused margin |p-tau| | 0.2088 | 0.1795 | 0.1678 | 0.1575 | 0.1571 |


## Verdict

**Referral works, and vacuity is what makes it work.** Balanced error falls from 0.2235 at full coverage to 0.1276 at 61% coverage — a 43% relative reduction — from a threshold fitted on dev and never retuned per test set.

**Vacuity substantially beats the free alternative.** The fused margin |p−τ| is the only uncertainty a non-evidential fusion of these same three branches would hand you at no cost, and it is far worse: err-AUROC 0.568 against vacuity's 0.771, E-AURC 0.2263 against 0.1624. Within this model, the Dirichlet quantity is doing real work that the decision score alone does not.

**Stream normalisation is a large, free improvement — and it is specific to this model's weakness.** Rank-normalising vacuity against the incoming batch cuts E-AURC from 0.1624 to 0.0951 (41% better) while leaving err-AUROC essentially unchanged (0.771 to 0.764). That pattern identifies the problem precisely: the ranking was already good, and what was costing us was the global threshold sitting in a different place on each domain. It is transductive — legitimate for corpus triage, not for scoring one video alone.

**What this report does not establish.** It covers the evidential arm only, so it cannot support a claim that evidential fusion is *required* for referral, or that it is the best available choice. The margin is the free baseline, not the only baseline — a non-evidential model with its own confidence estimate is a different comparison, and it is not made here. Any writing must say "vacuity beats the decision margin within our framework", never "EDL is needed for referral".

**And one correction to an earlier claim in this file.** A previous version said per-domain thresholds would likely reverse an unfavourable comparison. Half of that is now tested and true — normalising per incoming set is worth 41% of E-AURC. The other half is false: it does not improve err-AUROC at all (0.771 raw to 0.764 normalised), so calibration was never the whole story and the ranking quality of vacuity is what it is.
