# Choosing three branches

Every three-branch candidate, ranked by the DEVELOPMENT objective — the only basis on
which a deployed subset may be chosen. Table A2 ranks subsets on the test sets and its
own banner says that maximum is biased upward; this table is what that banner points to.

The four-branch set is the last row, for reference only.

| branches | dev v3.1 | 12 sets, no adv | 12 sets, with adv | DF40 (16 sets) |
|---|---:|---:|---:|---:|
| App+TD+LFS | 0.9056 | 0.9480 | 0.9441 | 0.9915 |
| App+TD+STT-order | 0.9045 | 0.9462 | 0.9427 | 0.9917 |
| App+TD+STT | 0.9039 | 0.9464 | 0.9434 | 0.9906 |
| TD+LFS+STT | 0.9009 | 0.9472 | 0.9452 | 0.9898 |
| TD+LFS+STT-order | 0.8986 | 0.9446 | 0.9423 | 0.9887 |
| App+LFS+STT-order | 0.8835 | 0.9407 | 0.9368 | 0.9941 |
| App+LFS+STT | 0.8834 | 0.9402 | 0.9366 | 0.9940 |
| **App+TD+LFS+STT** | 0.9065 | 0.9514 | 0.9482 | 0.9938 |

## What the numbers do and do not settle

The development spread across all 7 candidates is 0.0222, which is
WIDER than the measured seed noise of 0.0126 on this objective — so this table is
not uniformly inconclusive, and saying so would be the easy and wrong reading. It
separates into two groups:

- **5 candidates lie within seed noise of the best** (`App+TD+LFS`, `App+TD+STT-order`, `App+TD+STT`, `TD+LFS+STT`, `TD+LFS+STT-order`), spread 0.0070. Among these the numbers decide nothing.
- **2 are genuinely behind** (`App+LFS+STT-order`, `App+LFS+STT`), by more than noise.

Every tied candidate contains **TD**, so
that is the part the evidence fixes; the remaining slot is where the choice is free. The
twelve-set ordering also reverses between the two censuses, which is a second sign of
the same thing. That makes novelty the free variable rather than something to be argued
against the numbers.

## Where each branch comes from

| branch | provenance |
|---|---|
| App | adapted PE-Core-L/336 appearance — our LoRA+LayerNorm recipe over a standard view |
| TD | V-JEPA 2 LoRA temporal — trained by this campaign, new to the framework |
| LFS | INHERITED: the CLIP/β-VAE orthogonal residual from the DiCoME-lineage stage-23 export |
| STT-order | the same construction with learned slot positions — STT's ablation, not a view |
| STT | NEW: PE-Core block 16 + real relative timestamps, built by this campaign |

`LFS` is the one inherited view: it is the DiCoME-lineage artifact residual, read from
the stage-23 export. A three-branch framework that keeps it is carrying the baseline's
own representation into the contribution; one that drops it stands on App, TD and STT,
all three of which this campaign built or adapted. What that costs is the
`App+TD+LFS` row against the `App+TD+STT`
row above — read it there rather than from this paragraph.

