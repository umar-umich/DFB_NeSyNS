# Branch-3 candidates — what transfers, and the test that decides it cheaply

## First, the constraint that governs every candidate

Branch 3's failure is **not** about which feature it reads. The loss sits only on the fused
opinion; two strong views already produce a nearly correct one; so any non-vacuous contribution
from a weaker third view perturbs a good answer and *increases* the loss. Gradient descent drives
its evidence to zero, and a vacuous opinion is the identity element of the DS sum. Measured:
more head capacity made it **worse** (u 0.871 → 0.996), because a better head learns to be silent
more precisely. See `STAGE5_BRANCH3.md`.

**So no feature swap fixes Branch 3.** Supervision or the fusion operator has to change first.
Feature choice only matters *after* that.

## The 83-d hand-crafted forensic features — and a trap I nearly fell into

`preprocessing/forensic_helpers.py`, precomputed for 29 splits (all of FF++, Celeb-DF-v1/v2/v3,
DFDCP, DFDC, UADFV; only DFEval24 missing). Six mechanistically distinct families: region
forensics (30-d, SegFormer parsing), PPNC (8), CCNC (8), SRM (15), multi-scale noise (12),
spectral FFT (10).

**Within-dataset cross-validation makes them look excellent:**

| set | logistic | GBM |
|---|---:|---:|
| FF++ c23 | 0.9543 | 0.9661 |
| Celeb-DF-v2 | 0.9613 | **0.9804** |
| Celeb-DF-v3 FaceSwap | 0.9210 | 0.9412 |
| DFDCP | 0.9382 | **0.9819** |

Those numbers are **not comparable to anything in this project**, and quoting them would have
been a serious error. Each fold's training data came from the same dataset as its test data. Our
framework never sees a Celeb-DF or DFDC frame during training.

**Under the actual protocol — fit on FF++ c23 only, test zero-shot — they collapse:**

| test set | logistic frame | GBM frame | **logistic video** | **GBM video** |
|---|---:|---:|---:|---:|
| Celeb-DF-v2 | 0.6380 | 0.5806 | **0.6294** | 0.5737 |
| Celeb-DF-v3 FaceSwap | 0.4861 | 0.4942 | **0.5006** | 0.4977 |
| DFDCP | 0.5231 | 0.6217 | **0.5574** | 0.6809 |
| DFD | 0.5300 | 0.4881 | **0.5667** | 0.4892 |
| UADFV | 0.5079 | 0.5684 | **0.5131** | 0.6018 |

0.95 in-domain to 0.50 zero-shot is the signature of features that fingerprint **acquisition**
— compression, camera, preprocessing pipeline — rather than manipulation. CDFv3 FaceSwap lands at
exactly chance. For reference our CLIP-based views reach 0.88-0.96 video AUROC on these same sets
zero-shot.

**Verdict: not a viable Branch 3 under this protocol.** Best cell is DFDCP GBM at 0.6809, and
that arrives with two sets at chance. A view this weak is exactly what the vacuity attractor
eliminates, so it would be ignored even if wired in.

## The reusable result

The cheap test comes first, and it is now scripted:

> **`analysis/tbiom/probe_branch3_candidates.py` — fit a probe on FF++ frames only, test
> zero-shot on every OOD set, report frame AND video AUROC.** Minutes on cached features, no
> GPU, no training run. Only build a branch from a feature that clears this bar.

That would have saved the Stage-5 training run: the SDXL residual's 0.7047 was an *in-domain*
probe, and I let it stand in for transfer. It does not.

## Remaining candidates, ranked by what the evidence supports

1. **Fix the supervision, keep the SDXL residual.** Cheapest decisive experiment. A direct
   per-view loss on `e_proc` tests whether the branch has anything to give. Second variable
   under the brief -> ASK-UMAR.
2. **A frozen self-supervised face representation** (e.g. the FS-VFM expert that Step 7b found
   non-harmful). Representations transfer where hand-crafted statistics do not, which is the
   whole lesson above. Probe it first with the script.
3. **Feature concat rather than a branch.** Six stats (or 83) appended to the artifact vector,
   as the MR-VAE rate curve already is, sharing the trained head. Sidesteps vacuity entirely
   because there is no separate opinion to be ignored. Costs the interpretable third opinion.
4. **World models — not now.** Their forensic content is temporal physical implausibility, and
   this pipeline is frame-level end to end: new datamodule, new corpus, large frozen backbone,
   one score per clip. A different paper. And it would be the most expensive possible way to
   re-derive a vacuous branch while the supervision defect is unfixed.
