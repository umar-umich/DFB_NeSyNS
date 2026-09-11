# Step 8 — nested vs unified fusion over DiCoME's own two views

The DiCoME fork is gated on external experts surviving complementarity against DiCoME, and none did. But the step's other half — nested versus unified fusion — does not depend on that, and Step 1 gave a concrete reason to ask it: the artifact view ALONE beats DiCoME's fused output on Celeb-DF-v2, DFD and DFDC. A fusion whose output is worse than one of its inputs is mishandling them.

## 1. Are the two internal views complementary to each other?

Every membership test so far measured EXTERNAL experts against a DiCoME anchor. The two internal views are themselves a two-expert system and have never been put through the same gate. Semantic as anchor, artifact as expert, VALmix:

| P(art right \| sem wrong) | P(art wrong \| sem right) | margin | ceiling | realizable recovery |
|---:|---:|---:|---:|---:|
| 0.435 | 0.083 | +0.353 | 0.871 | 15.7% |

| domain | sem acc | art acc | margin |
|---|---:|---:|---:|
| CDFv2val | 0.849 | 0.911 | +0.593 |
| DFDCPval | 0.856 | 0.924 | +0.502 |
| DFEval24val | 0.611 | 0.587 | +0.081 |

## 2. Nested vs unified

| arm | AUROC | Δ vs nested | FPR_real@τ | d_RF |
|---|---:|---:|---:|---:|
| semantic alone | 0.8818 | -0.0035 | 0.314 | +0.314 |
| artifact alone | 0.8789 | -0.0063 | 0.127 | +0.368 |
| NESTED — DiCoME's own DS(sem, art) | 0.8852 | +0.0000 | 0.213 | +0.439 |
| UNIFIED — outer DS(sem, art) | 0.8841 | -0.0011 | 0.187 | +0.389 |
| UNIFIED — outer CCF(sem, art) | 0.8855 | +0.0002 | 0.187 | +0.270 |
| UNIFIED — CCF + applicability | 0.8765 | -0.0087 | 0.206 | +0.258 |

## Verdict

Best unified arm is **UNIFIED — outer CCF(sem, art)** at +0.0002 against DiCoME's own nested DS. Artifact alone is -0.0063 against it.

**Unified fusion does not beat the nested formulation on this split**, so DiCoME's own DS output is an acceptable single anchor opinion here. The Step-1 observation that artifact beats fused on several TEST sets does not reproduce as an advantage on VALmix, which is the honest place to check it — and that discrepancy is itself worth carrying, since Step 1's was a test-set observation and this is a development one.

The internal pair does not clear the membership bar either (margin +0.353, recovery 15.7% against 25%), so the negative result extends to DiCoME's own decomposition — it is not that we picked the wrong experts.
