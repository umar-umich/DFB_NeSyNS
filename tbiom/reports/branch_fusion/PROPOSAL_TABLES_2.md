# Proposal results, pass 2

Regenerated 2026-10-14T23:11:31-04:00. Census: **`nolipadv`** — the lip-sync / audio-driven exclusion
removed, and the adversarial subsets admitted. CDFv3 **5,418** videos, KoDF **2,239**.

`PROPOSAL_TABLES_1.md` holds the protocol-v3.1 census and is unchanged; this file is the
widened one. Only CDFv3 and KoDF differ between them — the other ten test sets contain
none of the excluded families, so their cells are identical in both.

| part | what it answers |
|---|---|
| I | what admitting each family cost, and where the false alarms come from |
| II | which three branches to take forward, if three is the number |
| III | every table on this census |
| IV | DF40 cross-manipulation — the structure of DiCoME Table 1's second half |

## One caveat, and it decides which table to quote where

**The baselines were never run on the admitted videos.** The GenD runs' prediction files
stop at the v3.1 census for CDFv3 and at the no-adversarial census for KoDF, and DiCoME's
CDFv3 export covers 1,918 videos against our 5,418. Widening the census widens OUR row and
leaves theirs where it was; Table 1 below names every cell this affects.

So: quote **Table 1 from `PROPOSAL_TABLES_1.md`** for the comparison against the field,
where every cell is on one census, and the tables here for everything about our own
system — the ablation, the fusion operator, calibration, and DF40.

---

# Part I — what the census costs

These are the only two test sets whose video census the scope touches. The other ten
contain none of the excluded families and are byte-identical under every scope.

The system is unchanged across the rows: `App + TD + LFS + STT` under PCR6+-R with the
weights fitted once on the development pool, which no scope touches. A difference below
is a difference in which videos are scored, and nothing else.

## CDFv3

| scope | videos (real / fake) | AUROC | FPR@TPR90 | App | TD | LFS | STT |
|---|---|---:|---:|---:|---:|---:|---:|
| v3.1 (published) | 178 / 4940 | **0.9612** | 0.1124 | 0.8935 | 0.9263 | 0.8801 | 0.8864 |
| + lip-sync / audio-driven | 178 / 5240 | **0.9614** | 0.1124 | 0.8893 | 0.9288 | 0.8790 | 0.8884 |
| + adversarial | 178 / 5240 | **0.9614** | 0.1124 | 0.8893 | 0.9288 | 0.8790 | 0.8884 |

`nolip` admits: `synthesis-TalkingFace-IP_LAP`

## KoDF

| scope | videos (real / fake) | AUROC | FPR@TPR90 | App | TD | LFS | STT |
|---|---|---:|---:|---:|---:|---:|---:|
| v3.1 (published) | 403 / 992 | **0.9436** | 0.0397 | 0.8563 | 0.9656 | 0.8548 | 0.9268 |
| + lip-sync / audio-driven | 403 / 1106 | **0.9494** | 0.0323 | 0.8692 | 0.9653 | 0.8647 | 0.9335 |
| + adversarial | 806 / 1433 | **0.9112** | 0.3834 | 0.8117 | 0.9298 | 0.7999 | 0.9093 |

`nolip` admits: `fake-audio-driven`
`nolipadv` admits: `adv-fake`, `adv-real`

## Where the false alarms come from

The threshold is the one that catches 90% of the fakes on the census WITHOUT the
adversarial subsets — the system as it would be deployed having never seen them — and
each genuine subset is then asked what fraction of it is flagged at that threshold.
This separates two things an aggregate FPR cannot: new fakes being missed, and new
genuine videos being accused.

### CDFv3

| subset | label | videos | flagged at that threshold |
|---|---|---:|---:|
| `YouTube-real` | genuine | 70 | 0.1286 |
| `Celeb-real` | genuine | 108 | 0.1019 |
| `synthesis-FaceSwap-SimSwap` | fake | 200 | 0.5700 |
| `synthesis-TalkingFace-SadTalker` | fake | 300 | 0.6333 |
| `synthesis-FaceReenact-DaGAN` | fake | 200 | 0.7500 |
| `synthesis-FaceReenact-LivePortrait` | fake | 200 | 0.8350 |
| `synthesis-TalkingFace-AniTalker` | fake | 300 | 0.8533 |
| `synthesis-TalkingFace-EchoMimic` | fake | 300 | 0.8567 |
| `synthesis-FaceReenact-FSRT` | fake | 200 | 0.8650 |
| `synthesis-FaceReenact-MCNET` | fake | 200 | 0.9100 |
| `synthesis-FaceReenact-TPSMM` | fake | 200 | 0.9100 |
| `synthesis-TalkingFace-IP_LAP` | fake | 300 | 0.9100 |
| `synthesis-FaceSwap-GHOST` | fake | 200 | 0.9400 |
| `synthesis-FaceSwap-MobileFaceSwap` | fake | 200 | 0.9450 |
| `synthesis-TalkingFace-FLOAT` | fake | 300 | 0.9467 |
| `synthesis-FaceSwap-Celeb-DF-v2` | fake | 340 | 0.9529 |
| `synthesis-FaceReenact-HyperReenact` | fake | 200 | 0.9700 |
| `synthesis-FaceReenact-LIA` | fake | 200 | 0.9700 |
| `synthesis-FaceSwap-HifiFace` | fake | 200 | 0.9950 |
| `synthesis-FaceSwap-BlendFace` | fake | 200 | 1.0000 |
| `synthesis-FaceSwap-InSwapper` | fake | 200 | 1.0000 |
| `synthesis-FaceSwap-UniFace` | fake | 200 | 1.0000 |
| `synthesis-TalkingFace-EDTalk` | fake | 300 | 1.0000 |
| `synthesis-TalkingFace-R3DPortrait` | fake | 300 | 1.0000 |

### KoDF

| subset | label | videos | flagged at that threshold |
|---|---|---:|---:|
| `adv-real` | genuine | 403 | 0.0546 |
| `real` | genuine | 403 | 0.0323 |
| `adv-fake` | fake | 327 | 0.4128 |
| `fake-fsgan` | fake | 205 | 0.6195 |
| `fake-fo` | fake | 352 | 0.9062 |
| `fake-audio-driven` | fake | 114 | 1.0000 |
| `fake-dffs` | fake | 223 | 1.0000 |
| `fake-dfl` | fake | 212 | 1.0000 |

At this fixed threshold the adversarial genuine videos are flagged at 0.0546 against 0.0323 for the ordinary ones — close. What moves the aggregate is the fake side: `adv-fake` is caught at only 0.4128, so holding TPR at 90% over the wider census forces a much lower threshold, and that is what floods the genuine videos.


---

# Part II — choosing three branches

Every three-branch candidate, ranked by the DEVELOPMENT objective — the only basis on
which a deployed subset may be chosen. Table A2 ranks subsets on the test sets and its
own banner says that maximum is biased upward; this table is what that banner points to.

The four-branch set is the last row, for reference only.

| branches | dev v3.1 | 12 sets | DF40 face swap | DF40 reenact | DF40 all |
|---|---:|---:|---:|---:|---:|
| App+TD+LFS | 0.9056 | 0.9441 | 0.9918 | 0.9456 | 0.9670 |
| App+TD+STT-order | 0.9045 | 0.9427 | 0.9922 | 0.9499 | 0.9695 |
| App+TD+STT | 0.9039 | 0.9434 | 0.9910 | 0.9385 | 0.9628 |
| TD+LFS+STT | 0.9009 | 0.9452 | 0.9894 | 0.9335 | 0.9594 |
| TD+LFS+STT-order | 0.8986 | 0.9423 | 0.9889 | 0.9548 | 0.9706 |
| App+LFS+STT-order | 0.8835 | 0.9368 | 0.9939 | 0.8934 | 0.9400 |
| App+LFS+STT | 0.8834 | 0.9366 | 0.9939 | 0.8755 | 0.9303 |
| **App+TD+LFS+STT** | 0.9065 | 0.9482 | 0.9936 | 0.9342 | 0.9617 |

### What the numbers do and do not settle

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

### Where each branch comes from

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


---

# Part III — every table on this census

Risk-coverage figure: `tbiom/reports/branch_fusion/scope_nolipadv/figures/uncertainty_riskcoverage.png`

## Figure — vacuity and risk-coverage

`figures/uncertainty_riskcoverage.png`, regenerated with the panels chosen by AUROC among the sets with at least forty errors: IDForge, FFIW, CDFv3, DFD.

## Top row — the separation holds

Vacuity split by whether the fused prediction is right. Misclassified videos carry
visibly more vacuity than correct ones on every panel, which is the claim DiCoME's
Figure 3 makes and it reproduces here:

| set | u(correct) | u(wrong) | ratio | errors |
|---|---:|---:|---:|---:|
| IDForge | 0.0059 | 0.0208 | 3.54x | 174 |
| FFIW | 0.0075 | 0.0254 | 3.40x | 295 |
| CDFv3 | 0.0056 | 0.0193 | 3.45x | 380 |
| DFD | 0.0034 | 0.0100 | 2.90x | 534 |

## Bottom row — is it actually a better rejector?

Separating errors in a histogram does not by itself make a quantity a better rejector
than the decision margin, which costs nothing and needs no evidential model. So the
verdict below is computed, not asserted. Area under the risk-coverage curve, lower is
better, both rejectors on the same coverage grid:

| set | vacuity (ours) | margin \|p-tau\| | oracle | vacuity better? |
|---|---:|---:|---:|---|
| IDForge | 0.0094 | 0.1085 | 0.0007 | yes |
| FFIW | 0.0433 | 0.1642 | 0.0047 | yes |
| CDFv3 | 0.0915 | 0.0938 | 0.0101 | yes |
| DFD | 0.0478 | 0.1201 | 0.0149 | yes |

Vacuity wins 4 of the 4 panels (IDForge, FFIW, CDFv3, DFD).

### What may be claimed from this

Both rows. Vacuity separates errors in the histogram AND orders them better
than the decision margin on every panel, so the evidential apparatus is paying
for itself as a rejector and not only as an interpretable quantity.

---

## Table A — branch ablation

Ten groups, every dataset, fused with **PCR6+-R**. No baseline rows: this table answers
what each branch contributes, and a SOTA column here would invite two questions to be
read off one row. The comparison against the field is Table 1.

Every branch group is computed on all 12 test sets.

## AUROC (higher is better)

| branch group | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| App | 0.9460 | 0.8893 | 0.9505 | 0.8371 | 0.9000 | 0.9760 | 0.8117 | 0.9950 | 0.9653 | 0.9708 | 0.9048 | 0.8636 | 0.9175 |
| TD | 0.7383 | 0.9288 | 0.7967 | 0.7009 | 0.7408 | 0.8011 | 0.9298 | 0.8899 | 0.9798 | 0.9456 | 0.8499 | 0.8378 | 0.8450 |
| LFS | 0.9736 | 0.8790 | 0.9338 | 0.8808 | 0.8727 | 0.9354 | 0.7999 | 0.9635 | 0.9920 | 0.9908 | 0.8595 | 0.7678 | 0.9041 |
| STT-order | 0.9661 | 0.8942 | 0.9524 | 0.8360 | 0.8887 | 0.9640 | 0.9035 | 0.9935 | 0.9969 | 0.9825 | 0.9012 | 0.8589 | 0.9282 |
| STT | 0.9683 | 0.8884 | 0.9485 | 0.8176 | 0.8976 | 0.9655 | 0.9093 | 0.9968 | 0.9954 | 0.9731 | 0.9056 | 0.8663 | 0.9277 |
| App+TD+LFS *(deployed)* | 0.9645 | 0.9623 | 0.9513 | 0.8596 | 0.9058 | 0.9693 | 0.9021 | 0.9950 | 0.9927 | 0.9883 | 0.9268 | 0.9119 | 0.9441 |
| App+TD+LFS+STT-order | 0.9738 | 0.9621 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9060 | 0.9961 | 0.9953 | 0.9867 | 0.9285 | 0.9210 | 0.9471 |
| App+TD+LFS+STT **(final)** | 0.9788 | 0.9614 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9112 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9482 |
| App+TD+LFS+STT-order+STT | 0.9738 | 0.9621 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9060 | 0.9961 | 0.9953 | 0.9867 | 0.9285 | 0.9210 | 0.9471 |
| App+LFS+STT-order+STT | 0.9847 | 0.9145 | 0.9528 | 0.8652 | 0.9224 | 0.9800 | 0.8631 | 0.9962 | 0.9939 | 0.9863 | 0.9146 | 0.8713 | 0.9371 |

## FPR at TPR = 90% (lower is better)

| branch group | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| App | 0.1910 | 0.3596 | 0.1708 | 0.4894 | 0.2565 | 0.0576 | 0.6960 | 0.0013 | 0.0730 | 0.0612 | 0.2542 | 0.4836 | 0.2579 |
| TD | 0.6180 | 0.2247 | 0.5868 | 0.7525 | 0.6304 | 0.4922 | 0.2419 | 0.3174 | 0.0393 | 0.1837 | 0.4922 | 0.4697 | 0.4207 |
| LFS | 0.0618 | 0.4213 | 0.2011 | 0.4289 | 0.3739 | 0.2242 | 0.7320 | 0.1145 | 0.0056 | 0.0000 | 0.3651 | 0.6554 | 0.2987 |
| STT-order | 0.1292 | 0.3708 | 0.1736 | 0.4592 | 0.3000 | 0.1101 | 0.3586 | 0.0057 | 0.0000 | 0.0000 | 0.2458 | 0.4799 | 0.2194 |
| STT | 0.0843 | 0.4101 | 0.2094 | 0.5110 | 0.3130 | 0.0974 | 0.3288 | 0.0020 | 0.0000 | 0.0408 | 0.2387 | 0.4842 | 0.2266 |
| App+TD+LFS *(deployed)* | 0.1011 | 0.0955 | 0.1653 | 0.3983 | 0.2870 | 0.0997 | 0.4342 | 0.0094 | 0.0112 | 0.0204 | 0.1773 | 0.2271 | 0.1689 |
| App+TD+LFS+STT-order | 0.0674 | 0.0674 | 0.1570 | 0.3978 | 0.2739 | 0.1032 | 0.4268 | 0.0037 | 0.0056 | 0.0204 | 0.1751 | 0.2281 | 0.1605 |
| App+TD+LFS+STT **(final)** | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.3834 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1557 |
| App+TD+LFS+STT-order+STT | 0.0674 | 0.0674 | 0.1570 | 0.3978 | 0.2739 | 0.1032 | 0.4268 | 0.0037 | 0.0056 | 0.0204 | 0.1751 | 0.2281 | 0.1605 |
| App+LFS+STT-order+STT | 0.0449 | 0.3146 | 0.1653 | 0.4078 | 0.2348 | 0.0519 | 0.5633 | 0.0003 | 0.0000 | 0.0000 | 0.2147 | 0.4423 | 0.2033 |


---

## Table A2 — all 31 combinations of the five branches

> **EXPLORATORY DIAGNOSTIC -- NOT A HELD-OUT RESULT. Every subset here was scored on every test set, so the best cell in this grid is a maximum over a large search on the test partitions and is biased upward. Use it to read structure and to choose table scope. The held-out numbers are the locked rows in test_eval.json, and the deployed subset must be chosen on development.**

## Dataset census

| set | videos | real | fake | prevalence |
|---|---:|---:|---:|---:|
| CDFv2 | 518 | 178 | 340 | 0.656 |
| CDFv3 | 5418 | 178 | 5240 | 0.967 |
| DFD | 3431 | 363 | 3068 | 0.894 |
| DFDC | 4704 | 2315 | 2389 | 0.508 |
| DFDCP | 654 | 230 | 424 | 0.648 |
| FFIW | 3470 | 1735 | 1735 | 0.500 |
| KoDF | 2239 | 806 | 1433 | 0.640 |
| IDForge | 5974 | 2987 | 2987 | 0.500 |
| DFDM | 1896 | 178 | 1718 | 0.906 |
| UADFV | 98 | 49 | 49 | 0.500 |
| DeepSpeak-1.1 | 2102 | 1416 | 686 | 0.326 |
| DeepSpeak-2.0 | 2572 | 1863 | 709 | 0.276 |

## AUROC by subset and test set

`worst` is the subset's weakest set, `spread` its best minus its worst -- a subset with a high macro and a wide spread is carried by a few sets.

| subset | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | macro | worst | spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| App+TD+LFS+STT | 0.9788 | 0.9614 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9112 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | **0.9482** | 0.8622 (DFDC) | 0.1345 |
| App+TD+LFS+STT-order | 0.9738 | 0.9621 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9060 | 0.9961 | 0.9953 | 0.9867 | 0.9285 | 0.9210 | **0.9471** | 0.8593 (DFDC) | 0.1368 |
| App+TD+LFS+STT-order+STT | 0.9738 | 0.9621 | 0.9577 | 0.8593 | 0.9090 | 0.9701 | 0.9060 | 0.9961 | 0.9953 | 0.9867 | 0.9285 | 0.9210 | **0.9471** | 0.8593 (DFDC) | 0.1368 |
| TD+LFS+STT-order+STT | 0.9782 | 0.9656 | 0.9543 | 0.8587 | 0.8895 | 0.9667 | 0.9346 | 0.9931 | 0.9981 | 0.9850 | 0.9243 | 0.9129 | **0.9467** | 0.8587 (DFDC) | 0.1394 |
| TD+LFS+STT | 0.9764 | 0.9680 | 0.9495 | 0.8566 | 0.8857 | 0.9638 | 0.9369 | 0.9904 | 0.9977 | 0.9846 | 0.9249 | 0.9078 | **0.9452** | 0.8566 (DFDC) | 0.1411 |
| App+TD+STT-order+STT | 0.9639 | 0.9577 | 0.9551 | 0.8365 | 0.9114 | 0.9713 | 0.9099 | 0.9970 | 0.9948 | 0.9825 | 0.9257 | 0.9252 | **0.9443** | 0.8365 (DFDC) | 0.1605 |
| App+TD+LFS | 0.9645 | 0.9623 | 0.9513 | 0.8596 | 0.9058 | 0.9693 | 0.9021 | 0.9950 | 0.9927 | 0.9883 | 0.9268 | 0.9119 | **0.9441** | 0.8596 (DFDC) | 0.1354 |
| App+TD+STT | 0.9592 | 0.9587 | 0.9530 | 0.8338 | 0.9099 | 0.9706 | 0.9119 | 0.9967 | 0.9939 | 0.9817 | 0.9262 | 0.9251 | **0.9434** | 0.8338 (DFDC) | 0.1628 |
| App+TD+STT-order | 0.9592 | 0.9598 | 0.9542 | 0.8350 | 0.9037 | 0.9658 | 0.9060 | 0.9964 | 0.9936 | 0.9850 | 0.9257 | 0.9282 | **0.9427** | 0.8350 (DFDC) | 0.1614 |
| TD+LFS+STT-order | 0.9739 | 0.9714 | 0.9486 | 0.8565 | 0.8774 | 0.9497 | 0.9331 | 0.9885 | 0.9976 | 0.9854 | 0.9172 | 0.9085 | **0.9423** | 0.8565 (DFDC) | 0.1412 |
| TD+STT-order+STT | 0.9603 | 0.9570 | 0.9383 | 0.8108 | 0.8858 | 0.9587 | 0.9399 | 0.9936 | 0.9970 | 0.9783 | 0.9188 | 0.9261 | **0.9387** | 0.8108 (DFDC) | 0.1861 |
| App+LFS+STT-order+STT | 0.9847 | 0.9145 | 0.9528 | 0.8652 | 0.9224 | 0.9800 | 0.8631 | 0.9962 | 0.9939 | 0.9863 | 0.9146 | 0.8713 | **0.9371** | 0.8631 (KoDF) | 0.1331 |
| App+LFS+STT-order | 0.9859 | 0.9156 | 0.9531 | 0.8732 | 0.9181 | 0.9819 | 0.8553 | 0.9953 | 0.9942 | 0.9888 | 0.9166 | 0.8643 | **0.9368** | 0.8553 (KoDF) | 0.1401 |
| LFS+STT-order+STT | 0.9834 | 0.9089 | 0.9532 | 0.8686 | 0.9064 | 0.9746 | 0.9018 | 0.9947 | 0.9972 | 0.9875 | 0.9103 | 0.8548 | **0.9368** | 0.8548 (DeepSpeak-2.0) | 0.1424 |
| App+LFS+STT | 0.9838 | 0.9134 | 0.9518 | 0.8656 | 0.9241 | 0.9807 | 0.8628 | 0.9963 | 0.9918 | 0.9854 | 0.9137 | 0.8699 | **0.9366** | 0.8628 (KoDF) | 0.1335 |
| LFS+STT | 0.9868 | 0.9090 | 0.9489 | 0.8703 | 0.9099 | 0.9742 | 0.9006 | 0.9932 | 0.9964 | 0.9850 | 0.9105 | 0.8480 | **0.9361** | 0.8480 (DeepSpeak-2.0) | 0.1485 |
| LFS+STT-order | 0.9826 | 0.9107 | 0.9533 | 0.8783 | 0.9026 | 0.9762 | 0.8950 | 0.9930 | 0.9978 | 0.9883 | 0.9058 | 0.8472 | **0.9359** | 0.8472 (DeepSpeak-2.0) | 0.1506 |
| App+TD | 0.9330 | 0.9607 | 0.9419 | 0.8250 | 0.8946 | 0.9591 | 0.9055 | 0.9935 | 0.9878 | 0.9808 | 0.9230 | 0.9236 | **0.9357** | 0.8250 (DFDC) | 0.1685 |
| TD+STT | 0.9515 | 0.9560 | 0.9329 | 0.8045 | 0.8836 | 0.9550 | 0.9430 | 0.9868 | 0.9942 | 0.9683 | 0.9178 | 0.9205 | **0.9345** | 0.8045 (DFDC) | 0.1897 |
| App+STT-order+STT | 0.9774 | 0.9077 | 0.9533 | 0.8436 | 0.9172 | 0.9766 | 0.8698 | 0.9967 | 0.9919 | 0.9813 | 0.9125 | 0.8782 | **0.9339** | 0.8436 (DFDC) | 0.1531 |
| App+STT | 0.9761 | 0.9063 | 0.9528 | 0.8418 | 0.9188 | 0.9772 | 0.8704 | 0.9969 | 0.9868 | 0.9746 | 0.9134 | 0.8788 | **0.9328** | 0.8418 (DFDC) | 0.1551 |
| App+STT-order | 0.9745 | 0.9073 | 0.9567 | 0.8489 | 0.9135 | 0.9777 | 0.8607 | 0.9962 | 0.9927 | 0.9838 | 0.9085 | 0.8736 | **0.9328** | 0.8489 (DFDC) | 0.1472 |
| STT-order+STT | 0.9693 | 0.8945 | 0.9516 | 0.8289 | 0.8977 | 0.9653 | 0.9069 | 0.9956 | 0.9965 | 0.9792 | 0.9044 | 0.8656 | **0.9296** | 0.8289 (DFDC) | 0.1675 |
| App+LFS | 0.9747 | 0.9059 | 0.9492 | 0.8678 | 0.9153 | 0.9806 | 0.8192 | 0.9951 | 0.9845 | 0.9863 | 0.9125 | 0.8568 | **0.9290** | 0.8192 (KoDF) | 0.1759 |
| STT-order | 0.9661 | 0.8942 | 0.9524 | 0.8360 | 0.8887 | 0.9640 | 0.9035 | 0.9935 | 0.9969 | 0.9825 | 0.9012 | 0.8589 | **0.9282** | 0.8360 (DFDC) | 0.1609 |
| STT | 0.9683 | 0.8884 | 0.9485 | 0.8176 | 0.8976 | 0.9655 | 0.9093 | 0.9968 | 0.9954 | 0.9731 | 0.9056 | 0.8663 | **0.9277** | 0.8176 (DFDC) | 0.1792 |
| TD+LFS | 0.9575 | 0.9670 | 0.9294 | 0.8489 | 0.8576 | 0.9174 | 0.9270 | 0.9621 | 0.9955 | 0.9863 | 0.9037 | 0.8746 | **0.9273** | 0.8489 (DFDC) | 0.1466 |
| TD+STT-order | 0.9438 | 0.9684 | 0.9125 | 0.7808 | 0.8501 | 0.9204 | 0.9407 | 0.9854 | 0.9972 | 0.9804 | 0.9082 | 0.9160 | **0.9253** | 0.7808 (DFDC) | 0.2163 |
| App | 0.9460 | 0.8893 | 0.9505 | 0.8371 | 0.9000 | 0.9760 | 0.8117 | 0.9950 | 0.9653 | 0.9708 | 0.9048 | 0.8636 | **0.9175** | 0.8117 (KoDF) | 0.1833 |
| LFS | 0.9736 | 0.8790 | 0.9338 | 0.8808 | 0.8727 | 0.9354 | 0.7999 | 0.9635 | 0.9920 | 0.9908 | 0.8595 | 0.7678 | **0.9041** | 0.7678 (DeepSpeak-2.0) | 0.2242 |
| TD | 0.7383 | 0.9288 | 0.7967 | 0.7009 | 0.7408 | 0.8011 | 0.9298 | 0.8899 | 0.9798 | 0.9456 | 0.8499 | 0.8378 | **0.8450** | 0.7009 (DFDC) | 0.2789 |
| ORACLE (best single per set, uses test labels) | 0.9736 | 0.9288 | 0.9524 | 0.8808 | 0.9000 | 0.9760 | 0.9298 | 0.9968 | 0.9969 | 0.9908 | 0.9056 | 0.8663 | **0.9415** | 0.8663 (DeepSpeak-2.0) | 0.1306 |

## Best single branch per set (the oracle's choice)

| set | best branch | AUROC | App | TD | LFS | STT-order | STT |
|---|---|---:|---:|---:|---:|---:|---:|
| CDFv2 | **LFS** | 0.9736 | 0.9460 | 0.7383 | 0.9736 | 0.9661 | 0.9683 |
| CDFv3 | **TD** | 0.9288 | 0.8893 | 0.9288 | 0.8790 | 0.8942 | 0.8884 |
| DFD | **STT-order** | 0.9524 | 0.9505 | 0.7967 | 0.9338 | 0.9524 | 0.9485 |
| DFDC | **LFS** | 0.8808 | 0.8371 | 0.7009 | 0.8808 | 0.8360 | 0.8176 |
| DFDCP | **App** | 0.9000 | 0.9000 | 0.7408 | 0.8727 | 0.8887 | 0.8976 |
| FFIW | **App** | 0.9760 | 0.9760 | 0.8011 | 0.9354 | 0.9640 | 0.9655 |
| KoDF | **TD** | 0.9298 | 0.8117 | 0.9298 | 0.7999 | 0.9035 | 0.9093 |
| IDForge | **STT** | 0.9968 | 0.9950 | 0.8899 | 0.9635 | 0.9935 | 0.9968 |
| DFDM | **STT-order** | 0.9969 | 0.9653 | 0.9798 | 0.9920 | 0.9969 | 0.9954 |
| UADFV | **LFS** | 0.9908 | 0.9708 | 0.9456 | 0.9908 | 0.9825 | 0.9731 |
| DeepSpeak-1.1 | **STT** | 0.9056 | 0.9048 | 0.8499 | 0.8595 | 0.9012 | 0.9056 |
| DeepSpeak-2.0 | **STT** | 0.8663 | 0.8636 | 0.8378 | 0.7678 | 0.8589 | 0.8663 |

---

## Table B — fusion operator on the final combination (App + TD + LFS + STT)

`App + TD + LFS + STT` under each decision rule. The cross-entropy row is fused
by mean-of-logits and NOT by DS or PCR6+: a CE branch has no Dirichlet, so its
uncertainty is a decision margin, and combining margins as belief masses would invent
evidence that was never measured. Mean-of-logits is what DiCoME's row (D) does.

**Mean appears twice on purpose.** F1 is a uniform mean and reads no reliability at all,
so comparing it against two fitted operators would confound the decision rule with the
weight fit. The second mean row is the same average on the fitted simplex, which is the
matched comparator for DS-C and PCR6+-R.

The cross-entropy row is on the same videos as every other row here.

## AUROC (higher is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.9465 | 0.9405 | 0.9570 | 0.8377 | 0.8922 | 0.9671 | 0.9204 | 0.9907 | 0.9886 | 0.9838 | 0.9303 | 0.9014 | 0.9380 |
| Mean (unweighted) | 0.9805 | 0.9589 | 0.9552 | 0.8583 | 0.9144 | 0.9739 | 0.9208 | 0.9951 | 0.9957 | 0.9850 | 0.9308 | 0.9158 | 0.9487 |
| Mean (fitted weights) | 0.9795 | 0.9601 | 0.9552 | 0.8602 | 0.9156 | 0.9747 | 0.9156 | 0.9957 | 0.9947 | 0.9863 | 0.9312 | 0.9147 | 0.9486 |
| DS-C | 0.9795 | 0.9690 | 0.9512 | 0.8578 | 0.9116 | 0.9748 | 0.9225 | 0.9966 | 0.9970 | 0.9867 | 0.9321 | 0.9222 | 0.9501 |
| PCR6+-R (ours) | 0.9788 | 0.9614 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9112 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9482 |

## FPR at TPR = 90% (lower is better)

| fusion rule | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cross-entropy (mean of logits) | 0.1011 | 0.1854 | 0.1460 | 0.5136 | 0.3304 | 0.0778 | 0.2543 | 0.0295 | 0.0225 | 0.0204 | 0.1942 | 0.3097 | 0.1821 |
| Mean (unweighted) | 0.0506 | 0.1404 | 0.1543 | 0.4086 | 0.2826 | 0.0657 | 0.2233 | 0.0117 | 0.0000 | 0.0204 | 0.1603 | 0.2405 | 0.1465 |
| Mean (fitted weights) | 0.0506 | 0.1124 | 0.1570 | 0.4039 | 0.2696 | 0.0646 | 0.2903 | 0.0094 | 0.0000 | 0.0204 | 0.1596 | 0.2480 | 0.1488 |
| DS-C | 0.0674 | 0.0899 | 0.1928 | 0.3991 | 0.2826 | 0.0715 | 0.2469 | 0.0054 | 0.0000 | 0.0204 | 0.1702 | 0.2163 | 0.1469 |
| PCR6+-R (ours) | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.3834 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1557 |


---

## Table 1 — detection performance

Per dataset. **AUROC higher is better, FPR lower is better.** FPR is read at a MATCHED
true-positive rate, so every method is compared at the same sensitivity and no
calibration difference enters: of the genuine videos, what fraction is flagged when the
method is tuned to catch 90% (resp. 95%) of the fakes.

This is an ROC property, like AUROC. Whether a threshold fixed in advance survives the
domain shift is a different and far less flattering question, answered in
`SOTA_OPERATING_POINTS.md`. The two must not be quoted as the same measurement.

`*in progress*`: not scored on that corpus yet, not zero and not inapplicable. Our row covers
every set. The baselines were never run on the three corpora packed on 2026-10-13, and
DiCoME's public checkpoints cover only the five sets its paper reports.

A ROW'S `mean` IS OVER ITS OWN COVERAGE, which is not a comparison. Our row spans twelve
sets, most baselines nine, DiCoME five -- and the three corpora only we cover are not the
easy ones, so our own mean is read off a harder average than anyone else's. The two
matched columns are the comparable figures:

- `mean (all)` -- every set every row in the table resolves. DiCoME's five-set coverage
  sets that width, so it is honest for all nine methods but narrow.
- `mean (wide)` -- every set that every method covering nine or more sets resolves. The
  narrow rows say in progress there rather than shrinking the column for everyone.

Rank methods on a matched column and say which one. The per-row mean belongs in the
per-dataset discussion, not in a ranking.

Every baseline cell is scored on the same videos as our row.

## AUROC (higher is better)

`mean (all)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.9788 | 0.9614 | 0.9569 | 0.8622 | 0.9112 | 0.9743 | 0.9112 | 0.9961 | 0.9966 | 0.9875 | 0.9277 | 0.9144 | 0.9482 | 0.9482 | 0.9482 |
| DiCoME (trained, their repo) | 0.9644 | 0.8435 | 0.9419 | 0.8825 | 0.8576 | 0.9511 | 0.7835 | 0.9701 | 0.9928 | 0.9842 | 0.8491 | 0.7630 | 0.8986 | 0.8986 | 0.8986 |
| GenD-CLIP | 0.9534 | 0.8484 | 0.9350 | 0.8708 | 0.9162 | 0.9278 | 0.8011 | 0.9839 | 0.9959 | 0.9921 | 0.8580 | 0.7445 | 0.9022 | 0.9022 | 0.9022 |
| GenD-PE | 0.9548 | 0.8953 | 0.9203 | 0.8158 | 0.8860 | 0.9330 | 0.8031 | 0.9808 | 0.9815 | 0.9758 | 0.8613 | 0.7865 | 0.8995 | 0.8995 | 0.8995 |
| GenD-DINO | 0.9177 | 0.8218 | 0.9165 | 0.8470 | 0.9114 | 0.9450 | 0.8675 | 0.9860 | 0.9903 | 0.9871 | 0.8403 | 0.7701 | 0.9001 | 0.9001 | 0.9001 |
| FSFM | 0.9554 | 0.8744 | 0.9219 | 0.8844 | 0.9061 | 0.9168 | 0.7731 | 0.9566 | 0.9790 | 0.9754 | 0.8601 | 0.7452 | 0.8957 | 0.8957 | 0.8957 |
| FSFM-Adapter | 0.8689 | 0.8342 | 0.7516 | 0.8318 | 0.8486 | 0.8565 | 0.7245 | 0.9077 | 0.8452 | 0.9933 | 0.7741 | 0.6550 | 0.8243 | 0.8243 | 0.8243 |
| Effort | 0.9312 | 0.7915 | 0.9276 | 0.8536 | 0.8617 | 0.9252 | 0.8180 | 0.9583 | 0.9729 | 0.9742 | 0.8396 | 0.6622 | 0.8763 | 0.8763 | 0.8763 |
| ForAda | 0.9544 | 0.7505 | 0.9395 | 0.8726 | 0.9289 | 0.9055 | 0.8133 | 0.9101 | 0.9713 | 0.9942 | 0.7597 | 0.6947 | 0.8746 | 0.8746 | 0.8746 |

## FPR at TPR = 90% (lower is better)

`mean (all)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0674 | 0.1124 | 0.1570 | 0.3948 | 0.2696 | 0.0646 | 0.3834 | 0.0067 | 0.0056 | 0.0204 | 0.1645 | 0.2222 | 0.1557 | 0.1557 | 0.1557 |
| DiCoME (trained, their repo) | 0.1011 | 0.5955 | 0.1901 | 0.3970 | 0.4783 | 0.1573 | 0.7457 | 0.0947 | 0.0112 | 0.0000 | 0.4449 | 0.6973 | 0.3261 | 0.3261 | 0.3261 |
| GenD-CLIP | 0.1685 | 0.4775 | 0.2507 | 0.4082 | 0.2217 | 0.2196 | 0.7084 | 0.0305 | 0.0056 | 0.0000 | 0.4541 | 0.7558 | 0.3084 | 0.3084 | 0.3084 |
| GenD-PE | 0.1404 | 0.3539 | 0.3003 | 0.5296 | 0.3348 | 0.1712 | 0.7022 | 0.0495 | 0.0449 | 0.0612 | 0.4011 | 0.6634 | 0.3127 | 0.3127 | 0.3127 |
| GenD-DINO | 0.2135 | 0.5955 | 0.3416 | 0.4985 | 0.2174 | 0.1481 | 0.5471 | 0.0171 | 0.0225 | 0.0204 | 0.4781 | 0.6715 | 0.3143 | 0.3143 | 0.3143 |
| FSFM | 0.1236 | 0.4494 | 0.3251 | 0.3434 | 0.3348 | 0.2565 | 0.7047 | 0.1570 | 0.0449 | 0.0204 | 0.4845 | 0.6575 | 0.3252 | 0.3252 | 0.3252 |
| FSFM-Adapter | 0.3427 | 0.4831 | 0.6556 | 0.5240 | 0.4391 | 0.4259 | 0.7978 | 0.3190 | 0.4326 | 0.0204 | 0.6250 | 0.7719 | 0.4864 | 0.4864 | 0.4864 |
| Effort | 0.2640 | 0.6292 | 0.2507 | 0.5132 | 0.4348 | 0.2582 | 0.6675 | 0.1336 | 0.0562 | 0.0000 | 0.5395 | 0.7950 | 0.3785 | 0.3785 | 0.3785 |
| ForAda | 0.1236 | 0.6854 | 0.2259 | 0.4095 | 0.2087 | 0.2859 | 0.6216 | 0.2929 | 0.0562 | 0.0000 | 0.7274 | 0.8159 | 0.3711 | 0.3711 | 0.3711 |

## FPR at TPR = 95% (lower is better)

`mean (all)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.1067 | 0.2303 | 0.2397 | 0.5892 | 0.3609 | 0.1487 | 0.6576 | 0.0144 | 0.0056 | 0.0204 | 0.2712 | 0.3548 | 0.2500 | 0.2500 | 0.2500 |
| DiCoME (trained, their repo) | 0.1629 | 0.8202 | 0.4353 | 0.5944 | 0.7130 | 0.2761 | 0.8697 | 0.1758 | 0.0225 | 0.0000 | 0.6970 | 0.8631 | 0.4692 | 0.4692 | 0.4692 |
| GenD-CLIP | 0.2809 | 0.6798 | 0.3747 | 0.5646 | 0.3478 | 0.3746 | 0.8486 | 0.0870 | 0.0112 | 0.0000 | 0.6893 | 0.8900 | 0.4290 | 0.4290 | 0.4290 |
| GenD-PE | 0.1910 | 0.5955 | 0.5289 | 0.7158 | 0.4522 | 0.3078 | 0.8300 | 0.1058 | 0.1011 | 0.1633 | 0.5360 | 0.8019 | 0.4441 | 0.4441 | 0.4441 |
| GenD-DINO | 0.3483 | 0.7865 | 0.5234 | 0.6873 | 0.3522 | 0.2565 | 0.7246 | 0.0807 | 0.0225 | 0.0408 | 0.6730 | 0.7837 | 0.4400 | 0.4400 | 0.4400 |
| FSFM | 0.1629 | 0.6067 | 0.4683 | 0.5123 | 0.4348 | 0.4081 | 0.8350 | 0.2266 | 0.1067 | 0.0612 | 0.6179 | 0.8164 | 0.4381 | 0.4381 | 0.4381 |
| FSFM-Adapter | 0.4775 | 0.6517 | 0.7741 | 0.6540 | 0.5435 | 0.5458 | 0.8908 | 0.4372 | 0.5281 | 0.0612 | 0.7938 | 0.8583 | 0.6013 | 0.6013 | 0.6013 |
| Effort | 0.3596 | 0.7978 | 0.5041 | 0.7201 | 0.5217 | 0.4063 | 0.8164 | 0.2417 | 0.1461 | 0.0816 | 0.7415 | 0.8959 | 0.5194 | 0.5194 | 0.5194 |
| ForAda | 0.2360 | 0.7921 | 0.3857 | 0.5862 | 0.3217 | 0.4323 | 0.7928 | 0.4054 | 0.1517 | 0.0408 | 0.8651 | 0.9034 | 0.4928 | 0.4928 | 0.4928 |


---

## Table 2 — probability quality and uncertainty

Per dataset. **All three lower is better.** ECE and Brier ask whether a reported
probability means what it says; E-AURC asks whether the model knows which of its own
answers to distrust, which the other two cannot see.

E-AURC subtracts the AURC an oracle ranking of the same predictions would reach, so the
base error rate drops out and what remains is the quality of the uncertainty ranking
alone — the only way to compare it across systems of different accuracy.

Ours ranks by the fused **vacuity**; the baselines rank by **1 − max p**, which is the
only uncertainty a cross-entropy detector emits. That asymmetry is the point of the
column, not a handicap we imposed.

`*in progress*`: not computable yet, and not zero or not applicable. Our row covers every
set. The baselines were never run on the three corpora packed on 2026-10-13 (UADFV,
DeepSpeak-1.1, DeepSpeak-2.0), and DiCoME's public checkpoints cover only the five
sets its paper reports; closing those cells needs a scoring pass, not a re-run.

A ROW'S `mean` IS OVER ITS OWN COVERAGE, which is not a comparison. Our row spans twelve
sets, most baselines nine, DiCoME five -- and the three corpora only we cover are not the
easy ones, so our own mean is read off a harder average than anyone else's. The two
matched columns are the comparable figures:

- `mean (all)` -- every set every row in the table resolves. DiCoME's five-set coverage
  sets that width, so it is honest for all nine methods but narrow.
- `mean (wide)` -- every set that every method covering nine or more sets resolves. The
  narrow rows say in progress there rather than shrinking the column for everyone.

Rank methods on a matched column and say which one. The per-row mean belongs in the
per-dataset discussion, not in a ranking.

## ECE-15 (lower is better)

`mean (all)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.1194 | 0.1732 | 0.1940 | 0.1296 | 0.1796 | 0.0623 | 0.1825 | 0.0505 | 0.0281 | 0.0988 | 0.0990 | 0.0933 | 0.1175 | 0.1175 | 0.1175 |
| DiCoME (trained, their repo) | 0.1119 | 0.3197 | 0.1191 | 0.0916 | 0.0480 | 0.0748 | 0.1163 | 0.0944 | 0.0450 | 0.1235 | 0.2111 | 0.2634 | 0.1349 | 0.1349 | 0.1349 |
| GenD-CLIP | 0.1180 | 0.1943 | 0.1162 | 0.0588 | 0.0641 | 0.0759 | 0.0840 | 0.1251 | 0.0597 | 0.1867 | 0.2112 | 0.2316 | 0.1271 | 0.1271 | 0.1271 |
| GenD-PE | 0.0654 | 0.2008 | 0.1929 | 0.1095 | 0.0979 | 0.0732 | 0.2525 | 0.0681 | 0.0385 | 0.1893 | 0.1399 | 0.1095 | 0.1281 | 0.1281 | 0.1281 |
| GenD-DINO | 0.0849 | 0.2662 | 0.2266 | 0.0681 | 0.0846 | 0.1054 | 0.2117 | 0.1150 | 0.0609 | 0.1407 | 0.1344 | 0.1387 | 0.1364 | 0.1364 | 0.1364 |
| FSFM | 0.1373 | 0.2912 | 0.2697 | 0.0850 | 0.1334 | 0.1498 | 0.2741 | 0.1159 | 0.0984 | 0.1627 | 0.0700 | 0.0585 | 0.1538 | 0.1538 | 0.1538 |
| FSFM-Adapter | 0.2054 | 0.3797 | 0.3975 | 0.1142 | 0.1470 | 0.1433 | 0.2682 | 0.1585 | 0.3383 | 0.2664 | 0.1589 | 0.1316 | 0.2257 | 0.2257 | 0.2257 |
| Effort | 0.1242 | 0.2506 | 0.0939 | 0.0870 | 0.1168 | 0.1400 | 0.0629 | 0.1826 | 0.0677 | 0.2045 | 0.2760 | 0.3393 | 0.1621 | 0.1621 | 0.1621 |
| ForAda | 0.0969 | 0.3657 | 0.1586 | 0.0944 | 0.1883 | 0.0848 | 0.2258 | 0.0658 | 0.0607 | 0.1403 | 0.0505 | 0.1013 | 0.1361 | 0.1361 | 0.1361 |

## Brier (lower is better)

`mean (all)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0730 | 0.0889 | 0.1406 | 0.1688 | 0.1626 | 0.0711 | 0.1584 | 0.0257 | 0.0150 | 0.0562 | 0.1162 | 0.1166 | 0.0994 | 0.0994 | 0.0994 |
| DiCoME (trained, their repo) | 0.0829 | 0.1971 | 0.0828 | 0.1498 | 0.1477 | 0.0939 | 0.1939 | 0.0830 | 0.0224 | 0.0634 | 0.1945 | 0.2498 | 0.1301 | 0.1301 | 0.1301 |
| GenD-CLIP | 0.1070 | 0.0980 | 0.0815 | 0.1503 | 0.1140 | 0.1151 | 0.1749 | 0.0722 | 0.0258 | 0.0817 | 0.1884 | 0.2294 | 0.1199 | 0.1199 | 0.1199 |
| GenD-PE | 0.0795 | 0.1182 | 0.1377 | 0.1878 | 0.1421 | 0.1115 | 0.2591 | 0.0637 | 0.0319 | 0.1366 | 0.1715 | 0.1580 | 0.1331 | 0.1331 | 0.1331 |
| GenD-DINO | 0.1101 | 0.1526 | 0.1546 | 0.1636 | 0.1155 | 0.1117 | 0.1989 | 0.0627 | 0.0223 | 0.0754 | 0.1700 | 0.1829 | 0.1267 | 0.1267 | 0.1267 |
| FSFM | 0.0914 | 0.1478 | 0.1539 | 0.1476 | 0.1394 | 0.1490 | 0.2609 | 0.0995 | 0.0399 | 0.0702 | 0.1360 | 0.1692 | 0.1337 | 0.1337 | 0.1337 |
| FSFM-Adapter | 0.1817 | 0.1844 | 0.2551 | 0.1830 | 0.1721 | 0.1813 | 0.2702 | 0.1548 | 0.1801 | 0.1147 | 0.1919 | 0.2073 | 0.1897 | 0.1897 | 0.1897 |
| Effort | 0.1234 | 0.1242 | 0.0763 | 0.1615 | 0.1567 | 0.1329 | 0.1664 | 0.1298 | 0.0421 | 0.1167 | 0.2269 | 0.3130 | 0.1475 | 0.1475 | 0.1475 |
| ForAda | 0.0915 | 0.2267 | 0.1060 | 0.1567 | 0.1612 | 0.1355 | 0.2312 | 0.1276 | 0.0440 | 0.0695 | 0.1707 | 0.2026 | 0.1436 | 0.1436 | 0.1436 |

## E-AURC (lower is better)

`mean (all)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).  
`mean (wide)`: 12 sets (CDFv2, CDFv3, DFD, DFDC, DFDCP, FFIW, KoDF, IDForge, DFDM, UADFV, DeepSpeak-1.1, DeepSpeak-2.0).

| method | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | FFIW | KoDF | IDForge | DFDM | UADFV | DeepSpeak-1.1 | DeepSpeak-2.0 | mean | mean (all) | mean (wide) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (App+TD+LFS+STT, PCR6+-R)** | 0.0167 | 0.0109 | 0.0278 | 0.0886 | 0.0571 | 0.0197 | 0.0362 | 0.0047 | 0.0016 | 0.0448 | 0.1297 | 0.1005 | 0.0449 | 0.0449 | 0.0449 |
| DiCoME (trained, their repo) | 0.0164 | 0.1091 | 0.0141 | 0.0670 | 0.0634 | 0.0262 | 0.0821 | 0.0154 | 0.0010 | 0.0123 | 0.1529 | 0.2099 | 0.0642 | 0.0642 | 0.0642 |
| GenD-CLIP | 0.0342 | 0.0266 | 0.0134 | 0.0720 | 0.0439 | 0.0353 | 0.0765 | 0.0124 | 0.0015 | 0.0131 | 0.1208 | 0.1970 | 0.0539 | 0.0539 | 0.0539 |
| GenD-PE | 0.0217 | 0.0466 | 0.0484 | 0.1177 | 0.0599 | 0.0417 | 0.0888 | 0.0107 | 0.0024 | 0.0452 | 0.0981 | 0.0899 | 0.0559 | 0.0559 | 0.0559 |
| GenD-DINO | 0.0427 | 0.0804 | 0.0773 | 0.0913 | 0.0420 | 0.0402 | 0.1363 | 0.0076 | 0.0020 | 0.0131 | 0.1110 | 0.1175 | 0.0634 | 0.0634 | 0.0634 |
| FSFM | 0.0227 | 0.0680 | 0.0630 | 0.0637 | 0.0507 | 0.0638 | 0.1354 | 0.0176 | 0.0026 | 0.0140 | 0.0657 | 0.1106 | 0.0565 | 0.0565 | 0.0565 |
| FSFM-Adapter | 0.0758 | 0.0899 | 0.2099 | 0.0968 | 0.0678 | 0.0879 | 0.2465 | 0.0423 | 0.0671 | 0.0046 | 0.1096 | 0.2008 | 0.1082 | 0.1082 | 0.1082 |
| Effort | 0.0416 | 0.0368 | 0.0120 | 0.0778 | 0.0627 | 0.0394 | 0.0706 | 0.0438 | 0.0037 | 0.0329 | 0.1748 | 0.3342 | 0.0775 | 0.0775 | 0.0775 |
| ForAda | 0.0203 | 0.1562 | 0.0262 | 0.0772 | 0.0634 | 0.0593 | 0.1103 | 0.0436 | 0.0038 | 0.0123 | 0.1232 | 0.2051 | 0.0751 | 0.0751 | 0.0751 |


---

# Part IV — DF40, every method on every generator

41 generator sets, 12 methods. Each generator is scored against the
real pool DF40 ships for ITS source, aggregated to video level by the project's rule,
so every row is on the same videos. AUROC.

This is the only DF40 table. It replaces `DF40_CROSSMANIP`, `DF40_CROSSMANIP_ALL` and
`DF40_ALL_METHODS`, which covered the same corpus with different rows and columns.

## Summary — family means

| method | face swap | reenactment | all sets | covered |
|---|---:|---:|---:|---:|
| **Ours (TD+LFS+STT-order)** | 0.9889 | 0.9548 | **0.9706** | 41 |
| **Ours (App+TD+LFS+STT)** | 0.9936 | 0.9342 | **0.9617** | 41 |
| **Ours (TD+LFS+STT)** | 0.9894 | 0.9335 | **0.9594** | 41 |
| GenD-PE | 0.9727 | 0.8825 | **0.9243** | 41 |
| GenD-CLIP | 0.9588 | 0.8836 | **0.9185** | 41 |
| GenD-DINO | 0.9580 | 0.8693 | **0.9104** | 41 |
| DiCoME (their repo) | 0.9793 | 0.7232 | **0.9005** | 13 |
| FSFM | 0.9623 | 0.8266 | **0.8895** | 41 |
| DiCoME (our training) | 0.9682 | 0.7792 | **0.8668** | 41 |
| FSFM-Adapter | 0.9236 | 0.8075 | **0.8613** | 41 |
| Effort | 0.9395 | 0.7789 | **0.8533** | 41 |
| ForAda | 0.9209 | 0.7447 | **0.8264** | 41 |

Reenactment is half of DF40 and the half DiCoME's published cross-manipulation table
does not report.

## Face Swap — 19 generator sets

| method | blendface_cdf | blendface_ff | deepfacelab | e4s_cdf | e4s_ff | facedancer_cdf | facedancer_ff | faceswap_cdf | faceswap_ff | fsgan_cdf | fsgan_ff | inswap_cdf | inswap_ff | mobileswap_cdf | mobileswap_ff | simswap_cdf | simswap_ff | uniface_cdf | uniface_ff | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *videos (real / fake)* | 178/640 | 140/140 | 49/100 | 178/292 | 140/115 | 178/648 | 140/140 | 178/637 | 140/140 | 178/616 | 140/140 | 178/424 | 140/118 | 178/637 | 140/858 | 178/641 | 140/140 | 178/626 | 140/140 | |
| **Ours (TD+LFS+STT-order)** | 0.9882 | 0.9901 | 0.9745 | 0.9969 | 0.9966 | 0.9832 | 0.9771 | 0.9989 | 0.9954 | 0.9935 | 0.9938 | 0.9762 | 0.9880 | 0.9679 | 0.9893 | 0.9898 | 0.9933 | 0.9979 | 0.9977 | **0.9889** |
| **Ours (App+TD+LFS+STT)** | 0.9930 | 0.9974 | 0.9798 | 0.9964 | 0.9993 | 0.9840 | 0.9912 | 0.9988 | 0.9988 | 0.9931 | 0.9973 | 0.9867 | 0.9978 | 0.9777 | 0.9980 | 0.9942 | 0.9990 | 0.9974 | 0.9987 | **0.9936** |
| **Ours (TD+LFS+STT)** | 0.9894 | 0.9922 | 0.9669 | 0.9933 | 0.9983 | 0.9752 | 0.9812 | 0.9988 | 0.9963 | 0.9930 | 0.9962 | 0.9792 | 0.9933 | 0.9722 | 0.9942 | 0.9878 | 0.9967 | 0.9954 | 0.9985 | **0.9894** |
| GenD-PE | 0.9456 | 0.9587 | 0.9447 | 0.9962 | 0.9899 | 0.9804 | 0.9621 | 0.9994 | 0.9832 | 0.9888 | 0.9862 | 0.9623 | 0.9804 | 0.9239 | 0.9778 | 0.9558 | 0.9665 | 0.9897 | 0.9904 | **0.9727** |
| GenD-CLIP | 0.9260 | 0.9281 | 0.9812 | 0.9937 | 0.9883 | 0.9740 | 0.9518 | 0.9986 | 0.9761 | 0.9804 | 0.9814 | 0.8791 | 0.9625 | 0.9322 | 0.9792 | 0.8849 | 0.9522 | 0.9655 | 0.9817 | **0.9588** |
| GenD-DINO | 0.9338 | 0.9451 | 0.9712 | 0.9783 | 0.9832 | 0.9552 | 0.9003 | 0.9923 | 0.9851 | 0.9791 | 0.9712 | 0.8973 | 0.9516 | 0.9174 | 0.9846 | 0.9245 | 0.9679 | 0.9767 | 0.9880 | **0.9580** |
| DiCoME (their repo) | 0.9505 | 0.9633 | 0.9627 | 0.9977 | 0.9952 | 0.9710 | 0.9860 | 0.9993 | 0.9885 | — | — | — | — | — | — | — | — | — | — | **0.9793** |
| FSFM | 0.9676 | 0.9652 | 0.9735 | 0.9350 | 0.9822 | 0.9126 | 0.9726 | 0.9855 | 0.9782 | 0.9857 | 0.9790 | 0.9530 | 0.9648 | 0.9192 | 0.9682 | 0.9222 | 0.9504 | 0.9872 | 0.9819 | **0.9623** |
| DiCoME (our training) | 0.9502 | 0.9505 | 0.9390 | 0.9949 | 0.9891 | 0.9657 | 0.9632 | 0.9991 | 0.9795 | 0.9892 | 0.9879 | 0.9433 | 0.9696 | 0.9571 | 0.9745 | 0.9339 | 0.9539 | 0.9758 | 0.9803 | **0.9682** |
| FSFM-Adapter | 0.9531 | 0.8834 | 0.9710 | 0.9679 | 0.9559 | 0.8874 | 0.8645 | 0.9903 | 0.9510 | 0.9881 | 0.9556 | 0.8853 | 0.8459 | 0.9272 | 0.8800 | 0.8823 | 0.8579 | 0.9683 | 0.9332 | **0.9236** |
| Effort | 0.8925 | 0.8777 | 0.9527 | 0.9952 | 0.9837 | 0.9803 | 0.9287 | 0.9962 | 0.9772 | 0.9786 | 0.9642 | 0.8618 | 0.9381 | 0.9114 | 0.9550 | 0.8486 | 0.9210 | 0.9229 | 0.9653 | **0.9395** |
| ForAda | 0.8278 | 0.8809 | 0.9614 | 0.8864 | 0.9687 | 0.8957 | 0.9462 | 0.9962 | 0.9737 | 0.9781 | 0.9748 | 0.8927 | 0.9492 | 0.9027 | 0.9791 | 0.7588 | 0.9131 | 0.8604 | 0.9515 | **0.9209** |

## Reenactment — 22 generator sets

| method | MRAA_cdf | MRAA_ff | danet_cdf | danet_ff | facevid2vid_cdf | facevid2vid_ff | fomm_cdf | fomm_ff | hyperreenact_cdf | hyperreenact_ff | lia_cdf | lia_ff | mcnet_cdf | mcnet_ff | one_shot_free_cdf | one_shot_free_ff | pirender_cdf | pirender_ff | sadtalker_cdf | sadtalker_ff | tpsm_cdf | tpsm_ff | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *videos (real / fake)* | 178/649 | 140/140 | 178/647 | 140/140 | 178/637 | 140/140 | 178/647 | 140/140 | 178/641 | 140/140 | 178/646 | 140/140 | 178/647 | 140/140 | 178/647 | 140/140 | 178/636 | 140/140 | 178/660 | 140/138 | 178/647 | 140/140 | |
| **Ours (TD+LFS+STT-order)** | 0.9929 | 0.9993 | 0.9030 | 0.9711 | 0.8766 | 0.9312 | 0.9484 | 0.9840 | 0.9845 | 0.9917 | 0.9967 | 1.0000 | 0.9055 | 0.9725 | 0.9266 | 0.9608 | 0.8903 | 0.9452 | 0.9063 | 0.9654 | 0.9747 | 0.9795 | **0.9548** |
| **Ours (App+TD+LFS+STT)** | 0.9936 | 1.0000 | 0.7750 | 0.9824 | 0.8585 | 0.9634 | 0.8997 | 0.9947 | 0.9809 | 0.9988 | 0.9977 | 1.0000 | 0.8183 | 0.9867 | 0.8672 | 0.9748 | 0.7949 | 0.9718 | 0.7976 | 0.9669 | 0.9391 | 0.9901 | **0.9342** |
| **Ours (TD+LFS+STT)** | 0.9924 | 0.9998 | 0.8070 | 0.9579 | 0.8473 | 0.9240 | 0.9026 | 0.9821 | 0.9850 | 0.9951 | 0.9966 | 1.0000 | 0.8147 | 0.9601 | 0.8721 | 0.9534 | 0.8195 | 0.9374 | 0.8909 | 0.9748 | 0.9572 | 0.9677 | **0.9335** |
| GenD-PE | 0.9823 | 0.9959 | 0.7523 | 0.8688 | 0.8628 | 0.8936 | 0.9537 | 0.9763 | 0.9108 | 0.9363 | 0.9635 | 0.9886 | 0.8383 | 0.9157 | 0.9416 | 0.9436 | 0.9151 | 0.9358 | 0.5682 | 0.6832 | 0.6883 | 0.9014 | **0.8825** |
| GenD-CLIP | 0.9563 | 0.9936 | 0.7352 | 0.9096 | 0.8482 | 0.9259 | 0.9352 | 0.9838 | 0.9141 | 0.9765 | 0.8970 | 0.9898 | 0.8042 | 0.9430 | 0.9029 | 0.9448 | 0.8672 | 0.9436 | 0.5788 | 0.7584 | 0.6980 | 0.9333 | **0.8836** |
| GenD-DINO | 0.8941 | 0.9857 | 0.7319 | 0.8911 | 0.8499 | 0.9316 | 0.9356 | 0.9870 | 0.8947 | 0.9682 | 0.8210 | 0.9394 | 0.7996 | 0.9270 | 0.9143 | 0.9593 | 0.8656 | 0.9490 | 0.5643 | 0.7027 | 0.6956 | 0.9171 | **0.8693** |
| DiCoME (their repo) | 0.9716 | 0.9983 | 0.3083 | 0.6147 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | **0.7232** |
| FSFM | 0.9903 | 0.9983 | 0.6032 | 0.7460 | 0.7771 | 0.8402 | 0.9214 | 0.9613 | 0.9671 | 0.9875 | 0.9343 | 0.9666 | 0.6529 | 0.8077 | 0.8822 | 0.9069 | 0.9070 | 0.9286 | 0.4851 | 0.6653 | 0.4812 | 0.7757 | **0.8266** |
| DiCoME (our training) | 0.9770 | 0.9937 | 0.4821 | 0.6764 | 0.6748 | 0.7776 | 0.8793 | 0.9609 | 0.7421 | 0.8483 | 0.9854 | 0.9921 | 0.5594 | 0.7680 | 0.8506 | 0.8815 | 0.8066 | 0.8833 | 0.5402 | 0.6071 | 0.5097 | 0.7458 | **0.7792** |
| FSFM-Adapter | 0.9809 | 0.9701 | 0.6928 | 0.6657 | 0.8277 | 0.7655 | 0.9474 | 0.9266 | 0.9614 | 0.9435 | 0.8597 | 0.8053 | 0.7083 | 0.6932 | 0.9316 | 0.8646 | 0.9292 | 0.8722 | 0.5961 | 0.5941 | 0.5660 | 0.6625 | **0.8075** |
| Effort | 0.8530 | 0.9483 | 0.6344 | 0.7490 | 0.7368 | 0.8092 | 0.8211 | 0.9124 | 0.8665 | 0.9452 | 0.7910 | 0.9146 | 0.6649 | 0.7752 | 0.7677 | 0.8222 | 0.7412 | 0.8172 | 0.5437 | 0.6064 | 0.6508 | 0.7652 | **0.7789** |
| ForAda | 0.8945 | 0.9654 | 0.4993 | 0.7453 | 0.6272 | 0.7784 | 0.7342 | 0.8844 | 0.7456 | 0.8955 | 0.7966 | 0.9272 | 0.5405 | 0.7785 | 0.6739 | 0.8143 | 0.6198 | 0.8274 | 0.5035 | 0.7304 | 0.6267 | 0.7748 | **0.7447** |

## Against DiCoME's published cross-manipulation table

Column order is DiCoME's. Our rows pool the two source halves, since the paper
reports one number per generator without saying which half it used. The published
rows are transcribed from the paper and were not re-measured.

| method | UniFace | BleFace | MobSwap | e4s | FaceDan | FSGAN | InSwap | SimSwap | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours (TD+LFS+STT-order)** | 0.9978 | 0.9891 | 0.9786 | 0.9968 | 0.9801 | 0.9937 | 0.9821 | 0.9916 | **0.9887** |
| **Ours (App+TD+LFS+STT)** | 0.9981 | 0.9952 | 0.9878 | 0.9978 | 0.9876 | 0.9952 | 0.9923 | 0.9966 | **0.9938** |
| **Ours (TD+LFS+STT)** | 0.9970 | 0.9908 | 0.9832 | 0.9958 | 0.9782 | 0.9946 | 0.9863 | 0.9923 | **0.9898** |
| GenD-PE | 0.9900 | 0.9522 | 0.9508 | 0.9930 | 0.9712 | 0.9875 | 0.9714 | 0.9612 | **0.9722** |
| GenD-CLIP | 0.9736 | 0.9270 | 0.9557 | 0.9910 | 0.9629 | 0.9809 | 0.9208 | 0.9186 | **0.9538** |
| GenD-DINO | 0.9823 | 0.9394 | 0.9510 | 0.9808 | 0.9277 | 0.9752 | 0.9245 | 0.9462 | **0.9534** |
| DiCoME (their repo) | — | 0.9569 | — | 0.9964 | 0.9785 | — | — | — | **0.9773** |
| FSFM | 0.9846 | 0.9664 | 0.9437 | 0.9586 | 0.9426 | 0.9824 | 0.9589 | 0.9363 | **0.9592** |
| DiCoME (our training) | 0.9780 | 0.9503 | 0.9658 | 0.9920 | 0.9644 | 0.9885 | 0.9565 | 0.9439 | **0.9674** |
| FSFM-Adapter | 0.9507 | 0.9182 | 0.9036 | 0.9619 | 0.8759 | 0.9719 | 0.8656 | 0.8701 | **0.9147** |
| Effort | 0.9441 | 0.8851 | 0.9332 | 0.9894 | 0.9545 | 0.9714 | 0.9000 | 0.8848 | **0.9328** |
| ForAda | 0.9059 | 0.8543 | 0.9409 | 0.9276 | 0.9210 | 0.9765 | 0.9210 | 0.8359 | **0.9104** |
| F3Net (ECCV'20) *(published)* | 0.809 | 0.808 | 0.867 | 0.494 | 0.717 | 0.845 | 0.757 | 0.674 | 0.746 |
| SPSL (CVPR'20) *(published)* | 0.747 | 0.748 | 0.885 | 0.514 | 0.666 | 0.812 | 0.643 | 0.665 | 0.710 |
| SRM (CVPR'21) *(published)* | 0.749 | 0.704 | 0.779 | 0.704 | 0.659 | 0.772 | 0.793 | 0.694 | 0.732 |
| CORE (CVPR'21) *(published)* | 0.871 | 0.843 | 0.959 | 0.679 | 0.774 | 0.958 | 0.855 | 0.724 | 0.833 |
| RECCE (CVPR'22) *(published)* | 0.898 | 0.832 | 0.925 | 0.683 | 0.848 | 0.949 | 0.848 | 0.768 | 0.844 |
| SBI (CVPR'22) *(published)* | 0.724 | 0.891 | 0.952 | 0.750 | 0.594 | 0.803 | 0.712 | 0.701 | 0.766 |
| UCF (ICCV'23) *(published)* | 0.831 | 0.827 | 0.950 | 0.731 | 0.862 | 0.937 | 0.809 | 0.647 | 0.824 |
| IID (CVPR'23) *(published)* | 0.839 | 0.789 | 0.888 | 0.766 | 0.844 | 0.927 | 0.789 | 0.644 | 0.811 |
| LSDA (CVPR'24) *(published)* | 0.872 | 0.875 | 0.930 | 0.694 | 0.721 | 0.939 | 0.855 | 0.793 | 0.835 |
| ProDet (NIPS'24) *(published)* | 0.908 | 0.929 | 0.975 | 0.771 | 0.747 | 0.928 | 0.837 | 0.844 | 0.867 |
| Effort (ICML'25) *(published)* | 0.962 | 0.873 | 0.953 | 0.983 | 0.926 | 0.957 | 0.936 | 0.926 | 0.940 |
| GenD (WACV'26) *(published)* | 0.974 | 0.935 | 0.976 | 0.983 | 0.942 | 0.974 | 0.943 | 0.938 | 0.958 |
| DiCoME *(published)* | 0.982 | 0.963 | 0.978 | 0.991 | 0.972 | 0.989 | 0.970 | 0.965 | 0.976 |

