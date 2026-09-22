# Grouped fusion on the exposed suites (ten test sets and DF40)

The recipe is frozen by `tbiom/configs/grouped_fusion_preregister.yaml` before these are scored. Both DF40 means the
brief asks for are reported -- family-balanced (the mean of the four family x source cell means) and generator-macro
(the plain mean over all 41 sets) -- with their weights stated, and never one family alone.

One check here is decisive for the novelty question and is reported before any headline: whether grouped-R, under the
discount its own fit chooses, is numerically identical to the two-view T+OM fusion. It is, whenever the fit sets the
appearance weight to zero.

Wall clock 0.4 minutes, CPU. Detector seed 42.

## 0. The identity check, before anything else

| split | fitted discount (T, APP, OM) | grouped-R identical to T+OM on all 41 DF40 sets |
|---|---|---|
| split0 | (0.25, 0, 1) | **yes** |
| split1 | (0.25, 0.25, 1) | no |
| split2 | (0.25, 0, 1) | **yes** |
| split3 | (0.25, 0, 1) | **yes** |
| split4 | (0.25, 0, 1) | **yes** |

**In 4 of 5 folds the fitted discount sets the appearance weight to zero, and grouped-R is then bit-for-bit the two-view Dempster fusion of the temporal and off-manifold views.** This is not a numerical coincidence: an evidence-scaled opinion with r = 0 carries no evidence, its canonical weights are both 1, and the minimum of 1 and w_OM is w_OM, so the cautious inner step returns the off-manifold opinion unchanged and the outer Dempster step is the only thing left. Under its own fitted discount the operator is therefore, in most folds, not a new combination rule at all — it is the decision to drop the appearance view. Any claim built on the fitted-discount numbers has to answer that first, which is why T+OM is carried as an arm in every table below.

## 1. Ten test sets, video AUROC

| arm | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **grouped-R** | 0.9536 | 0.9674 | 0.9267 | 0.8487 | 0.8584 | 0.7218 | 0.9139 | 0.9625 | 0.9611 | 0.9956 | 0.9109 |
| grouped-TA-R | 0.9610 | 0.9624 | 0.9349 | 0.8643 | 0.8684 | 0.7244 | 0.9300 | 0.9593 | 0.9675 | 0.9961 | 0.9168 |
| grouped-TO-R | 0.9532 | 0.9605 | 0.9330 | 0.8551 | 0.8612 | 0.7248 | 0.9220 | 0.9583 | 0.9627 | 0.9954 | 0.9126 |
| Dempster-R *(baseline)* | 0.9598 | 0.9616 | 0.9338 | 0.8626 | 0.8656 | 0.7252 | 0.9282 | 0.9574 | 0.9656 | 0.9956 | 0.9155 |
| AVG-R *(baseline)* | 0.9542 | 0.9632 | 0.9330 | 0.8578 | 0.8642 | 0.7236 | 0.9241 | 0.9602 | 0.9650 | 0.9960 | 0.9141 |
| CAUT-R | 0.9481 | 0.9622 | 0.9317 | 0.8497 | 0.8574 | 0.7184 | 0.9187 | 0.9630 | 0.9627 | 0.9960 | 0.9108 |
| PCR6-R | 0.9607 | 0.9613 | 0.9334 | 0.8594 | 0.8633 | 0.7241 | 0.9272 | 0.9577 | 0.9648 | 0.9954 | 0.9147 |
| GCP-AO-R | 0.9569 | 0.9667 | 0.9290 | 0.8491 | 0.8577 | 0.7224 | 0.9171 | 0.9621 | 0.9619 | 0.9955 | 0.9118 |
| GCY-AO-R | 0.9597 | 0.9653 | 0.9303 | 0.8493 | 0.8572 | 0.7221 | 0.9188 | 0.9611 | 0.9620 | 0.9953 | 0.9121 |
| GAP-AO-R | 0.9454 | 0.9615 | 0.9205 | 0.8450 | 0.8544 | 0.7201 | 0.9153 | 0.9585 | 0.9583 | 0.9946 | 0.9074 |
| T only | 0.7384 | 0.9263 | 0.7967 | 0.7009 | 0.7409 | 0.6733 | 0.8011 | 0.9656 | 0.8899 | 0.9798 | 0.8213 |
| T+APP | 0.9338 | 0.9533 | 0.9127 | 0.8362 | 0.8453 | 0.7186 | 0.9119 | 0.9548 | 0.9520 | 0.9929 | 0.9012 |
| T+OM | 0.9543 | 0.9676 | 0.9271 | 0.8487 | 0.8585 | 0.7224 | 0.9142 | 0.9628 | 0.9613 | 0.9957 | 0.9113 |

## 2. DF40, both means the brief requires

Family-balanced = the mean of the four family x source cell means, each cell weighted 1/4. Generator-macro = the plain mean over all 41 generator sets, each set weighted 1/41, which weights the Celeb-DF-sourced cells more because they contain more generators.

| arm | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced | generator-macro |
|---|---:|---:|---:|---:|---:|---:|
| **grouped-R** | 0.9706 | 0.9812 | 0.9366 | 0.9644 | 0.9632 | 0.9621 |
| grouped-TA-R | 0.9745 | 0.9856 | 0.9213 | 0.9603 | 0.9604 | 0.9588 |
| grouped-TO-R | 0.9708 | 0.9849 | 0.9185 | 0.9612 | 0.9588 | 0.9573 |
| Dempster-R *(baseline)* | 0.9731 | 0.9854 | 0.9174 | 0.9607 | 0.9591 | 0.9575 |
| AVG-R *(baseline)* | 0.9729 | 0.9846 | 0.9267 | 0.9628 | 0.9618 | 0.9604 |
| CAUT-R | 0.9707 | 0.9812 | 0.9296 | 0.9615 | 0.9608 | 0.9595 |
| PCR6-R | 0.9725 | 0.9851 | 0.9166 | 0.9602 | 0.9586 | 0.9569 |
| GCP-AO-R | 0.9706 | 0.9821 | 0.9333 | 0.9634 | 0.9624 | 0.9612 |
| GCY-AO-R | 0.9707 | 0.9828 | 0.9268 | 0.9604 | 0.9602 | 0.9588 |
| GAP-AO-R | 0.9673 | 0.9836 | 0.9238 | 0.9636 | 0.9596 | 0.9582 |
| T only | 0.8857 | 0.9019 | 0.9355 | 0.9550 | 0.9195 | 0.9212 |
| T+APP | 0.9612 | 0.9843 | 0.9089 | 0.9613 | 0.9539 | 0.9522 |
| T+OM | 0.9707 | 0.9810 | 0.9368 | 0.9643 | 0.9632 | 0.9622 |

## 3. grouped-R against the arms that matter, per DF40 family

| contrast | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced | generator-macro |
|---|---:|---:|---:|---:|---:|---:|
| grouped-R − Dempster-R | -0.0025 | -0.0043 | +0.0192 | +0.0038 | +0.0040 | +0.0046 |
| grouped-R − AVG-R | -0.0023 | -0.0035 | +0.0099 | +0.0017 | +0.0014 | +0.0018 |
| grouped-R − T+OM | -0.0001 | +0.0001 | -0.0003 | +0.0002 | -0.0000 | -0.0000 |
| grouped-R − CAUT-R | -0.0001 | -0.0000 | +0.0070 | +0.0030 | +0.0024 | +0.0026 |

The row against T+OM is the one to read first: where it is zero, the grouped operator is doing nothing that dropping the appearance view does not already do.

## 4. Fitted discounts

| arm | split 0 | split 1 | split 2 | split 3 | split 4 |
|---|---|---|---|---|---|
| grouped-R | (0.25, 0, 1) | (0.25, 0.25, 1) | (0.25, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) |
| grouped-TA-R | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) |
| grouped-TO-R | (0.5, 0.5, 1) | (0.5, 0.5, 1) | (0.5, 0.25, 1) | (0.5, 0.25, 1) | (0.5, 0.25, 1) |
| Dempster-R | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) |
| AVG-R | (0.25, 0.5, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) |
| CAUT-R | (0.25, 0, 1) | (0.25, 0.5, 1) | (0.25, 0.5, 1) | (0.25, 0, 1) | (0.25, 0, 1) |
| PCR6-R | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0.25, 1) | (0.25, 0, 1) |
| GCP-AO-R | (0.25, 0, 1) | (0.25, 0.25, 1) | (0.25, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) |
| GCY-AO-R | (0.25, 0, 1) | (0.25, 0.25, 1) | (0.25, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) |
| GAP-AO-R | (0.25, 1, 1) | (0.25, 1, 1) | (0.25, 1, 1) | (0.25, 0.75, 1) | (0.25, 0.75, 1) |
| T only | (0.25, 0, 1) | (0.25, 0, 1) | (0.75, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) |
| T+APP | (0.25, 1, 0) | (0.25, 1, 0) | (0.25, 0.75, 1) | (0.25, 1, 0) | (0.25, 1, 0) |
| T+OM | (0.25, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) | (0.25, 0, 1) |

Both suites are exposed: CDFv2 and DeepfakeEval2024 informed development, and DF40 informed the choice of this operator. Nothing here is a confirmation set (`GROUPED_REFERENCE_MANIFEST.md` §5).

