# Cross-manipulation results — DF40 face-swap generators

OMP_NUM_THREADS=8 python analysis/tbiom/df40_analysis.py

Each generator is scored against the genuine pool DF40 ships for its source. Our system fuses the three reference
views with DS-R, the weights fitted per split on the development folds exactly as everywhere else, so the numbers are
comparable with every other table; the logistic comparator is reported beside it. DiCoME is our own trained
checkpoint (the same one in the baseline matrix), replayed through its own two-view DS fusion, so both sides are
scored on identical videos with identical aggregation.

Two generators DiCoME's published cross-manipulation table does not list — the classical graphics **FaceSwap** and
**DeepFaceLab** — are included and marked, together with the FF++-sourced version of every generator.

Every number below uses the reference manifest `tbiom/reports/REFERENCE_MANIFEST.md` (three resolved checkpoints, DS-R definition, fold splits). Script `analysis/tbiom/temporal/finish_line.py`.

## Face swap — Celeb-DF-sourced generators

| generator | videos (real / fake) | **ours, DS-R** | ours, logistic | DiCoME (our training) | Δ ours − DiCoME | temporal view | appearance | off-manifold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BlendFace | 178 / 640 | **0.9801** | 0.9757 | 0.9502 | +0.0299 | 0.9371 | 0.9444 | 0.9516 |
| DeepFaceLab ★ | 49 / 100 | **0.9610** | 0.9302 | 0.9390 | +0.0220 | 0.9161 | 0.9349 | 0.9378 |
| e4s | 178 / 292 | **0.9892** | 0.9920 | 0.9949 | -0.0057 | 0.8055 | 0.9931 | 0.9939 |
| FaceDancer | 178 / 648 | **0.9714** | 0.9809 | 0.9657 | +0.0058 | 0.8598 | 0.9553 | 0.9744 |
| FaceSwap ★ | 178 / 637 | **0.9995** | 0.9928 | 0.9991 | +0.0004 | 0.9981 | 0.9997 | 0.9974 |
| FSGAN | 178 / 616 | **0.9906** | 0.9817 | 0.9892 | +0.0014 | 0.9393 | 0.9782 | 0.9938 |
| InSwapper | 178 / 424 | **0.9551** | 0.9526 | 0.9433 | +0.0117 | 0.8872 | 0.9398 | 0.9429 |
| MobileFaceSwap | 178 / 637 | **0.9584** | 0.9570 | 0.9571 | +0.0013 | 0.7988 | 0.9454 | 0.9641 |
| SimSwap | 178 / 641 | **0.9416** | 0.9463 | 0.9339 | +0.0077 | 0.8006 | 0.9247 | 0.9380 |
| UniFace | 178 / 626 | **0.9837** | 0.9806 | 0.9758 | +0.0079 | 0.9137 | 0.9727 | 0.9747 |
| **mean over 10 generators** | | **0.9731** | 0.9690 | 0.9648 | +0.0082 | 0.8856 | 0.9588 | 0.9668 |

## Face swap — FaceForensics++-sourced generators

| generator | videos (real / fake) | **ours, DS-R** | ours, logistic | DiCoME (our training) | Δ ours − DiCoME | temporal view | appearance | off-manifold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BlendFace | 140 / 140 | **0.9858** | 0.9795 | 0.9505 | +0.0353 | 0.9490 | 0.9499 | 0.9470 |
| e4s | 140 / 115 | **0.9927** | 0.9902 | 0.9891 | +0.0037 | 0.8081 | 0.9892 | 0.9881 |
| FaceDancer | 140 / 140 | **0.9713** | 0.9673 | 0.9632 | +0.0081 | 0.8330 | 0.9629 | 0.9616 |
| FaceSwap ★ | 140 / 140 | **0.9948** | 0.9923 | 0.9795 | +0.0153 | 0.9856 | 0.9800 | 0.9782 |
| FSGAN | 140 / 140 | **0.9935** | 0.9915 | 0.9879 | +0.0056 | 0.9453 | 0.9868 | 0.9879 |
| InSwapper | 140 / 118 | **0.9783** | 0.9685 | 0.9696 | +0.0087 | 0.8993 | 0.9706 | 0.9654 |
| MobileFaceSwap | 140 / 858 | **0.9848** | 0.9790 | 0.9745 | +0.0103 | 0.9165 | 0.9734 | 0.9740 |
| SimSwap | 140 / 140 | **0.9749** | 0.9648 | 0.9539 | +0.0210 | 0.8577 | 0.9551 | 0.9500 |
| UniFace | 140 / 140 | **0.9928** | 0.9908 | 0.9803 | +0.0125 | 0.9228 | 0.9804 | 0.9783 |
| **mean over 9 generators** | | **0.9854** | 0.9804 | 0.9720 | +0.0134 | 0.9019 | 0.9720 | 0.9701 |

## Reenactment — Celeb-DF-sourced generators

| generator | videos (real / fake) | **ours, DS-R** | ours, logistic | DiCoME (our training) | Δ ours − DiCoME | temporal view | appearance | off-manifold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MRAA | 178 / 649 | **0.9934** | 0.9930 | 0.9770 | +0.0164 | 0.9724 | 0.9738 | 0.9780 |
| DaNet | 178 / 647 | **0.8645** | 0.9083 | 0.4821 | +0.3823 | 0.9522 | 0.4599 | 0.5289 |
| FaceVid2Vid | 178 / 637 | **0.8403** | 0.8966 | 0.6748 | +0.1655 | 0.8899 | 0.6505 | 0.7141 |
| FOMM | 178 / 647 | **0.9341** | 0.9260 | 0.8793 | +0.0547 | 0.9104 | 0.8673 | 0.8870 |
| HyperReenact | 178 / 641 | **0.9708** | 0.9786 | 0.7421 | +0.2287 | 0.9865 | 0.7253 | 0.7674 |
| LIA | 178 / 646 | **0.9977** | 0.9981 | 0.9854 | +0.0123 | 0.9818 | 0.9793 | 0.9873 |
| MCNet | 178 / 647 | **0.8662** | 0.9077 | 0.5594 | +0.3068 | 0.9411 | 0.5371 | 0.5992 |
| One-Shot Free-View | 178 / 647 | **0.9102** | 0.9031 | 0.8506 | +0.0596 | 0.8878 | 0.8410 | 0.8570 |
| PIRender | 178 / 636 | **0.8626** | 0.8766 | 0.8066 | +0.0559 | 0.8133 | 0.7855 | 0.8287 |
| SadTalker | 178 / 660 | **0.8857** | 0.8801 | 0.5402 | +0.3455 | 0.9629 | 0.5531 | 0.5159 |
| TPSM | 178 / 647 | **0.9661** | 0.9690 | 0.5097 | +0.4564 | 0.9915 | 0.5080 | 0.5191 |
| **mean over 11 generators** | | **0.9174** | 0.9306 | 0.7280 | +0.1895 | 0.9354 | 0.7164 | 0.7439 |

## Reenactment — FaceForensics++-sourced generators

| generator | videos (real / fake) | **ours, DS-R** | ours, logistic | DiCoME (our training) | Δ ours − DiCoME | temporal view | appearance | off-manifold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MRAA | 140 / 140 | **0.9993** | 0.9984 | 0.9937 | +0.0056 | 0.9807 | 0.9938 | 0.9930 |
| DaNet | 140 / 140 | **0.9546** | 0.9496 | 0.6764 | +0.2782 | 0.9726 | 0.6761 | 0.6855 |
| FaceVid2Vid | 140 / 140 | **0.9058** | 0.9121 | 0.7776 | +0.1282 | 0.9119 | 0.7778 | 0.7818 |
| FOMM | 140 / 140 | **0.9792** | 0.9730 | 0.9609 | +0.0183 | 0.9363 | 0.9624 | 0.9552 |
| HyperReenact | 140 / 140 | **0.9827** | 0.9835 | 0.8483 | +0.1343 | 0.9818 | 0.8499 | 0.8462 |
| LIA | 140 / 140 | **0.9999** | 1.0000 | 0.9921 | +0.0079 | 0.9929 | 0.9923 | 0.9913 |
| MCNet | 140 / 140 | **0.9566** | 0.9540 | 0.7680 | +0.1886 | 0.9677 | 0.7703 | 0.7706 |
| One-Shot Free-View | 140 / 140 | **0.9442** | 0.9354 | 0.8815 | +0.0627 | 0.9274 | 0.8843 | 0.8733 |
| PIRender | 140 / 140 | **0.9274** | 0.9089 | 0.8833 | +0.0441 | 0.8840 | 0.8861 | 0.8734 |
| SadTalker | 140 / 138 | **0.9503** | 0.9445 | 0.6071 | +0.3431 | 0.9731 | 0.6169 | 0.5798 |
| TPSM | 140 / 140 | **0.9677** | 0.9705 | 0.7458 | +0.2219 | 0.9767 | 0.7467 | 0.7491 |
| **mean over 11 generators** | | **0.9607** | 0.9573 | 0.8304 | +0.1303 | 0.9550 | 0.8324 | 0.8272 |

★ = not listed in DiCoME's published cross-manipulation table. Every generator here is scored against the genuine pool DF40 ships for its source (`DF40-cdf-real`, `DF40-ff-real`, and DeepFaceLab's own pool), so a generator's AUROC is never computed against another source's genuine videos.

Our rows are means over the five fitted splits (the fusion weights differ per split); DiCoME is a single checkpoint, so its rows carry no spread. Both sides use per-frame opinions averaged per video and then fused — the same aggregation as every other table. DF40 was never trained on by either system; its generators are unseen manipulations built on FF++ and Celeb-DF source videos, which both systems have seen as training or development content.

One detector seed and one temporal-branch seed unless stated. CDFv2 and DeepfakeEval2024 informed development and are not untouched confirmation data. FaceShifter is sealed (`tbiom/docs/SEALED_BENCHMARKS.md`).
