# PCR6+ on the exposed suites

Ten test sets, the processed conditions, and all 41 DF40 generator sets, for every arm of the fixed matrix and every
detector configuration. Labelled exposed benchmark evaluation: CDFv2 and DeepfakeEval2024 informed development and
DF40 has informed operator design in three studies now, so none of this confirms anything.

Both DF40 means are reported with their weights: family-balanced gives each of the four family x source cells weight
1/4; generator-macro gives each of the 41 sets weight 1/41, which favours the Celeb-DF-sourced cells because they
contain more generators.

Wall clock 1.2 minutes, CPU.

## Configuration `42`

### Ten test sets, video AUROC

| arm | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **PCR6P-R** | 0.9583 | 0.9639 | 0.9307 | 0.8530 | 0.8597 | 0.7235 | 0.9217 | 0.9600 | 0.9631 | 0.9954 | 0.9129 |
| PCR6-R | 0.9596 | 0.9618 | 0.9318 | 0.8553 | 0.8606 | 0.7235 | 0.9245 | 0.9583 | 0.9636 | 0.9954 | 0.9134 |
| PCR6-pairwise-R | 0.9607 | 0.9613 | 0.9334 | 0.8594 | 0.8633 | 0.7241 | 0.9272 | 0.9577 | 0.9648 | 0.9954 | 0.9147 |
| Yager-R | 0.9602 | 0.9635 | 0.9310 | 0.8507 | 0.8576 | 0.7223 | 0.9210 | 0.9600 | 0.9624 | 0.9953 | 0.9124 |
| Averaging-R | 0.9542 | 0.9632 | 0.9330 | 0.8578 | 0.8642 | 0.7236 | 0.9241 | 0.9602 | 0.9650 | 0.9960 | 0.9141 |
| Dempster-R | 0.9598 | 0.9616 | 0.9338 | 0.8626 | 0.8656 | 0.7252 | 0.9282 | 0.9574 | 0.9656 | 0.9956 | 0.9155 |
| T+OM (PCR6+) | 0.9575 | 0.9669 | 0.9294 | 0.8489 | 0.8576 | 0.7231 | 0.9174 | 0.9624 | 0.9621 | 0.9955 | 0.9121 |
| T+APP (PCR6+) | 0.9348 | 0.9515 | 0.9116 | 0.8346 | 0.8450 | 0.7174 | 0.9112 | 0.9540 | 0.9505 | 0.9923 | 0.9003 |

### PCR6+-R per test set, at the frozen threshold

| test set | AUROC | FPR | TPR | ECE | NLL | Brier | E-AURC (vacuity) | err-AUROC vacuity | err-AUROC K | err-AUROC entropy | risk@90 | risk@80 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CDFv2 | 0.9583 | 0.0404 | 0.6853 | 0.2544 | 0.5159 | 0.1776 | 0.1119 | 0.6409 | 0.6646 | 0.6668 | 0.2163 | 0.2082 |
| CDFv3 | 0.9639 | 0.0404 | 0.7960 | 0.2830 | 0.5174 | 0.1705 | 0.0547 | 0.7683 | 0.5420 | 0.6418 | 0.1741 | 0.1454 |
| DFD | 0.9307 | 0.1234 | 0.7984 | 0.2202 | 0.4558 | 0.1536 | 0.0470 | 0.7902 | 0.7154 | 0.7499 | 0.1692 | 0.1394 |
| DFDC | 0.8530 | 0.1906 | 0.7425 | 0.1053 | 0.5298 | 0.1671 | 0.0983 | 0.6722 | 0.6673 | 0.7082 | 0.2082 | 0.1982 |
| DFDCP | 0.8597 | 0.1791 | 0.7193 | 0.1805 | 0.5695 | 0.1881 | 0.1046 | 0.6746 | 0.6426 | 0.6684 | 0.2363 | 0.2221 |
| DFEval24 | 0.7235 | 0.1907 | 0.4819 | 0.1880 | 0.7598 | 0.2506 | 0.1855 | 0.6268 | 0.5676 | 0.6135 | 0.3315 | 0.3107 |
| FFIW | 0.9217 | 0.1150 | 0.7531 | 0.0987 | 0.4049 | 0.1323 | 0.0749 | 0.6994 | 0.7540 | 0.7851 | 0.1616 | 0.1490 |
| KoDF | 0.9600 | 0.0263 | 0.8796 | 0.1284 | 0.3452 | 0.0965 | 0.0199 | 0.8208 | 0.6551 | 0.7177 | 0.0597 | 0.0522 |
| IDForge | 0.9631 | 0.0670 | 0.8206 | 0.0784 | 0.2855 | 0.0916 | 0.0314 | 0.7935 | 0.7819 | 0.8459 | 0.0991 | 0.0815 |
| DFDM | 0.9954 | 0.0371 | 0.9710 | 0.0602 | 0.1080 | 0.0307 | 0.0034 | 0.8973 | 0.8695 | 0.9077 | 0.0165 | 0.0062 |

### DF40, both required means

| arm | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced (1/4 each) | generator-macro (1/41 each) |
|---|---:|---:|---:|---:|---:|---:|
| **PCR6P-R** | 0.9713 | 0.9834 | 0.9245 | 0.9616 | 0.9602 | 0.9588 |
| PCR6-R | 0.9716 | 0.9845 | 0.9176 | 0.9593 | 0.9582 | 0.9566 |
| PCR6-pairwise-R | 0.9725 | 0.9851 | 0.9166 | 0.9602 | 0.9586 | 0.9569 |
| Yager-R | 0.9708 | 0.9831 | 0.9215 | 0.9585 | 0.9585 | 0.9570 |
| Averaging-R *(baseline)* | 0.9729 | 0.9846 | 0.9267 | 0.9628 | 0.9618 | 0.9604 |
| Dempster-R *(baseline)* | 0.9731 | 0.9854 | 0.9174 | 0.9607 | 0.9591 | 0.9575 |
| T+OM (PCR6+) | 0.9707 | 0.9819 | 0.9337 | 0.9633 | 0.9624 | 0.9613 |
| T+APP (PCR6+) | 0.9602 | 0.9845 | 0.9036 | 0.9605 | 0.9522 | 0.9504 |

### DF40 per generator, PCR6+-R against the baselines

| generator | family | source | **PCR6+-R** | PCR6-R | Dempster-R | Averaging-R |
|---|---|---|---:|---:|---:|---:|
| blendface | face swap | cdf | **0.9790** | 0.9787 | 0.9801 | 0.9806 |
| deepfacelab | face swap | cdf | **0.9577** | 0.9589 | 0.9610 | 0.9640 |
| e4s | face swap | cdf | **0.9841** | 0.9851 | 0.9892 | 0.9869 |
| facedancer | face swap | cdf | **0.9710** | 0.9709 | 0.9714 | 0.9721 |
| faceswap | face swap | cdf | **0.9993** | 0.9994 | 0.9995 | 0.9996 |
| fsgan | face swap | cdf | **0.9912** | 0.9911 | 0.9906 | 0.9919 |
| inswap | face swap | cdf | **0.9523** | 0.9526 | 0.9551 | 0.9544 |
| mobileswap | face swap | cdf | **0.9571** | 0.9582 | 0.9584 | 0.9564 |
| simswap | face swap | cdf | **0.9396** | 0.9401 | 0.9416 | 0.9399 |
| uniface | face swap | cdf | **0.9813** | 0.9813 | 0.9837 | 0.9833 |
| blendface | face swap | ff | **0.9846** | 0.9844 | 0.9858 | 0.9865 |
| e4s | face swap | ff | **0.9910** | 0.9925 | 0.9927 | 0.9907 |
| facedancer | face swap | ff | **0.9690** | 0.9713 | 0.9713 | 0.9699 |
| faceswap | face swap | ff | **0.9949** | 0.9948 | 0.9948 | 0.9952 |
| fsgan | face swap | ff | **0.9934** | 0.9939 | 0.9935 | 0.9939 |
| inswap | face swap | ff | **0.9739** | 0.9757 | 0.9783 | 0.9763 |
| mobileswap | face swap | ff | **0.9824** | 0.9831 | 0.9848 | 0.9845 |
| simswap | face swap | ff | **0.9708** | 0.9722 | 0.9749 | 0.9738 |
| uniface | face swap | ff | **0.9911** | 0.9922 | 0.9928 | 0.9909 |
| MRAA | reenactment | cdf | **0.9927** | 0.9925 | 0.9934 | 0.9937 |
| danet | reenactment | cdf | **0.8852** | 0.8688 | 0.8645 | 0.8834 |
| facevid2vid | reenactment | cdf | **0.8512** | 0.8390 | 0.8403 | 0.8581 |
| fomm | reenactment | cdf | **0.9337** | 0.9308 | 0.9341 | 0.9408 |
| hyperreenact | reenactment | cdf | **0.9759** | 0.9723 | 0.9708 | 0.9755 |
| lia | reenactment | cdf | **0.9974** | 0.9975 | 0.9977 | 0.9976 |
| mcnet | reenactment | cdf | **0.8819** | 0.8673 | 0.8662 | 0.8846 |
| one_shot_free | reenactment | cdf | **0.9096** | 0.9059 | 0.9102 | 0.9177 |
| pirender | reenactment | cdf | **0.8652** | 0.8618 | 0.8626 | 0.8707 |
| sadtalker | reenactment | cdf | **0.9020** | 0.8862 | 0.8857 | 0.8982 |
| tpsm | reenactment | cdf | **0.9751** | 0.9712 | 0.9661 | 0.9734 |
| MRAA | reenactment | ff | **0.9987** | 0.9988 | 0.9993 | 0.9992 |
| danet | reenactment | ff | **0.9592** | 0.9546 | 0.9546 | 0.9580 |
| facevid2vid | reenactment | ff | **0.9071** | 0.9017 | 0.9058 | 0.9103 |
| fomm | reenactment | ff | **0.9756** | 0.9766 | 0.9792 | 0.9783 |
| hyperreenact | reenactment | ff | **0.9839** | 0.9825 | 0.9827 | 0.9835 |
| lia | reenactment | ff | **0.9998** | 0.9998 | 0.9999 | 0.9998 |
| mcnet | reenactment | ff | **0.9602** | 0.9562 | 0.9566 | 0.9598 |
| one_shot_free | reenactment | ff | **0.9438** | 0.9414 | 0.9442 | 0.9486 |
| pirender | reenactment | ff | **0.9222** | 0.9207 | 0.9274 | 0.9314 |
| sadtalker | reenactment | ff | **0.9570** | 0.9525 | 0.9503 | 0.9526 |
| tpsm | reenactment | ff | **0.9705** | 0.9677 | 0.9677 | 0.9691 |

## Configuration `T1337`

### Ten test sets, video AUROC

| arm | CDFv2 | CDFv3 | DFD | DFDC | DFDCP | DFEval24 | FFIW | KoDF | IDForge | DFDM | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **PCR6P-R** | 0.9512 | 0.9592 | 0.9315 | 0.8566 | 0.8749 | 0.7243 | 0.9209 | 0.9660 | 0.9542 | 0.9944 | 0.9133 |
| PCR6-R | 0.9526 | 0.9589 | 0.9323 | 0.8561 | 0.8744 | 0.7244 | 0.9213 | 0.9659 | 0.9546 | 0.9945 | 0.9135 |
| PCR6-pairwise-R | 0.9515 | 0.9565 | 0.9315 | 0.8578 | 0.8741 | 0.7242 | 0.9230 | 0.9642 | 0.9547 | 0.9941 | 0.9132 |
| Yager-R | 0.9536 | 0.9584 | 0.9330 | 0.8556 | 0.8733 | 0.7241 | 0.9216 | 0.9658 | 0.9550 | 0.9945 | 0.9135 |
| Averaging-R | 0.9502 | 0.9573 | 0.9323 | 0.8628 | 0.8806 | 0.7235 | 0.9228 | 0.9652 | 0.9551 | 0.9950 | 0.9145 |
| Dempster-R | 0.9506 | 0.9585 | 0.9307 | 0.8595 | 0.8774 | 0.7239 | 0.9209 | 0.9655 | 0.9540 | 0.9946 | 0.9136 |
| T+OM (PCR6+) | 0.9512 | 0.9592 | 0.9315 | 0.8566 | 0.8749 | 0.7243 | 0.9209 | 0.9660 | 0.9542 | 0.9944 | 0.9133 |
| T+APP (PCR6+) | 0.9254 | 0.9454 | 0.9129 | 0.8360 | 0.8596 | 0.7196 | 0.9112 | 0.9608 | 0.9372 | 0.9906 | 0.8999 |

### PCR6+-R per test set, at the frozen threshold

| test set | AUROC | FPR | TPR | ECE | NLL | Brier | E-AURC (vacuity) | err-AUROC vacuity | err-AUROC K | err-AUROC entropy | risk@90 | risk@80 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CDFv2 | 0.9512 | 0.0270 | 0.5400 | 0.2170 | 0.4696 | 0.1594 | 0.1792 | 0.6086 | 0.8146 | 0.8471 | 0.2955 | 0.2892 |
| CDFv3 | 0.9592 | 0.0270 | 0.6948 | 0.2722 | 0.4853 | 0.1616 | 0.0594 | 0.8253 | 0.6154 | 0.8156 | 0.2494 | 0.2084 |
| DFD | 0.9315 | 0.0512 | 0.6866 | 0.2319 | 0.4845 | 0.1647 | 0.0736 | 0.7750 | 0.8243 | 0.8589 | 0.2527 | 0.2216 |
| DFDC | 0.8566 | 0.1088 | 0.5974 | 0.1182 | 0.5243 | 0.1689 | 0.1398 | 0.6197 | 0.7253 | 0.7599 | 0.2465 | 0.2335 |
| DFDCP | 0.8749 | 0.1009 | 0.6005 | 0.1603 | 0.5269 | 0.1755 | 0.1418 | 0.6693 | 0.7159 | 0.7825 | 0.2683 | 0.2450 |
| DFEval24 | 0.7243 | 0.1103 | 0.3347 | 0.1817 | 0.7347 | 0.2469 | 0.2100 | 0.6105 | 0.6130 | 0.6643 | 0.3683 | 0.3494 |
| FFIW | 0.9209 | 0.0686 | 0.6578 | 0.0811 | 0.3904 | 0.1270 | 0.0818 | 0.7146 | 0.7503 | 0.8292 | 0.1737 | 0.1594 |
| KoDF | 0.9660 | 0.0154 | 0.8204 | 0.1058 | 0.2736 | 0.0787 | 0.0404 | 0.7692 | 0.7835 | 0.8305 | 0.1161 | 0.1027 |
| IDForge | 0.9542 | 0.0566 | 0.7328 | 0.0688 | 0.2983 | 0.0962 | 0.0496 | 0.7659 | 0.8229 | 0.8837 | 0.1339 | 0.1159 |
| DFDM | 0.9944 | 0.0247 | 0.9468 | 0.0502 | 0.0987 | 0.0272 | 0.0044 | 0.9247 | 0.8964 | 0.9575 | 0.0194 | 0.0099 |

### DF40, both required means

| arm | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced (1/4 each) | generator-macro (1/41 each) |
|---|---:|---:|---:|---:|---:|---:|
| **PCR6P-R** | 0.9733 | 0.9814 | 0.9363 | 0.9693 | 0.9651 | 0.9641 |
| PCR6-R | 0.9735 | 0.9816 | 0.9344 | 0.9685 | 0.9645 | 0.9634 |
| PCR6-pairwise-R | 0.9730 | 0.9826 | 0.9288 | 0.9680 | 0.9631 | 0.9619 |
| Yager-R | 0.9735 | 0.9816 | 0.9326 | 0.9677 | 0.9639 | 0.9628 |
| Averaging-R *(baseline)* | 0.9741 | 0.9828 | 0.9325 | 0.9694 | 0.9647 | 0.9636 |
| Dempster-R *(baseline)* | 0.9735 | 0.9818 | 0.9339 | 0.9691 | 0.9646 | 0.9635 |
| T+OM (PCR6+) | 0.9733 | 0.9814 | 0.9363 | 0.9693 | 0.9651 | 0.9641 |
| T+APP (PCR6+) | 0.9626 | 0.9815 | 0.9146 | 0.9681 | 0.9567 | 0.9553 |

### DF40 per generator, PCR6+-R against the baselines

| generator | family | source | **PCR6+-R** | PCR6-R | Dempster-R | Averaging-R |
|---|---|---|---:|---:|---:|---:|
| blendface | face swap | cdf | **0.9825** | 0.9822 | 0.9827 | 0.9828 |
| deepfacelab | face swap | cdf | **0.9637** | 0.9633 | 0.9654 | 0.9676 |
| e4s | face swap | cdf | **0.9817** | 0.9820 | 0.9841 | 0.9844 |
| facedancer | face swap | cdf | **0.9736** | 0.9742 | 0.9729 | 0.9733 |
| faceswap | face swap | cdf | **0.9993** | 0.9993 | 0.9994 | 0.9994 |
| fsgan | face swap | cdf | **0.9923** | 0.9925 | 0.9914 | 0.9924 |
| inswap | face swap | cdf | **0.9530** | 0.9532 | 0.9537 | 0.9546 |
| mobileswap | face swap | cdf | **0.9558** | 0.9574 | 0.9547 | 0.9548 |
| simswap | face swap | cdf | **0.9474** | 0.9477 | 0.9470 | 0.9474 |
| uniface | face swap | cdf | **0.9833** | 0.9832 | 0.9837 | 0.9844 |
| blendface | face swap | ff | **0.9844** | 0.9845 | 0.9852 | 0.9855 |
| e4s | face swap | ff | **0.9887** | 0.9891 | 0.9891 | 0.9886 |
| facedancer | face swap | ff | **0.9630** | 0.9639 | 0.9627 | 0.9648 |
| faceswap | face swap | ff | **0.9943** | 0.9939 | 0.9944 | 0.9939 |
| fsgan | face swap | ff | **0.9927** | 0.9931 | 0.9925 | 0.9931 |
| inswap | face swap | ff | **0.9693** | 0.9692 | 0.9699 | 0.9728 |
| mobileswap | face swap | ff | **0.9813** | 0.9813 | 0.9820 | 0.9833 |
| simswap | face swap | ff | **0.9705** | 0.9702 | 0.9712 | 0.9737 |
| uniface | face swap | ff | **0.9886** | 0.9891 | 0.9891 | 0.9897 |
| MRAA | reenactment | cdf | **0.9936** | 0.9935 | 0.9937 | 0.9937 |
| danet | reenactment | cdf | **0.9049** | 0.9021 | 0.8994 | 0.8944 |
| facevid2vid | reenactment | cdf | **0.8877** | 0.8832 | 0.8840 | 0.8857 |
| fomm | reenactment | cdf | **0.9518** | 0.9505 | 0.9517 | 0.9537 |
| hyperreenact | reenactment | cdf | **0.9648** | 0.9635 | 0.9622 | 0.9597 |
| lia | reenactment | cdf | **0.9956** | 0.9956 | 0.9955 | 0.9954 |
| mcnet | reenactment | cdf | **0.8979** | 0.8936 | 0.8935 | 0.8917 |
| one_shot_free | reenactment | cdf | **0.9319** | 0.9305 | 0.9322 | 0.9350 |
| pirender | reenactment | cdf | **0.8922** | 0.8910 | 0.8912 | 0.8945 |
| sadtalker | reenactment | cdf | **0.9080** | 0.9044 | 0.9018 | 0.8896 |
| tpsm | reenactment | cdf | **0.9706** | 0.9708 | 0.9675 | 0.9640 |
| MRAA | reenactment | ff | **0.9967** | 0.9966 | 0.9972 | 0.9975 |
| danet | reenactment | ff | **0.9681** | 0.9676 | 0.9669 | 0.9661 |
| facevid2vid | reenactment | ff | **0.9328** | 0.9303 | 0.9329 | 0.9343 |
| fomm | reenactment | ff | **0.9795** | 0.9790 | 0.9802 | 0.9822 |
| hyperreenact | reenactment | ff | **0.9751** | 0.9742 | 0.9749 | 0.9747 |
| lia | reenactment | ff | **0.9991** | 0.9991 | 0.9993 | 0.9993 |
| mcnet | reenactment | ff | **0.9642** | 0.9630 | 0.9633 | 0.9632 |
| one_shot_free | reenactment | ff | **0.9609** | 0.9602 | 0.9612 | 0.9626 |
| pirender | reenactment | ff | **0.9407** | 0.9391 | 0.9419 | 0.9456 |
| sadtalker | reenactment | ff | **0.9687** | 0.9684 | 0.9660 | 0.9625 |
| tpsm | reenactment | ff | **0.9768** | 0.9763 | 0.9764 | 0.9757 |

## Configuration `armB42`


### DF40, both required means

| arm | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced (1/4 each) | generator-macro (1/41 each) |
|---|---:|---:|---:|---:|---:|---:|
| **PCR6P-R** | 0.9577 | 0.9771 | 0.9468 | 0.9723 | 0.9635 | 0.9630 |
| PCR6-R | 0.9576 | 0.9776 | 0.9454 | 0.9712 | 0.9630 | 0.9624 |
| PCR6-pairwise-R | 0.9577 | 0.9771 | 0.9468 | 0.9723 | 0.9635 | 0.9630 |
| Yager-R | 0.9572 | 0.9775 | 0.9439 | 0.9701 | 0.9622 | 0.9616 |
| Averaging-R *(baseline)* | 0.9602 | 0.9784 | 0.9445 | 0.9722 | 0.9638 | 0.9632 |
| Dempster-R *(baseline)* | 0.9589 | 0.9768 | 0.9484 | 0.9730 | 0.9643 | 0.9638 |
| T+OM (PCR6+) | 0.9577 | 0.9771 | 0.9468 | 0.9723 | 0.9635 | 0.9630 |
| T+APP (PCR6+) | 0.9575 | 0.9789 | 0.9441 | 0.9726 | 0.9633 | 0.9626 |

### DF40 per generator, PCR6+-R against the baselines

| generator | family | source | **PCR6+-R** | PCR6-R | Dempster-R | Averaging-R |
|---|---|---|---:|---:|---:|---:|
| blendface | face swap | cdf | **0.9680** | 0.9674 | 0.9693 | 0.9702 |
| deepfacelab | face swap | cdf | **0.9600** | 0.9596 | 0.9616 | 0.9632 |
| e4s | face swap | cdf | **0.9724** | 0.9725 | 0.9755 | 0.9778 |
| facedancer | face swap | cdf | **0.9719** | 0.9720 | 0.9738 | 0.9763 |
| faceswap | face swap | cdf | **0.9992** | 0.9992 | 0.9993 | 0.9992 |
| fsgan | face swap | cdf | **0.9884** | 0.9884 | 0.9886 | 0.9903 |
| inswap | face swap | cdf | **0.9316** | 0.9313 | 0.9327 | 0.9346 |
| mobileswap | face swap | cdf | **0.9324** | 0.9333 | 0.9326 | 0.9328 |
| simswap | face swap | cdf | **0.8913** | 0.8911 | 0.8924 | 0.8930 |
| uniface | face swap | cdf | **0.9620** | 0.9616 | 0.9629 | 0.9642 |
| blendface | face swap | ff | **0.9837** | 0.9836 | 0.9844 | 0.9840 |
| e4s | face swap | ff | **0.9882** | 0.9887 | 0.9874 | 0.9877 |
| facedancer | face swap | ff | **0.9579** | 0.9593 | 0.9570 | 0.9596 |
| faceswap | face swap | ff | **0.9948** | 0.9951 | 0.9948 | 0.9942 |
| fsgan | face swap | ff | **0.9914** | 0.9923 | 0.9902 | 0.9918 |
| inswap | face swap | ff | **0.9632** | 0.9622 | 0.9634 | 0.9675 |
| mobileswap | face swap | ff | **0.9784** | 0.9784 | 0.9785 | 0.9812 |
| simswap | face swap | ff | **0.9534** | 0.9538 | 0.9523 | 0.9552 |
| uniface | face swap | ff | **0.9833** | 0.9847 | 0.9830 | 0.9844 |
| MRAA | reenactment | cdf | **0.9867** | 0.9866 | 0.9873 | 0.9877 |
| danet | reenactment | cdf | **0.9246** | 0.9213 | 0.9277 | 0.9206 |
| facevid2vid | reenactment | cdf | **0.9207** | 0.9193 | 0.9223 | 0.9224 |
| fomm | reenactment | cdf | **0.9544** | 0.9536 | 0.9555 | 0.9578 |
| hyperreenact | reenactment | cdf | **0.9813** | 0.9807 | 0.9824 | 0.9806 |
| lia | reenactment | cdf | **0.9896** | 0.9894 | 0.9901 | 0.9896 |
| mcnet | reenactment | cdf | **0.9306** | 0.9281 | 0.9334 | 0.9307 |
| one_shot_free | reenactment | cdf | **0.9451** | 0.9448 | 0.9463 | 0.9476 |
| pirender | reenactment | cdf | **0.8937** | 0.8930 | 0.8947 | 0.8962 |
| sadtalker | reenactment | cdf | **0.9124** | 0.9076 | 0.9155 | 0.8819 |
| tpsm | reenactment | cdf | **0.9755** | 0.9751 | 0.9776 | 0.9745 |
| MRAA | reenactment | ff | **0.9981** | 0.9980 | 0.9984 | 0.9986 |
| danet | reenactment | ff | **0.9718** | 0.9705 | 0.9728 | 0.9710 |
| facevid2vid | reenactment | ff | **0.9391** | 0.9377 | 0.9404 | 0.9419 |
| fomm | reenactment | ff | **0.9811** | 0.9809 | 0.9813 | 0.9830 |
| hyperreenact | reenactment | ff | **0.9917** | 0.9910 | 0.9921 | 0.9916 |
| lia | reenactment | ff | **0.9994** | 0.9994 | 0.9994 | 0.9994 |
| mcnet | reenactment | ff | **0.9746** | 0.9738 | 0.9761 | 0.9765 |
| one_shot_free | reenactment | ff | **0.9532** | 0.9518 | 0.9544 | 0.9572 |
| pirender | reenactment | ff | **0.9359** | 0.9332 | 0.9376 | 0.9409 |
| sadtalker | reenactment | ff | **0.9659** | 0.9640 | 0.9655 | 0.9514 |
| tpsm | reenactment | ff | **0.9841** | 0.9832 | 0.9851 | 0.9831 |

## Configuration `armB1337`


### DF40, both required means

| arm | swap CDF | swap FF++ | reenact CDF | reenact FF++ | family-balanced (1/4 each) | generator-macro (1/41 each) |
|---|---:|---:|---:|---:|---:|---:|
| **PCR6P-R** | 0.9687 | 0.9804 | 0.9503 | 0.9767 | 0.9690 | 0.9685 |
| PCR6-R | 0.9684 | 0.9803 | 0.9494 | 0.9764 | 0.9686 | 0.9681 |
| PCR6-pairwise-R | 0.9690 | 0.9809 | 0.9495 | 0.9765 | 0.9690 | 0.9684 |
| Yager-R | 0.9673 | 0.9790 | 0.9505 | 0.9759 | 0.9682 | 0.9677 |
| Averaging-R *(baseline)* | 0.9695 | 0.9807 | 0.9513 | 0.9775 | 0.9698 | 0.9692 |
| Dempster-R *(baseline)* | 0.9697 | 0.9816 | 0.9505 | 0.9775 | 0.9698 | 0.9692 |
| T+OM (PCR6+) | 0.9665 | 0.9780 | 0.9540 | 0.9770 | 0.9689 | 0.9685 |
| T+APP (PCR6+) | 0.9672 | 0.9795 | 0.9529 | 0.9774 | 0.9692 | 0.9688 |

### DF40 per generator, PCR6+-R against the baselines

| generator | family | source | **PCR6+-R** | PCR6-R | Dempster-R | Averaging-R |
|---|---|---|---:|---:|---:|---:|
| blendface | face swap | cdf | **0.9786** | 0.9782 | 0.9795 | 0.9795 |
| deepfacelab | face swap | cdf | **0.9627** | 0.9627 | 0.9653 | 0.9669 |
| e4s | face swap | cdf | **0.9848** | 0.9845 | 0.9868 | 0.9840 |
| facedancer | face swap | cdf | **0.9833** | 0.9831 | 0.9835 | 0.9827 |
| faceswap | face swap | cdf | **0.9992** | 0.9992 | 0.9993 | 0.9993 |
| fsgan | face swap | cdf | **0.9936** | 0.9935 | 0.9933 | 0.9934 |
| inswap | face swap | cdf | **0.9394** | 0.9389 | 0.9410 | 0.9422 |
| mobileswap | face swap | cdf | **0.9391** | 0.9389 | 0.9397 | 0.9378 |
| simswap | face swap | cdf | **0.9267** | 0.9261 | 0.9282 | 0.9285 |
| uniface | face swap | cdf | **0.9796** | 0.9794 | 0.9805 | 0.9808 |
| blendface | face swap | ff | **0.9843** | 0.9844 | 0.9858 | 0.9848 |
| e4s | face swap | ff | **0.9887** | 0.9886 | 0.9897 | 0.9864 |
| facedancer | face swap | ff | **0.9616** | 0.9610 | 0.9626 | 0.9621 |
| faceswap | face swap | ff | **0.9929** | 0.9928 | 0.9928 | 0.9925 |
| fsgan | face swap | ff | **0.9921** | 0.9920 | 0.9920 | 0.9920 |
| inswap | face swap | ff | **0.9640** | 0.9638 | 0.9659 | 0.9660 |
| mobileswap | face swap | ff | **0.9803** | 0.9802 | 0.9820 | 0.9816 |
| simswap | face swap | ff | **0.9713** | 0.9713 | 0.9736 | 0.9717 |
| uniface | face swap | ff | **0.9885** | 0.9885 | 0.9897 | 0.9888 |
| MRAA | reenactment | cdf | **0.9915** | 0.9913 | 0.9919 | 0.9915 |
| danet | reenactment | cdf | **0.9285** | 0.9272 | 0.9289 | 0.9316 |
| facevid2vid | reenactment | cdf | **0.9328** | 0.9316 | 0.9354 | 0.9370 |
| fomm | reenactment | cdf | **0.9627** | 0.9622 | 0.9641 | 0.9654 |
| hyperreenact | reenactment | cdf | **0.9799** | 0.9794 | 0.9800 | 0.9801 |
| lia | reenactment | cdf | **0.9908** | 0.9907 | 0.9912 | 0.9908 |
| mcnet | reenactment | cdf | **0.9308** | 0.9293 | 0.9325 | 0.9354 |
| one_shot_free | reenactment | cdf | **0.9500** | 0.9491 | 0.9516 | 0.9531 |
| pirender | reenactment | cdf | **0.9102** | 0.9091 | 0.9122 | 0.9144 |
| sadtalker | reenactment | cdf | **0.9006** | 0.8986 | 0.8955 | 0.8915 |
| tpsm | reenactment | cdf | **0.9753** | 0.9751 | 0.9720 | 0.9735 |
| MRAA | reenactment | ff | **0.9973** | 0.9972 | 0.9976 | 0.9975 |
| danet | reenactment | ff | **0.9738** | 0.9736 | 0.9745 | 0.9758 |
| facevid2vid | reenactment | ff | **0.9571** | 0.9568 | 0.9591 | 0.9585 |
| fomm | reenactment | ff | **0.9860** | 0.9856 | 0.9870 | 0.9871 |
| hyperreenact | reenactment | ff | **0.9844** | 0.9839 | 0.9857 | 0.9859 |
| lia | reenactment | ff | **0.9997** | 0.9997 | 0.9997 | 0.9996 |
| mcnet | reenactment | ff | **0.9753** | 0.9751 | 0.9765 | 0.9774 |
| one_shot_free | reenactment | ff | **0.9656** | 0.9651 | 0.9667 | 0.9666 |
| pirender | reenactment | ff | **0.9499** | 0.9489 | 0.9524 | 0.9530 |
| sadtalker | reenactment | ff | **0.9710** | 0.9709 | 0.9684 | 0.9670 |
| tpsm | reenactment | ff | **0.9835** | 0.9834 | 0.9847 | 0.9844 |

