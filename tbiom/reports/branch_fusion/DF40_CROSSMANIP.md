# DF40 cross-manipulation — App+TD+LFS+STT fused with PCR6+-R

The structure of DiCoME Table 1's second half, measured on our videos.

**Both framework sizes are reported.** The four-branch set and the best three-branch set
(`App+TD+LFS`) each carry their own weights, fitted on development and
applied unchanged here, so the question of whether STT earns its place is answered on
generators no branch was trained on rather than on the twelve test sets alone.

**Both source halves are reported.** DF40 ships each generator twice — built from
Celeb-DF sources (`cdf`) and from FaceForensics++ sources (`ff`) — and DiCoME's paper
reports one number per generator without saying which. Picking the more flattering half
would be a choice this table hides, so neither is picked.

`DiCoME (our training)` is our own trained DiCoME checkpoint on the IDENTICAL videos and
is the comparison to read. The published rows below are transcribed from the paper on a
census we cannot inspect and are context, not a measured difference.

## Celeb-DF-sourced generators

| generator | videos (real / fake) | **App+TD+LFS+STT** | **App+TD+LFS** | DiCoME (our training) | Δ (four) | App | LFS | STT | TD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UniFace | 178 / 626 | **0.9974** | **0.9945** | 0.9758 | +0.0216 | 0.9968 | 0.9747 | 0.9969 | 0.9137 |
| BleFace | 178 / 640 | **0.9930** | **0.9899** | 0.9502 | +0.0429 | 0.9786 | 0.9516 | 0.9775 | 0.9371 |
| MobSwap | 178 / 637 | **0.9777** | **0.9693** | 0.9571 | +0.0206 | 0.9561 | 0.9641 | 0.9479 | 0.7988 |
| e4s | 178 / 292 | **0.9964** | **0.9913** | 0.9949 | +0.0015 | 0.9980 | 0.9939 | 0.9968 | 0.8055 |
| FaceDan | 178 / 648 | **0.9840** | **0.9831** | 0.9657 | +0.0184 | 0.9852 | 0.9744 | 0.9487 | 0.8598 |
| FSGAN | 178 / 616 | **0.9931** | **0.9903** | 0.9892 | +0.0039 | 0.9816 | 0.9938 | 0.9891 | 0.9393 |
| InSwap | 178 / 424 | **0.9867** | **0.9828** | 0.9433 | +0.0434 | 0.9855 | 0.9429 | 0.9760 | 0.8872 |
| SimSwap | 178 / 641 | **0.9942** | **0.9861** | 0.9339 | +0.0603 | 0.9967 | 0.9380 | 0.9942 | 0.8006 |
| **mean over 8** | | **0.9903** | **0.9859** | 0.9638 | +0.0266 | 0.9848 | 0.9667 | 0.9784 | 0.8677 |

## FaceForensics++-sourced generators

| generator | videos (real / fake) | **App+TD+LFS+STT** | **App+TD+LFS** | DiCoME (our training) | Δ (four) | App | LFS | STT | TD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UniFace | 140 / 140 | **0.9987** | **0.9986** | 0.9803 | +0.0185 | 0.9971 | 0.9783 | 0.9954 | 0.9228 |
| BleFace | 140 / 140 | **0.9974** | **0.9973** | 0.9505 | +0.0469 | 0.9954 | 0.9470 | 0.9866 | 0.9490 |
| MobSwap | 140 / 858 | **0.9980** | **0.9978** | 0.9745 | +0.0235 | 0.9974 | 0.9740 | 0.9928 | 0.9165 |
| e4s | 140 / 115 | **0.9993** | **0.9988** | 0.9891 | +0.0102 | 0.9988 | 0.9881 | 0.9966 | 0.8081 |
| FaceDan | 140 / 140 | **0.9912** | **0.9909** | 0.9632 | +0.0280 | 0.9919 | 0.9616 | 0.9741 | 0.8330 |
| FSGAN | 140 / 140 | **0.9973** | **0.9966** | 0.9879 | +0.0095 | 0.9956 | 0.9879 | 0.9857 | 0.9453 |
| InSwap | 140 / 118 | **0.9978** | **0.9978** | 0.9696 | +0.0282 | 0.9980 | 0.9654 | 0.9931 | 0.8993 |
| SimSwap | 140 / 140 | **0.9990** | **0.9989** | 0.9539 | +0.0452 | 0.9982 | 0.9500 | 0.9951 | 0.8577 |
| **mean over 8** | | **0.9974** | **0.9971** | 0.9711 | +0.0262 | 0.9965 | 0.9690 | 0.9899 | 0.8915 |

## Against the published cross-manipulation table

Column order is DiCoME's. Each of our groups contributes three rows — the two
source halves and their pool; every other row is transcribed from the paper and was
not re-measured here, on a census we cannot inspect.

| method | UniFace | BleFace | MobSwap | e4s | FaceDan | FSGAN | InSwap | SimSwap | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **App+TD+LFS+STT** (cdf) | 0.9974 | 0.9930 | 0.9777 | 0.9964 | 0.9840 | 0.9931 | 0.9867 | 0.9942 | 0.9903 |
| **App+TD+LFS+STT** (ff) | 0.9987 | 0.9974 | 0.9980 | 0.9993 | 0.9912 | 0.9973 | 0.9978 | 0.9990 | 0.9974 |
| **App+TD+LFS+STT** (both halves) | 0.9981 | 0.9952 | 0.9878 | 0.9978 | 0.9876 | 0.9952 | 0.9923 | 0.9966 | 0.9938 |
| **App+TD+LFS** (cdf) | 0.9945 | 0.9899 | 0.9693 | 0.9913 | 0.9831 | 0.9903 | 0.9828 | 0.9861 | 0.9859 |
| **App+TD+LFS** (ff) | 0.9986 | 0.9973 | 0.9978 | 0.9988 | 0.9909 | 0.9966 | 0.9978 | 0.9989 | 0.9971 |
| **App+TD+LFS** (both halves) | 0.9965 | 0.9936 | 0.9836 | 0.9950 | 0.9870 | 0.9934 | 0.9903 | 0.9925 | 0.9915 |
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

