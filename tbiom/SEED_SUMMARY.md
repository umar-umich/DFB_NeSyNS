# Seed summary — three seeds per configuration

Mean over the six OOD sets. P0-DS reference **0.8633** (mean real-side FPR 0.247). `tau` frozen per run on its own FF++ val; all probabilities are the evidential expectation `alpha/S`, one convention throughout.

## Per configuration

| configuration | views fused | seeds | mean AUROC | spread | vs P0-DS | mean FPR_real |
|---|---|---|---:|---:|---:|---:|
| arm B, 3-view | semantic + artifact + fsvfm | 3 | **0.8782** | ±0.0012 | +0.0149 | 0.135 |
| arm C, 2-view | artifact + fsvfm | 3 | **0.8837** | ±0.0056 | +0.0204 | 0.168 |
| arm C, 3-view | semantic + artifact + fsvfm | 3 | **0.8783** | ±0.0083 | +0.0150 | 0.175 |
| arm B, 2-view | artifact + fsvfm | 3 | **0.8723** | ±0.0003 | +0.0089 | 0.132 |

## Per seed

| configuration | seed 42 | seed 1337 | seed 7 | mean | range |
|---|---:|---:|---:|---:|---:|
| arm B, 3-view | 0.8768 | 0.8784 | 0.8793 | **0.8782** | 0.0025 |
| arm C, 2-view | 0.8784 | 0.8832 | 0.8895 | **0.8837** | 0.0111 |
| arm C, 3-view | 0.8701 | 0.8783 | 0.8866 | **0.8783** | 0.0166 |
| arm B, 2-view | 0.8719 | 0.8726 | 0.8723 | **0.8723** | 0.0007 |

## Per dataset, the leading configuration's mean over seeds

| dataset | P0-DS | arm B 3-view | arm C 2-view | best |
|---|---:|---:|---:|---|
| CDFv2 | 0.9646 | 0.9505 | 0.9685 | C, 2-view |
| CDFv3 | 0.8409 | 0.9008 | 0.9092 | C, 2-view |
| DFD | 0.9421 | 0.9486 | 0.9437 | B, 3-view |
| DFDC | 0.8828 | 0.8816 | 0.8892 | C, 2-view |
| DFDCP | 0.8573 | 0.9002 | 0.9040 | C, 2-view |
| DFEval24 | 0.6922 | 0.6875 | 0.6877 | C, 2-view |

## Verdict

- arm B 3-view: **0.8782** ±0.0012 over 3 seeds
- arm C 2-view: **0.8837** ±0.0056 over 3 seeds
- gap **-0.0055**, largest half-range **±0.0056**

**Not separable.** The gap is smaller than the seed spread, so neither configuration is demonstrably better. Both beat P0-DS by a margin that clears the noise band, and that is the claim the data supports. Choose on grounds other than this number -- arm B fuses all three views, which is the simpler story and needs no inference-time surgery.
