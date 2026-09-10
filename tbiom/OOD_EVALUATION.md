# OOD evaluation — frozen detector, per test set

Frozen configuration: **arm B, three-view, DS fusion, per-branch EDL**, over seeds 42 / 1337 / 7. `tau` is frozen per seed at its own EER on FF++ val; no OOD set was consulted to set anything here. P0-DS is the published baseline read on the same convention.

## Detector

| test set | P0-DS AUROC | frozen AUROC | spread | ΔAUROC | P0-DS FPR_real | frozen FPR_real | ΔFPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CDFv2 | 0.9646 | 0.9505 | ±0.0028 | **-0.0141** | 0.129 | 0.054 | -0.075 |
| CDFv3 | 0.8409 | 0.9008 | ±0.0108 | **+0.0599** | 0.129 | 0.054 | -0.075 |
| DFD | 0.9421 | 0.9486 | ±0.0029 | +0.0065 | 0.174 | 0.025 | -0.149 |
| DFDC | 0.8828 | 0.8816 | ±0.0078 | -0.0012 | 0.317 | 0.188 | -0.129 |
| DFDCP | 0.8573 | 0.9002 | ±0.0042 | **+0.0429** | 0.391 | 0.287 | -0.104 |
| DFEval24 | 0.6922 | 0.6875 | ±0.0044 | -0.0047 | 0.341 | 0.202 | -0.139 |
| **mean** | 0.8633 | **0.8782** | | **+0.0149** | 0.247 | **0.135** | -0.112 |

Bold ΔAUROC marks |Δ| ≥ 0.01, the band a single seed can produce. Smaller differences are reported but not claimed.

## Overlay, on the same six sets

| test set | detector AUROC | rule coverage | driver claims | intervention survival |
|---|---:|---:|---:|---:|
| CDFv2 | 0.9505 | 0.598 | 40 | 0.950 |
| CDFv3 | 0.9008 | 0.518 | 263 | 0.920 |
| DFD | 0.9486 | 0.599 | 144 | 0.917 |
| DFDC | 0.8816 | 0.672 | 61 | 0.934 |
| DFDCP | 0.9002 | 0.557 | 20 | 0.950 |
| DFEval24 | 0.6875 | 0.506 | 3 | 1.000 |
| **all** | **0.8782** | 0.575 | 531 | **0.925** |

## What the two tables say together

- **Where the gain comes from.** The +0.0149 mean is not spread evenly. It is carried by CDFv3 and DFDCP (+0.0599 and +0.0429), paid for partly on CDFv2 (-0.0141), and flat within the seed band on DFD, DFDC, DFEval24. Two clear wins, one clear loss, three ties.
- **The real-side improvement is the uniform one.** FPR_real drops on all six sets, mean 0.247 → 0.135, with no set worse. That is the more robust claim than the AUROC mean.
- **CDFv2 and CDFv3 report the same FPR_real by construction**, not by coincidence: both draw their real videos from the identical Celeb-real pool (5,692 frames, byte-identical keys apart from the dataset prefix), so any threshold scores them identically on the real side. Only their fake pools differ.
- **Deepfake-Eval-2024 remains the stated limitation.** At 0.6875 it is 0.19 below the next-worst set, and its shortfall against P0-DS (−0.0047) is inside the seed band, so the honest statement is that the frozen model does not improve on in-the-wild 2024 content rather than that it loses there. The overlay agrees: 3 driver claims on 814 videos, because the branches almost never hold decisive evidence on this set.
- **Overlay coverage does not track detector accuracy.** DFDC has the highest coverage (0.672) at a mid-pack AUROC, CDFv2 the highest AUROC at 0.598 coverage. The overlay is not simply narrating the samples the detector finds easy.

Traces: `tbiom/SYMBOLIC_TRACES.md`. Overlay construction and validation: `tbiom/SYMBOLIC_OVERLAY.md`. Freeze: `tbiom/BRANCH3_REPORT.md` Phase F.
