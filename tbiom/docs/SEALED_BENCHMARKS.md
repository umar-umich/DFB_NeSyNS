# Sealed benchmarks

| dataset | sealed on | source on disk | rule |
|---|---|---|---|
| **FaceShifter** (GenD release `extracted/test/FSh`) | 2026-09-17 (UMAR) | `/data/kutub/datasets/GEND/extracted/test/FSh` | Do not build, read, score or look at it until the final recipe is fixed for confirmation reporting. |

UADFV remains uninspected but is **not** the confirmation benchmark (near-ceiling performance). The GenD-repo baselines
were run on a *DeepfakeBench* FaceShifter split in May 2026 (`GenD_NeSy/results/baselines/*/FaceShifter_*`) — no DISCERN,
DiCoME or temporal-branch model has seen FaceShifter, and those May baseline files must not be opened either.
