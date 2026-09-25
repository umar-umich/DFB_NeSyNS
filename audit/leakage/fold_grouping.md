# Task 5 — fold grouping

Connected components are built by `exp1_fusion_audit.groups_for`, which unions a video with the source videos it derives from, so a fake links its sources and every clip of a source lands in one component.

Development pool: **2728 components** over 5814 videos.

| fold | components in the selection half |
|---|---:|
| 0 | 1364 |
| 1 | 1364 |
| 2 | 1364 |
| 3 | 1364 |
| 4 | 1364 |

Components appearing in more than one fold's selection half: **1364**.

This is expected and is not a leak by itself: `GroupShuffleSplit` is run with five different seeds, so a component is in the selection half of some folds and the fitting half of others. The property that matters is that within any single fold a component is never in both halves, which Task 3c verifies directly.

First 10 components and the folds whose selection half contains them:

- `0` -> folds [0, 1, 2, 3, 4]
- `1` -> folds [0, 1, 2, 3, 4]
- `2` -> folds [0, 1, 2, 3, 4]
- `3` -> folds [0, 1, 2, 3, 4]
- `4` -> folds [0, 1, 2, 3, 4]
- `5` -> folds [0, 1, 2, 3, 4]
- `6` -> folds [0, 1, 2, 3, 4]
- `7` -> folds [0, 1, 2, 3, 4]
- `8` -> folds [0, 1, 2, 3, 4]
- `9` -> folds [0, 1, 2, 3, 4]

