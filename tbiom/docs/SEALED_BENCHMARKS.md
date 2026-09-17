# Sealed benchmarks

| dataset | sealed on | source on disk | rule |
|---|---|---|---|
| **FaceShifter** (GenD release `extracted/test/FSh`) | 2026-09-17 (UMAR) | `/data/kutub/datasets/GEND/extracted/test/FSh` | Do not build, read, score or look at it until the final recipe is fixed for confirmation reporting. |

UADFV remains uninspected but is **not** the confirmation benchmark (near-ceiling performance). The GenD-repo baselines
were run on a *DeepfakeBench* FaceShifter split in May 2026 (`GenD_NeSy/results/baselines/*/FaceShifter_*`) — no DISCERN,
DiCoME or temporal-branch model has seen FaceShifter, and those May baseline files must not be opened either.

## Source-split audit (2026-09-17, metadata only)

The finish-line brief requires auditing FaceShifter's exact source split *before* assigning it a confirmation role,
because the official FF++ FaceShifter release manipulates the original FF++ videos and can therefore share source
content with training. The audit used **directory names only** — no frames were decoded, no labels beyond the
`real/` and `fake/` folder names were read, and no model was run on it.

- The release holds 140 fake videos (`NNN_MMM`, target\_source) and 140 real videos (`NNN`).
- Its 140 distinct FF++ identifiers are **exactly the FF++ test-split identifiers** used by our `FFpp` evaluation list
  (`eval_adaptation/data/splits/FFpp/*.txt`): subset check passes in both directions, 0 left over on either side.
- Overlap with our FF++ **training** source ids (720): **0**. Overlap with our FF++ **validation** source ids (140): **0**.

So for this deployment FaceShifter is both an unseen manipulation method *and* source-disjoint from the training and
validation content. Describe it as **unseen manipulation on held-out FF++ source videos**; still do not call it
"zero-leakage" in general, since that property belongs to this particular split pairing, not to the release.

The seal stands: no scoring, no reading, until the final recipe is fixed.
