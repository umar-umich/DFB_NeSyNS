# CROSS-EXAM upgrade — execution state

Brief: `tbiom/docs/discern-v2-crossexam-upgrade-brief.md`. Started 2026-10-01.
Code in `upgrade/`, reports in `analysis/upgrade/`, logs in `logs/upgrade/`.

**Deviation from the brief's lane table, recorded as the brief requires.** The brief assigns three
lanes (GPUs 1/2/3). UMAR restricted this run to **GPU 1 only** (2026-10-01). GPU 1 also carries
kutub's `train.py --name rr_pr` (17.9 GiB, 94% util at start), so every lane is serialised onto one
shared card. Ordering is by critical path: longest extraction in the background, cached and CPU work
alongside. Throughput is measured on the first extraction and the plan re-estimated from it.

Priority if the window runs short: Parts 1, 2, 3, 6 before 4, 5, 7, so Part 8 always has a
configuration to assemble.

## Status

| Part | What | Status | Evidence |
|---|---|---|---|
| 1 | IBDC on current views (J1s, lambda sweeps, J2b verdict) | TODO(run) | |
| 2a | Threshold transfer vs ranking | TODO(run) | |
| 2b | Feature-space factor profile (ESP adaptation) | TODO(run) | |
| 3a | Extraction: DINOv3, PE-Spatial | TODO(run) | |
| 3b | Frozen screen | TODO(run) | |
| 3c | Adaptation of shortlist (L and N recipes) | TODO(run) | |
| 4 | Processing suppression (MC/PD/LEACE/RND) | TODO(run) | |
| 5 | Label-preserving consistency (C-AUG, C-CONS) | TODO(run) | |
| 6a | Extraction: V-JEPA 2.1 | TODO(run) | |
| 6b | Global swap vs TD | TODO(run) | |
| 6c | Regional evidence (R-REG vs R-GLOB, R-POOL) | TODO(run) | |
| 7 | Fillers (stability, mapping, neuron ESP) | TODO(run) | |
| 8 | Final assembly, 3 seeds, tests, tables | TODO(run) | |

## Log
- 2026-10-01: scaffolding created; GPU-1-only deviation recorded.
