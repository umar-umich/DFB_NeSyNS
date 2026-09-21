# Stage 3 — composed processing on the deployed spatial recipe: frozen before launch

Written 2026-09-21 at UMAR's direction, after the Stage 3 pre-check (`tbiom/reports/STAGE3_PRECHECK.md`) found that
Stage 1B's winner — composition on the `sbi_arm` family — does not beat the deployed spatial views as a drop-in.
This pilot asks the question on the deployed recipe itself. It goes beyond the brief's Stage 3, which only combines
changes that already passed; it is recorded as an additional matched pilot, and nothing here was chosen after seeing
its results.

## Why the recipe must change

The deployed spatial views come from the warm+gate detector (`warm_dicome_gate_seed42`, epoch 19): warm-started
from our DiCoME run and then **frozen** (`freeze_views: true`), so no training-time augmentation can reach them
(Experiment 5 found exactly this). Both arms below therefore unfreeze the views; that change is common to both and is
itself measured against the deployed checkpoint.

## Arms (one seed each, seed 42)

| arm | config | differs from the deployed config by |
|---|---|---|
| **UB** | `warm_dicome_gate_unfrozenB_seed42.yaml` | `freeze_views: false`, `training_arm: B` |
| **UE** | `warm_dicome_gate_unfrozenE_seed42.yaml` | `freeze_views: false`, `training_arm: E` |

Same warm start (`init_views_from` DiCoME epoch 5), same learning rate (1e-4), 20 epochs, batch 128, 8 data-loader
workers, face-prior gate unchanged (the face prior stays outside the reference fusion). Checkpoints are selected by
the unchanged v3.1 development rule. Each arm's Appearance and Off-manifold views are fused with the frozen reference
temporal branch; DS-R and the logistic comparator are refitted per arm on the same 5 source-grouped folds.

## Questions and acceptance, development only, DS-R primary (logistic reported beside it)

Paired source-group bootstrap, 1000 draws per split; "significant" means the 95% CI excludes zero in ≥ 3 of 5 splits.

1. **Composition effect on this recipe (UE − UB):** reported with the Stage 1B criteria — processed Δ ≥ +0.005
   significant, clean Δ ≥ −0.005, reenactment not significantly negative, clean FPR Δ ≤ +0.02.
2. **Adoption into the deployed system (UE − deployed):** UE replaces the deployed spatial views only if it meets
   the same four conditions against the deployed checkpoint. This is the decision that matters for the paper.
3. **Unfreezing effect (UB − deployed):** reported, never a gate.

If (2) passes, the combined recipe is confirmed across training seeds (1337 and 7) before it is reported, as the
brief's Stage 3 requires. No test set enters any decision.

## Also run at UMAR's direction (2026-09-21): adopt arm E as an override, confirmed across seeds

UMAR asked for both routes. Alongside the pilot above, the Stage-1B arm E (`sbi_armE`, composed processing, trainable
views, no warm start) is carried forward **as an override of the frozen criteria** — the pre-check found it does not
beat the deployed spatial views under DS-R (clean −0.0051, processed +0.0037 not significant, reenactment −0.0087).
It is confirmed across training seeds as the brief's Stage 3 requires:

* `sbi_armE_seed1337.yaml` and `sbi_armE_seed7.yaml` are `sbi_armE.yaml` with only `seed` and `run_name` changed.
* Each seed is selected by the unchanged v3.1 rule, fused with the frozen reference temporal branch, and compared with
  the deployed system on the same folds: E − deployed for clean, processed and reenactment, per seed, with the spread.
* Nothing gates this route; it is reported exactly as measured and labelled as an override. If route (2) passes its
  adoption criterion, its result — not this override — is what the paper adopts.
