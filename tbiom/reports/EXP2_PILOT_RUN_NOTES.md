# Experiment 2 pilot — run notes

`warm_dicome_gate_B_seed42` (DISCERN_Ext config commit `78337e6`): warm+gate recipe + arm-B processing mix.

- 2026-09-16 20:54 first launch stopped within minutes (kutub's burst on GPU 1); run directory removed; restarted 21:08.
- 2026-09-16 22:05 CPU oversubscription on the shared box (our extraction jobs) starved the run; fixed by thread caps.
- **2026-09-17 00:12 deviation:** stopped after 9 completed epochs (global step 8100) and resumed from `last.ckpt` with
  `num_workers=8` instead of 2 (UMAR: speed up the pilot). Recipe, initialisation, seed, trainable modules, losses,
  20-epoch budget and the development-only checkpoint rule are unchanged. What changes: the per-worker RNG stream
  that draws the processing settings for epochs 10-20 (worker count enters the per-worker seed), so the exact
  sequence of processed samples in those epochs differs from a 2-worker run. The processing DISTRIBUTION is
  unchanged. The unaugmented comparison (`warm_dicome_gate_seed42`) used 2 workers throughout.
