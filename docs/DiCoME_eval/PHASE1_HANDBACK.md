# DISCERN v2 — Phase 1 hand-back

Closes the deliverable list in `claude-code-discern-v2-phase1.md`. Written 2026-08-15.

## Confirmed paths

| What | Path | Note |
|---|---|---|
| **DISCERN repo** | `/data/umar/Repos/DFB_NeSyNS` | Confirmed by Umar 2026-08-15. It is this repo — **not** a sibling `../DISCERN/`. Closes open gate #3 in `README-discern-v2-integration.md`. |
| DiCoME repo (exports) | `/data/umar/Repos/DiCoME` | Read-only source of the per-sample pilot exports. Override with `DICOME_ROOT` if it moves. |
| SDXL-VAE checkout | `/data/umar/Repos/DiCoME/eval_adaptation/data/models/sdxl-vae` | Frozen; the checkout the P2a pilot used. |
| Conda env — analyses | `discern_ext` | numpy/pandas/sklearn/scipy/matplotlib. |
| Conda env — torch tests | `dfb_nesy` | For `training/networks/discern_v2/test_discern_v2.py`. |

## Delivered

**Track A** (`analysis/discern_v2/`) — scripts plus written verdicts. Generated CSV/plot
output dirs are gitignored so a given run's numbers can't be mistaken for the run of record.

| File | Purpose |
|---|---|
| `common.py` | Loading, frozen-threshold discipline, LOGO fold hygiene |
| `a1_complementarity.py` | rescue / harm / error-overlap / evidence correlation |
| `a2_gate.py` | Oracle headroom + A2a gate; writes `A2b_PROTOCOL.md` |
| `a3_rate_response.py` | Q1 separability / Q2 family structure / Q3 error-predictiveness |
| `test_folds.py` | 6 tests pinning fold construction |
| `SELECTION.md` | All seven selection questions, answered |
| `A3_VERDICT.md` | The P1d rate-response verdict |

**Track B** (`training/networks/discern_v2/`, `training/config/discern_v2/`)

- `dirichlet.py` — the single shared utility deriving alpha/S/p/u centrally.
- `branches.py` — Visual / Manifold / Process, independently toggleable.
- `projectors.py` — beta_vae (control), deterministic_ae (P1a), beta_tcvae (P1b),
  mr_vae (P1d). P1c/WAE unported by decision.
- `process_residual.py` — P2a frozen SDXL-VAE cycle, caching hazard enforced in code.
- `test_discern_v2.py` — 23 tests including port-parity checks.
- D0 (explicit v1 restore) + D1-V / D1-M / D1-VM + D2 + D3 configs.

## Headline results

- **Recommended D1 manifold branch: P1b (β-TCVAE)** — best on all eight inverted rows, best
  standalone AUROC (0.858), largest oracle headroom. 🟡 Awaiting sign-off.
- **P1d loses privileged status** — its rate response adds +0.012 on forgery separability
  and is *negative* on both applicability questions.
- **D4 leans negative** — real oracle headroom (+0.055 to +0.094 BA) that an observable gate
  largely fails to recover, badly so on DF40.

## A DiCoME fix that was required

`experiments/common/analysis/export_features.py` could never export P1d. The pilot loader
deliberately pops `ae_operator` from `sys.modules` (pilots share that module name and would
shadow each other), and the beta-grid lookup then indexed that exact key — a guaranteed
`KeyError`, hit only by P1d since it is the only pilot with a rate response. It now resolves
`BETA_GRID` from the projector class's defining globals, which also removes the silent
variant where another pilot's restored module would have supplied the *wrong* grid. This is
the one change made outside the DISCERN repo; DiCoME's config and model code are untouched.

## Not done, deliberately

Per the explicit stop points: no D0–D5 training launched, no D4 fusion built, P1c unported,
and P3a / P4 / SBI / VALmix untouched.

## Outstanding

**FF++ threshold export** (approved, not yet run) — without it no threshold can come from
the permitted protocol source, so `frozen_threshold()` falls back to a documented 0.5 and
the threshold-free analyses carry every conclusion:

```
cd /data/umar/Repos/DiCoME
python experiments/common/analysis/export_features.py P0-DS --source FFpp
```

**Next runnable ladder rung: D0**, using `training/config/discern_v2/D0_v1_reproduction.yaml`.
