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

Evidence base: **8 eval sources** (FFpp, DF40, CDFv3, CDFv2, DFEval24, DFDC, DFDCP, DFD),
threshold **0.5110** frozen from FFpp EER.

- **D1 manifold branch: P1b (β-TCVAE)**, confirmed by three independent lines — wins all
  eight DF40 inverted rows, reproduces the ordering on CDFv3's inverted row
  (0.420 → 0.686, ahead of P2a and P1d), best standalone AUROC (0.8581), largest oracle
  headroom (+0.094 BA CDFv3).
- **D4 arm: P1d (MR-VAE)**, on the matched `D3_full_p1d.yaml` baseline. It is the only
  projector a gate can exploit, in both A2a and A2b.
- **P1d loses its *rate-response* justification** — +0.012 incremental on forgery
  separability, negative on both applicability questions, replicated on CDFv3. It stays in
  the D4 arm on gate-recovery evidence, not on R(x).
- **D4 is defensible, not negative** (revised). A gate trained only on FF++ and frozen
  recovers a mean **+0.106** of oracle headroom with P1d, positive on 5 of 7 OOD sources.
- **Build D4 with a conservative routing threshold.** A2c: tau = 0.95 cuts worst-source harm
  72%, halves routing to 47%, and costs only ~20% of the mean gain.
- **The gate harms the in-the-wild source.** DFEval24 is negative under every configuration
  tested; tau = 0.95 reduces it to −0.0066 BA but does not eliminate it. For a reliability
  paper this belongs in the abstract, not a footnote.
- **Scope limit:** per-generator inversion is only measurable on DF40 and CDFv3. The other
  five sources label everything as one method — a metadata limitation, not a null result.

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

**D0 is running** (launched 2026-08-15 02:38, GPU 0) — see `D0_v1_reproduction.yaml` for the
command and the verified v1 flag state. Its numbers become the ladder's reference once
frozen.

**D1–D3 cannot be launched.** The `discern_v2` branches are scaffolding: nothing under
`training/detectors/` calls `build_branches()`, so the D1/D2/D3 configs are flag manifests
exactly as D0's was. Running the ladder requires wiring the branch contract into the
detector and trainer — a real architectural change to the training path, deliberately not
started without an explicit go-ahead. **This is the single blocker for all remaining
D-ladder work.**

**D4** is unblocked on evidence but still needs a greenlight, and should be built at
tau ≈ 0.95 against `D3_full_p1d.yaml`.
