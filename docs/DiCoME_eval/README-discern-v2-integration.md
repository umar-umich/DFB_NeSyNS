# DISCERN v2 Integration — Phase Tracker

**Date opened**: 2026-08-14
**Status**: DiCoME exploratory phase FROZEN. DISCERN v2 integration OPEN.
**Paper target**: T-BIOM journal extension of DISCERN (IJCB 2026).

---

## North star (the sentence that governs every decision below)

> We are no longer searching for the component with the highest average AUC. We are
> searching for complementary forensic representations whose regions of expertise can be
> identified from observable evidence and exploited without harming samples the primary
> detector already handles correctly.

Everything in this phase is in service of that claim. If an experiment does not help
establish or falsify it, it is out of scope for this cycle.

## Phase success criterion (explicit — not SOTA)

The bar for this phase is **not** beating a benchmark. It is:

> Do the manifold and process-residual mechanisms that showed complementary behavior as
> standalone pilots transfer into DISCERN **without destroying the strong CLIP visual
> baseline** — and can an *observable* gate recover a meaningful fraction of the ideal
> per-sample routing gain?

If the answer to the second half is no (oracle gain large but unrecoverable by a cheap
gate), that is a legitimate negative result and D4/D5 shrink to an honest "applicability
is not inferable here" section rather than a week of reasoner engineering.

---

## Three parallel tracks

The intellectual foundation (Track A) and the engineering (Track B) run **at the same
time**, not in sequence. Track C (writing) starts now because most of it does not depend
on final results. This is how we reconcile schedule pressure with the discipline of not
pre-selecting branches.

| Track | What it is | Blocks what | Owner split |
|---|---|---|---|
| **A — Analysis / selection gate** | complementarity + oracle + realizable-gate + rate-response re-confirm | Which manifold projector enters D1; go/no-go for D4/D5 | Claude Code writes, Umar runs |
| **B — Integration scaffolding + D-ladder** | branch interfaces, ported mechanisms, D0→D3 | The final system | Claude Code writes code; Umar runs D0–D5 |
| **C — Paper writing** | intro, related work, motivation, v1 limitations, pilot analysis, protocol | Nothing — proceeds in parallel | Umar (Claude assists on request) |

---

## Track A — the analysis that selects the branches

Do **not** commit the manifold projector or the process branch on aggregate AUC. Let
sample-level behavior decide. Track A produces three artifacts:

1. **Complementarity matrices** across `{P0, P1a, P1b, P1c, P1d, P2a, P3a, P4}`:
   per-sample **rescue** (P0 wrong → operator right), **harm** (P0 right → operator
   wrong), **error overlap**, and evidence correlation.

2. **Oracle headroom + realizable gate** (feasibility vs deployment):
   - **Oracle metrics are accuracy, balanced accuracy, union-of-correct coverage
     (`P(∃b: ŷ_b = y)`), and remaining shared-error rate** — not oracle-routed AUROC. A
     label-aware oracle makes AUROC conceptually awkward (the routed score depends on the
     label); if computed, it goes in supplementary, labeled *label-peeking / non-deployable*.
   - Realizable gate = a deliberately **simple** head (logistic / shallow MLP) on a
     **rich** feature state — evidence, vacuity, cross-branch conflict, specialist residual
     response. **Not P0 confidence alone**: P0 is *confidently wrong* on the inverted rows
     (danet 0.31, mcnet 0.38, tpsm 0.40). The gate itself reports AUROC normally.
   - **Two levels**, because they prove different claims:
     - **A2a feasibility** (this phase): leave-one-generator-out OOD CV. Answers "can
       observable evidence predict applicability at all?" This is the **Phase-1 go/no-go
       for building D4**.
     - **A2b deployment-valid** (spec now, run later): gate trained only on permitted
       protocol data (FF++ train / holdouts / approved augmentation), with **no generator
       ID, dataset ID, or family label as inputs**. A2a is an optimistic ceiling; A2b is
       the real deployment test.
   - **Gate-Recovery** (balanced-accuracy terms):
     `(BA_gate − BA_base) / (BA_oracle − BA_base)`, per candidate set, per held-out
     generator. Large oracle + high A2a recovery → build D4/D5. Large oracle + near-zero
     recovery → negative result.
   - **Threshold discipline**: one threshold from the permitted train/val protocol, frozen
     across all OOD sources — never tuned per OOD source (that leaks the target and breaks
     the zero-shot claim). If no single threshold works, report threshold-free score
     analyses alongside.

3. **Rate-response + UMAP on the P1 suite** (cheapest — run first). Evaluates `R(x)` on
   **three** questions, because clean family clusters are not the only thing that justifies
   P1d: (a) forgery separability (real/fake linear probe), (b) family structure
   (silhouette / Fisher), (c) complementarity / error-predictiveness (P0-error prediction,
   rescue/harm separation). **P1d survives if `R(x)` adds incremental forensic/applicability
   information on any of these — even without family clusters** (e.g. it may separate
   P0-correct from P0-error without naming the family). Handoff records family separation
   as already negative; this re-confirms and broadens the test rather than trusting memory.

**Selection logic**: the projector for D1 is chosen by rate-response incremental signal
and A2a gate-recovery per candidate — **not** unconditional AUC. Best standalone ≠ best
specialist for a gated architecture: P1b's +0.146 rescue is a live candidate precisely
because a working gate could suppress its harmful regime.

---

## Track B — integration scaffolding and the D-ladder

### Branch contract (all evidence branches output the same shape)

Branches output `evidence` + `features` + `diagnostics`; a **single shared DISCERN
utility** derives `α = e+1`, `S = Σα`, `p = α/S`, `u = K/S` centrally — not each branch on
its own — so DiCoME-ported and DISCERN-native branches can't drift. The public interface
still exposes all four:

```
evidence: [B, 2]      # Dirichlet evidence, real/fake
alpha:    [B, 2]      # derived centrally (α = e+1)
p:        [B, 2]      # derived centrally (α / S)
vacuity:  [B]         # derived centrally (K / S)
features: [...]       # diagnostic representation for analysis/gate
```

Three branch classes implement it: `VisualEvidenceBranch`, `ManifoldEvidenceBranch`
(swappable projector: `beta_vae | deterministic_ae | beta_tcvae | mr_vae` — `beta_vae` is
the original DiCoME β-VAE ported as the manifold **control**; the rest are P1a / P1b / P1d),
`ProcessEvidenceBranch` (P2a SDXL-VAE residual). Each branch is independently toggleable so
the D1-V / D1-M / D1-VM ablation is a config switch.

### First integrated architecture

```
CLIP Visual  +  Manifold Residual (swappable)  +  LDM-AE Process Residual  →  DISCERN EDL
```

No P3a, no P4 in the initial T-BIOM implementation.

### Integration ladder

| Step | Config | Isolates | Gate to advance |
|---|---|---|---|
| **D0** | DISCERN v1 reproduction (**explicit v1 flags**: `semantic_v1=true`, `structural_sem_v1=true`, `manifold_v2=false`, `process_v2=false`) | reference numbers | numbers frozen |
| **D1** | CLIP + Manifold, run as **D1-V** (visual only) / **D1-M** (manifold only) / **D1-VM** (both) | does e_man add to e_vis, or make the visual head redundant | manifold evidence non-degenerate; VM beats V on the inversion rows |
| **D2** | CLIP + Process (P2a) | does process residual survive integration | process evidence non-degenerate |
| **D3** | CLIP + Manifold + Process, ordinary EDL fusion | do the complements survive together | no collapse of the CLIP baseline |
| **D4** | D3 + applicability-aware fusion (q_man, q_proc) | can a gate keep the rescue without the heygen/conventional harm | **gated on A2a feasibility green** |
| **D5** | D4 + shared-ignorance-safe IBDC + risk/defer | reliability contribution | — |

D4 is **gated on Track A**: if Gate-Recovery is near zero, do not build it as a
performance claim — report the negative result and keep D3 as the honest system.

### Port, don't rewrite

**Port from DiCoME** → into the DISCERN repo: projector implementation, residual
construction, evidential head design (if useful), SDXL-VAE reconstruction preprocessing,
P2a residual extraction, analysis/export utilities. **Port from the validated pilot code
— do not reimplement from memory.**

**Keep from DISCERN**: data pipeline, CLIP branch, EDL abstraction, video aggregation,
losses/training framework, calibration evaluation, uncertainty, selective prediction,
final decision pipeline. This preserves the journal-extension lineage.

### Disable behind config flags (do not delete — needed for ablations)

```
semantic_v1        = false   # FACS predicate reasoning
structural_sem_v1  = false   # class-conditioned SEMs
handcrafted_freq   = false   # FFT / SRM / DCT features
ccv                = false   # CCV experiments
symbolic_rules_v1  = false   # old symbolic rules
manifold_v2        = true
process_v2         = true
```

---

## Parked / explicitly not this cycle

| Item | Disposition | Why |
|---|---|---|
| **P3a (LaRE²-inspired)** | Diagnostic / negative result, not a branch | SD 3.5 MMDiT predicts velocity on a 16-ch latent — the epsilon-residual LaRE² computes does not exist there. Scope any paper claim to "our SD3.5 velocity proxy did not help," or build a genuine epsilon-residual on SDXL later. |
| **P4 (LGrad)** | Park | Weights are ProGAN-objects + bedroom-StyleGAN, off-distribution for faces — result is inconclusive, not negative. Hunting/training face-domain weights is a separate project. |
| **SBI / blending** | Defer to one training ablation (D3 vs D3+SBI) | Given FS↔FR specialization, could reinforce swap artifacts and hurt heterogeneous generalization. Test after D3. |
| **VALmix full-suite retrain** | Defer to a controlled selection-protocol experiment | Retraining the whole suite changes the protocol and makes Deepfake-Eval-2024 no longer strictly zero-shot. Instead run a dedicated FF++-val-selection vs leak-free-OOD-selection experiment later; keep current comparisons. |
| **WACV side paper** | Shelved this cycle | Realistic scope 15–18 days; folds into T-BIOM related-work with no degradation. Do not let it dictate T-BIOM experiments. |

---

## Coding-agent contract

- 🟢 **YOU-RUN (Claude Code)**: code, configs, scripts, analysis, plots.
- 🔴 **UMAR-RUNS**: training runs (D0–D5), evaluation runs, dataset conversions.
- 🟡 **ASK-UMAR**: any change to hyperparameters, seed, eval sources, data paths, scope —
  and the **projector selection** for D1 (comes out of Track A).
- **No invented numbers** — every result cell filled from an actual run.
- **Port from validated pilot code, not from memory.**
- **Freeze before measuring** — never modify a checkpoint mid-evaluation.
- **Second seeds only for the finalists** (P0 + the chosen manifold + P2a), not the whole
  suite. Reserve for deltas that land inside the 0.005–0.015 single-seed band.

---

## Open gates / decisions

1. **Rate-response status** — resolved by Track A task A3 (re-confirmation). Working
   assumption per handoff: already negative. If A3 re-confirms, P1d loses privileged
   status.
2. **T-BIOM deadline** — needed to calibrate the Track C writing cadence and the
   second-seed budget. *(Awaiting Umar.)*
3. **DISCERN repo path** — v2 integration happens in the DISCERN repo (assumed
   `../DISCERN/` by convention, mirroring `../DiCoME/`). *Confirm before Track B lands
   files.*

---

## Table 4 finding to carry into the paper

Every pilot reaches 99% of peak in-domain val AUROC at epoch 0 and ends at train AUROC
1.0000; between-pilot peak spread 0.00093 vs within-pilot epoch std 0.00046 (S/N 2.0×).
**In-domain FF++ validation cannot rank cross-domain detectors.** For a paper whose
identity is *reliability*, selecting checkpoints on a procedure shown to be arbitrary is a
soft spot — the leak-free OOD selection protocol is a candidate contribution in its own
right. This is analysis/writing material, not a reason to retrain the suite now.