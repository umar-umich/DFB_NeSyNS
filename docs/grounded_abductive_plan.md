# Full Plan — "VLM Proposes, Forensics Disposes": Grounded Abductive Forgery Reasoning

**Provisional name:** PVR (Propose–Verify–Reason) over the Forgery-Evidence
Homogeneity (FEH) axis.
**Gate:** build this only if `killtest_FEH_spec.md` returns PASS. If FALSIFIED,
this document is void.
**Target venues:** WACV / AAAI (method), ACM TOMM special issue "Towards
Responsible and Explainable Multi-Modal Fusion" (the safety/explainability fit).

---

## 1. The design principle that fixes the last framework

The previous system was complex *and* underperformed because the symbolic
modules (SAE, DAGMA causal discovery, concept head, EDL/CMEF/PBAS/IBDC, evidence
fusion) sat **on the AUC critical path** — the classifier consumed their outputs.
Weak interpretable signal in the path → AUC down + complexity up.

**New wiring (one rule): decouple.**
- The **neural detector owns AUC** (your existing PE-Core/CLIP linear probe, GenD
  recipe — untouched).
- The **neuro-symbolic layer owns** family assignment (swap vs synthetic) +
  faithful explanation + a *bounded, calibrated correction* applied only where
  verified evidence is strong.
- Interpretability can therefore **never cost AUC**. Worst case: free
  explanations. Best case: grounded corrections raise cross-dataset AUC.

**Discarded from the old repo:** SAE, DAGMA causal discovery, concept head,
EDL/CMEF/PBAS/IBDC, evidence fusion. Not trusted, not detection-grade, not needed.

---

## 2. Why VLM + forensics, and why it's safe

Two orthogonal failure modes → one complementary system:
- Generative VLMs: rich semantics + localization, **near-chance detectors**,
  **hallucinate** rationales. (semantics, no metrology)
- Forensic probes: precise metrology, **no semantics**, brittle alone.
  (metrology, no semantics)

**Safety by construction:** every symbolic predicate must be confirmed by a
deterministic measurement. A VLM claim with no forensic confirmation is
**discarded** — so the system is robust to hallucination, and explanations are
**faithful by definition** (an explanation citing an artifact the probe did not
fire is a detectable bug, not a plausible story).

---

## 3. Architecture (3 stages, deliberately small)

```
            ┌─────────────────────────── NEURAL PATH (owns AUC) ───────────────────────────┐
 frame ───► │  PE-Core/CLIP backbone ─► linear probe ─► p_fake  (your existing detector)    │
            └──────────────────────────────────────────────────────────────────────────────┘
                                          │ (unchanged)
            ┌──────────────────────── NEURO-SYMBOLIC OVERLAY (owns family+proof) ───────────┐
 frame ───► │  PROPOSE  │  VERIFY                 │  REASON                                   │
            │  (offline │  per-region grounding   │  tiny fixed logic over verified preds     │
            │   VLM →   │  PE-Spatial scores pred │  → family ∈ {real,swap,synthetic}         │
            │   pred    │  + forensic probe       │  → proof = [(region,artifact,sim,measure)]│
            │   bank)   │  confirms → V map       │  → bounded correction to p_fake           │
            └──────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1 — PROPOSE (generative VLM, **offline, once**)
- Run Qwen2.5-VL / InternVL over a sample of train frames to **discover a compact,
  human-readable predicate bank**: ~30–60 `(artifact-type, natural-language
  description, swap/synthetic-leaning)` entries. This is the "unexplored VLM
  power" — the *vocabulary is learned, not hand-listed by us*.
- Dedup/cluster the generated predicates into the final bank. Inspect it. Freeze.
- **Not on the inference path** → reproducible, no API at test time.

### Stage 2 — VERIFY (PE-Spatial + forensic, runtime, fast, differentiable)
- For each region (from face-parsing, reuse `precompute_forensic_features.py`'s
  SegFormer `jonathandinu/face-parsing`):
  - **Ground:** PE-Spatial-L14 dense patch features · text-embedding of each
    predicate → per-region semantic score `S`.
  - **Measure:** the matching forensic probe from `forensic_helpers.py`
    (`boundary_gradient`, `dct_hf_energy`, `extract_srm`, `extract_spectral`,
    `laplacian_variance`, …) → magnitude `m`.
  - **Confirm:** predicate is TRUE iff `S>τ_s AND m>τ_m`. Output the verified
    evidence map `V` over regions/patches.

### Stage 3 — REASON (tiny logic, ~5 rules, no DAGMA/SAE)
- From `V`, compute FEH scalars (magnitude `μ`, concentration `C`) — same as the
  kill-test but now on *verified* evidence.
- Rules:
  - `¬high(μ)`                       → **real**
  - `high(μ) ∧ localized(C)`         → **face-swap**
  - `high(μ) ∧ diffuse(C)`           → **fully-synthetic**
- **Output:** family + **proof** (the confirmed `(region, artifact, S, m)` list) +
  a bounded correction `p_fake ← clip(p_fake + λ·g(V))`, where `g` is calibrated
  (Platt/conformal) and `λ` capped so the overlay cannot tank AUC.

Optional (only if ablation justifies): a swap-advocate vs synthetic-advocate
**debate**, adjudicated by which side's predicates actually fired.

---

## 4. Contributions, mechanisms, and falsification (fewer, defensible)

| # | Contribution | Mechanism | Experiment / metric | FALSIFIED if… |
|---|---|---|---|---|
| C1 | **Hallucination-safe, faithful explanations** | VLM claim gated by forensic measurement | **Faithfulness** = % of cited artifacts whose probe fired; vs a free-text VLM baseline (MARE/M2F2-Det-style) | A free-text VLM matches faithfulness, or gating doesn't reduce unfaithful citations |
| C2 | **Unified swap + synthetic, single training regime** | FEH axis (localized vs diffuse) over *verified* evidence | Single model, cross-dataset AUC on **both** families; swap-vs-synthetic AUC | A same-capacity neural baseline trained once matches on both families |
| C3 | **AUC-preserving interpretability** | decoupled wiring; bounded calibrated correction | AUC with vs without overlay (Δ≥0 required); per-dataset | Overlay reduces AUC on any held-out set |

C1 + C2 are the headline. C3 is the safety guarantee that makes the paper
credible. Drop the debate/agent entirely unless an ablation shows it lifts the
**ambiguous subset** specifically — do not ship "agentic" as decoration.

---

## 5. Experiments

**Train:** FF++ c23 (swap) + DiFF (synthetic) + FF++ real — *one* training run,
*one* model. **Cross-dataset test:** Celeb-DF-v2, DFDC/DFDCP, FaceShifter (swap);
DiffusionFace / DeepFakeFace (synthetic). Report per-family AUC + combined.

**Core tables/figures:**
1. FEH anchor figure (from kill-test, on verified evidence).
2. Cross-dataset AUC vs SOTA generalizable detectors (CLIP/PE linear probe,
   UNITE, Effort, a recent VLM-explainer).
3. Swap-vs-synthetic confusion (the unification claim).
4. Faithfulness metric vs free-text VLM baseline (C1).

**Ablations (each maps to a contribution):**
- − forensic verification (trust VLM) → tests C1.
- − semantic grounding (forensic only) → quantifies what VLM adds (Stage A vs B).
- − overlay (neural only) → tests C3 (AUC must not rise when overlay removed by
  more than calibration noise; overlay must not drop it).
- learned bank vs hand bank → justifies the generative-VLM offline step.
- − debate/agent (if included) → must help the ambiguous subset or it's cut.

---

## 6. Reuse vs new

**Reused:** PE-Core/CLIP backbone + linear probe (neural path); FAD DCT high-band
(`frequency_extractor.py`); forensic probes (`forensic_helpers.py`) now applied
**per VLM-proposed region**; SegFormer face-parsing
(`precompute_forensic_features.py`); dataset loaders/indices.

**New:** offline VLM predicate-bank discovery; PE-Spatial per-region grounding;
predicate verification gate; FEH reasoning layer; calibrated bounded correction;
faithfulness metric.

**Discarded:** §1 list (SAE, DAGMA, concept head, EDL/CMEF/PBAS/IBDC, fusion).

---

## 7. Risks & honest flags

- **C2 collapse:** if high-quality blended swaps read as "diffuse," the
  unification weakens. The kill-test surfaces this first; if it happens, reframe
  C2 around the *verified* (semantic+forensic) map, which still localizes a seam
  even when raw frequency is compressed-out.
- **Generative-VLM bank quality (speculative):** the learned predicate bank may
  contain vague/unmeasurable predicates. Mitigation: keep only predicates that
  have a matching forensic probe; this is a hard filter, not a hope.
- **Debate/agent = decoration risk:** gated behind its own ablation. Default OFF.

---

## 8. Build order
1. **Kill-test** (`killtest_FEH_spec.md`) — gate.  ← do this first
2. Offline VLM predicate-bank discovery + dedup + inspect.
3. PE-Spatial region grounding + forensic verification → `V` map.
4. FEH reasoning layer + calibrated correction.
5. Single-regime training + cross-dataset eval (tables 1–3).
6. Faithfulness metric + ablations (table 4, §5).
7. (Optional, gated) debate/agent.
