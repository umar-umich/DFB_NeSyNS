# CURRENT STATE — read this before anything else
July 16, 2026 · This one page is the decision ledger. Where any other document disagrees with this page, **this page wins.**

## What this project is (one paragraph)
A causal certification framework for deepfake detection, built inside the existing DeepfakeBench checkout. Frozen SOTA detectors (ForAda, Effort, FSFM, GenD) own the verdict and the accuracy numbers. An MLLM proposes candidate evidence claims in JSON. A dual-instrument counterfactual gate certifies each claim: spatial claims by repairing the cited region with paired-real pixels and re-querying the CLIP-family detector; spectral claims by frequency-domain interventions against a frequency detector. The output shows the detector's verdict plus ONLY certified claims, abstaining (`rationale_ungrounded`) when nothing certifies — that is the hallucination-free guarantee. The gate's labels train the proposer via offline DPO (LoRA, proposer only, strictly after the audit freeze) — the causal training signal is the paper's central novelty. Target venue: IEEE TIFS.

## Document map (what to read, what to ignore)
| File | Status |
|---|---|
| `CLAUDE_CODE_VEG_IMPLEMENTATION_v2.md` | **EXECUTE THIS.** The task list, gates, and component specs. |
| `VEG_glossary_and_examples.md` | Reference: full forms + the worked example (target pipeline behavior). Its Step-4 "graph nodes" wording is legacy; the record schema in Implementation v2 Task 7 governs. |
| `VEG_MASTER_PLAN_v2.md` | Background/context only. Superseded in parts (see ledger below). |
| `docs/PLANNING.md`, `docs/CODEBOOK.md` (rev-3, in repo) | Authoritative for predicate vocabulary, controls, claim scope, design rules. |
| Phase0/Phase1 files, Implementation v1, Protocol v1, Framework-SOTA doc | **STALE. Do not execute.** Kept for provenance only. |

## Decision ledger (everything decided after Master Plan v2)
1. **Venue:** TIFS primary (training arm included). WACV lean variant optional later. The 40-day WACV clock is gone; internal pace ~10–12 weeks.
2. **Training is IN:** offline DPO on gate labels (certified ≻ rejected), LoRA on the proposer only, after the audit freeze, gated by Pilot 1L (positive-pool size). Rule-4 amendment approved: detectors and instruments stay frozen; the proposer may be preference-tuned.
3. **Dual-instrument gate adopted:** motivated by Study 1B (CLIP detectors ignore spectral cues, Δp≈0, while spectral separates at AUC 0.94–0.99). Pilot S is the make-or-break gate; pre-committed fallback = single instrument + spectral as characterization.
4. **Lean cuts (all confirmed by Umar):**
   - Standalone claim parser → JSON-structured prompting + ~30-line validator (schema-validity ≥95% gate).
   - Evidence **graph → REMOVED entirely**, from paper and dissertation alike. At K≤4 claims per image the edges carry no information (contradiction = pairwise check; corroboration = counting; attribution = the 2R arm that failed to beat a CLIP probe). Claims live in per-image **Certified Evidence Records** (JSON) — the minimum storage read by the DPO builder, the output assembly, and the paper tables. One future-work sentence may note that video (temporal edges) could re-earn graph structure.
   - Symbolic solver → ~50-line output assembly (verdict passthrough; rationale = certified claims; abstention on empty). No Scallop/DeepProbLog.
   - Temperature calibration / ECE → struck from the critical path. Reporting rules instead: NM values are raw per-detector score drops, never probabilities, never compared across detectors; one limitations sentence.
   - Attribution (2R-wide) → removed (own pilot evidence).
5. **Naming:** framework will be renamed **CEC (Causal Evidence Certification)** or **CEV (Causal Evidence Verification)** — Umar decides; `veg/` stays as the working package name meanwhile; no "graph" language anywhere.
6. **Frozen anchors (do not re-derive):** CGT validated on 4 detectors, GT-repair drops 0.479 / 0.713 / 0.625 / 0.839; controls inert; inpainting and retrieval repair DEAD as verifiers; MLLM-as-verdict near chance (hence proposer-not-judge); 14/376 single-region necessity on full-face swaps (hence 1L gates DPO); FF++ edits family = abstention set (Option A).
7. **Locked params:** K=4 claims, order-of-listing, T=0 + fixed seed, $100 API cap, per-detector margins from pilot numbers (confirm once via 🟡, then frozen).

## Assets on the server
DeepfakeBench checkout with DISCERN implemented · FF++ and DF40 preprocessed (frames, landmarks, masks) · Phase-1 pilot results + harness in `results/{pilot, pilotrev3, pilot1_5, pilot2, pilot2r}` (PORT the harness, never rewrite) · `Repos/DeepfakeJudge` cloned · 2× H200, 148 GB each · Umar handles all downloads (🔴).

## First action
Execute Implementation v2, **Task 0 (recon)**: inventory detectors/weights, locate the pilot harness, map the preprocessed layouts — then STOP and report to Umar before writing any code.
