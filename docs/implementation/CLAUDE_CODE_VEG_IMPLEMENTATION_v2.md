# CLAUDE_CODE — VEG Lean Implementation inside DeepfakeBench (v2, supersedes v1)
**Read fully first. Companions: `VEG_MASTER_PLAN_v2.md`, `VEG_glossary_and_examples.md` (worked example = target behavior), `docs/PLANNING.md` + `docs/CODEBOOK.md` (authoritative vocab, controls, claim scope).**

**Environment:** DeepfakeBench checkout on server; DISCERN implemented inside it; FF++ and DF40 preprocessed (frames, landmarks, masks). 2× H200 148 GB. Phase-1 pilot results in `results/{pilot,pilotrev3,pilot1_5,pilot2,pilot2r}` — the harness that produced them is the canonical repair/control/spectral code: **PORT, never rewrite.** `Repos/DeepfakeJudge` cloned.

**Naming:** the framework will be renamed **CEC (Causal Evidence Certification)** or **CEV (Causal Evidence Verification)** — decision pending. Until Umar picks, keep `veg/` as the working package name; use no "graph" language anywhere in code, comments, or outputs (the graph was removed as a component; claims live in per-image Certified Evidence Records).

## The four components (everything else was cut deliberately — do not resurrect)
```
1. DETECTORS  frozen SOTA (ForAda, Effort, FSFM, GenD) via wrapper      → verdict [ACCURACY]
2. PROPOSER   MLLM, JSON-structured output (no separate parser)         → candidate claims
3. GATE       dual-instrument CGT (spatial: CLIP-det repair; spectral:
              freq-det interventions)                                   → certified/rejected/untestable + NM
4. ASSEMBLY   ~50 lines: detector verdict + certified claims only;
              abstain (`rationale_ungrounded`) when nothing certifies   [HALLUCINATION-FREE GUARANTEE]
TRAINING ARM  gate labels → DPO preferences → LoRA-tune the PROPOSER only, after audit freeze
              [HALLUCINATION-FREE AT THE ROOT]
```
**Cut and why (answer reviewers/lab from this):** standalone parser → JSON prompting + validator (we control the prompt). Evidence graph → per-image **Certified Evidence Record** (JSON): the minimum storage read by DPO builder, assembly, and paper tables; cross-claim graph reasoning = dissertation track. Symbolic solver → assembly function (verdict is passthrough; rationale is the certified list; abstention is an emptiness check). Temperature calibration → struck from critical path (monotone rescale; can't change certify/reject; thresholds were pilot-derived on raw scores). **Recorded reporting rules (put in params.yaml and obey in all outputs):** NM values are raw per-detector score drops; never present as probabilities; never compare or average across detectors; one limitations sentence in the paper.

## Operating contract
🟡 [ASK-UMAR] decisions · 🔴 [UMAR-RUNS] heavy jobs (downloads, full-split runs, DPO) · 🟢 [YOU-RUN] quick work. Status line after every task. Log all decisions/gates in `veg/LOG.md` (UTC). Invariants: **freeze before you measure · training only after the audit freeze · every component earns a number · no invented numbers.**

## Layout
```
veg/
  LOG.md
  registration/   frozen_prompts/ · params.yaml · detectors.yaml · pins.yaml · seeds
  data/pairing.py
  masks/regions.py
  instruments/spatial.py · instruments/spectral.py
  repair/repair.py · repair/controls.py
  proposer/infer.py · proposer/validate.py
  gate/certify.py                 # → records/
  records/                        # per-image Certified Evidence Records (JSONL)
  assembly/output.py
  dpo/build_prefs.py · dpo/train_dpo.py
  eval/metrics.py · eval/tables.py · eval/figures.py
  pilots/pilot_s/ · pilots/pilot_1l/ · pilots/pilot_d/
  scripts/                        # one job = one CLI
```

## TASK 0 — Recon (🟢 first, code later)
(1) DeepfakeBench registry: which of ForAda/Effort/FSFM/GenD exist here with weights; how to call single-image inference programmatically (reuse their config+transforms, not the CLI). → `registration/detectors.yaml`. (2) Locate Phase-1 harness source (repair, controls, spectral interventions, re-alignment). (3) Preprocessed layout: FF++ frame/landmark/mask patterns + compression levels; DF40 families on disk. (4) 🟡 present recon summary; missing detectors/harness are blockers.

## TASK 1 — Registration + pins
🟢 `params.yaml`: K=4 claims, order-of-listing, T=0 + seed, control battery spec, per-detector certify margins + control-inertness thresholds (source: pilot numbers; 🟡 confirm the two threshold values once, then frozen), $100 API cap, the NM reporting rules above. `frozen_prompts/`: **one** proposer prompt = Type-B content + embedded codebook vocab + strict JSON output schema; verdict re-query prompt. 🟡 pin the **proposer** (2 candidates, InternVL/Qwen-VL class, VRAM estimates for one H200). 🔴 download.

## TASK 2 — Detector wrapper (calibration struck)
🟢 `instruments/spatial.py`: `SpatialInstrument.p_fake(img)->float` + batched variant, wrapping each frozen detector through DeepfakeBench's own preprocessing/normalization — a mismatched transform silently changes every number. Smoke: known fake ≫ known real per detector.

## TASK 3 — Pairing + masks
🟢 `data/pairing.py`: FF++ `<method>/<id1_id2>/<frame>` → `original/<id1>/<frame>`; port the pilot re-alignment; `verify_pair()`. 🟢 `masks/regions.py`: location vocab → binary masks from **existing DeepfakeBench landmarks** (eyes; nose+mouth; jaw/neck band; inner-face union; background=complement) + codebook post-processing. BiSeNet upgrade only if Pilot 1L shows landmark regions too coarse (🟡 then).

## TASK 4 — Repair + controls (PORT; acceptance test mandatory)
🟢 Port Poisson-blend repair + three controls (LPIPS-matched corruption, wrong-region, real-repair offset). **Acceptance:** reproduce effort 257_420/762 Δtrue=+0.659, Δctrl=−0.001 within tolerance. 🔴 run it; paste the Δs. No downstream work until this passes.

## TASK 5 — Spectral instrument + PILOT S
🟢 `instruments/spectral.py`: port Study-1B interventions (notch, checkerboard suppression, residual renormalization) + freq-detector wrapper. 🟡 propose 2 public-weight candidates (NPR-class, FreqNet-class) + fallback (linear probe on DISCERN spectral bank; rule-4 amendment already approved). 🔴 download + smoke. **PILOT S** on ~100 fakes + 100 reals: (i) freq detector separates on our crops (AUC), (ii) responds to spectral interventions (Δ ≫ 0) where CLIP detectors ≈ 0. 🔴 run. Gate: pass → dual-instrument; fail → 🟡 fallback probe or single-instrument + characterization (pre-committed).

## TASK 6 — Proposer (JSON out) + validator
🟢 `proposer/infer.py`: HF inference, frozen prompt, T=0, seed; disk cache keyed (image, model, prompt-hash) — interventions×models is the cost center, cache everything. 🟢 `proposer/validate.py` (~30 lines): JSON well-formed; artifact ∈ vocab (else OTHER); location maps to a mask (else null→spectral-or-untestable). **Gate:** schema-validity ≥95% on a 100-image probe; one recorded prompt revision allowed. 🔴 probe run.

## TASK 7 — Certification gate → Certified Evidence Records
🟢 `gate/certify.py`, per (image, claim): route (spatial→mask+repair vs CLIP det; whole-face spectral→intervention vs freq det; else UNTESTABLE) → Δ = p_fake(orig) − p_fake(intervened) on the matched instrument → controls → label CERTIFIED (Δ≥margin ∧ controls inert) / REJECTED / UNTESTABLE. Emit per-image record:
```json
{"image":"FFpp/DF/371_367/000","detector":"forada","p_fake":0.918,
 "claims":[{"id":"c1","artifact":"Blending Artifacts","location":"jaw",
   "label":"CERTIFIED","NM":0.78,"controls":{"corr":0.02,"wrong":0.01,"offset":0.03},
   "instrument":"forada"}, ...],
 "provenance":{"proposer":"<pin>","prompt_hash":"...","seed":17}}
```
Resumable, deterministic, per-image JSONL under `records/`. This file is what DPO, assembly, and every table read.

## TASK 8 — Output assembly (+ the guarantee)
🟢 `assembly/output.py` (~50 lines): verdict = detector (passthrough, never modified); rationale = CERTIFIED claims verbalized from templates; empty → `rationale_ungrounded` abstention exactly as the glossary Step-6 blocks; UNTESTABLE count surfaced. At inference the spectral branch may live-check spectral claims (no paired real needed). Optional post-v1: ≤10-pair contradiction flags (🟡 before adding).

## TASK 9 — Audit run + FREEZE
🟢 `scripts/run_audit.py`: per proposer-model: 30-image control re-validation first (**gate:** offset<15% ∧ true-region ≥ corruption+20pp, else model dropped+reported) → N_shared → N_full; 10-image timing probe → 🔴 hour estimates → 🔴 full run. **FREEZE:** hash `records/`, commit, log. Tasks 10–11 blocked until this commit exists.

## TASK 10 — PILOT 1L (gates DPO)
🟢+🔴 On localized manipulations (FF++ NT; DF40 localized slices if on disk): region-level repair recovers ≥70% of pixel-mask NM? Also yields the count of CERTIFIED-capable claims on localized data (the DPO positive pool). *(Attribution/2R and graph structure: REMOVED from paper and dissertation alike — pilot 2R evidence; do not build. One future-work sentence only.)*

## TASK 11 — Training arm (the novelty; only after freeze, gated by 1L)
🟢 `dpo/build_prefs.py`: records → pairs `{prompt: <image+frozen prompt>, chosen: <CERTIFIED claim text/JSON>, rejected: <REJECTED claim>}`, same image; report pair counts + positive-source breakdown (localized vs spectral). 🟡 if positives scarce. 🟢 `dpo/train_dpo.py`: TRL `DPOTrainer`, LoRA (r=16, attn+MLP targets, β≈0.1 start), bf16, one H200; config frozen in registration. 🔴 train (~hours). **Re-certify (PILOT D):** tuned proposer on held-out through the untouched gate. 🔴 run; paste: certification pass-rate before/after · claim diversity across predicate families (detector-mimic check) · region P/R vs GT masks · optional standalone-verdict accuracy of the MLLM before/after (bonus row, not headline). Any honest outcome publishes; fallback pre-committed.

## TASK 12 — Eval outputs (lean set)
`eval/`: T2 validation (frozen numbers) · T3 detector AUC via DeepfakeBench's own test harness (inherited, disclaimed) · T4 audit table (per proposer: certified/rejected/untestable rates, raw-score NM per detector) · T5 correlations (certification rate vs CHAIR via TriDF protocol; vs DeepfakeJudge-7B scores from `Repos/DeepfakeJudge`) · T6 DPO before/after · F1 strips (reuse pilot plotting) · F4 spectral disconnect · F5 certification-rate bars · qualitative certified-report figures. Every number traceable to `records/` + commit hash.

## Order
0 → 1 → 2 → 3 → 4(acceptance) → 5(Pilot S gate) → 6(validity gate) → 7 → 8 → 9(FREEZE) → 10(1L gate) → 11(Pilot D) → 12 throughout.

## If you lose the thread
Re-read master plan + `docs/PLANNING.md` + `veg/LOG.md`. Spine: **detector decides · MLLM proposes · two instruments certify counterfactually · assembly shows only certified evidence and abstains otherwise · DPO teaches the proposer to cite load-bearing evidence · nothing trained ever touches a verdict.** When in doubt, [ASK-UMAR].
