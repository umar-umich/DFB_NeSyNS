CONTEXT
Repo: DFB_NeSyNS. Phase 1/1.5 complete: probes p1-p4b done, CCV numerically
hardened, per-sample CSVs in logs/probes/*/per_sample_*.csv with per-frame
rows (video_id, frame_idx, label, p_fake, uncertainty, S_spatial, S_concept,
S_causal, p_fake_spatial, p_fake_concept, p_fake_causal, disagreement_d,
gates, 18 per-rule violation columns). Decisions: EDL kept; substrate stream
kept; causal branch = CCV (SCM archived as comparison-only); full-training
architecture below. GLOBAL RULES unchanged: never delete (attic/), never
touch configs/retained_predicates.yaml, every task ends with changed-files
summary + verification command. CRITICAL RULE: no code path may read OOD
test datasets during any training, threshold selection, or hyperparameter
choice. Thresholds/calibration use ONLY an FF++ train-fold split. Add an
assert/guard in any new calibration script that its input dataset == FF++.

TASK 1 — Rule-evidence layer (S1)
New concept_branch option evidence_head: 'mlp' (current) | 'rule_linear'.
rule_linear: Ev_rules = softplus(W @ nu + b), W shape (2, k), non-negativity
NOT forced on W (signed weights are the interpretation: positive-to-fake
rules vs positive-to-real). Substrate handled per substrate_mode as before;
when substrate_mode='both' and evidence_head='rule_linear', substrate goes
through its own small linear head and the two evidences ADD, so rule
contributions remain isolable. Expose per-rule contribution
(W[:,j] * nu[:,j]) in the branch output dict as 'rule_contributions' (B, k, 2).

TASK 2 — Vacuity-weighted IBDC (S2)
In NeSyEvidentialLoss._disagreement_calibration: compute per-branch vacuity
V_b = K/S_b and commitment q_b = 1 - V_b. Weight each pairwise term by
q_i*q_j: d = sum_{i<j} q_i q_j (1-cos(p_i,p_j)) / (sum_{i<j} q_i q_j + eps),
q detached along with d. Config flag edl.ibdc_version: 'v1'|'v2' (default v1
for reproducibility). Log mean q_b per branch as diagnostics.

TASK 3 — Symbolic-guided hard-sample weighting (S9)
Config flag training.symbolic_reweight: {enabled: false, factor: 2.0,
warmup_epochs: 8}. When enabled, after warmup: per training batch, compute
spatial-stream argmax and fused symbolic signal argmax (concept+causal
evidence sum); samples where spatial is WRONG and symbolic is RIGHT get loss
weight *= factor (applied to the fused EDL NLL term only). Uses labels
already in the batch — training data only. Log the per-epoch fraction of
reweighted samples.

TASK 4 — Full-training configs (training/config/detector/full/)
Common: nEpochs 100, early_stopping patience 20, tta.enabled false (single
tta:on eval pass allowed later, reported separately), test_dataset all six,
per-sample CSV logging on, seeds via CLI arg (3407, 42, 1024).
f1_lean:      spatial + EDL + concept(substrate_only, mlp head). No causal.
f2_full:      spatial + EDL + concept(both, rule_linear) + CCV + NeSy-EDL,
              ibdc_version v2, symbolic_reweight enabled.
f3_ibdc_v1:   f2_full but ibdc_version v1, symbolic_reweight disabled.
              (isolates S2+S9 jointly vs f2; that's accepted for now)
Sanity-load all three; print launch commands per seed.

TASK 5 — Dual uncertainty + post-hoc risk head (S4)
scripts/risk_head.py: reads per-sample video-level CSVs. Computes per-video
V (mean per-branch vacuity, commitment-weighted), C (q-weighted pairwise JS
divergence between branch probs), Q proxy (frame-score variance + mean
|p-0.5|), T (temporal instability: std of p_fake over frames). Trains
logistic regression R = sigma(w·[V,C,Q,T]+b) on FF++ leave-one-manipulation-
out error labels (train on 3 manip types' errors, validate on 4th), then
EVALUATES frozen R on OOD CSVs: error-prediction AUROC per dataset,
risk-coverage curves, E-AURC, coverage@risk. Outputs results/risk_report.md
+ figures. Enforce the FF++-only-fitting guard.

TASK 6 — Intervention audit + error analysis
scripts/audit.py: per dataset, video level: benefit rate (spatial wrong ->
fused right), harm rate (spatial right -> fused wrong), net correction;
broken down by dataset and (for FF++) manipulation type; per-rule breakdown
using rule_contributions where available.
scripts/error_analysis.py: for misclassified OOD videos (from existing
CSVs): distributions of V, C, R, top firing rules, FP vs FN separated.
Pure analysis — writes results/error_analysis.md. No mechanism may import
from this script.

TASK 7 — Risk-gated selective inference (S8)
Inference wrapper scripts/selective_inference.py: pass 1 = standard 32-frame
eval; videos with R >= tau (tau chosen on the FF++ calibration split at a
target coverage, e.g. 90%) get pass 2 = 64-frame dense resampling + re-eval;
final score = pass-2 result for escalated videos. Report AUC/ECE before vs
after escalation per dataset, and escalation rate. tau selection code must
run only on FF++ (guard from CRITICAL RULE).

TASK 8 — Structured Evidence Record (S7)
scripts/evidence_record.py: for a given video_id, emit JSON + markdown:
verdict, p_fake, V, C, R, decision (decide/escalate/defer), top-5
rule_contributions with names from retained_predicates.yaml, branch
evidence bars, CCV anomaly-group scores, counterfactual residual, temporal
profile (p_fake and d per frame). Optional --render flag: send the JSON to
a local/apised LLM with the instruction to write 4 sentences referencing
ONLY provided fields; then run a faithfulness check that every number in
the prose appears in the JSON (reject and fall back to template otherwise).

TASK 9 — Temporal ladder (Phase 3, data prep now)
scripts/temporal_ladder.py: from per-frame CSVs build per-video trajectory
tensors (p_fake, d, 18 rules, anomaly scores; ordered by frame_idx). Rung A:
frame-mean (reference). Rung B: transparent stats (mean, var, max, change-
point strength, persistence) + logistic head. Rung C: 1-layer GRU (hidden
32) over the sequence. Train B/C heads on FF++ CSVs only; evaluate frozen
on OOD CSVs. Report video AUC per rung per dataset.

TASK 10 — Metrics bug
In the eval metrics code, find why W1-sep_* columns were byte-identical to
W1-conf_* in pre-fix runs but differ post-fix. Fix or document; add a unit
test with synthetic scores where the two metrics must differ.