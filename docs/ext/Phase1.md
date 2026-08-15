CONTEXT
Repo: DFB_NeSyNS (DeepfakeBench-derived deepfake detection). Key files:
detectors/nesy_defake_detector.py (NeSyDeFakeHybridDetector),
networks/nesy_defake/{concept_branch.py, improved_scm_branch.py, ccv_branch.py,
causal_branch_factory.py, fusion.py, losses/nesy_edl_loss.py,
semantic/consistency_rules_v8.py, foundation_models/*},
training/config/detector/full_defakenet_18rules.yaml (reference config),
configs/retained_predicates.yaml (frozen 18-predicate set — NEVER modify).

GLOBAL RULES
- Never delete files. Move dead code to attic/ preserving relative paths.
- Never modify configs/retained_predicates.yaml or consistency_rules_v8.py logic.
- Every change must keep the existing full_defakenet_18rules.yaml runnable.
- After each task, print a summary of files changed and a verification command.

TASK 1 — Archive dead code
Move to attic/: foundation_models/frequency_feature_extractor.py,
foundation_models/temporal_feature_extractor.py, SimplifiedCausalBranch
(split out of concept_branch.py; keep ConceptBranch), and any imports of
them. Remove the frequency: block from the reference YAML (active_branches
is ['spatial']). Fix all resulting import errors.
Verify: python -c "from detectors.nesy_defake_detector import NeSyDeFakeHybridDetector"

TASK 2 — ConceptBranch substrate_mode
Add config option concept_branch.substrate_mode: one of
'both' (default, current behavior: concat [x_sem || violations]),
'rules_only' (input = violations only, input_dim = rules_dim),
'substrate_only' (input = x_sem only, input_dim = combined_dim;
violations still computed and returned for logging/SCM use, just
excluded from the MLP input). predicate_mask must keep working in all modes.

TASK 3 — Concept evidence kill-switch
Add config option concept_branch.evidence_in_fusion: true|false (default true).
When false: the concept branch still runs (its violations still feed the
ImprovedSCM identity sub-graph), but its evidence is excluded from
EvidenceFusion AND from PBAS branch_evidences AND from the IBDC pair set.
This isolates the concept branch's *evidence* contribution from its
violations-as-SCM-input role.

TASK 4 — Per-sample CSV logging
In the test/eval loop, dump one CSV per (run, dataset) to
logs/<run_name>/per_sample_<dataset>.csv with columns:
video_id, frame_idx, label, p_fake, uncertainty, S_spatial, S_concept,
S_causal (Dirichlet strengths; empty if branch absent), p_fake_spatial,
p_fake_concept, p_fake_causal (per-branch Dirichlet means), disagreement_d
(pairwise cosine, same formula as IBDC), concept_gate, causal_gate.
Also write a video-level aggregate CSV (mean over frames).
This must work for every ablation_mode.

TASK 5 — Probe configs
Generate 8 configs in training/config/detector/probes/, each derived from
full_defakenet_18rules.yaml with these GLOBAL overrides:
nEpochs: 6, early_stopping.enabled: false, tta.enabled: false,
save_epoch: 6, test_dataset: [FaceForensics++, Celeb-DF-v2],
manualSeed: 3407, log_dir: logs/probes/<name>.
Per-probe deltas:
p1_spatial_ce:   ablation_spatial_only: true,  ablation_mode: spatial_ce
p2_spatial_edl:  ablation_mode: spatial_edl
p3_concept:      ablation_mode: concept_edl, substrate_mode: both
p3a_rules_only:  ablation_mode: concept_edl, substrate_mode: rules_only
p3b_substrate:   ablation_mode: concept_edl, substrate_mode: substrate_only
p4_full_scm:     ablation_mode: causal_edl, causal_branch.type: improved_scm
p4a_causal_only: ablation_mode: causal_edl, improved_scm,
                 concept_branch.evidence_in_fusion: false
p4b_full_ccv:    ablation_mode: causal_edl, causal_branch.type: ccv
Sanity-check every config loads and the model constructs (dims: 58-d
substrate, 18 rules, 83-d forensic; identity SCM input = 32+26+18 = 76).

TASK 6 — Run manifest
Write probes/README.md: one launch command per probe, expected runtime,
and a results table skeleton (probe | FF++ frame AUC | CDFv2 frame AUC |
CDFv2 video AUC) to fill in by hand.