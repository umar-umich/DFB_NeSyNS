"""
Generate the 3 full-training configs from the reference config.

Source : configs/ablations/full_defakenet_18rules.yaml
Output : training/config/detector/full/<name>.yaml  (this directory)

Each config = reference + GLOBAL full-training overrides + per-config deltas.
Re-run to regenerate all three deterministically:

    cd training && python config/detector/full/_generate_full.py

Configs (Phase-2 Task 4):
  f1_lean     spatial + EDL + concept(substrate_only, mlp head). No causal.
  f2_full     spatial + EDL + concept(both, rule_linear) + CCV + NeSy-EDL,
              ibdc_version v2, symbolic_reweight enabled.
  f3_ibdc_v1  f2_full but ibdc_version v1, symbolic_reweight disabled
              (isolates S2+S9 jointly vs f2).

Seeds are passed at launch via `train.py --seed {3407,42,1024}` (overrides
manualSeed); the file default is 3407.
"""
import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
REFERENCE = os.path.join(REPO, 'configs', 'ablations', 'full_defakenet_18rules.yaml')

FF = 'FaceForensics++'
ALL_SIX = ['FaceForensics++', 'Celeb-DF-v2', 'Celeb-DF-v3',
           'DeepFakeDetection', 'DFDC', 'DFDCP']

# Global overrides applied to every full config.
#
# NOTE (2026-08-04): training-time eval / early-stopping runs on a small
# VALIDATION set = FaceForensics++ + Celeb-DF-v2. The trainer evaluates
# `test_dataset` every epoch and early-stops on the average AUC:
#   - FF++ alone would let the model overfit to the training distribution, so we
#     add ONE cross-dataset (Celeb-DF-v2) to select for generalization;
#   - averaging in ALL the OOD sets instead would both bias every OOD number and
#     waste most of each epoch on ~335k OOD frames (DFDC alone is 132k).
# Celeb-DF-v2 is therefore a VALIDATION set (used for selection), not a clean
# held-out number; the other four sets (CDFv3, DeepFakeDetection, DFDC, DFDCP)
# stay held-out and are evaluated POST-HOC via test.py --test_dataset (see
# full/README.md "Evaluate on all six").
VAL_DATASETS = [FF, 'Celeb-DF-v2']
GLOBAL = {
    'nEpochs': 100,
    'manualSeed': 3407,               # overridden per-run via `--seed`
    'test_dataset': VAL_DATASETS,     # training-time eval/early-stop validation
    # nested (dotted) overrides handled by set_dotted():
    'early_stopping.enabled': True,
    'early_stopping.patience': 20,
    'tta.enabled': False,
}

# Per-config deltas. log_dir is filled in per name.
CONFIGS = {
    'f1_lean': {
        'ablation_spatial_only': False,
        'ablation_mode': 'concept_edl',
        'concept_branch.substrate_mode': 'substrate_only',
        'concept_branch.evidence_head': 'mlp',
        'causal_branch.enabled': False,
        'edl.nesy_fusion': True,
    },
    'f2_full': {
        'ablation_spatial_only': False,
        'ablation_mode': 'causal_edl',
        'concept_branch.substrate_mode': 'both',
        'concept_branch.evidence_head': 'rule_linear',
        'causal_branch.enabled': True,
        'causal_branch.type': 'ccv',
        'edl.nesy_fusion': True,
        'edl.ibdc_version': 'v2',
        'training.symbolic_reweight.enabled': True,
        'training.symbolic_reweight.factor': 2.0,
        'training.symbolic_reweight.warmup_epochs': 8,
        'training.feature_augment.enabled': True,
        'training.feature_augment.gauss_std': 0.05,
        'training.feature_augment.feature_dropout': 0.1,
        'training.feature_augment.std_file':
            'configs/feature_stats/ff_train_feature_std.npz',
    },
    # f2_ibdc_v3: exact copy of f2_full with the one-sided-hinge IBDC (v3).
    'f2_ibdc_v3': {
        'ablation_spatial_only': False,
        'ablation_mode': 'causal_edl',
        'concept_branch.substrate_mode': 'both',
        'concept_branch.evidence_head': 'rule_linear',
        'causal_branch.enabled': True,
        'causal_branch.type': 'ccv',
        'edl.nesy_fusion': True,
        'edl.ibdc_version': 'v3',
        'training.symbolic_reweight.enabled': True,
        'training.symbolic_reweight.factor': 2.0,
        'training.symbolic_reweight.warmup_epochs': 8,
        'training.feature_augment.enabled': True,
        'training.feature_augment.gauss_std': 0.05,
        'training.feature_augment.feature_dropout': 0.1,
        'training.feature_augment.std_file':
            'configs/feature_stats/ff_train_feature_std.npz',
    },
    # f2_noaug: exact copy of f2_full with feature augmentation OFF (nothing
    # else different) — the T1 ablation control.
    'f2_noaug': {
        'ablation_spatial_only': False,
        'ablation_mode': 'causal_edl',
        'concept_branch.substrate_mode': 'both',
        'concept_branch.evidence_head': 'rule_linear',
        'causal_branch.enabled': True,
        'causal_branch.type': 'ccv',
        'edl.nesy_fusion': True,
        'edl.ibdc_version': 'v2',
        'training.symbolic_reweight.enabled': True,
        'training.symbolic_reweight.factor': 2.0,
        'training.symbolic_reweight.warmup_epochs': 8,
        'training.feature_augment.enabled': False,
        'training.feature_augment.gauss_std': 0.05,
        'training.feature_augment.feature_dropout': 0.1,
        'training.feature_augment.std_file':
            'configs/feature_stats/ff_train_feature_std.npz',
    },
    'f3_ibdc_v1': {
        'ablation_spatial_only': False,
        'ablation_mode': 'causal_edl',
        'concept_branch.substrate_mode': 'both',
        'concept_branch.evidence_head': 'rule_linear',
        'causal_branch.enabled': True,
        'causal_branch.type': 'ccv',
        'edl.nesy_fusion': True,
        'edl.ibdc_version': 'v1',
        'training.symbolic_reweight.enabled': False,
        'training.symbolic_reweight.factor': 2.0,
        'training.symbolic_reweight.warmup_epochs': 8,
    },
}


def set_dotted(cfg: dict, key: str, value):
    """Set a possibly-dotted key ('a.b.c') into a nested dict, creating dicts."""
    parts = key.split('.')
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise TypeError(f"cannot descend into non-dict at {p} for key {key}")
    node[parts[-1]] = value


def build(name: str, deltas: dict) -> dict:
    with open(REFERENCE) as f:
        cfg = yaml.safe_load(f)
    for k, v in GLOBAL.items():
        set_dotted(cfg, k, v)
    set_dotted(cfg, 'log_dir', f'logs/full/{name}')
    for k, v in deltas.items():
        set_dotted(cfg, k, v)
    return cfg


def main():
    header = (
        "# =============================================================================\n"
        "# FULL-TRAINING CONFIG (generated) — {name}\n"
        "# Derived from configs/ablations/full_defakenet_18rules.yaml by\n"
        "# training/config/detector/full/_generate_full.py. Do not hand-edit;\n"
        "# change the generator and re-run.\n"
        "# Deltas: {deltas}\n"
        "# Global: nEpochs=100, early_stopping.patience=20, tta.enabled=false,\n"
        "#         test_dataset=all six, log_dir=logs/full/{name}.\n"
        "#         Seed per run via `train.py --seed {{3407,42,1024}}`.\n"
        "# =============================================================================\n")
    for name, deltas in CONFIGS.items():
        cfg = build(name, deltas)
        path = os.path.join(HERE, f'{name}.yaml')
        with open(path, 'w') as f:
            f.write(header.format(name=name, deltas=deltas))
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
