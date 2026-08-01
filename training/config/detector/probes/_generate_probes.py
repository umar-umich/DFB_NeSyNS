"""
Generate the 8 ablation probe configs from the reference config.

Source : configs/ablations/full_defakenet_18rules.yaml
Output : training/config/detector/probes/<name>.yaml  (this directory)

Each probe = the reference config + the GLOBAL probe overrides + the per-probe
deltas below. Re-run this script to regenerate all 8 deterministically:

    cd training && python config/detector/probes/_generate_probes.py
"""
import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
REFERENCE = os.path.join(REPO, 'configs', 'ablations', 'full_defakenet_18rules.yaml')

# Global overrides applied to every probe.
GLOBAL = {
    'nEpochs': 6,
    'save_epoch': 6,
    'test_dataset': ['FaceForensics++', 'Celeb-DF-v2'],
    'manualSeed': 3407,
    # nested (dotted) overrides handled by set_dotted():
    'early_stopping.enabled': False,
    'tta.enabled': False,
}

# Per-probe deltas. log_dir is filled in per name.
PROBES = {
    'p1_spatial_ce':   {'ablation_spatial_only': True,  'ablation_mode': 'spatial_ce'},
    'p2_spatial_edl':  {'ablation_spatial_only': False, 'ablation_mode': 'spatial_edl'},
    'p3_concept':      {'ablation_spatial_only': False, 'ablation_mode': 'concept_edl',
                        'concept_branch.substrate_mode': 'both'},
    'p3a_rules_only':  {'ablation_spatial_only': False, 'ablation_mode': 'concept_edl',
                        'concept_branch.substrate_mode': 'rules_only'},
    'p3b_substrate':   {'ablation_spatial_only': False, 'ablation_mode': 'concept_edl',
                        'concept_branch.substrate_mode': 'substrate_only'},
    'p4_full_scm':     {'ablation_spatial_only': False, 'ablation_mode': 'causal_edl',
                        'causal_branch.type': 'improved_scm'},
    'p4a_causal_only': {'ablation_spatial_only': False, 'ablation_mode': 'causal_edl',
                        'causal_branch.type': 'improved_scm',
                        'concept_branch.evidence_in_fusion': False},
    'p4b_full_ccv':    {'ablation_spatial_only': False, 'ablation_mode': 'causal_edl',
                        'causal_branch.type': 'ccv'},
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
    set_dotted(cfg, 'log_dir', f'logs/probes/{name}')
    for k, v in deltas.items():
        set_dotted(cfg, k, v)
    return cfg


def main():
    header = (
        "# =============================================================================\n"
        "# PROBE CONFIG (generated) — {name}\n"
        "# Derived from configs/ablations/full_defakenet_18rules.yaml by\n"
        "# training/config/detector/probes/_generate_probes.py. Do not hand-edit;\n"
        "# change the generator and re-run.\n"
        "# Deltas: {deltas}\n"
        "# Global: nEpochs=6, save_epoch=6, early_stopping.enabled=false,\n"
        "#         tta.enabled=false, manualSeed=3407,\n"
        "#         test_dataset=[FaceForensics++, Celeb-DF-v2], log_dir=logs/probes/{name}\n"
        "# =============================================================================\n")
    for name, deltas in PROBES.items():
        cfg = build(name, deltas)
        path = os.path.join(HERE, f'{name}.yaml')
        with open(path, 'w') as f:
            f.write(header.format(name=name, deltas=deltas))
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
