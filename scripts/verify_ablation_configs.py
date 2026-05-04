"""
scripts/verify_ablation_configs.py
==================================
Pre-flight check for the ablation YAML files under configs/ablations/.

For each ablation it asserts:
  (a) The frozen 18-rule retained set is referenced
      (`concept_branch.consistency_rules_version == v8_retained`
       and `retained_predicates_yaml` points at
       configs/retained_predicates.yaml — except where the symbolic
       stream is intentionally absent, e.g. visual_edl_only).
  (b) The component the ablation claims to disable is actually
      disabled in the YAML (correct lambda or flag).
  (c) Every other field matches the reference
      configs/ablations/full_defakenet_18rules.yaml — i.e. each
      ablation overrides only the minimum it should.

Outputs a summary table to stdout. Exits non-zero if any config fails.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ABLATIONS_DIR = REPO_ROOT / 'configs' / 'ablations'
RETAINED_YAML = REPO_ROOT / 'configs' / 'retained_predicates.yaml'
FULL_NAME = 'full_defakenet_18rules'

# Per-ablation expected diff vs. the full config. Each value is a list of
# (dotted_path, expected_value) pairs; a value of `cm.PRESENT` means the key
# must exist (not None / not missing) regardless of value.
PRESENT = object()
EXPECTED: Dict[str, List[Tuple[str, Any]]] = {
    'no_ibdc':         [('edl.disagreement_weight', 0.0)],
    'no_pbas':         [('edl.aux_weight', 0.0)],
    'no_cmef':         [('edl.cmef_disable_modulation', True)],
    'no_causal':       [('ablation_mode', 'concept_edl')],
    'no_symbolic':     [('concept_branch.disabled', True)],
    'visual_edl_only': [('ablation_mode', 'spatial_edl'),
                        ('edl.nesy_fusion', False),
                        ('edl.aux_weight', 0.0),
                        ('edl.disagreement_weight', 0.0)],
}

# Ablations where v8_retained is intentionally absent (no symbolic stream).
SKIP_RETAINED_CHECK = {'visual_edl_only'}


def _get(blob: dict, dotted: str, missing=None):
    cur = blob
    for k in dotted.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return missing
        cur = cur[k]
    return cur


def _flatten(blob: dict, prefix: str = '') -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in blob.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _check_retained(blob: dict) -> List[str]:
    errs: List[str] = []
    cb = blob.get('concept_branch') or {}
    if cb.get('consistency_rules_version') != 'v8_retained':
        errs.append("concept_branch.consistency_rules_version != 'v8_retained'")
    yaml_path = cb.get('retained_predicates_yaml')
    if not yaml_path:
        errs.append('concept_branch.retained_predicates_yaml is unset')
    else:
        # Resolve relative to repo root
        resolved = (REPO_ROOT / yaml_path).resolve()
        if resolved != RETAINED_YAML.resolve():
            errs.append(
                f'retained_predicates_yaml -> {resolved}, expected '
                f'{RETAINED_YAML}')
    if cb.get('rules_dim') != 18:
        errs.append(f'concept_branch.rules_dim != 18 '
                    f'(got {cb.get("rules_dim")})')
    return errs


def _check_overrides(blob: dict, expected: List[Tuple[str, Any]]) -> List[str]:
    errs: List[str] = []
    for path, want in expected:
        got = _get(blob, path, missing='__MISSING__')
        if want is PRESENT:
            if got == '__MISSING__' or got is None:
                errs.append(f'{path}: missing (expected non-None)')
        elif got != want:
            errs.append(f'{path}: got {got!r}, expected {want!r}')
    return errs


def _check_minimal_diff(full_blob: dict, ablation_blob: dict,
                        expected: List[Tuple[str, Any]],
                        is_retained_skipped: bool) -> List[str]:
    """Every key not listed in `expected` and not exempt should match the
    full config. Exemptions: the v8_retained substrate keys for ablations
    that also tweak rules_dim; the `disabled` flag on no_symbolic, etc.
    """
    full_flat = _flatten(full_blob)
    abl_flat = _flatten(ablation_blob)

    expected_keys = {p for p, _ in expected}
    # These keys are always allowed to differ (defaults / new flags / per-run knobs).
    always_allowed = {
        'concept_branch.disabled',
        'edl.cmef_disable_modulation',
        'mixed_precision',           # per-run AMP toggle, not architectural
    }
    if is_retained_skipped:
        # visual_edl_only: rules_dim / version / yaml not required
        always_allowed |= {
            'concept_branch.rules_dim',
            'concept_branch.consistency_rules_version',
            'concept_branch.retained_predicates_yaml',
        }

    errs: List[str] = []
    keys = set(full_flat) | set(abl_flat)
    for k in sorted(keys):
        if k in expected_keys or k in always_allowed:
            continue
        if full_flat.get(k) != abl_flat.get(k):
            errs.append(
                f'unexpected drift at {k}: full={full_flat.get(k)!r}, '
                f'ablation={abl_flat.get(k)!r}')
    return errs


def _row(name: str, status: str, problems: List[str]) -> str:
    cell = '✓' if status == 'ok' else ('—' if status == 'skip' else '✗')
    return f'  [{cell}] {name:<28s}  {", ".join(problems) if problems else status}'


def main() -> int:
    full_path = ABLATIONS_DIR / f'{FULL_NAME}.yaml'
    if not full_path.exists():
        print(f'ERROR: reference full config missing: {full_path}',
              file=sys.stderr)
        return 2
    full_blob = _load(full_path)

    # Ensure the retained YAML actually exists and references 18 rules.
    if not RETAINED_YAML.exists():
        print(f'ERROR: retained predicates YAML missing: {RETAINED_YAML}',
              file=sys.stderr)
        return 2
    retained_blob = _load(RETAINED_YAML)
    if int(retained_blob.get('retained_count', -1)) != 18:
        print(f'ERROR: {RETAINED_YAML} retained_count != 18', file=sys.stderr)
        return 2

    rows: List[str] = ['', '── Ablation config verification ──']
    failures: List[str] = []

    # The full config itself must reference v8_retained too.
    full_errs = _check_retained(full_blob)
    if full_errs:
        rows.append(_row(FULL_NAME, 'fail', full_errs))
        failures.append(FULL_NAME)
    else:
        rows.append(_row(FULL_NAME, 'ok', []))

    for name, expected in EXPECTED.items():
        path = ABLATIONS_DIR / f'{name}.yaml'
        if not path.exists():
            rows.append(_row(name, 'fail', [f'config file missing: {path}']))
            failures.append(name)
            continue
        blob = _load(path)
        problems: List[str] = []

        if name not in SKIP_RETAINED_CHECK:
            problems.extend(_check_retained(blob))

        problems.extend(_check_overrides(blob, expected))
        problems.extend(_check_minimal_diff(
            full_blob, blob, expected,
            is_retained_skipped=(name in SKIP_RETAINED_CHECK)))

        if problems:
            rows.append(_row(name, 'fail', problems))
            failures.append(name)
        else:
            rows.append(_row(name, 'ok', []))

    print('\n'.join(rows))
    print()
    if failures:
        print(f'FAILED: {len(failures)} config(s): {", ".join(failures)}')
        return 1
    print('All ablation configs verified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
