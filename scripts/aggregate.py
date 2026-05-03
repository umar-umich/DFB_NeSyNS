"""
scripts/aggregate.py
====================
Read every ablation / faithfulness / selective JSON produced by the
upstream runners and emit ``results/ABLATION_SUMMARY.md`` in the
paper-ready three-table format.

Inputs:
    results/ablations/<name>/metrics.json     (one per ablation)
    results/faithfulness/CDFv2.json
    results/selective/CDFv2.json

Output:
    results/ABLATION_SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Display labels and the order they appear in Table 4. The first row is
# the baseline (GenD-CLIP); it has no metrics.json — its numbers come
# from the GenD-CLIP per-method summary if present.
COMPONENT_TABLE_ORDER = [
    ('GenD-CLIP (visual + softmax)',         None),
    ('Visual stream + EDL only',             'visual_edl_only'),
    ('DeFakeNet - IBDC',                     'no_ibdc'),
    ('DeFakeNet - CMEF',                     'no_cmef'),
    ('DeFakeNet - PBAS',                     'no_pbas'),
    ('DeFakeNet - Causal stream',            'no_causal'),
    ('DeFakeNet - Symbolic stream',          'no_symbolic'),
    ('Full DeFakeNet',                       'full_defakenet'),
]


def _fmt(v, nd=4, scale=1.0, missing='—'):
    if v is None:
        return missing
    try:
        if v != v:           # NaN
            return missing
    except TypeError:
        return missing
    return f'{v * scale:.{nd}f}'


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _gend_clip_video_metrics(repo: Path) -> dict | None:
    """Pull the published / measured GenD-CLIP video-level numbers from
    ``results/baselines/GenD-CLIP/CDFv2_metrics.json`` if available.
    """
    p = repo / 'results' / 'baselines' / 'GenD-CLIP' / 'CDFv2_metrics.json'
    blob = _load_json(p)
    if blob is None or 'video_level' not in blob:
        return None
    v = blob['video_level']
    return {
        'auc_video': v.get('auc_video'),
        'ece_video': v.get('ece_video'),
        'eaurc_video': v.get('eaurc_video'),
        'cw_at_09_video': v.get('cw_at_09_video'),
    }


def _component_row(label: str, blob: dict | None) -> str:
    if blob is None:
        return f'| {label:<37s} | —     | —      | —           | —      |'
    auc   = blob.get('auc_video')
    ece   = blob.get('ece_video')
    eaurc = blob.get('eaurc_video')
    cw    = blob.get('cw_at_09_video')
    return (
        f'| {label:<37s} | '
        f'{_fmt(auc,   nd=2, scale=100):>5} | '
        f'{_fmt(ece,   nd=4):<6} | '
        f'{_fmt(eaurc, nd=2, scale=100):<11} | '
        f'{_fmt(cw,    nd=4):<6} |'
    )


def _build_component_table(repo: Path, ablations_dir: Path) -> list[str]:
    rows = [
        '## Component Ablation (Table 4)',
        '| Setting                              | AUC   | ECE    | E-AURC x100 | CW@0.9 |',
        '|--------------------------------------|-------|--------|-------------|--------|',
    ]
    gend = _gend_clip_video_metrics(repo)
    for label, key in COMPONENT_TABLE_ORDER:
        if key is None:
            rows.append(_component_row(label, gend))
        else:
            blob = _load_json(ablations_dir / key / 'metrics.json')
            rows.append(_component_row(label, blob))
    rows.append('')
    return rows


def _build_faithfulness_table(faith_path: Path) -> list[str]:
    rows = [
        '## Faithfulness (Table 5)',
        '| Intervention             | k=1   | k=3   | k=5   | Flip rate (%) |',
        '|--------------------------|-------|-------|-------|---------------|',
    ]
    blob = _load_json(faith_path)
    if blob is None:
        rows.append('| Random predicates        | —     | —     | —     | —             |')
        rows.append('| Top-k firing (cited)     | —     | —     | —     | —             |')
        rows.append('')
        return rows
    metrics = blob.get('metrics', {})

    def cell(kind: str, k: int, key: str, scale: float = 1.0) -> str:
        m = metrics.get(f'{kind}_k{k}')
        if not m:
            return '—'
        v = m.get(key)
        return _fmt(v, nd=2, scale=scale)

    def avg_flip(kind: str) -> str:
        vals = [metrics.get(f'{kind}_k{k}', {}).get('flip_rate_pct')
                for k in (1, 3, 5)]
        vals = [v for v in vals if v is not None]
        if not vals:
            return '—'
        return f'{sum(vals) / len(vals):.2f}'

    # The "k=1/k=3/k=5" cells report drop_rate (Ev_sym > 50 % drop, in %).
    rows.append(
        f'| Random predicates        | '
        f'{cell("random", 1, "drop_rate_pct"):<5} | '
        f'{cell("random", 3, "drop_rate_pct"):<5} | '
        f'{cell("random", 5, "drop_rate_pct"):<5} | '
        f'{avg_flip("random"):<13} |')
    rows.append(
        f'| Top-k firing (cited)     | '
        f'{cell("topk", 1, "drop_rate_pct"):<5} | '
        f'{cell("topk", 3, "drop_rate_pct"):<5} | '
        f'{cell("topk", 5, "drop_rate_pct"):<5} | '
        f'{avg_flip("topk"):<13} |')
    rows.append('')
    return rows


def _build_selective_table(sel_path: Path,
                           gend_video_blob: dict | None) -> list[str]:
    rows = [
        '## Selective Prediction (10% abstention on CDFv2)',
        '| Method                         | Full coverage AUC | At 90% coverage AUC |',
        '|--------------------------------|-------------------|---------------------|',
    ]
    blob = _load_json(sel_path) or {}
    methods = (blob.get('methods') or {})
    full = methods.get('full_defakenet') or {}
    gend = methods.get('GenD-CLIP') or {}

    rows.append(
        f'| Full DeFakeNet                 | '
        f'{_fmt(full.get("auc_full_coverage"), nd=2, scale=100):<17} | '
        f'{_fmt(full.get("auc_at_90pct_coverage"), nd=2, scale=100):<19} |')

    # GenD-CLIP full-coverage AUC: prefer the value computed by
    # selective JSON; fall back to the GenD baseline metrics file.
    full_cov = gend.get('auc_full_coverage')
    if full_cov is None and gend_video_blob is not None:
        full_cov = gend_video_blob.get('auc_video')
    rows.append(
        f'| GenD-CLIP softmax              | '
        f'{_fmt(full_cov, nd=2, scale=100):<17} | '
        f'{_fmt(gend.get("auc_at_90pct_coverage"), nd=2, scale=100):<19} |')
    rows.append('')
    return rows


def _build_runlog(ablations_dir: Path,
                  faith_path: Path,
                  sel_path: Path) -> list[str]:
    rows = ['## Run log']
    if ablations_dir.exists():
        for sub in sorted(ablations_dir.iterdir()):
            mp = sub / 'metrics.json'
            if mp.exists():
                rows.append(f'- `{sub.name}` ✓ metrics.json present')
            else:
                rows.append(f'- `{sub.name}` ✗ metrics.json MISSING')
    rows.append(f'- faithfulness  '
                f'{"✓" if faith_path.exists() else "✗"} '
                f'{faith_path}')
    rows.append(f'- selective     '
                f'{"✓" if sel_path.exists() else "✗"} '
                f'{sel_path}')
    rows.append('')
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, default=REPO_ROOT)
    p.add_argument('--ablations-dir', type=Path,
                   default=REPO_ROOT / 'results' / 'ablations')
    p.add_argument('--faithfulness-json', type=Path,
                   default=REPO_ROOT / 'results' / 'faithfulness' / 'CDFv2.json')
    p.add_argument('--selective-json', type=Path,
                   default=REPO_ROOT / 'results' / 'selective' / 'CDFv2.json')
    p.add_argument('--out-path', type=Path,
                   default=REPO_ROOT / 'results' / 'ABLATION_SUMMARY.md')
    args = p.parse_args()

    lines = ['# Ablation Results - DeFakeNet on CDFv2 (video-level)', '']
    lines += _build_component_table(args.repo, args.ablations_dir)
    lines += _build_faithfulness_table(args.faithfulness_json)
    gend_video = _gend_clip_video_metrics(args.repo)
    lines += _build_selective_table(args.selective_json, gend_video)
    lines += _build_runlog(args.ablations_dir, args.faithfulness_json,
                           args.selective_json)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text('\n'.join(lines))
    print(f'[aggregate] wrote {args.out_path}', flush=True)


if __name__ == '__main__':
    main()
