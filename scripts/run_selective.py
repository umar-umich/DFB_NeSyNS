"""
scripts/run_selective.py
========================
Selective-prediction comparison: full DeFakeNet vs. GenD-CLIP on CDFv2.

Reads two existing per-sample CSVs and computes:
  • Full-coverage AUC.
  • AUC at 90 % coverage (10 % abstention) — sort by confidence
    descending, keep the top 90 %, recompute AUC.

Both metrics are reported at the *video* level (mean ``prob_fake``
across the video's frames, mean confidence across the same frames).

Inputs (paths follow the spec):
    a) results/ablations/full_defakenet/per_sample.csv
    b) results/baselines/GenD-CLIP/CDFv2_per_sample.csv

Output (incremental):
    results/selective/CDFv2.json

Both per-sample CSVs are produced as a byproduct of the rest of the
pipeline:
  - (a) by ``scripts/run_ablation_eval.py``;
  - (b) by ``scripts/calibration_metrics.py`` (the canonical baseline
    aggregator). If file (b) is missing, this runner exits with a
    clear hint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import calibration_metrics as cm  # verbatim


def _aggregate_video(per_sample: pd.DataFrame
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (video_prob_fake, video_label, video_confidence) arrays.
    Confidence at video level = mean of frame-level confidences for
    that video (matching the convention in calibration_metrics.py).
    """
    grouped = per_sample.groupby('video_id', sort=False)
    prob = grouped['prob_fake'].mean().to_numpy()
    labels = grouped['label'].first().to_numpy()
    if 'confidence' in per_sample.columns:
        conf = grouped['confidence'].mean().to_numpy()
    else:
        conf = np.maximum(prob, 1.0 - prob)
    return prob, labels, conf


def _selective_auc(prob_fake: np.ndarray,
                   labels: np.ndarray,
                   confidence: np.ndarray,
                   coverage: float) -> tuple[float, int]:
    """Sort descending by confidence, keep the top ``coverage`` fraction,
    recompute AUC on the retained subset. Returns (auc, n_kept)."""
    n = len(prob_fake)
    if n == 0:
        return float('nan'), 0
    n_keep = max(1, int(round(coverage * n)))
    order = np.argsort(-confidence, kind='mergesort')[:n_keep]
    return cm.compute_auc(prob_fake[order], labels[order]), int(n_keep)


def _evaluate(name: str, csv_path: Path) -> dict:
    if not csv_path.exists():
        return {'method': name,
                'available': False,
                'reason': f'missing per-sample csv: {csv_path}'}
    df = pd.read_csv(csv_path)
    needed = {'video_id', 'label', 'prob_fake'}
    missing = needed - set(df.columns)
    if missing:
        return {'method': name,
                'available': False,
                'reason': f'csv missing columns {missing}'}

    v_prob, v_labels, v_conf = _aggregate_video(df)
    auc_full = cm.compute_auc(v_prob, v_labels)
    auc_90, n_keep = _selective_auc(v_prob, v_labels, v_conf, coverage=0.90)
    return {
        'method': name,
        'available': True,
        'csv_path': str(csv_path),
        'n_videos': int(len(v_prob)),
        'auc_full_coverage': auc_full,
        'auc_at_90pct_coverage': auc_90,
        'n_videos_at_90pct': n_keep,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--full-csv', type=Path,
                   default=REPO_ROOT / 'results' / 'ablations'
                                     / 'full_defakenet' / 'per_sample.csv')
    p.add_argument('--gendclip-csv', type=Path,
                   default=REPO_ROOT / 'results' / 'baselines'
                                     / 'GenD-CLIP' / 'CDFv2_per_sample.csv')
    p.add_argument('--out-path', type=Path,
                   default=REPO_ROOT / 'results' / 'selective' / 'CDFv2.json')
    args = p.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'dataset': 'CDFv2',
        'level': 'video',
        'methods': {
            'full_defakenet': _evaluate('full_defakenet', args.full_csv),
            'GenD-CLIP': _evaluate('GenD-CLIP', args.gendclip_csv),
        },
    }
    with open(args.out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[run_selective] wrote {args.out_path}', flush=True)
    for name, m in summary['methods'].items():
        if m.get('available'):
            print(f'  {name:18s} '
                  f'AUC@full={m["auc_full_coverage"]:.4f} '
                  f'AUC@0.9 ={m["auc_at_90pct_coverage"]:.4f} '
                  f'(n_videos={m["n_videos"]})', flush=True)
        else:
            print(f'  {name:18s} UNAVAILABLE — {m["reason"]}', flush=True)


if __name__ == '__main__':
    main()
