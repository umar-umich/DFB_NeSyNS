"""
scripts/run_ablation_eval.py
============================
Wrapper around the existing inference script (training/test.py) for a
single ablation checkpoint. Runs inference on CDFv2, normalises the
predictions to the canonical per-sample schema, and computes the
calibration / selective-prediction metrics defined in
``scripts/calibration_metrics.py`` (imported verbatim).

Outputs (incremental writes — no batched flush):
    results/ablations/{ablation-name}/per_sample.csv
    results/ablations/{ablation-name}/metrics.json

The runner reuses model loading and the inference loop from
``training/test.py`` via subprocess; it does not re-implement either.
The Dirichlet ``u = K/S`` is not surfaced in ``test_predictions.csv``
today (the inference loop drops it), so this runner does not record
per-sample uncertainty. Adding a column to ``run_inference()`` is the
one-line patch needed to expose it; left out here on purpose to keep
this wrapper non-invasive.

Usage:
    python scripts/run_ablation_eval.py \\
        --ablation-name full_defakenet \\
        --checkpoint-path checkpoints/defakenet_full.pth
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure scripts/ is on sys.path so calibration_metrics is importable.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import calibration_metrics as cm  # verbatim metric helpers


DEFAULT_DETECTOR_CFG = (
    REPO_ROOT / 'training' / 'config' / 'detector'
    / 'nesy_defake_ablation4_causal.yaml'
)
DEFAULT_TARGET_DATASET = 'Celeb-DF-v2'
CALIB_DATASET_TOKEN = 'CDFv2'  # canonical name used by calibration_metrics


def _run_inference(
    detector_cfg: Path,
    weights_path: Path,
    target_dataset: str,
    work_dir: Path,
) -> Path:
    """Invoke training/test.py via subprocess. Returns the directory
    containing test_predictions.csv for the requested dataset."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / 'training' / 'test.py'),
        '--detector_path', str(detector_cfg),
        '--weights_path', str(weights_path),
        '--test_dataset', target_dataset,
        '--output_dir', str(work_dir),
        '--no_interpretability',
    ]
    print(f'[run_ablation_eval] inference: {" ".join(cmd)}', flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(
            f'inference subprocess exited with code {proc.returncode}')
    pred_csv = work_dir / target_dataset / 'test_predictions.csv'
    if not pred_csv.exists():
        raise RuntimeError(
            f'expected predictions at {pred_csv} but file is missing')
    return pred_csv


def _metrics_block_with_aliases(prob_fake, labels, level: str) -> dict:
    """Run the verbatim metric block and tag the keys with the level."""
    block = cm.metrics_block(prob_fake, labels)
    return {
        f'n_{level}': int(block['n']),
        f'auc_{level}': block['auc'],
        f'acc_{level}': block['acc'],
        f'ece_{level}': block['ece'],
        f'cw_at_09_{level}': block['cw_at_09'],
        f'aurc_{level}': block['aurc'],
        f'eaurc_{level}': block['eaurc'],
        # Reliability data carried for downstream plotting (frame only)
        f'bin_edges_{level}': block['bin_edges'],
        f'bin_accuracies_{level}': block['bin_accuracies'],
        f'bin_confidences_{level}': block['bin_confidences'],
        f'bin_counts_{level}': block['bin_counts'],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--ablation-name', required=True)
    p.add_argument('--checkpoint-path', required=True, type=Path)
    p.add_argument('--detector-cfg', type=Path, default=DEFAULT_DETECTOR_CFG)
    p.add_argument('--target-dataset', default=DEFAULT_TARGET_DATASET,
                   help='Dataset name as listed in test_config.yaml '
                        '(default: Celeb-DF-v2).')
    p.add_argument('--results-dir', type=Path,
                   default=REPO_ROOT / 'results' / 'ablations')
    p.add_argument('--keep-inference-dir', action='store_true',
                   help='Keep the raw test.py output directory.')
    args = p.parse_args()

    out_dir = (args.results_dir / args.ablation_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    work_dir = out_dir / '_inference_raw'
    pred_csv = _run_inference(
        detector_cfg=args.detector_cfg,
        weights_path=args.checkpoint_path,
        target_dataset=args.target_dataset,
        work_dir=work_dir,
    )

    # ── Per-sample frame DataFrame in the canonical schema ──────────────
    per_sample = cm.predictions_to_per_sample(pred_csv, CALIB_DATASET_TOKEN)
    per_sample_csv = out_dir / 'per_sample.csv'
    per_sample.to_csv(per_sample_csv, index=False)
    print(f'[run_ablation_eval] wrote {per_sample_csv} '
          f'({len(per_sample)} frames, '
          f'{per_sample["video_id"].nunique()} videos)', flush=True)

    # ── Frame + video metrics ───────────────────────────────────────────
    prob_fake = per_sample['prob_fake'].to_numpy()
    labels = per_sample['label'].to_numpy()
    frame_block = _metrics_block_with_aliases(prob_fake, labels, 'frame')

    v_prob, v_labels, n_videos = cm.aggregate_video_level(per_sample)
    video_block = _metrics_block_with_aliases(v_prob, v_labels, 'video')
    video_block['n_video'] = int(n_videos)

    metrics = {
        'ablation': args.ablation_name,
        'dataset': CALIB_DATASET_TOKEN,
        'checkpoint': str(args.checkpoint_path),
        'detector_cfg': str(args.detector_cfg),
        'target_dataset_token': args.target_dataset,
        'n_frames': int(len(per_sample)),
        'n_videos': int(n_videos),
        **{k: v for k, v in frame_block.items()
           if not k.startswith('bin_')},
        **{k: v for k, v in video_block.items()
           if not k.startswith('bin_')},
        'reliability_frame': {
            'bin_edges': frame_block['bin_edges_frame'],
            'bin_accuracies': frame_block['bin_accuracies_frame'],
            'bin_confidences': frame_block['bin_confidences_frame'],
            'bin_counts': frame_block['bin_counts_frame'],
        },
    }

    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'[run_ablation_eval] wrote {metrics_path}', flush=True)

    if not args.keep_inference_dir and work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
