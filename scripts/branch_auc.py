"""
scripts/branch_auc.py
=====================
Offline per-branch standalone AUC from the per-sample diagnostic CSVs.

Each probe's test run writes `logs/test/<probe>_<timestamp>/per_sample_<dataset>.csv`
(see training/per_sample_logger.py) with, per frame:

    label, p_fake, p_fake_spatial, p_fake_concept, p_fake_causal, ...

This script reads those files *as they already are on disk* — no model rerun —
and reports, for every (run, dataset), the frame- and video-level ROC-AUC of
each branch's OWN Dirichlet-mean p_fake:

    fused    -> p_fake            (the model's fused prediction, for reference)
    spatial  -> p_fake_spatial
    concept  -> p_fake_concept
    causal   -> p_fake_causal

A branch that never ran (all-blank column) is skipped. Non-finite entries
(NaN / inf — e.g. the CCV causal branch on DFDC / DeepFakeDetection before the
numerical hardening) are dropped per-branch and counted, so a partially-broken
column still yields an AUC over its finite rows instead of erroring.

Video-level AUC aggregates each branch score by `video_id` (mean over that
video's finite frames), matching the video aggregate used elsewhere.

Usage (from the repo root):
    python scripts/branch_auc.py                       # scans logs/test/*
    python scripts/branch_auc.py --logs_dir logs/test  # explicit root
    python scripts/branch_auc.py --run logs/test/p4b_full_ccv_2026-07-30-...  # one run
    python scripts/branch_auc.py --out logs/probes_summary/branch_auc.csv
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

# branch label -> per-sample CSV column holding that branch's p_fake
BRANCHES = [
    ('fused',   'p_fake'),
    ('spatial', 'p_fake_spatial'),
    ('concept', 'p_fake_concept'),
    ('causal',  'p_fake_causal'),
]

_TS_RE = re.compile(r'_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$')


def probe_name(run_basename):
    """Strip the trailing _YYYY-MM-DD-HH-MM-SS timestamp to get the probe name."""
    return _TS_RE.sub('', run_basename)


def _f(x):
    """Parse a cell to float; blank / unparseable -> nan."""
    if x is None or x == '':
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def _auc(labels, scores):
    """ROC-AUC over finite scores; (auc, n, n_pos, n_dropped) — auc None if undefined."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores) & np.isfinite(labels)
    n_dropped = int((~finite).sum())
    labels, scores = labels[finite], scores[finite]
    n = len(labels)
    n_pos = int((labels == 1).sum())
    if n < 2 or n_pos == 0 or n_pos == n:
        return None, n, n_pos, n_dropped          # single-class or empty
    return float(roc_auc_score(labels, scores)), n, n_pos, n_dropped


def read_frame_csv(path):
    """Load a per_sample_<dataset>.csv -> (labels, {col: np.array}, video_ids)."""
    labels, vids = [], []
    cols = {c: [] for _, c in BRANCHES}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                labels.append(int(row['label']))
            except (KeyError, ValueError):
                continue
            vids.append(row.get('video_id', ''))
            for _, c in BRANCHES:
                cols[c].append(_f(row.get(c, '')))
    labels = np.asarray(labels, dtype=float)
    cols = {c: np.asarray(v, dtype=float) for c, v in cols.items()}
    return labels, cols, vids


def aggregate_video(labels, score, vids):
    """Mean each branch score over a video's finite frames -> (vlabels, vscores)."""
    acc = defaultdict(lambda: {'label': None, 'vals': []})
    for lab, sc, vid in zip(labels, score, vids):
        rec = acc[vid]
        rec['label'] = lab
        if np.isfinite(sc):
            rec['vals'].append(sc)
    vlabels, vscores = [], []
    for rec in acc.values():
        vlabels.append(rec['label'])
        vscores.append(np.mean(rec['vals']) if rec['vals'] else np.nan)
    return np.asarray(vlabels, dtype=float), np.asarray(vscores, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs_dir', default='logs/test',
                    help='root scanned for <run>/per_sample_<dataset>.csv')
    ap.add_argument('--run', default=None,
                    help='single run dir (overrides --logs_dir scan)')
    ap.add_argument('--out', default='logs/probes_summary/branch_auc.csv')
    args = ap.parse_args()

    if args.run:
        frame_csvs = sorted(glob.glob(os.path.join(args.run, 'per_sample_*.csv')))
    else:
        frame_csvs = sorted(glob.glob(
            os.path.join(args.logs_dir, '*', 'per_sample_*.csv')))
    # exclude the video-aggregate CSVs (per_sample_video_<dataset>.csv)
    frame_csvs = [p for p in frame_csvs
                  if not os.path.basename(p).startswith('per_sample_video_')]

    rows = []
    for path in frame_csvs:
        run_dir = os.path.basename(os.path.dirname(path))
        dataset = os.path.basename(path)[len('per_sample_'):-len('.csv')]
        labels, cols, vids = read_frame_csv(path)
        if labels.size == 0:
            continue
        for bname, col in BRANCHES:
            score = cols[col]
            if not np.isfinite(score).any():
                continue                    # branch absent (all blank)
            f_auc, f_n, f_pos, f_drop = _auc(labels, score)
            vlab, vsc = aggregate_video(labels, score, vids)
            v_auc, v_n, v_pos, v_drop = _auc(vlab, vsc)
            rows.append({
                'probe': probe_name(run_dir),
                'run_dir': run_dir,
                'dataset': dataset,
                'branch': bname,
                'frame_auc': '' if f_auc is None else f'{f_auc:.4f}',
                'video_auc': '' if v_auc is None else f'{v_auc:.4f}',
                'n_frames': f_n,
                'n_frames_nonfinite': f_drop,
                'n_videos': v_n,
                'n_videos_nonfinite': v_drop,
            })

    cols_out = ['probe', 'run_dir', 'dataset', 'branch', 'frame_auc', 'video_auc',
                'n_frames', 'n_frames_nonfinite', 'n_videos', 'n_videos_nonfinite']
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols_out)
        w.writeheader()
        w.writerows(rows)

    # console: grouped table
    print(f'Per-branch standalone AUC — {len(rows)} (probe x dataset x branch) rows')
    print(f'{"probe":18} {"dataset":18} {"branch":8} '
          f'{"frameAUC":>9} {"videoAUC":>9} {"nonfin(f/v)":>12}')
    for r in rows:
        nf = f"{r['n_frames_nonfinite']}/{r['n_videos_nonfinite']}"
        print(f'{r["probe"]:18} {r["dataset"]:18} {r["branch"]:8} '
              f'{r["frame_auc"] or "-":>9} {r["video_auc"] or "-":>9} {nf:>12}')
    print(f'\nWritten to {args.out}')


if __name__ == '__main__':
    main()
