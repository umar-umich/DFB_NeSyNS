"""
scripts/validate_predicates.py
==============================
Discriminative-gap predicate selection for the NeSy-DeFake symbolic stream.

Protocol
--------
1. Load FF++ training-fold precomputed fast-semantic features only.
2. Deterministic 90/10 video-level split (seed=42). The 10% slice is the
   *validation slice for predicate selection*. NEVER touched by detector
   training, NEVER touched by test or cross-dataset evaluation.
3. Apply the V8 candidate predicate set
   (``networks.nesy_defake.semantic.consistency_rules_v8``) to every
   frame in the validation slice.
4. Per predicate p, compute:
       gap_p   = |mean(v_p | y=fake) - mean(v_p | y=real)|
       auc_p   = single-predicate AUROC on the validation slice
       mean/std per class, sample counts
5. Emit ``results/predicate_gaps.csv``,
   ``results/predicate_gaps_sorted.csv``,
   ``results/predicate_gap_barplot.png``,
   ``results/predicate_gap_distribution.png``,
   ``results/run_log.txt``.

Leakage prevention (REPEAT — also in predicate_selection_decision.md):
The 10% validation slice for predicate selection is never used during
detector training; only the 90% training slice. No predicate gap is
recomputed on test or cross-dataset data at any point. The retained
set is fixed BEFORE the first test evaluation runs.

Reproducibility
---------------
- Splits use ``seed=42``.
- Each run logs its CLI invocation, git commit hash, and the dataset
  manifest hash to ``results/run_log.txt``.

Usage
-----
    python scripts/validate_predicates.py \\
        --ff_json /data/umar/Datasets/preprocessed/dataset_json/FaceForensics++.json \\
        --out_dir results
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Repo root → enable `networks.*` imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TRAINING_DIR = os.path.join(REPO_ROOT, 'training')
sys.path.insert(0, TRAINING_DIR)

from networks.nesy_defake.semantic.consistency_rules_v8 import (  # noqa: E402
    CrossAttributeConsistencyRulesV8,
    CANDIDATE_RULE_NAMES_V8,
    NUM_CANDIDATE_RULES_V8,
    RULE_CATEGORIES_V8,
)


# Class-name → binary label, mirrors training/config/train_config.yaml
LABEL_DICT = {
    'FF-real': 0,
    'FF-DF':   1, 'FF-F2F': 1, 'FF-FS': 1, 'FF-NT': 1,
}

logger = logging.getLogger('validate_predicates')


# ═══════════════════════════════════════════════════════════════════════
#  Path inference
# ═══════════════════════════════════════════════════════════════════════

def _fast_semantic_path(frame_path: str, video_id: str) -> str:
    """Map a frame path to its fast_semantic .pt file:
        .../<branch>/c23/frames/<vid>/<frame>.png
        →
        .../<branch>/c23/fast_semantic/<vid>.pt
    """
    # Walk up to the c23 dir, then descend to fast_semantic/<vid>.pt
    frames_idx = frame_path.find('/frames/')
    if frames_idx < 0:
        raise ValueError(f'Cannot locate /frames/ in {frame_path}')
    base = frame_path[:frames_idx]
    return os.path.join(base, 'fast_semantic', f'{video_id}.pt')


def _hash_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()[:16]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ['git', '-C', REPO_ROOT, 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return 'unknown'


# ═══════════════════════════════════════════════════════════════════════
#  Manifest loading
# ═══════════════════════════════════════════════════════════════════════

def load_train_manifest(ff_json: str) -> List[Tuple[str, str, str, int]]:
    """Return a list of (class_name, video_id, fast_semantic_path, label)
    for every video in the FF++ TRAIN split.
    """
    with open(ff_json) as f:
        d = json.load(f)
    ff = d['FaceForensics++']
    out: List[Tuple[str, str, str, int]] = []
    for cls in LABEL_DICT:
        train = ff[cls]['train']['c23']
        for vid_id, info in train.items():
            frames = info.get('frames') or []
            if not frames:
                continue
            try:
                pt_path = _fast_semantic_path(frames[0], vid_id)
            except ValueError:
                continue
            out.append((cls, vid_id, pt_path, LABEL_DICT[cls]))
    out.sort()  # deterministic order before splitting
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Deterministic 90/10 video-level split
# ═══════════════════════════════════════════════════════════════════════

def split_videos(records: List[Tuple[str, str, str, int]],
                 val_frac: float = 0.10,
                 seed: int = 42
                 ) -> Tuple[List, List]:
    """90/10 split *per class* so both slices stay class-balanced. Uses
    a deterministic permutation seeded with ``seed``.
    """
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List] = {}
    for r in records:
        by_class.setdefault(r[0], []).append(r)
    train, val = [], []
    for cls, recs in by_class.items():
        recs_sorted = sorted(recs, key=lambda x: x[1])  # by video_id
        idx = rng.permutation(len(recs_sorted))
        n_val = max(1, int(round(len(recs_sorted) * val_frac)))
        val_idx = set(idx[:n_val].tolist())
        for i, r in enumerate(recs_sorted):
            (val if i in val_idx else train).append(r)
    return train, val


# ═══════════════════════════════════════════════════════════════════════
#  Feature loading
# ═══════════════════════════════════════════════════════════════════════

def load_features(records, log_every: int = 200
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Load fast_semantic features for the supplied video records.

    Returns:
        feats: (N_frames_total, 58)
        labels: (N_frames_total,)
    """
    chunks_X, chunks_y = [], []
    n_missing = 0
    for i, (cls, vid, pt_path, lbl) in enumerate(records):
        if not os.path.exists(pt_path):
            n_missing += 1
            continue
        try:
            blob = torch.load(pt_path, map_location='cpu')
            feats = blob['features'] if isinstance(blob, dict) else blob
            if not isinstance(feats, torch.Tensor):
                continue
            feats = feats.float().numpy()
            if feats.ndim != 2 or feats.shape[1] != 58:
                continue
            chunks_X.append(feats)
            chunks_y.append(np.full(feats.shape[0], lbl, dtype=np.int64))
        except Exception as exc:
            logger.warning(f'Failed loading {pt_path}: {exc}')
            n_missing += 1
            continue
        if (i + 1) % log_every == 0:
            logger.info(f'  loaded {i + 1}/{len(records)} videos')
    if n_missing:
        logger.warning(f'{n_missing}/{len(records)} videos missing or '
                       f'unloadable.')
    if not chunks_X:
        raise RuntimeError('No features loaded — paths likely wrong.')
    return np.concatenate(chunks_X, axis=0), np.concatenate(chunks_y, axis=0)


# ═══════════════════════════════════════════════════════════════════════
#  Predicate evaluation
# ═══════════════════════════════════════════════════════════════════════

def compute_violations(feats: np.ndarray) -> np.ndarray:
    """Apply the v8 candidate set to a (N, 58) feature matrix and return
    a (N, 28) violation matrix.
    """
    mod = CrossAttributeConsistencyRulesV8().eval()
    with torch.no_grad():
        out = mod(torch.from_numpy(feats).float())
    return out.numpy()


def per_predicate_stats(violations: np.ndarray, labels: np.ndarray
                        ) -> List[dict]:
    """Compute per-predicate gap, AUC, mean, std, sample counts, and
    sanity flags.
    """
    from sklearn.metrics import roc_auc_score

    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())
    rows: List[dict] = []
    for j, name in enumerate(CANDIDATE_RULE_NAMES_V8):
        col = violations[:, j]
        real = col[labels == 0]
        fake = col[labels == 1]

        sanity = []
        if np.isnan(col).any():
            sanity.append('NaN')
        if np.all(col == col[0]):
            sanity.append('constant')
        elif col.max() < 1e-9:
            sanity.append('all_zero')
        sanity_str = '|'.join(sanity) if sanity else 'ok'

        m_real = float(real.mean()) if real.size else 0.0
        m_fake = float(fake.mean()) if fake.size else 0.0
        s_real = float(real.std())  if real.size else 0.0
        s_fake = float(fake.std())  if fake.size else 0.0
        gap = abs(m_fake - m_real)

        # Single-predicate AUC: higher violation = more likely fake?
        # We compute both directions and take the max (because some
        # predicates may have inverted polarity on the validation set).
        try:
            auc_pos = float(roc_auc_score(labels, col))
            auc = max(auc_pos, 1.0 - auc_pos)
        except Exception:
            auc = float('nan')

        rows.append(dict(
            predicate=name,
            category=RULE_CATEGORIES_V8.get(name, 'other'),
            mean_real=m_real, mean_fake=m_fake,
            std_real=s_real,  std_fake=s_fake,
            gap=gap, auc=auc,
            n_real=n_real, n_fake=n_fake,
            sanity=sanity_str,
        ))
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  CSV / plot output
# ═══════════════════════════════════════════════════════════════════════

CSV_COLS = [
    'predicate', 'category', 'mean_real', 'mean_fake',
    'std_real', 'std_fake', 'gap', 'auc',
    'n_real', 'n_fake', 'sanity',
]


def write_csv(rows: List[dict], path: str) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{r[k]:.6f}' if isinstance(r[k], float) else r[k])
                        for k in CSV_COLS})


def plot_barplot(rows: List[dict], save_path: str,
                 top_n_highlight: int = 18) -> None:
    rows_sorted = sorted(rows, key=lambda r: -r['gap'])
    names = [r['predicate'] for r in rows_sorted]
    gaps = np.array([r['gap'] for r in rows_sorted])
    colors = ['#1f77b4' if i < top_n_highlight else '#bbbbbb'
              for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(names)), 5.0))
    ax.bar(range(len(names)), gaps, color=colors,
           edgecolor='white', linewidth=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel(r'Discriminative gap $|\bar v_{\rm fake}-\bar v_{\rm real}|$',
                  fontsize=10)
    ax.set_title(f'Per-predicate discriminative gap '
                 f'(top-{top_n_highlight} retained → blue)', fontsize=11)
    for i, g in enumerate(gaps):
        ax.text(i, g + 0.005, f'{g:.3f}', ha='center', va='bottom',
                fontsize=6.5, color='#222')
    ax.set_ylim(0, gaps.max() * 1.18 + 1e-6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_distribution(rows: List[dict], save_path: str) -> None:
    gaps = np.array([r['gap'] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(gaps, bins=12, color='#1f77b4', edgecolor='white')
    ax.set_xlabel('Discriminative gap')
    ax.set_ylabel('# predicates')
    ax.set_title('Distribution of predicate gaps (FF++ val slice)',
                 fontsize=11)
    ax.axvline(np.median(gaps), color='red', ls='--', lw=1,
               label=f'median = {float(np.median(gaps)):.3f}')
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ff_json', type=str,
                        default='/data/umar/Datasets/preprocessed/'
                                'dataset_json/FaceForensics++.json')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(REPO_ROOT, 'results'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_frac', type=float, default=0.10)
    parser.add_argument('--top_highlight', type=int, default=18,
                        help='Highlight count in barplot (visual only).')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s')

    logger.info('Loading FF++ training-fold manifest …')
    records = load_train_manifest(args.ff_json)
    logger.info(f'  total train videos: {len(records)}')

    logger.info(f'Splitting (seed={args.seed}, val_frac={args.val_frac}) …')
    _, val_records = split_videos(records, args.val_frac, args.seed)
    logger.info(f'  val-slice videos: {len(val_records)}')

    logger.info('Loading val-slice features …')
    feats, labels = load_features(val_records)
    logger.info(f'  val-slice frames: {feats.shape[0]} '
                f'(real={int((labels==0).sum())}, '
                f'fake={int((labels==1).sum())})')

    logger.info('Computing violations under v8 candidate set …')
    viol = compute_violations(feats)
    logger.info(f'  violations: {viol.shape}')

    logger.info('Computing per-predicate stats …')
    rows = per_predicate_stats(viol, labels)
    rows_sorted = sorted(rows, key=lambda r: -r['gap'])

    csv_unsorted = os.path.join(args.out_dir, 'predicate_gaps.csv')
    csv_sorted   = os.path.join(args.out_dir, 'predicate_gaps_sorted.csv')
    write_csv(rows, csv_unsorted)
    write_csv(rows_sorted, csv_sorted)
    logger.info(f'  wrote {csv_unsorted} and {csv_sorted}')

    plot_barplot(rows, os.path.join(args.out_dir, 'predicate_gap_barplot.png'),
                 top_n_highlight=args.top_highlight)
    plot_distribution(rows, os.path.join(
        args.out_dir, 'predicate_gap_distribution.png'))
    logger.info('  wrote barplot + distribution PNGs')

    # Run log
    log_path = os.path.join(args.out_dir, 'run_log.txt')
    invocation = ' '.join(sys.argv)
    with open(log_path, 'a') as f:
        f.write('=' * 78 + '\n')
        f.write(f'timestamp:        {datetime.datetime.now().isoformat()}\n')
        f.write(f'invocation:       {invocation}\n')
        f.write(f'git_commit:       {_git_commit()}\n')
        f.write(f'manifest_path:    {args.ff_json}\n')
        f.write(f'manifest_sha256:  {_hash_file(args.ff_json)}\n')
        f.write(f'seed:             {args.seed}\n')
        f.write(f'val_frac:         {args.val_frac}\n')
        f.write(f'val_videos:       {len(val_records)}\n')
        f.write(f'val_frames:       {feats.shape[0]}\n')
        f.write(f'val_real_frames:  {int((labels==0).sum())}\n')
        f.write(f'val_fake_frames:  {int((labels==1).sum())}\n')
        f.write(f'n_candidates:     {NUM_CANDIDATE_RULES_V8}\n')
        f.write('top-5 by gap:\n')
        for r in rows_sorted[:5]:
            f.write(f'  {r["predicate"]:30s} gap={r["gap"]:.4f} '
                    f'auc={r["auc"]:.4f} cat={r["category"]}\n')
        f.write('bottom-5 by gap:\n')
        for r in rows_sorted[-5:]:
            f.write(f'  {r["predicate"]:30s} gap={r["gap"]:.4f} '
                    f'auc={r["auc"]:.4f} cat={r["category"]}\n')
        sanity_flagged = [r for r in rows if r['sanity'] != 'ok']
        f.write(f'sanity_flagged:   {len(sanity_flagged)}\n')
        for r in sanity_flagged:
            f.write(f'  {r["predicate"]} → {r["sanity"]}\n')
        f.write('\n')
    logger.info(f'  appended run record to {log_path}')

    # Stdout summary
    print('\n──────── Summary ────────')
    print(f'candidates : {NUM_CANDIDATE_RULES_V8}')
    print(f'val frames : {feats.shape[0]}')
    print('top-5 by gap:')
    for r in rows_sorted[:5]:
        print(f'  {r["predicate"]:30s} gap={r["gap"]:.4f}  '
              f'auc={r["auc"]:.4f}  cat={r["category"]}')


if __name__ == '__main__':
    main()
