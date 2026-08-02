"""
scripts/compute_feature_stats.py  —  Phase-3 Task 1 helper.

Computes the per-feature standard deviation of the symbolic inputs
(precomputed_attrs, 58-d; forensic_features, 83-d) over the **FF++ train split**
and freezes it to an .npz that FeatureAugment loads. This makes the training-only
Gaussian noise scale-correct per feature dimension.

CRITICAL RULE guard: the config's train_dataset must be FaceForensics++ only —
the std is a training-time statistic and must never see OOD data.

Usage (repo root, dfb_nesy):
    python scripts/compute_feature_stats.py \
        --config training/config/detector/full/f2_full.yaml \
        --out configs/feature_stats/ff_train_feature_std.npz \
        [--max_batches 0]     # 0 = use the whole train split
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'training'))
from dataset.nesy_defake_dataset import NeSyDeFakeDataset

FF = 'FaceForensics++'


def _is_ff_only(train_ds):
    if isinstance(train_ds, str):
        return train_ds == FF
    if isinstance(train_ds, (list, tuple)):
        return list(train_ds) == [FF]
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',
                    default='training/config/detector/full/f2_full.yaml')
    ap.add_argument('--out',
                    default='configs/feature_stats/ff_train_feature_std.npz')
    ap.add_argument('--max_batches', type=int, default=0,
                    help='0 = whole train split; else cap for a quick estimate')
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    # Mirror train.py's config assembly: merge train_config.yaml (dataset paths,
    # dataset_json_folder, lmdb, ...). It carries no train_dataset, so the
    # detector config's FF++ train split is preserved.
    tc = 'training/config/train_config.yaml'
    if os.path.exists(tc):
        config.update(yaml.safe_load(open(tc)))

    # ── FF++-only guard ───────────────────────────────────────────────────
    if not _is_ff_only(config.get('train_dataset')):
        raise SystemExit(
            f"[GUARD] feature std must be computed on {FF} train ONLY; "
            f"config train_dataset={config.get('train_dataset')!r}.")

    # Do not let augmentation perturb the statistic we're measuring.
    config.setdefault('training', {}).setdefault('feature_augment', {})
    config['training']['feature_augment']['enabled'] = False

    loader = NeSyDeFakeDataset.prepare_data_loader(config, mode='train')

    attrs_sum = attrs_sqsum = None
    fore_sum = fore_sqsum = None
    n = 0
    for i, batch in enumerate(loader):
        if args.max_batches and i >= args.max_batches:
            break
        a = batch.get('precomputed_attrs')
        f = batch.get('forensic_features')
        if a is None or f is None:
            continue
        a = a.float().view(a.shape[0], -1).cpu().numpy()
        f = f.float().view(f.shape[0], -1).cpu().numpy()
        if attrs_sum is None:
            attrs_sum = np.zeros(a.shape[1]); attrs_sqsum = np.zeros(a.shape[1])
            fore_sum = np.zeros(f.shape[1]);  fore_sqsum = np.zeros(f.shape[1])
        attrs_sum += a.sum(0); attrs_sqsum += (a ** 2).sum(0)
        fore_sum += f.sum(0);  fore_sqsum += (f ** 2).sum(0)
        n += a.shape[0]
        if i % 20 == 0:
            print(f'  batch {i}  frames={n}', flush=True)

    if n == 0:
        raise SystemExit("no frames with precomputed_attrs/forensic_features found")

    attrs_std = np.sqrt(np.maximum(attrs_sqsum / n - (attrs_sum / n) ** 2, 0))
    fore_std = np.sqrt(np.maximum(fore_sqsum / n - (fore_sum / n) ** 2, 0))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out, attrs_std=attrs_std.astype(np.float32),
             forensic_std=fore_std.astype(np.float32),
             n_frames=np.int64(n), source=FF + ':train')
    print(f'\nFF++ train frames: {n}')
    print(f'attrs_std (58): min={attrs_std.min():.4g} max={attrs_std.max():.4g} '
          f'mean={attrs_std.mean():.4g}')
    print(f'forensic_std (83): min={fore_std.min():.4g} max={fore_std.max():.4g} '
          f'mean={fore_std.mean():.4g}')
    print(f'Written {args.out}')


if __name__ == '__main__':
    main()
