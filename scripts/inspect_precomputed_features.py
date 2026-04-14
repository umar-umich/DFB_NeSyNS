#!/usr/bin/env python3
"""
inspect_precomputed_features.py
================================
Load precomputed .pt files and report:
  1. Which feature columns are all-zero (dead features)
  2. Per-feature statistics (mean, std, min, max, % zeros)
  3. Original vs augmented sample comparison (augmentation invariance check)

Usage:
  python scripts/inspect_precomputed_features.py \
      --fast_semantic /data/umar/Datasets/preprocessed/FaceForensics++/original_sequences/youtube/c23/fast_semantic/001.pt \
      --fast_semantic_aug /data/umar/Datasets/preprocessed/FaceForensics++/original_sequences/youtube/c23/fast_semantic/001_aug1.pt \
      --forensic /data/umar/Datasets/preprocessed/FaceForensics++/original_sequences/youtube/c23/forensic_features/001.pt \
      --forensic_aug /data/umar/Datasets/preprocessed/FaceForensics++/original_sequences/youtube/c23/forensic_features/001_aug1.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

# ── Feature name registries ──────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))

from networks.nesy_defake.semantic.refined_attributes import FAST_FEATURE_NAMES
from forensic_helpers import FEATURE_NAMES as FORENSIC_FEATURE_NAMES


def load_pt(path):
    """Load a .pt file and return the features tensor + frame paths."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        features = data.get('features', None)
        frame_paths = data.get('frame_paths', [])
    elif isinstance(data, torch.Tensor):
        features = data
        frame_paths = []
    else:
        raise ValueError(f"Unexpected .pt format: {type(data)}")
    return features.float(), frame_paths


def analyze_features(features, feature_names, label):
    """Print per-feature statistics for a (N, D) feature tensor."""
    N, D = features.shape
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  Shape: ({N} frames, {D} features)")
    print(f"{'='*80}")

    if D != len(feature_names):
        print(f"  WARNING: {D} features but {len(feature_names)} names — "
              f"using indices for missing names")
        while len(feature_names) < D:
            feature_names = list(feature_names) + [f"feat_{len(feature_names)}"]

    # Gather stats
    dead_features = []
    active_features = []

    print(f"\n  {'Idx':>4}  {'Feature Name':<35}  {'Mean':>9}  {'Std':>9}  "
          f"{'Min':>9}  {'Max':>9}  {'%Zero':>6}  {'Status'}")
    print(f"  {'─'*4}  {'─'*35}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*6}  {'─'*10}")

    for i in range(D):
        col = features[:, i]
        mean = col.mean().item()
        std = col.std().item()
        mn = col.min().item()
        mx = col.max().item()
        pct_zero = (col == 0).float().mean().item() * 100

        # Determine status
        all_zero = (col == 0).all().item()
        all_same = (std < 1e-8)

        if all_zero:
            status = "DEAD"
            dead_features.append((i, feature_names[i]))
        elif all_same:
            status = "CONST"
            dead_features.append((i, feature_names[i]))
        else:
            status = "OK"
            active_features.append((i, feature_names[i]))

        # Color-code for terminal
        marker = "  " if status == "OK" else ">>"
        print(f"{marker}[{i:>3}]  {feature_names[i]:<35}  {mean:>9.4f}  {std:>9.4f}  "
              f"{mn:>9.4f}  {mx:>9.4f}  {pct_zero:>5.1f}%  {status}")

    print(f"\n  Summary: {len(active_features)}/{D} active, "
          f"{len(dead_features)}/{D} dead/constant")

    if dead_features:
        print(f"\n  Dead/constant features:")
        for idx, name in dead_features:
            print(f"    [{idx:>3}] {name}")

    return dead_features, active_features


def compare_original_vs_augmented(feat_orig, feat_aug, feature_names, label):
    """Compare original vs augmented features to check augmentation invariance."""
    N = min(feat_orig.shape[0], feat_aug.shape[0])
    D = feat_orig.shape[1]

    feat_orig = feat_orig[:N]
    feat_aug = feat_aug[:N]

    print(f"\n{'='*80}")
    print(f"  {label} — Original vs Augmented ({N} matched frames)")
    print(f"{'='*80}")

    if D != len(feature_names):
        feature_names = list(feature_names)
        while len(feature_names) < D:
            feature_names.append(f"feat_{len(feature_names)}")

    # Per-feature: mean absolute difference, correlation
    print(f"\n  {'Idx':>4}  {'Feature Name':<35}  {'MAD':>9}  {'MaxDiff':>9}  "
          f"{'Corr':>7}  {'Verdict'}")
    print(f"  {'─'*4}  {'─'*35}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*20}")

    sensitive_features = []
    invariant_features = []

    for i in range(D):
        orig_col = feat_orig[:, i]
        aug_col = feat_aug[:, i]

        diff = (orig_col - aug_col).abs()
        mad = diff.mean().item()
        max_diff = diff.max().item()

        # Correlation (skip if either is constant)
        if orig_col.std() < 1e-8 or aug_col.std() < 1e-8:
            corr = float('nan')
            corr_str = "  N/A"
        else:
            corr = torch.corrcoef(torch.stack([orig_col, aug_col]))[0, 1].item()
            corr_str = f"{corr:>7.3f}"

        # Verdict based on correlation and MAD
        both_zero = (orig_col == 0).all() and (aug_col == 0).all()
        if both_zero:
            verdict = "BOTH DEAD"
        elif np.isnan(corr):
            verdict = "CONST/DEAD"
        elif corr > 0.95 and mad < 0.05:
            verdict = "INVARIANT"
            invariant_features.append((i, feature_names[i], corr, mad))
        elif corr > 0.80:
            verdict = "MOSTLY INVARIANT"
            invariant_features.append((i, feature_names[i], corr, mad))
        elif corr > 0.50:
            verdict = "PARTIALLY SENSITIVE"
            sensitive_features.append((i, feature_names[i], corr, mad))
        else:
            verdict = "AUG-SENSITIVE"
            sensitive_features.append((i, feature_names[i], corr, mad))

        marker = "  " if "INVARIANT" in verdict else ">>"
        print(f"{marker}[{i:>3}]  {feature_names[i]:<35}  {mad:>9.5f}  {max_diff:>9.5f}  "
              f"{corr_str}  {verdict}")

    print(f"\n  Summary: {len(invariant_features)} invariant/mostly-invariant, "
          f"{len(sensitive_features)} sensitive")

    if sensitive_features:
        print(f"\n  Augmentation-SENSITIVE features (corr < 0.80):")
        for idx, name, corr, mad in sensitive_features:
            corr_str = f"{corr:.3f}" if not np.isnan(corr) else "N/A"
            print(f"    [{idx:>3}] {name:<35}  corr={corr_str}  MAD={mad:.5f}")

    if invariant_features:
        print(f"\n  Augmentation-INVARIANT features (corr >= 0.80):")
        for idx, name, corr, mad in invariant_features[:10]:
            print(f"    [{idx:>3}] {name:<35}  corr={corr:.3f}  MAD={mad:.5f}")
        if len(invariant_features) > 10:
            print(f"    ... and {len(invariant_features) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect precomputed feature .pt files')
    parser.add_argument('--fast_semantic', type=str, default=None,
                        help='Path to fast_semantic .pt (original)')
    parser.add_argument('--fast_semantic_aug', type=str, default=None,
                        help='Path to fast_semantic .pt (augmented)')
    parser.add_argument('--forensic', type=str, default=None,
                        help='Path to forensic_features .pt (original)')
    parser.add_argument('--forensic_aug', type=str, default=None,
                        help='Path to forensic_features .pt (augmented)')
    args = parser.parse_args()

    if not any([args.fast_semantic, args.forensic]):
        # Default paths
        base = "/data/umar/Datasets/preprocessed/FaceForensics++/original_sequences/youtube/c23"
        args.fast_semantic = f"{base}/fast_semantic/001.pt"
        args.fast_semantic_aug = f"{base}/fast_semantic/001_aug1.pt"
        args.forensic = f"{base}/forensic_features/001.pt"
        args.forensic_aug = f"{base}/forensic_features/001_aug1.pt"
        print("Using default paths (FaceForensics++ video 001)")

    # ── Fast Semantic Features ────────────────────────────────────────
    if args.fast_semantic and os.path.exists(args.fast_semantic):
        feat, paths = load_pt(args.fast_semantic)
        analyze_features(feat, list(FAST_FEATURE_NAMES),
                         f"Fast Semantic (original): {args.fast_semantic}")

        if args.fast_semantic_aug and os.path.exists(args.fast_semantic_aug):
            feat_aug, _ = load_pt(args.fast_semantic_aug)
            analyze_features(feat_aug, list(FAST_FEATURE_NAMES),
                             f"Fast Semantic (augmented): {args.fast_semantic_aug}")
            compare_original_vs_augmented(
                feat, feat_aug, list(FAST_FEATURE_NAMES),
                "Fast Semantic: Original vs Augmented")
        elif args.fast_semantic_aug:
            print(f"\n  Augmented file not found: {args.fast_semantic_aug}")
    elif args.fast_semantic:
        print(f"\n  File not found: {args.fast_semantic}")

    # ── Forensic Features ─────────────────────────────────────────────
    if args.forensic and os.path.exists(args.forensic):
        feat, paths = load_pt(args.forensic)
        analyze_features(feat, list(FORENSIC_FEATURE_NAMES),
                         f"Forensic Features (original): {args.forensic}")

        if args.forensic_aug and os.path.exists(args.forensic_aug):
            feat_aug, _ = load_pt(args.forensic_aug)
            analyze_features(feat_aug, list(FORENSIC_FEATURE_NAMES),
                             f"Forensic Features (augmented): {args.forensic_aug}")
            compare_original_vs_augmented(
                feat, feat_aug, list(FORENSIC_FEATURE_NAMES),
                "Forensic Features: Original vs Augmented")
        elif args.forensic_aug:
            print(f"\n  Augmented file not found: {args.forensic_aug}")
    elif args.forensic:
        print(f"\n  File not found: {args.forensic}")

    print(f"\n{'='*80}")
    print("  Done.")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
