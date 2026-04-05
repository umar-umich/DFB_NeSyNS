#!/usr/bin/env python3
"""
analyze_feature_stability.py
=============================
Analyze FaceBench semantic feature stability across video frames.

Extracts 211 attributes from ALL 32 frames of sample videos, then measures:
  1. Top-K feature overlap (Jaccard) across frame pairs
  2. Core features: intersection of top-K across all frames
  3. Per-attribute variance (stable vs volatile attributes)
  4. Key-frame coverage: how many features does {1}, {1,16,32}, {1,8,16,24,32}
     capture vs all-32-frames union?
  5. Real vs fake attribute distribution comparison

Tests the hypothesis: if >90% of top-K features are shared across frames,
we can extract from 3 key frames and propagate to save ~10x LLM cost.

Usage:
  python preprocessing/analyze_feature_stability.py \
      --detector_path training/config/detector/nesy_defake.yaml \
      --n_videos 2 \
      --top_k 50 \
      --image_batch_size 8
"""

import argparse
import os
import sys
import time
from itertools import combinations

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config_utils import load_config


# ── Sample video selection ────────────────────────────────────────────────

def find_sample_videos(n_per_class: int = 2) -> dict:
    """
    Find sample frame directories for analysis.
    Returns dict with keys like 'FF++_real_000' -> {'frames_dir': ..., 'label': ...}
    """
    base = '/data/umar/Datasets/preprocessed'
    samples = {}

    # FF++ real
    ff_real = os.path.join(base, 'FaceForensics++/original_sequences/youtube/c23/frames')
    if os.path.isdir(ff_real):
        vids = sorted(os.listdir(ff_real))[:n_per_class]
        for v in vids:
            d = os.path.join(ff_real, v)
            if os.path.isdir(d):
                samples[f'FF++_real_{v}'] = {
                    'frames_dir': d, 'dataset': 'FF++',
                    'label': 'real', 'video_id': v,
                }

    # FF++ fake (FaceSwap)
    ff_fake = os.path.join(base, 'FaceForensics++/manipulated_sequences/FaceSwap/c23/frames')
    if os.path.isdir(ff_fake):
        vids = sorted(os.listdir(ff_fake))[:n_per_class]
        for v in vids:
            d = os.path.join(ff_fake, v)
            if os.path.isdir(d):
                samples[f'FF++_fake_{v}'] = {
                    'frames_dir': d, 'dataset': 'FF++',
                    'label': 'fake', 'video_id': v,
                }

    # CelebDF real
    cdf_real = os.path.join(base, 'Celeb-DF-v2/Celeb-real/frames')
    if os.path.isdir(cdf_real):
        vids = sorted(os.listdir(cdf_real))[:n_per_class]
        for v in vids:
            d = os.path.join(cdf_real, v)
            if os.path.isdir(d):
                samples[f'CelebDF_real_{v}'] = {
                    'frames_dir': d, 'dataset': 'CelebDF',
                    'label': 'real', 'video_id': v,
                }

    # CelebDF fake
    cdf_fake = os.path.join(base, 'Celeb-DF-v2/Celeb-synthesis/frames')
    if os.path.isdir(cdf_fake):
        vids = sorted(os.listdir(cdf_fake))[:n_per_class]
        for v in vids:
            d = os.path.join(cdf_fake, v)
            if os.path.isdir(d):
                samples[f'CelebDF_fake_{v}'] = {
                    'frames_dir': d, 'dataset': 'CelebDF',
                    'label': 'fake', 'video_id': v,
                }

    return samples


def load_frames(frames_dir: str) -> list:
    """Load all image paths from a frames directory, sorted."""
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    paths = sorted([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    return paths


# ── Analysis functions ────────────────────────────────────────────────────

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def analyze_video(scores_mat: np.ndarray, attr_names: list,
                  top_k_values: list = [40, 50, 60],
                  video_name: str = '') -> dict:
    """
    Analyze feature stability for one video.

    Args:
        scores_mat: (n_frames, 211) attribute probabilities
        attr_names: list of 211 attribute names
        top_k_values: list of K values to test
        video_name: for display

    Returns:
        dict with analysis results
    """
    n_frames, n_attrs = scores_mat.shape
    results = {'video': video_name, 'n_frames': n_frames}

    # ── Per-attribute statistics ──────────────────────────────────
    attr_mean = scores_mat.mean(axis=0)    # (211,)
    attr_std = scores_mat.std(axis=0)      # (211,)
    attr_min = scores_mat.min(axis=0)      # (211,)
    attr_max = scores_mat.max(axis=0)      # (211,)

    results['attr_mean'] = attr_mean
    results['attr_std'] = attr_std
    results['attr_range'] = attr_max - attr_min

    # Attributes with P>0.5 across all frames
    always_present = (scores_mat > 0.5).all(axis=0)
    never_present = (scores_mat <= 0.5).all(axis=0)
    sometimes_present = ~always_present & ~never_present

    results['n_always_present'] = always_present.sum()
    results['n_never_present'] = never_present.sum()
    results['n_sometimes_present'] = sometimes_present.sum()

    # ── Top-K stability analysis ─────────────────────────────────
    for K in top_k_values:
        # Top-K indices per frame
        topk_per_frame = []
        for fi in range(n_frames):
            topk_idx = set(np.argsort(scores_mat[fi])[::-1][:K])
            topk_per_frame.append(topk_idx)

        # Pairwise Jaccard similarity
        pair_jaccards = []
        for i, j in combinations(range(n_frames), 2):
            pair_jaccards.append(jaccard(topk_per_frame[i], topk_per_frame[j]))

        # Core features: appear in top-K of ALL frames
        core = topk_per_frame[0]
        for s in topk_per_frame[1:]:
            core = core & s

        # Union: appear in top-K of ANY frame
        union = set()
        for s in topk_per_frame:
            union = union | s

        results[f'topk_{K}'] = {
            'jaccard_mean': np.mean(pair_jaccards),
            'jaccard_min': np.min(pair_jaccards),
            'jaccard_std': np.std(pair_jaccards),
            'core_count': len(core),
            'core_pct': len(core) / K * 100,
            'union_count': len(union),
            'union_pct': len(union) / n_attrs * 100,
            'core_attrs': sorted([attr_names[i] for i in core],
                                  key=lambda a: -attr_mean[attr_names.index(a)]),
        }

        # ── Key-frame coverage ───────────────────────────────────
        # How well does extracting from N key frames capture the union?
        strategies = {
            '1_frame': [0],                              # frame 0 only
            '3_frames': [0, n_frames // 2, n_frames - 1],  # first, mid, last
            '5_frames': list(np.linspace(0, n_frames - 1, 5, dtype=int)),
        }
        coverage = {}
        for strat_name, indices in strategies.items():
            strat_union = set()
            for idx in indices:
                strat_union |= topk_per_frame[idx]
            coverage[strat_name] = {
                'n_keyframes': len(indices),
                'captured': len(strat_union & union),
                'total_union': len(union),
                'coverage_pct': len(strat_union & union) / len(union) * 100
                                if union else 100.0,
            }
        results[f'topk_{K}_coverage'] = coverage

    # ── Rank correlation across frames ────────────────────────────
    # Spearman rank correlation of full 211-d vectors between frames
    from scipy.stats import spearmanr
    rank_corrs = []
    # Sample pairs to avoid O(n^2) cost with many frames
    pairs = list(combinations(range(n_frames), 2))
    if len(pairs) > 100:
        rng = np.random.RandomState(42)
        pairs = [pairs[i] for i in rng.choice(len(pairs), 100, replace=False)]
    for i, j in pairs:
        corr, _ = spearmanr(scores_mat[i], scores_mat[j])
        rank_corrs.append(corr)
    results['rank_corr_mean'] = np.mean(rank_corrs)
    results['rank_corr_min'] = np.min(rank_corrs)

    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Analyze FaceBench feature stability across video frames')
    parser.add_argument('--detector_path', type=str, required=True)
    parser.add_argument('--n_videos', type=int, default=2,
                        help='Videos per class (real/fake) per dataset')
    parser.add_argument('--top_k', type=int, nargs='+', default=[40, 50, 60],
                        help='Top-K values to analyze')
    parser.add_argument('--image_batch_size', type=int, default=8,
                        help='Images batched through CLIP vision tower '
                             '(LLM always loops per image)')
    parser.add_argument('--attr_batch_size', type=int, default=112,
                        help='Attributes per LLM forward pass')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--frames_dir', type=str, default=None,
                        help='Analyze a specific frames directory instead of '
                             'auto-selecting samples')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Save raw results to .npz file')
    args = parser.parse_args()

    config = load_config(args.detector_path)

    # ── Load model ────────────────────────────────────────────────
    print(f"\nLoading SemanticPrecomputer on {args.device}...")
    from precompute_semantic_features import SemanticPrecomputer
    precomputer = SemanticPrecomputer(
        config, args.device, attr_batch_size=args.attr_batch_size)
    attr_names = precomputer.attr_names

    # ── Select videos ─────────────────────────────────────────────
    if args.frames_dir:
        samples = {
            os.path.basename(args.frames_dir): {
                'frames_dir': args.frames_dir,
                'dataset': 'custom',
                'label': 'unknown',
                'video_id': os.path.basename(args.frames_dir),
            }
        }
    else:
        samples = find_sample_videos(n_per_class=args.n_videos)

    if not samples:
        print("ERROR: No sample videos found!")
        sys.exit(1)

    print(f"\nSelected {len(samples)} videos for analysis:")
    for name, info in samples.items():
        n = len(load_frames(info['frames_dir']))
        print(f"  {name:<40s} {n:>3d} frames  [{info['label']}]")

    # ── Extract features ──────────────────────────────────────────
    all_results = []
    all_scores = {}  # video_name -> (n_frames, 211)

    for vid_name, vid_info in samples.items():
        frame_paths = load_frames(vid_info['frames_dir'])
        if not frame_paths:
            print(f"\n  WARNING: No frames in {vid_info['frames_dir']}, skipping")
            continue

        print(f"\n{'─' * 60}")
        print(f"  Extracting: {vid_name} ({len(frame_paths)} frames)")
        print(f"{'─' * 60}")

        t0 = time.time()
        result = precomputer.process_video(
            frame_paths, image_batch_size=args.image_batch_size)
        elapsed = time.time() - t0

        if result is None:
            print(f"  ERROR: extraction returned None")
            continue

        scores_mat = result['features'].numpy()  # (n_frames, 211)
        print(f"  Extracted in {elapsed:.1f}s "
              f"({elapsed/len(frame_paths):.2f}s/frame)")

        all_scores[vid_name] = scores_mat

        # Run analysis
        r = analyze_video(scores_mat, attr_names, args.top_k, vid_name)
        r['label'] = vid_info['label']
        r['dataset'] = vid_info['dataset']
        all_results.append(r)

    # ── Print per-video reports ───────────────────────────────────
    for r in all_results:
        print(f"\n{'=' * 75}")
        print(f"  VIDEO: {r['video']}  ({r['n_frames']} frames, "
              f"{r['dataset']} {r['label']})")
        print(f"{'=' * 75}")

        print(f"\n  Attribute presence (P>0.5 threshold):")
        print(f"    Always present:    {r['n_always_present']:>3d} / 211")
        print(f"    Never present:     {r['n_never_present']:>3d} / 211")
        print(f"    Sometimes (noisy): {r['n_sometimes_present']:>3d} / 211")

        print(f"\n  Rank correlation across frames:")
        print(f"    Mean Spearman ρ:   {r['rank_corr_mean']:.4f}")
        print(f"    Min Spearman ρ:    {r['rank_corr_min']:.4f}")

        for K in args.top_k:
            tk = r[f'topk_{K}']
            print(f"\n  Top-{K} Feature Stability:")
            print(f"    Pairwise Jaccard:  {tk['jaccard_mean']:.3f} "
                  f"(min={tk['jaccard_min']:.3f}, std={tk['jaccard_std']:.3f})")
            print(f"    Core (in ALL):     {tk['core_count']:>3d} / {K} "
                  f"({tk['core_pct']:.1f}%)")
            print(f"    Union (in ANY):    {tk['union_count']:>3d} / 211 "
                  f"({tk['union_pct']:.1f}%)")

            cov = r[f'topk_{K}_coverage']
            print(f"    Key-frame coverage of union ({tk['union_count']} attrs):")
            for strat, c in cov.items():
                print(f"      {strat:<12s}: {c['captured']:>3d}/{c['total_union']} "
                      f"({c['coverage_pct']:.1f}%) from {c['n_keyframes']} frames")

            if tk['core_attrs']:
                n_show = min(20, len(tk['core_attrs']))
                print(f"    Core top-{K} attributes (stable, showing top {n_show}):")
                for attr in tk['core_attrs'][:n_show]:
                    ai = attr_names.index(attr)
                    print(f"      {attr.replace('_', ' '):<35s} "
                          f"mean={r['attr_mean'][ai]:.3f} "
                          f"std={r['attr_std'][ai]:.4f}")

        # Most volatile
        print(f"\n  Most VOLATILE attributes (highest std):")
        volatile_idx = np.argsort(r['attr_std'])[::-1][:10]
        for i in volatile_idx:
            print(f"    {attr_names[i].replace('_', ' '):<35s} "
                  f"std={r['attr_std'][i]:.4f}, "
                  f"mean={r['attr_mean'][i]:.3f}, "
                  f"range={r['attr_range'][i]:.3f}")

    # ── Aggregate summary ─────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n\n{'#' * 75}")
        print(f"  AGGREGATE SUMMARY ({len(all_results)} videos)")
        print(f"{'#' * 75}")

        for K in args.top_k:
            print(f"\n  Top-{K} Stability Summary:")
            print(f"  {'Video':<40s} {'Jaccard':>8s} {'Core':>6s} "
                  f"{'3fr-cov':>8s} {'Label':>6s}")
            print(f"  {'─' * 72}")

            jaccards_real, jaccards_fake = [], []
            core_pcts_real, core_pcts_fake = [], []
            cov3_real, cov3_fake = [], []

            for r in all_results:
                tk = r[f'topk_{K}']
                cov3 = r[f'topk_{K}_coverage']['3_frames']['coverage_pct']
                print(f"  {r['video']:<40s} {tk['jaccard_mean']:>7.3f} "
                      f"{tk['core_pct']:>5.1f}% "
                      f"{cov3:>7.1f}% "
                      f"{r['label']:>6s}")

                if r['label'] == 'real':
                    jaccards_real.append(tk['jaccard_mean'])
                    core_pcts_real.append(tk['core_pct'])
                    cov3_real.append(cov3)
                else:
                    jaccards_fake.append(tk['jaccard_mean'])
                    core_pcts_fake.append(tk['core_pct'])
                    cov3_fake.append(cov3)

            print(f"\n  Averages:")
            if jaccards_real:
                print(f"    Real: Jaccard={np.mean(jaccards_real):.3f}, "
                      f"Core={np.mean(core_pcts_real):.1f}%, "
                      f"3-frame-cov={np.mean(cov3_real):.1f}%")
            if jaccards_fake:
                print(f"    Fake: Jaccard={np.mean(jaccards_fake):.3f}, "
                      f"Core={np.mean(core_pcts_fake):.1f}%, "
                      f"3-frame-cov={np.mean(cov3_fake):.1f}%")

            all_jaccards = [r[f'topk_{K}']['jaccard_mean'] for r in all_results]
            all_core = [r[f'topk_{K}']['core_pct'] for r in all_results]
            all_cov3 = [r[f'topk_{K}_coverage']['3_frames']['coverage_pct']
                        for r in all_results]

            print(f"    ALL:  Jaccard={np.mean(all_jaccards):.3f}, "
                  f"Core={np.mean(all_core):.1f}%, "
                  f"3-frame-cov={np.mean(all_cov3):.1f}%")

        # ── Hypothesis verdict ────────────────────────────────────
        K_main = args.top_k[1] if len(args.top_k) > 1 else args.top_k[0]
        avg_core = np.mean([r[f'topk_{K_main}']['core_pct']
                            for r in all_results])
        avg_cov3 = np.mean([r[f'topk_{K_main}_coverage']['3_frames'][
            'coverage_pct'] for r in all_results])
        avg_jaccard = np.mean([r[f'topk_{K_main}']['jaccard_mean']
                               for r in all_results])

        print(f"\n{'=' * 75}")
        print(f"  HYPOTHESIS VERDICT (Top-{K_main}):")
        print(f"{'=' * 75}")
        print(f"  Q: Do >90% of top-{K_main} features stay stable across "
              f"all 32 frames?")
        if avg_core >= 90:
            print(f"  A: YES — {avg_core:.1f}% core stability. "
                  f"Safe to extract from 3 key frames.")
        elif avg_core >= 80:
            print(f"  A: MOSTLY — {avg_core:.1f}% core stability. "
                  f"3-frame coverage is {avg_cov3:.1f}%. "
                  f"Reasonable with union strategy.")
        elif avg_core >= 60:
            print(f"  A: PARTIAL — {avg_core:.1f}% core stability. "
                  f"3-frame coverage is {avg_cov3:.1f}%. "
                  f"Use 5 key frames for safety.")
        else:
            print(f"  A: NO — {avg_core:.1f}% core stability. "
                  f"Features are too volatile for key-frame propagation. "
                  f"Extract all frames.")

        print(f"\n  Mean pairwise Jaccard:   {avg_jaccard:.3f}")
        print(f"  Mean core intersection: {avg_core:.1f}%")
        print(f"  Mean 3-frame coverage:  {avg_cov3:.1f}%")

        # ── Recommendation ────────────────────────────────────────
        print(f"\n  RECOMMENDATION:")
        if avg_cov3 >= 95:
            print(f"  Extract from 3 key frames (first, mid, last).")
            print(f"  Use UNION of their top-{K_main} as the active feature set.")
            print(f"  For remaining 29 frames, only extract those features.")
        elif avg_cov3 >= 85:
            print(f"  Extract from 5 key frames (evenly spaced).")
            print(f"  Use UNION of their top-{K_main} as the active feature set.")
        else:
            print(f"  Feature instability too high for key-frame strategy.")
            print(f"  Extract all 211 attributes from all frames.")
            print(f"  Consider: store raw probabilities, let the model learn "
                  f"what matters.")

        # ── Real vs Fake comparison ───────────────────────────────
        real_scores = [all_scores[r['video']] for r in all_results
                       if r['label'] == 'real' and r['video'] in all_scores]
        fake_scores = [all_scores[r['video']] for r in all_results
                       if r['label'] == 'fake' and r['video'] in all_scores]

        if real_scores and fake_scores:
            # Average over frames, then over videos
            real_avg = np.mean([s.mean(axis=0) for s in real_scores], axis=0)
            fake_avg = np.mean([s.mean(axis=0) for s in fake_scores], axis=0)
            diff = fake_avg - real_avg

            print(f"\n  Real vs Fake attribute differences (|Δ| > 0.1):")
            big_diff_idx = np.where(np.abs(diff) > 0.1)[0]
            if len(big_diff_idx) > 0:
                sorted_diff = big_diff_idx[np.argsort(np.abs(diff[big_diff_idx]))[::-1]]
                for i in sorted_diff[:15]:
                    direction = "fake↑" if diff[i] > 0 else "real↑"
                    print(f"    {attr_names[i].replace('_', ' '):<35s} "
                          f"Δ={diff[i]:+.3f} ({direction})  "
                          f"real={real_avg[i]:.3f} fake={fake_avg[i]:.3f}")
            else:
                print(f"    No attributes with |Δ| > 0.1 between real and fake")

    # ── Save raw results ──────────────────────────────────────────
    if args.save_results:
        save_dict = {}
        for vid_name, scores in all_scores.items():
            safe_name = vid_name.replace('/', '_').replace(' ', '_')
            save_dict[safe_name] = scores
        save_dict['attr_names'] = np.array(attr_names, dtype=object)
        np.savez_compressed(args.save_results, **save_dict)
        print(f"\n  Raw results saved to: {args.save_results}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
