"""
NeSyDeFake Dataset Statistics Verification Script

This script validates and reports comprehensive statistics about your dataset:
- Video counts
- Frame counts
- Clip generation
- Batch construction
- Memory estimates
- Data distribution

Usage:
    python test_dataset_stats.py --config /path/to/detector/config.yaml
"""

import os
import sys
import yaml
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

# Add DeepfakeBench to path (adjust as needed)
# sys.path.insert(0, '/path/to/DeepfakeBench')

import torch
from torch.utils.data import DataLoader

# Import your dataset
from dataset.nesy_defake_dataset import NeSyDeFakeDataset


class DatasetStatistics:
    """Comprehensive dataset statistics calculator."""
    
    def __init__(self, config: dict, mode: str = 'train'):
        self.config = config
        self.mode = mode
        self.stats = defaultdict(lambda: defaultdict(int))
        
    def analyze_dataset(self) -> Dict:
        """Run complete dataset analysis."""
        print("\n" + "="*80)
        print(f"DATASET ANALYSIS: {self.mode.upper()} MODE")
        print("="*80)
        
        # Create dataset
        print("\n[1/6] Creating dataset...")
        dataset = NeSyDeFakeDataset(self.config, mode=self.mode)
        
        # Basic statistics
        print("\n[2/6] Computing basic statistics...")
        basic_stats = self._compute_basic_stats(dataset)
        self._print_basic_stats(basic_stats)
        
        # Sample inspection
        print("\n[3/6] Inspecting sample structure...")
        sample_stats = self._inspect_samples(dataset)
        self._print_sample_stats(sample_stats)
        
        # Batch statistics
        print("\n[4/6] Analyzing batch construction...")
        batch_stats = self._analyze_batches(dataset)
        self._print_batch_stats(batch_stats)
        
        # Memory estimation
        print("\n[5/6] Estimating memory usage...")
        memory_stats = self._estimate_memory(dataset)
        self._print_memory_stats(memory_stats)
        
        # Data distribution
        print("\n[6/6] Analyzing data distribution...")
        distribution_stats = self._analyze_distribution(dataset)
        self._print_distribution_stats(distribution_stats)
        
        # Summary
        self._print_summary(basic_stats, batch_stats, memory_stats)
        
        return {
            'basic': basic_stats,
            'sample': sample_stats,
            'batch': batch_stats,
            'memory': memory_stats,
            'distribution': distribution_stats,
        }
    
    def _compute_basic_stats(self, dataset: NeSyDeFakeDataset) -> Dict:
        """Compute basic dataset statistics."""
        stats = {
            'num_videos': getattr(dataset, 'num_videos', 0),
            'num_clips': len(dataset),
            'clip_size': dataset.target_clip_size,
            'resolution': dataset.resolution,
            'expected_clips_per_video': self.config['frame_num'][self.mode] // self.config['clip_size'],
        }
        
        # Verify the ratio
        if stats['num_videos'] > 0:
            stats['actual_clips_per_video'] = stats['num_clips'] / stats['num_videos']
        else:
            stats['actual_clips_per_video'] = 0
            
        return stats
    
    def _inspect_samples(self, dataset: NeSyDeFakeDataset, num_samples: int = 5) -> Dict:
        """Inspect individual sample structure."""
        samples = []
        
        num_to_check = min(num_samples, len(dataset))
        indices = np.linspace(0, len(dataset) - 1, num_to_check, dtype=int)
        
        for idx in indices:
            try:
                sample = dataset[idx]
                sample_info = {
                    'index': int(idx),
                    'temporal_shape': tuple(sample['temporal_clip'].shape),
                    'spatial_shape': tuple(sample['spatial_frame'].shape),
                    'frequency_shape': tuple(sample['frequency_frame'].shape),
                    'raw_shape': tuple(sample['raw_frame'].shape),
                    'label': int(sample['label']),
                    'video_name': sample.get('video_name', 'unknown'),
                }
                samples.append(sample_info)
            except Exception as e:
                print(f"    ⚠ Error loading sample {idx}: {e}")
                samples.append({'index': int(idx), 'error': str(e)})
        
        return {'samples': samples}
    
    def _analyze_batches(self, dataset: NeSyDeFakeDataset) -> Dict:
        """Analyze batch construction."""
        batch_size = self.config['train_batchSize'] if self.mode == 'train' else self.config['test_batchSize']
        
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Single worker for testing
            collate_fn=dataset.collate_fn,
        )
        
        stats = {
            'batch_size': batch_size,
            'num_batches': len(loader),
            'num_workers': self.config['workers'],
            'total_iterations_per_epoch': len(loader),
        }
        
        # Inspect first batch
        try:
            first_batch = next(iter(loader))
            stats['first_batch_shapes'] = {
                'temporal_clip': tuple(first_batch['temporal_clip'].shape),
                'spatial_frame': tuple(first_batch['spatial_frame'].shape),
                'frequency_frame': tuple(first_batch['frequency_frame'].shape),
                'raw_frame': tuple(first_batch['raw_frame'].shape),
                'label': tuple(first_batch['label'].shape),
            }
            stats['first_batch_label_dist'] = {
                'real': int((first_batch['label'] == 0).sum()),
                'fake': int((first_batch['label'] == 1).sum()),
            }
        except Exception as e:
            stats['first_batch_error'] = str(e)
        
        return stats
    
    def _estimate_memory(self, dataset: NeSyDeFakeDataset) -> Dict:
        """Estimate memory usage."""
        batch_size = self.config['train_batchSize'] if self.mode == 'train' else self.config['test_batchSize']
        clip_size = dataset.target_clip_size
        resolution = dataset.resolution
        
        # Calculate per-batch memory (float32)
        bytes_per_element = 4  # float32
        
        temporal_memory = batch_size * clip_size * 3 * resolution * resolution * bytes_per_element
        spatial_memory = batch_size * 3 * resolution * resolution * bytes_per_element
        frequency_memory = spatial_memory
        raw_memory = spatial_memory
        
        total_batch_memory = temporal_memory + spatial_memory + frequency_memory + raw_memory
        
        # Estimate total dataset size
        total_clips = len(dataset)
        approx_dataset_memory = total_clips * total_batch_memory / batch_size
        
        return {
            'temporal_per_batch_mb': temporal_memory / (1024**2),
            'spatial_per_batch_mb': spatial_memory / (1024**2),
            'frequency_per_batch_mb': frequency_memory / (1024**2),
            'raw_per_batch_mb': raw_memory / (1024**2),
            'total_per_batch_mb': total_batch_memory / (1024**2),
            'total_per_batch_gb': total_batch_memory / (1024**3),
            'estimated_dataset_size_gb': approx_dataset_memory / (1024**3),
        }
    
    def _analyze_distribution(self, dataset: NeSyDeFakeDataset) -> Dict:
        """Analyze label distribution and video representation."""
        label_counts = defaultdict(int)
        video_names = set()
        
        # Sample 1000 clips or all if fewer
        num_to_sample = min(1000, len(dataset))
        indices = np.random.choice(len(dataset), num_to_sample, replace=False)
        
        for idx in indices:
            try:
                sample = dataset[idx]
                label_counts[int(sample['label'])] += 1
                video_names.add(sample.get('video_name', f'video_{idx}'))
            except Exception as e:
                print(f"    ⚠ Error in distribution analysis at index {idx}: {e}")
        
        total_sampled = sum(label_counts.values())
        
        return {
            'num_sampled': num_to_sample,
            'label_distribution': dict(label_counts),
            'label_percentages': {
                label: (count / total_sampled * 100) if total_sampled > 0 else 0
                for label, count in label_counts.items()
            },
            'unique_videos_sampled': len(video_names),
        }
    
    def _print_basic_stats(self, stats: Dict):
        """Print basic statistics."""
        print("\n┌─ BASIC STATISTICS " + "─" * 58)
        print(f"│ Mode:                    {self.mode}")
        print(f"│ Total Videos:            {stats['num_videos']:,}")
        print(f"│ Total Clips:             {stats['num_clips']:,}")
        print(f"│ Clips per Video:         {stats['actual_clips_per_video']:.2f} (expected: {stats['expected_clips_per_video']})")
        print(f"│ Clip Size (frames):      {stats['clip_size']}")
        print(f"│ Resolution:              {stats['resolution']}x{stats['resolution']}")
        
        # Verification
        if abs(stats['actual_clips_per_video'] - stats['expected_clips_per_video']) < 0.1:
            print(f"│ Status:                  ✓ CORRECT (4× increase achieved)")
        else:
            print(f"│ Status:                  ✗ MISMATCH (check clip extraction logic)")
        print("└" + "─" * 78)
    
    def _print_sample_stats(self, stats: Dict):
        """Print sample inspection results."""
        print("\n┌─ SAMPLE INSPECTION " + "─" * 57)
        for i, sample in enumerate(stats['samples']):
            if 'error' in sample:
                print(f"│ Sample {sample['index']}: ERROR - {sample['error']}")
            else:
                print(f"│ Sample {sample['index']}:")
                print(f"│   Video:      {sample['video_name']}")
                print(f"│   Label:      {sample['label']} ({'Real' if sample['label'] == 0 else 'Fake'})")
                print(f"│   Temporal:   {sample['temporal_shape']}")
                print(f"│   Spatial:    {sample['spatial_shape']}")
                print(f"│   Frequency:  {sample['frequency_shape']}")
                print(f"│   Raw:        {sample['raw_shape']}")
                if i < len(stats['samples']) - 1:
                    print("│   " + "─" * 70)
        print("└" + "─" * 78)
    
    def _print_batch_stats(self, stats: Dict):
        """Print batch statistics."""
        print("\n┌─ BATCH CONSTRUCTION " + "─" * 56)
        print(f"│ Batch Size:              {stats['batch_size']}")
        print(f"│ Total Batches:           {stats['num_batches']:,}")
        print(f"│ Iterations per Epoch:    {stats['total_iterations_per_epoch']:,}")
        print(f"│ Num Workers:             {stats['num_workers']}")
        
        if 'first_batch_shapes' in stats:
            print(f"│")
            print(f"│ First Batch Shapes:")
            for key, shape in stats['first_batch_shapes'].items():
                print(f"│   {key:20s} {shape}")
            
            print(f"│")
            print(f"│ First Batch Label Distribution:")
            dist = stats['first_batch_label_dist']
            print(f"│   Real: {dist['real']:3d} | Fake: {dist['fake']:3d}")
        
        print("└" + "─" * 78)
    
    def _print_memory_stats(self, stats: Dict):
        """Print memory statistics."""
        print("\n┌─ MEMORY ESTIMATION " + "─" * 57)
        print(f"│ Per-Batch Memory Usage:")
        print(f"│   Temporal Stream:       {stats['temporal_per_batch_mb']:>8.2f} MB")
        print(f"│   Spatial Stream:        {stats['spatial_per_batch_mb']:>8.2f} MB")
        print(f"│   Frequency Stream:      {stats['frequency_per_batch_mb']:>8.2f} MB")
        print(f"│   Raw Stream:            {stats['raw_per_batch_mb']:>8.2f} MB")
        print(f"│   {'─' * 40}")
        print(f"│   Total per Batch:       {stats['total_per_batch_mb']:>8.2f} MB ({stats['total_per_batch_gb']:.3f} GB)")
        print(f"│")
        print(f"│ Estimated Dataset Size:  {stats['estimated_dataset_size_gb']:.2f} GB")
        print("└" + "─" * 78)
    
    def _print_distribution_stats(self, stats: Dict):
        """Print distribution statistics."""
        print("\n┌─ DATA DISTRIBUTION " + "─" * 57)
        print(f"│ Samples Analyzed:        {stats['num_sampled']:,}")
        print(f"│ Unique Videos Seen:      {stats['unique_videos_sampled']:,}")
        print(f"│")
        print(f"│ Label Distribution:")
        for label, count in sorted(stats['label_distribution'].items()):
            pct = stats['label_percentages'][label]
            label_name = 'Real' if label == 0 else 'Fake'
            print(f"│   {label_name:6s} (label={label}): {count:5d} clips ({pct:5.1f}%)")
        print("└" + "─" * 78)
    
    def _print_summary(self, basic: Dict, batch: Dict, memory: Dict):
        """Print final summary."""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        # Training efficiency
        clips_per_video = basic['actual_clips_per_video']
        expected_clips = basic['expected_clips_per_video']
        
        print(f"\n✓ Data Utilization:")
        print(f"  • {basic['num_videos']:,} videos → {basic['num_clips']:,} clips")
        print(f"  • {clips_per_video:.1f}× increase (expected {expected_clips}×)")
        
        if abs(clips_per_video - expected_clips) < 0.1:
            print(f"  • Status: ✓✓✓ OPTIMAL (all frames utilized)")
        else:
            print(f"  • Status: ⚠ Check clip extraction logic")
        
        print(f"\n✓ Training Load:")
        print(f"  • {batch['num_batches']:,} batches per epoch")
        print(f"  • {batch['batch_size']} clips per batch")
        print(f"  • {batch['batch_size'] * basic['clip_size']} frames processed per iteration")
        
        print(f"\n✓ Memory Requirements:")
        print(f"  • {memory['total_per_batch_mb']:.1f} MB per batch")
        print(f"  • {memory['total_per_batch_gb']:.3f} GB GPU memory (batch only)")
        print(f"  • Recommendation: {'Use batch_size=' + str(batch['batch_size'])}")
        
        # Comparison with OLD approach
        print(f"\n✓ Improvement vs OLD (1 clip/video):")
        old_clips = basic['num_videos']
        new_clips = basic['num_clips']
        old_batches = old_clips // batch['batch_size']
        new_batches = batch['num_batches']
        
        print(f"  • Training samples: {old_clips:,} → {new_clips:,} (+{(new_clips/old_clips - 1)*100:.0f}%)")
        print(f"  • Batches per epoch: {old_batches:,} → {new_batches:,} (+{(new_batches/old_batches - 1)*100:.0f}%)")
        print(f"  • Memory per batch: UNCHANGED (same batch size)")
        
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Test NeSyDeFake dataset statistics')
    parser.add_argument('--detector_config', type=str, required=True,
                        help='Path to detector config YAML')
    parser.add_argument('--train_config', type=str, 
                        default='./training/config/train_config.yaml',
                        help='Path to train config YAML')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Dataset mode to analyze')
    args = parser.parse_args()
    
    # Load configs
    print("Loading configuration files...")
    with open(args.detector_config, 'r') as f:
        detector_config = yaml.safe_load(f)
    
    with open(args.train_config, 'r') as f:
        train_config = yaml.safe_load(f)
    
    # Merge configs (detector config takes precedence)
    config = {**train_config, **detector_config}
    
    # Override for testing if needed
    if args.mode == 'test' and 'test_dataset' not in config:
        config['test_dataset'] = config.get('train_dataset', ['FaceForensics++'])
    
    print(f"Analyzing {args.mode} dataset...")
    print(f"  Datasets: {config.get('train_dataset' if args.mode == 'train' else 'test_dataset')}")
    
    # Run analysis
    analyzer = DatasetStatistics(config, mode=args.mode)
    stats = analyzer.analyze_dataset()
    
    # Save results
    output_file = f'dataset_stats_{args.mode}.yaml'
    with open(output_file, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    print(f"\n✓ Full statistics saved to: {output_file}")


if __name__ == '__main__':
    main()