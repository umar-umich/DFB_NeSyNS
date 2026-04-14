#!/usr/bin/env python3
"""
Remove stray non-frame entries from dataset JSON files.

Filters out paths that are not numeric-named frames (e.g. demo_plots,
730_facebench_precomputed.png, .txt files). Valid frames have purely
numeric stems like 000.png, 296.jpg OR are inside frames_aug_N dirs.

Usage:
    python scripts/clean_dataset_json.py /data/umar/Datasets/preprocessed/dataset_json/FaceForensics++.json
    python scripts/clean_dataset_json.py /data/umar/Datasets/preprocessed/dataset_json/FaceForensics++_augmented.json
"""

import json
import os
import re
import sys


def is_valid_frame(path: str) -> bool:
    """Check if a path looks like a valid frame (e.g. 000.png, 296.jpg)."""
    fname = path.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
    stem, _, ext = fname.rpartition('.')
    if not ext or ext.lower() not in ('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'):
        return False
    # Stem must be purely numeric (e.g. 000, 296)
    return stem.isdigit()


def clean_json(data: dict) -> tuple:
    """Remove invalid frame paths in-place. Returns (count, list of removed paths)."""
    removed = 0
    removed_entries = []
    for top_key in data:
        for label_key in data[top_key]:
            for mode_key in data[top_key][label_key]:
                inner = data[top_key][label_key][mode_key]
                if not isinstance(inner, dict):
                    continue
                # Handle both direct video dicts and compression-level nesting
                for sub_key in inner:
                    sub = inner[sub_key]
                    if isinstance(sub, dict) and 'frames' in sub:
                        # Direct video entry
                        bad = [p for p in sub['frames'] if not is_valid_frame(p)]
                        removed_entries.extend(bad)
                        removed += len(bad)
                        sub['frames'] = [p for p in sub['frames'] if is_valid_frame(p)]
                    elif isinstance(sub, dict):
                        # Compression level (c23, c40, etc.)
                        for vid_id, vid_data in sub.items():
                            if isinstance(vid_data, dict) and 'frames' in vid_data:
                                bad = [p for p in vid_data['frames'] if not is_valid_frame(p)]
                                removed_entries.extend(bad)
                                removed += len(bad)
                                vid_data['frames'] = [p for p in vid_data['frames'] if is_valid_frame(p)]
    return removed, removed_entries


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/clean_dataset_json.py <json_file> [json_file2 ...]")
        sys.exit(1)

    for json_path in sys.argv[1:]:
        if not os.path.exists(json_path):
            print(f"  Not found: {json_path}")
            continue

        with open(json_path) as f:
            data = json.load(f)

        removed, removed_entries = clean_json(data)

        if removed > 0:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            # Write removed entries to a log file for reference
            log_path = json_path.rsplit('.', 1)[0] + '_removed_entries.txt'
            with open(log_path, 'w') as f:
                for entry in removed_entries:
                    f.write(entry + '\n')
            print(f"  {json_path}: removed {removed} stray entries")
            print(f"  Removed entries logged to: {log_path}")
        else:
            print(f"  {json_path}: clean, no changes needed")


if __name__ == '__main__':
    main()
