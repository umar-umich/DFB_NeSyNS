#!/usr/bin/env python3
"""
aggregate_results.py — NeSyDeFake Ablation Results Aggregator  (v2)
=====================================================================
Parses per-dataset metrics from training logs and prints a table with:
    AUC | Acc | Real_Acc | Fake_Acc
reported separately for FF++ and CelebDF-v2.

Log format (actual from trainer):
    testing-metric, auc: 0.9726  testing-metric, acc: 0.9330 ... acc_real:0.8; acc_fake:0.966

USAGE
-----
    python aggregate_results.py
    python aggregate_results.py --log_dir ./ablation_results/logs
    python aggregate_results.py --out ./results.csv
"""

import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

# ── Regex patterns matched to actual log format ───────────────────────────────
_NUM = r"[-+]?\d*\.?\d+"

METRIC_PATTERNS = {
    "auc":      re.compile(rf"testing-metric,\s*auc:\s*({_NUM})", re.IGNORECASE),
    "acc":      re.compile(rf"testing-metric,\s*acc:\s*({_NUM})", re.IGNORECASE),
    "real_acc": re.compile(rf"acc_real:\s*({_NUM})", re.IGNORECASE),
    "fake_acc": re.compile(rf"acc_fake:\s*({_NUM})", re.IGNORECASE),
}

DATASET_PATTERN = re.compile(r"dataset:\s*([\w\+\-\.v]+)", re.IGNORECASE)

TARGET_DATASETS = {
    "FaceForensics++": "FF++",
    "Celeb-DF-v2":     "CelebDF-v2",
}

TAG_LABELS = {
    "T":     "Temporal only",
    "S":     "Spatial only",
    "F":     "Frequency only",
    "T_S":   "Temporal + Spatial",
    "T_F":   "Temporal + Frequency",
    "S_F":   "Spatial + Frequency",
    "T_S_F": "All branches (T+S+F)",
}
TAG_ORDER = ["T", "S", "F", "T_S", "T_F", "S_F", "T_S_F"]


def parse_log(path):
    """Return best-epoch metrics per dataset: {display_name: {metric: float}}"""
    snapshots = defaultdict(list)

    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if "testing-metric" not in line.lower():
                continue
            ds_match = DATASET_PATTERN.search(line)
            if not ds_match:
                continue
            ds_name = ds_match.group(1)

            snap = {}
            for metric, pat in METRIC_PATTERNS.items():
                m = pat.search(line)
                if m:
                    val = float(m.group(1))
                    if val > 1.5:
                        val /= 100.0
                    snap[metric] = val

            if snap:
                snapshots[ds_name].append(snap)

    result = {}
    for log_name, display_name in TARGET_DATASETS.items():
        snaps = snapshots.get(log_name, [])
        if snaps:
            result[display_name] = max(snaps, key=lambda s: s.get("auc", 0.0))
        else:
            result[display_name] = {}

    avg_snaps = snapshots.get("avg", [])
    if avg_snaps:
        result["avg"] = max(avg_snaps, key=lambda s: s.get("auc", 0.0))

    return result


def derive_tag(filename):
    stem = Path(filename).stem
    return stem[4:] if stem.startswith("exp_") else None


def build_rows(log_dir):
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    tag_to_data = {}
    for log_file in sorted(log_path.glob("exp_*.log")):
        tag = derive_tag(log_file.name)
        if tag is None:
            continue
        data = parse_log(log_file)
        tag_to_data[tag] = data

        parts = []
        for ds in list(TARGET_DATASETS.values()) + ["avg"]:
            m = data.get(ds, {})
            auc = f"{m['auc']*100:.2f}" if "auc" in m else "N/A"
            parts.append(f"{ds}:auc={auc}")
        print(f"  Parsed [{tag:>7s}] — " + "  |  ".join(parts))

    rows = []
    for tag in TAG_ORDER:
        if tag in tag_to_data:
            rows.append({"tag": tag, "label": TAG_LABELS.get(tag, tag), "data": tag_to_data[tag]})
    for tag, data in tag_to_data.items():
        if tag not in TAG_ORDER:
            rows.append({"tag": tag, "label": tag, "data": data})
    return rows


def fmt(val):
    if val is None:
        return "—"
    return f"{val * 100:.2f}"


def get_val(data, dataset, metric):
    return fmt(data.get(dataset, {}).get(metric))


def print_table(rows):
    datasets    = list(TARGET_DATASETS.values())
    metric_keys = ["auc", "acc", "real_acc", "fake_acc"]
    metric_hdrs = ["AUC", "Acc", "R.Acc", "F.Acc"]

    exp_w  = max(len("Experiment"), max(len(r["label"]) for r in rows)) + 2
    col_w  = 8

    def pad(s, w):
        return str(s).ljust(w)

    def sep_line(char="-"):
        total = exp_w + (col_w + 3) * len(datasets) * len(metric_keys) + 3 * (len(datasets) - 1)
        return char * total

    # Dataset header
    ds_block_w = col_w * len(metric_keys) + 3 * (len(metric_keys) - 1)
    hdr1 = pad("Experiment", exp_w) + " | " + " || ".join(
        ds.center(ds_block_w) for ds in datasets
    )
    # Metric sub-header
    hdr2 = pad("", exp_w) + " | " + " || ".join(
        " | ".join(pad(h, col_w) for h in metric_hdrs)
        for _ in datasets
    )

    print("\n" + hdr1)
    print(hdr2)
    print(sep_line())

    for row in rows:
        tag    = row["tag"]
        marker = "*" if tag == "T_S_F" else " "
        label  = pad(marker + " " + row["label"], exp_w)
        data   = row["data"]

        ds_blocks = []
        for ds in datasets:
            cells = " | ".join(pad(get_val(data, ds, mk), col_w) for mk in metric_keys)
            ds_blocks.append(cells)

        print(label + " | " + " || ".join(ds_blocks))

    print(sep_line())
    print("  * = full model baseline | R.Acc = Real Acc | F.Acc = Fake Acc\n")


def write_csv(rows, out_path):
    datasets    = list(TARGET_DATASETS.values())
    metric_keys = ["auc", "acc", "real_acc", "fake_acc"]

    fieldnames = ["Experiment", "Tag"]
    for ds in datasets:
        for mk in metric_keys:
            fieldnames.append(f"{ds}_{mk.upper()}")

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {"Experiment": row["label"], "Tag": row["tag"]}
            for ds in datasets:
                for mk in metric_keys:
                    record[f"{ds}_{mk.upper()}"] = get_val(row["data"], ds, mk)
            writer.writerow(record)

    print(f"CSV saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./ablation_results/logs")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    print(f"\nScanning logs in: {args.log_dir}\n")
    rows = build_rows(args.log_dir)

    print("\n" + "=" * 70)
    print("  NeSyDeFake Branch Ablation — Results Summary")
    print("=" * 70)
    print_table(rows)

    out_path = args.out or str(Path(args.log_dir) / "ablation_table.csv")
    write_csv(rows, out_path)


if __name__ == "__main__":
    main()