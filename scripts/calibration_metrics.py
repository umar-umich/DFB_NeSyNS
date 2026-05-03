"""
Calibration & selective-prediction metrics for deepfake-detection baselines.

For each (method, dataset) pair this script:
  1. Reads `runs/test/{method}--{dataset}-test_split/test_predictions.csv`
     (one row per pre-sampled frame; the test manifests already encode
      the 32-frame uniform sample with seed=42 across all methods).
  2. Writes the canonical per-sample CSV
     `results/baselines/{method}/{dataset}_per_sample.csv` with columns
     sample_id, video_id, frame_idx, label, prediction, confidence, prob_fake.
  3. Computes ECE / CW@0.9 / AURC / E-AURC at the frame and video level
     and saves `results/baselines/{method}/{dataset}_metrics.json`.
  4. Runs sanity checks (label convention, confidence range, bin counts,
     CDFv2 AUC ±1.5pt of the published value when applicable).
  5. Aggregates into `results/baselines/MASTER_SUMMARY.md` and
     `results/baselines/per_method_summaries/{method}_SUMMARY.md`.

The metric helpers below MUST NOT be modified — they are the reference
implementations specified in the task brief and match DeFakeNet's
reliability-diagram protocol exactly (10 bins, max(p, 1-p) confidence).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# NumPy 2.x removed `np.trapz` (renamed to `np.trapezoid`). The brief mandates
# that the metric functions below stay verbatim, so we restore the old name
# here rather than edit them.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Verbatim metric functions — DO NOT MODIFY.
# ---------------------------------------------------------------------------


def compute_ece(probs, labels, n_bins=10):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(int)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_accs[i] = correct[mask].mean()
            bin_confs[i] = confidences[mask].mean()
            bin_counts[i] = mask.sum()

    total = bin_counts.sum()
    ece = (bin_counts / max(total, 1) * np.abs(bin_accs - bin_confs)).sum()
    return float(ece), bin_accs.tolist(), bin_confs.tolist(), bin_counts.astype(int).tolist()


def compute_cw_at_threshold(probs, labels, threshold=0.9):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(int)
    correct = (predictions == labels)
    high_conf_wrong = (confidences > threshold) & (~correct)
    return float(high_conf_wrong.sum() / len(probs))


def compute_aurc_eaurc(probs, labels):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(int)
    correct = (predictions == labels).astype(float)
    n = len(probs)

    order = np.argsort(-confidences)
    sorted_correct = correct[order]
    cum_errors = np.cumsum(1.0 - sorted_correct)
    coverage = np.arange(1, n + 1) / n
    selective_risk = cum_errors / np.arange(1, n + 1)
    aurc = np.trapz(selective_risk, coverage)

    oracle_correct = np.sort(correct)[::-1]
    oracle_cum_errors = np.cumsum(1.0 - oracle_correct)
    oracle_selective_risk = oracle_cum_errors / np.arange(1, n + 1)
    aurc_oracle = np.trapz(oracle_selective_risk, coverage)

    eaurc = aurc - aurc_oracle
    return float(aurc), float(eaurc)


# ---------------------------------------------------------------------------
# AUC (sklearn-free) and ACC.
# ---------------------------------------------------------------------------


def compute_auc(prob_fake: np.ndarray, labels: np.ndarray) -> float:
    """Mann–Whitney U based ROC-AUC. Returns NaN when only one class present."""
    prob_fake = np.asarray(prob_fake, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(prob_fake, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(prob_fake) + 1)
    sorted_scores = prob_fake[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    rank_pos_sum = ranks[pos_mask].sum()
    auc = (rank_pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_acc(prob_fake: np.ndarray, labels: np.ndarray) -> float:
    pred = (np.asarray(prob_fake) >= 0.5).astype(int)
    return float((pred == np.asarray(labels)).mean())


# ---------------------------------------------------------------------------
# Path → (video_id, frame_idx) parsing.
# ---------------------------------------------------------------------------

# CDFv2  paths look like:  ./datasets/CDFv2/Celeb-real/id0_0001/frame_0000.png
# DFDC   paths look like:  /.../DFDC/fake/aamrozxzsq/000.png
# DFDCP  paths look like:  /.../DFDCP/fake-A/<vid>/000.png
# DFD    paths look like:  /.../DeepFakeDetection/fake/<vid>/000.png
# FSh    paths look like:  /.../FaceShifter/fake-FH/<vid>/000.png
# CDFv3  paths look like:  /.../Celeb-DF-v3/Celeb-synthesis/FaceSwap/<method>/frames/<vid>/000.png
#                          /.../Celeb-DF-v3/Celeb-real/frames/<vid>/000.png

_FRAME_NUM_RE = re.compile(r"(?:frame_)?(\d+)")


def parse_frame_idx(stem: str) -> int:
    m = _FRAME_NUM_RE.search(stem)
    if m is None:
        raise ValueError(f"Cannot parse frame index from: {stem}")
    return int(m.group(1))


def parse_video_and_frame(path: str, dataset: str) -> tuple[str, int]:
    """Return (video_id, frame_idx) using the dataset directory layout."""
    p = Path(path)
    frame_idx = parse_frame_idx(p.stem)

    if dataset == "CDFv2":
        # .../CDFv2/<split>/<video>/frame_xxxx.png
        return f"{p.parent.parent.name}/{p.parent.name}", frame_idx
    if dataset in {"DFDC", "DFDCP", "DeepFakeDetection", "FaceShifter"}:
        # .../<dataset>/<bucket>/<video>/xxx.png
        return f"{p.parent.parent.name}/{p.parent.name}", frame_idx
    if dataset == "CDFv3":
        # Symlinked layout produced by scripts/prepare_cdfv3.py:
        #   .../CDFv3/<source>/<video>/<frame>.png
        # where <source> is one of {real-Celeb, real-YouTube, fake-<method>}.
        # Fall back to the raw on-disk layout for completeness.
        return f"{p.parent.parent.name}/{p.parent.name}", frame_idx
    raise ValueError(f"Unknown dataset for path parsing: {dataset}")


def dataset_id_prefix(dataset: str) -> str:
    return {
        "CDFv2": "cdfv2",
        "CDFv3": "cdfv3",
        "DFDC": "dfdc",
        "DFDCP": "dfdcp",
        "DeepFakeDetection": "dfd",
        "FaceShifter": "fsh",
    }[dataset]


# ---------------------------------------------------------------------------
# Per-sample CSV construction.
# ---------------------------------------------------------------------------


def predictions_to_per_sample(pred_csv: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    expected = {"files", "labels", "prob_class_0", "prob_class_1"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{pred_csv} missing columns: {missing}")

    video_ids = []
    frame_idxs = []
    for f in df["files"].astype(str):
        v, fi = parse_video_and_frame(f, dataset)
        video_ids.append(v)
        frame_idxs.append(fi)

    prob_fake = df["prob_class_1"].astype(float).to_numpy()
    confidences = np.maximum(prob_fake, 1.0 - prob_fake)
    predictions = (prob_fake >= 0.5).astype(int)
    labels = df["labels"].astype(int).to_numpy()

    prefix = dataset_id_prefix(dataset)
    sample_ids = [
        f"{prefix}_{v.replace('/', '__')}_frame{fi:04d}" for v, fi in zip(video_ids, frame_idxs)
    ]

    out = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "video_id": video_ids,
            "frame_idx": frame_idxs,
            "label": labels,
            "prediction": predictions,
            "confidence": confidences,
            "prob_fake": prob_fake,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Frame-level + video-level metric computation.
# ---------------------------------------------------------------------------


def metrics_block(prob_fake: np.ndarray, labels: np.ndarray) -> dict:
    ece, bin_accs, bin_confs, bin_counts = compute_ece(prob_fake, labels, n_bins=10)
    cw = compute_cw_at_threshold(prob_fake, labels, threshold=0.9)
    aurc, eaurc = compute_aurc_eaurc(prob_fake, labels)
    return {
        "n": int(len(prob_fake)),
        "auc": compute_auc(prob_fake, labels),
        "acc": compute_acc(prob_fake, labels),
        "ece": ece,
        "cw_at_09": cw,
        "aurc": aurc,
        "eaurc": eaurc,
        "bin_edges": np.linspace(0, 1, 11).tolist(),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
    }


def aggregate_video_level(per_sample: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    grouped = per_sample.groupby("video_id", sort=False)
    video_prob_fake = grouped["prob_fake"].mean().to_numpy()
    video_labels = grouped["label"].first().to_numpy()
    n_videos = int(per_sample["video_id"].nunique())
    return video_prob_fake, video_labels, n_videos


def evaluate_pair(
    method: str,
    dataset: str,
    pred_csv: Path,
    out_dir: Path,
    cdfv2_published_auc: float | None,
) -> tuple[dict, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    per_sample = predictions_to_per_sample(pred_csv, dataset)

    csv_path = out_dir / f"{dataset}_per_sample.csv"
    per_sample.to_csv(csv_path, index=False)

    prob_fake = per_sample["prob_fake"].to_numpy()
    labels = per_sample["label"].to_numpy()

    frame = metrics_block(prob_fake, labels)
    frame["n_frames"] = frame["n"]
    frame["n_videos"] = int(per_sample["video_id"].nunique())

    v_prob, v_labels, n_videos = aggregate_video_level(per_sample)
    video = metrics_block(v_prob, v_labels)
    video["n_frames"] = frame["n_frames"]
    video["n_videos"] = n_videos

    # ---- Sanity checks --------------------------------------------------
    confs = np.maximum(prob_fake, 1.0 - prob_fake)
    if confs.min() < 0.5 - 1e-9 or confs.max() > 1.0 + 1e-9:
        warnings.append(
            f"confidence out of [0.5, 1.0]: min={confs.min():.4f} max={confs.max():.4f}"
        )

    if int(np.sum(frame["bin_counts"])) != int(frame["n"]):
        warnings.append(
            f"bin_counts sum {np.sum(frame['bin_counts'])} != n_samples {frame['n']}"
        )

    fake_videos = per_sample[per_sample["label"] == 1]
    if len(fake_videos) > 0:
        rng = np.random.default_rng(42)
        sampled_v = rng.choice(fake_videos["video_id"].unique(), size=1)[0]
        v_prob_sample = float(per_sample.loc[per_sample["video_id"] == sampled_v, "prob_fake"].mean())
        if v_prob_sample <= 0.5:
            warnings.append(
                f"sanity: random fake video {sampled_v} has video_prob_fake={v_prob_sample:.3f} <= 0.5"
            )

    if dataset == "CDFv2" and cdfv2_published_auc is not None:
        diff = abs(video["auc"] - cdfv2_published_auc)
        if diff > 0.015:
            warnings.append(
                f"CDFv2 video-AUC {video['auc']:.4f} differs from published "
                f"{cdfv2_published_auc:.4f} by {diff:.4f} (>0.015)"
            )

    # ---- JSON output ---------------------------------------------------
    out = OrderedDict(
        method=method,
        dataset=dataset,
        n_frames=frame["n_frames"],
        n_videos=video["n_videos"],
        frame_level={
            "n_frames": frame["n_frames"],
            "n_videos": frame["n_videos"],
            "auc_frame": frame["auc"],
            "acc_frame": frame["acc"],
            "ece_frame": frame["ece"],
            "cw_at_09_frame": frame["cw_at_09"],
            "aurc_frame": frame["aurc"],
            "eaurc_frame": frame["eaurc"],
        },
        video_level={
            "n_videos": video["n_videos"],
            "auc_video": video["auc"],
            "acc_video": video["acc"],
            "ece_video": video["ece"],
            "cw_at_09_video": video["cw_at_09"],
            "aurc_video": video["aurc"],
            "eaurc_video": video["eaurc"],
        },
        reliability_frame={
            "bin_edges": frame["bin_edges"],
            "bin_accuracies": frame["bin_accuracies"],
            "bin_confidences": frame["bin_confidences"],
            "bin_counts": frame["bin_counts"],
        },
        reliability_video={
            "bin_edges": video["bin_edges"],
            "bin_accuracies": video["bin_accuracies"],
            "bin_confidences": video["bin_confidences"],
            "bin_counts": video["bin_counts"],
        },
        per_sample_csv=str(csv_path.relative_to(csv_path.parents[2])),
        warnings=warnings,
    )

    json_path = out_dir / f"{dataset}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    if warnings:
        with open(out_dir / "ERRORS.txt", "a") as f:
            for w in warnings:
                f.write(f"[{method}/{dataset}] {w}\n")

    return out, warnings


# ---------------------------------------------------------------------------
# Frame-sampling manifest.
# ---------------------------------------------------------------------------


def build_frame_manifest(
    per_sample_frames: dict[tuple[str, str], pd.DataFrame],
    out_path: Path,
):
    """Save (dataset -> video_id -> sorted list of frame indices)."""
    manifest: dict = {}
    seen: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for (method, dataset), df in per_sample_frames.items():
        for v, fi in zip(df["video_id"], df["frame_idx"]):
            seen[dataset][v].add(int(fi))

    for dataset, videos in seen.items():
        manifest[dataset] = {v: sorted(list(s)) for v, s in videos.items()}

    out = {
        "seed": 42,
        "n_frames_per_video": 32,
        "note": (
            "Frames listed here are the ones evaluated by every baseline for the "
            "given video. The DeepfakeBench-prepared manifests already encode a "
            "deterministic 32-frame uniform sample, so the manifest is read out "
            "from the predictions rather than regenerated. All baselines see the "
            "same frames for any given video."
        ),
        "datasets": manifest,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Driver / orchestration.
# ---------------------------------------------------------------------------

# (method_name, run_name_prefix, citation, checkpoint_path)
METHODS: list[tuple[str, str, str, str]] = [
    ("GenD-CLIP", "GenD_CLIP", "yermakov2026gend", "yermandy/GenD_CLIP_L_14"),
    ("ForAda", "ForAda", "cui2025forada", "weights/ForAda/ForAdackpt_best.pth"),
    ("GenD-DINO", "GenD_DINO", "yermakov2026gend", "yermandy/GenD_DINOv3_L"),
    ("GenD-PE", "GenD_PE", "yermakov2026gend", "yermandy/GenD_PE_L"),
    ("Effort", "Effort", "yan2025effort", "weights/Effort/effort_clip_L14_trainOn_FaceForensic.pth"),
    ("FSFM-simple", "FSFM", "wang2025fsfm", "weights/FS-VFM/FS-VFM-ViT-L.pth"),
    ("FSFM-Adapter", "FSFM_Adapter", "wang2025fsfm", "weights/FS-VFM/FS-VFM-ViT-L-Adapter.pth"),
]

DATASETS: list[tuple[str, str]] = [
    # (canonical_name, run_dir_dataset_token)
    ("CDFv2", "CDFv2"),
    ("DFDC", "DFDC"),
    ("DFDCP", "DFDCP"),
    ("DeepFakeDetection", "DeepFakeDetection"),
    ("FaceShifter", "FaceShifter"),
    ("CDFv3", "CDFv3"),
]

# Published CDFv2 video-level AUCs (best available from each paper / run-summary).
# These set the ±1.5pt sanity guard. They are *targets*, not reported numbers.
PUBLISHED_CDFV2_AUC = {
    "GenD-CLIP": 0.9512,
    "GenD-DINO": 0.9156,
    "GenD-PE": 0.9549,
    "Effort": 0.9302,
    "ForAda": 0.9489,
    "FSFM-simple": 0.9442,
    "FSFM-Adapter": None,  # No published CDFv2 number wired in repo
}


def fmt(v: float, nd: int = 4) -> str:
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "—"
    return f"{v:.{nd}f}"


def write_master_summary(rows_frame: list[dict], rows_video: list[dict], notes: list[str], out_dir: Path):
    lines: list[str] = []
    lines.append("# Cross-Dataset Calibration & Selective-Prediction Metrics\n")
    lines.append(
        "Frame-level uses every evaluated frame; video-level averages "
        "`prob_fake` over the 32 frames per video and applies the same metric "
        "functions. ECE / CW@0.9 / AURC / E-AURC follow the verbatim "
        "definitions in `scripts/calibration_metrics.py`. Confidence is "
        "`max(prob_fake, 1 - prob_fake)`.\n"
    )

    def render(rows: list[dict]) -> list[str]:
        out = [
            "| Method | Dataset | n | AUC | ACC | ECE | CW@0.9 | AURC | E-AURC |",
            "|--------|---------|---|-----|-----|-----|--------|------|--------|",
        ]
        for r in rows:
            out.append(
                f"| {r['method']} | {r['dataset']} | {r['n']} | "
                f"{fmt(r['auc'])} | {fmt(r['acc'])} | {fmt(r['ece'])} | "
                f"{fmt(r['cw_at_09'])} | {fmt(r['aurc'])} | {fmt(r['eaurc'])} |"
            )
        return out

    lines.append("## Frame-Level\n")
    lines.extend(render(rows_frame))
    lines.append("")
    lines.append("## Video-Level\n")
    lines.extend(render(rows_video))
    lines.append("")

    if notes:
        lines.append("## Per-method notes\n")
        for n in notes:
            lines.append(n)
            lines.append("")

    (out_dir / "MASTER_SUMMARY.md").write_text("\n".join(lines))


def write_per_method_summary(method: str, frame_rows: list[dict], video_rows: list[dict], out_dir: Path, notes_block: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = method.replace(" ", "_")
    lines = [f"# {method} — per-dataset summary\n", notes_block, "\n## Frame-Level\n"]
    lines.append("| Dataset | n | AUC | ACC | ECE | CW@0.9 | AURC | E-AURC |")
    lines.append("|---------|---|-----|-----|-----|--------|------|--------|")
    for r in frame_rows:
        lines.append(
            f"| {r['dataset']} | {r['n']} | {fmt(r['auc'])} | {fmt(r['acc'])} | "
            f"{fmt(r['ece'])} | {fmt(r['cw_at_09'])} | {fmt(r['aurc'])} | {fmt(r['eaurc'])} |"
        )
    lines.append("\n## Video-Level\n")
    lines.append("| Dataset | n | AUC | ACC | ECE | CW@0.9 | AURC | E-AURC |")
    lines.append("|---------|---|-----|-----|-----|--------|------|--------|")
    for r in video_rows:
        lines.append(
            f"| {r['dataset']} | {r['n']} | {fmt(r['auc'])} | {fmt(r['acc'])} | "
            f"{fmt(r['ece'])} | {fmt(r['cw_at_09'])} | {fmt(r['aurc'])} | {fmt(r['eaurc'])} |"
        )
    (out_dir / f"{safe}_SUMMARY.md").write_text("\n".join(lines) + "\n")


METHOD_NOTES: dict[str, str] = {
    "GenD-CLIP": (
        "- Citation: Yermakov 2026 (GenD)\n"
        "- Backbone: CLIP ViT-L/14 (HuggingFace `yermandy/GenD_CLIP_L_14`)\n"
        "- Output: 2 logits → softmax → `prob_fake = p[1]`\n"
        "- Preprocessing: standard DeepfakeBench (RetinaFace, 1.3× margin, 224²)\n"
    ),
    "GenD-DINO": (
        "- Citation: Yermakov 2026 (GenD)\n"
        "- Backbone: DINOv3-L (HuggingFace `yermandy/GenD_DINOv3_L`)\n"
        "- Output: 2 logits → softmax → `prob_fake = p[1]`\n"
        "- Preprocessing: standard DeepfakeBench (RetinaFace, 1.3× margin, 224²)\n"
    ),
    "GenD-PE": (
        "- Citation: Yermakov 2026 (GenD)\n"
        "- Backbone: Perception Encoder L (HuggingFace `yermandy/GenD_PE_L`)\n"
        "- Output: 2 logits → softmax → `prob_fake = p[1]`\n"
        "- Preprocessing: standard DeepfakeBench (RetinaFace, 1.3× margin, 224²)\n"
    ),
    "Effort": (
        "- Citation: Yan 2025 (Effort, orthogonal subspace)\n"
        "- Checkpoint: `weights/Effort/effort_clip_L14_trainOn_FaceForensic.pth`\n"
        "- Output: 2 logits → softmax → `prob_fake = p[1]`\n"
    ),
    "ForAda": (
        "- Citation: Cui 2025 (Forensics Adapter)\n"
        "- Checkpoint: `weights/ForAda/ForAdackpt_best.pth`\n"
        "- Output: 2 logits → softmax → `prob_fake = p[1]`\n"
    ),
    "FSFM-simple": (
        "- Citation: Wang 2025 (FSFM)\n"
        "- Checkpoint: `weights/FS-VFM/FS-VFM-ViT-L.pth`\n"
        "- Output: 2 logits, swapped to `[real, fake]` inside `src/model/FSFM.py:54` and softmaxed; "
        "`prob_fake = p[1]`. Preprocessing uses the FSFM zoom-in (`zoom_factor=1.3`).\n"
    ),
    "FSFM-Adapter": (
        "- Citation: Wang 2025 (FSFM, FS-Adapter variant)\n"
        "- Checkpoint: `weights/FS-VFM/FS-VFM-ViT-L-Adapter.pth`\n"
        "- Output: 2 logits, same `[real, fake]` swap as FSFM-simple\n"
    ),
}


def collect_rows(method: str, datasets_done: list[tuple[str, dict]]):
    frame_rows: list[dict] = []
    video_rows: list[dict] = []
    for dataset, m in datasets_done:
        f = m["frame_level"]
        v = m["video_level"]
        frame_rows.append(
            dict(method=method, dataset=dataset, n=f["n_frames"], auc=f["auc_frame"], acc=f["acc_frame"],
                 ece=f["ece_frame"], cw_at_09=f["cw_at_09_frame"], aurc=f["aurc_frame"], eaurc=f["eaurc_frame"])
        )
        video_rows.append(
            dict(method=method, dataset=dataset, n=v["n_videos"], auc=v["auc_video"], acc=v["acc_video"],
                 ece=v["ece_video"], cw_at_09=v["cw_at_09_video"], aurc=v["aurc_video"], eaurc=v["eaurc_video"])
        )
    return frame_rows, video_rows


def git_commit_hash(repo: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def gpu_info() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip().splitlines()[0] if r.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def cuda_version() -> str:
    try:
        import torch  # local import; only needed for the manifest

        return f"cuda={torch.version.cuda}"
    except Exception:
        return "torch-not-importable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--methods", nargs="*", default=None,
                        help="subset of method names to evaluate (default: all)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="subset of dataset names to evaluate (default: all)")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    runs_dir = Path(args.runs_dir or repo / "runs/test")
    results_dir = Path(args.results_dir or repo / "results/baselines")
    results_dir.mkdir(parents=True, exist_ok=True)

    method_filter = set(args.methods) if args.methods else None
    dataset_filter = set(args.datasets) if args.datasets else None

    manifest_frames: dict[tuple[str, str], pd.DataFrame] = {}
    per_method_results: dict[str, list[tuple[str, dict]]] = OrderedDict()
    failures: list[tuple[str, str, str]] = []  # (method, dataset, reason)
    all_warnings: list[str] = []
    timings: dict[str, dict] = {}

    rows_frame: list[dict] = []
    rows_video: list[dict] = []

    run_start = datetime.now(timezone.utc)

    for method, run_prefix, citation, ckpt in METHODS:
        if method_filter and method not in method_filter:
            continue
        method_dir = results_dir / method.replace(" ", "_")
        method_start = datetime.now(timezone.utc)
        per_method_results[method] = []
        for dataset, ds_token in DATASETS:
            if dataset_filter and dataset not in dataset_filter:
                continue
            run_dir = runs_dir / f"{run_prefix}--{ds_token}-test_split"
            pred_csv = run_dir / "test_predictions.csv"
            if not pred_csv.exists():
                failures.append((method, dataset, f"missing predictions at {pred_csv}"))
                continue
            try:
                m, warnings = evaluate_pair(
                    method=method,
                    dataset=dataset,
                    pred_csv=pred_csv,
                    out_dir=method_dir,
                    cdfv2_published_auc=PUBLISHED_CDFV2_AUC.get(method),
                )
            except Exception as e:  # noqa: BLE001
                failures.append((method, dataset, f"exception: {e}"))
                continue
            per_method_results[method].append((dataset, m))
            for w in warnings:
                all_warnings.append(f"{method}/{dataset}: {w}")
            manifest_frames[(method, dataset)] = pd.read_csv(method_dir / f"{dataset}_per_sample.csv")

        method_end = datetime.now(timezone.utc)
        timings[method] = {
            "metrics_start_utc": method_start.isoformat(),
            "metrics_end_utc": method_end.isoformat(),
            "metrics_compute_seconds": (method_end - method_start).total_seconds(),
            "inference_runs_dir": str(runs_dir.relative_to(repo)),
            "note": (
                "Total per-method runtime corresponds to model inference, "
                "available in runs/test/_logs and runs/test/<exp>/metrics.csv. "
                "metrics_compute_seconds covers only the post-hoc metrics "
                "pipeline that consumes test_predictions.csv."
            ),
            "checkpoint": ckpt,
            "citation_key": citation,
        }
        f_rows, v_rows = collect_rows(method, per_method_results[method])
        rows_frame.extend(f_rows)
        rows_video.extend(v_rows)
        write_per_method_summary(
            method=method,
            frame_rows=f_rows,
            video_rows=v_rows,
            out_dir=results_dir / "per_method_summaries",
            notes_block=METHOD_NOTES.get(method, ""),
        )

    # Frame-sampling manifest (built from any method's CSVs; all methods share frames per video).
    if manifest_frames:
        # Pick the first dataset entries per method to enumerate; we only need frame indices per video.
        build_frame_manifest(manifest_frames, results_dir.parent / "frame_sampling_seed42.json")

    # Notes block for master summary.
    notes_lines: list[str] = []
    for method, _, citation, ckpt in METHODS:
        if method_filter and method not in method_filter:
            continue
        body = METHOD_NOTES.get(method, "")
        ds_done = [d for d, _ in per_method_results.get(method, [])]
        ds_failed = [(d, r) for (m, d, r) in failures if m == method]
        sanity_warn = [w for w in all_warnings if w.startswith(f"{method}/")]
        block = [f"### {method} (`{citation}`, ckpt: `{ckpt}`)\n", body]
        if ds_done:
            block.append(f"- Datasets evaluated: {', '.join(ds_done)}")
        if ds_failed:
            block.append(
                "- Failed datasets: "
                + ", ".join(f"{d} ({r})" for d, r in ds_failed)
            )
        if sanity_warn:
            block.append(f"- Sanity-check warnings: {len(sanity_warn)}")
            for w in sanity_warn:
                block.append(f"  - {w}")
        else:
            block.append("- Sanity-check warnings: 0")
        notes_lines.append("\n".join(block))

    write_master_summary(rows_frame, rows_video, notes_lines, results_dir)

    # ---- Reproducibility manifest --------------------------------------
    try:
        import torch  # local

        torch_version = torch.__version__
    except Exception:
        torch_version = "unknown"

    run_manifest = {
        "git_commit": git_commit_hash(repo),
        "run_started_utc": run_start.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch_version,
        "cuda": cuda_version(),
        "gpu": gpu_info(),
        "seed": 42,
        "methods": [
            {
                "method": method,
                "run_prefix": run_prefix,
                "citation_key": citation,
                "checkpoint": ckpt,
                **timings.get(method, {}),
            }
            for method, run_prefix, citation, ckpt in METHODS
            if not method_filter or method in method_filter
        ],
        "datasets": [
            {
                "name": dname,
                "manifests": _dataset_manifest_paths(repo, dname),
            }
            for dname, _ in DATASETS
            if not dataset_filter or dname in dataset_filter
        ],
        "failures": [
            {"method": m, "dataset": d, "reason": r} for m, d, r in failures
        ],
        "warnings": all_warnings,
    }

    with open(results_dir / "RUN_MANIFEST.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    print("\nDone.")
    print(f"  master summary : {results_dir / 'MASTER_SUMMARY.md'}")
    print(f"  manifests      : {results_dir / 'RUN_MANIFEST.json'}")
    if failures:
        print("Failures:")
        for m, d, r in failures:
            print(f"  - {m} / {d}: {r}")
    if all_warnings:
        print(f"Warnings: {len(all_warnings)}")


def _dataset_manifest_paths(repo: Path, dataset: str) -> list[str]:
    base = repo / "config/datasets" / dataset / "test_split"
    if not base.exists():
        return []
    return sorted(str(p.relative_to(repo)) for p in base.glob("*.txt"))


if __name__ == "__main__":
    main()
