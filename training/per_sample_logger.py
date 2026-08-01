"""
per_sample_logger.py
====================
Per-sample diagnostic CSV logging for the NeSy-DeFake eval loop (test.py).

Dumps, per (run, dataset), one row per frame with the fused prediction and the
per-branch evidential breakdown, plus a video-level aggregate (mean over frames).

Columns (per frame):
  video_id, frame_idx, label, p_fake, uncertainty,
  S_spatial, S_concept, S_causal          # Dirichlet strengths, blank if branch absent
  p_fake_spatial, p_fake_concept, p_fake_causal   # per-branch Dirichlet means
  disagreement_d                          # mean pairwise (1 - cos), IBDC formula
  concept_gate, causal_gate               # fusion gates (blank if absent)

Design notes:
  * Per-branch S / p_fake are read from each branch's OWN evidence in the
    prediction dict (spatial_evidence / concept_evidence / causal_evidence), so a
    branch that ran but was withheld from fusion (concept under
    evidence_in_fusion=false) still logs its own contribution.
  * disagreement_d uses the FUSED `branch_evidences` set — the same set IBDC
    consumes — so it matches the training-time disagreement signal exactly.
  * Every field degrades gracefully: absent keys become blank cells, so this
    works for every ablation_mode (spatial_ce has only p_fake).
"""

import csv
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Column order for the per-sample CSV.
FRAME_COLUMNS = [
    'video_id', 'frame_idx', 'label', 'p_fake', 'uncertainty',
    'S_spatial', 'S_concept', 'S_causal',
    'p_fake_spatial', 'p_fake_concept', 'p_fake_causal',
    'disagreement_d', 'concept_gate', 'causal_gate',
]

# Numeric columns aggregated (mean) at video level.
_NUMERIC = [
    'p_fake', 'uncertainty', 'S_spatial', 'S_concept', 'S_causal',
    'p_fake_spatial', 'p_fake_concept', 'p_fake_causal',
    'disagreement_d', 'concept_gate', 'causal_gate',
]

_NAN = float('nan')


def _np(t):
    return t.detach().float().cpu().numpy()


def _branch_S(evidence):
    """Dirichlet strength S = sum(alpha) = sum(evidence) + K, per sample (B,)."""
    return _np((evidence + 1.0).sum(dim=1))


def _branch_p_fake(evidence):
    """Per-branch Dirichlet mean for the fake class = alpha_fake / S, (B,)."""
    alpha = evidence + 1.0
    return _np(alpha[:, 1] / alpha.sum(dim=1))


def _disagreement(branch_evidences):
    """Mean pairwise (1 - cos(Dirichlet-mean_i, Dirichlet-mean_j)) — the IBDC formula.

    Returns (B,) numpy array, or None if fewer than two branches are present.
    """
    evs = [ev for ev in branch_evidences.values() if ev is not None]
    if len(evs) < 2:
        return None
    probs = []
    for ev in evs:
        alpha = ev + 1.0
        probs.append(alpha / alpha.sum(dim=1, keepdim=True))
    B = probs[0].shape[0]
    dis = torch.zeros(B, device=probs[0].device)
    n_pairs = 0
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            dis = dis + (1.0 - F.cosine_similarity(probs[i], probs[j], dim=1))
            n_pairs += 1
    return _np((dis / max(n_pairs, 1)).clamp(0, 1))


def _scalar_broadcast(value, batch_size):
    """A per-batch scalar gate broadcast to every row; NaN if absent."""
    if value is None:
        return np.full(batch_size, _NAN, dtype=np.float64)
    if isinstance(value, torch.Tensor):
        value = float(value.detach().float().mean().item())
    return np.full(batch_size, float(value), dtype=np.float64)


def collect_batch(predictions: dict) -> dict:
    """Extract the per-sample numeric columns for one batch → {col: (B,) array}.

    Robust to any ablation_mode: missing keys yield all-NaN columns.
    """
    prob = _np(predictions['prob']).reshape(-1)
    B = prob.shape[0]

    def col(key, fn):
        ev = predictions.get(key)
        return fn(ev) if ev is not None else np.full(B, _NAN)

    out = {
        'p_fake': prob,
        'uncertainty': (_np(predictions['uncertainty']).reshape(-1)
                        if predictions.get('uncertainty') is not None
                        else np.full(B, _NAN)),
        'S_spatial': col('spatial_evidence', _branch_S),
        'S_concept': col('concept_evidence', _branch_S),
        'S_causal': col('causal_evidence', _branch_S),
        'p_fake_spatial': col('spatial_evidence', _branch_p_fake),
        'p_fake_concept': col('concept_evidence', _branch_p_fake),
        'p_fake_causal': col('causal_evidence', _branch_p_fake),
        'concept_gate': _scalar_broadcast(predictions.get('concept_gate'), B),
        'causal_gate': _scalar_broadcast(predictions.get('causal_gate'), B),
    }
    be = predictions.get('branch_evidences')
    d = _disagreement(be) if isinstance(be, dict) else None
    out['disagreement_d'] = d if d is not None else np.full(B, _NAN)

    # Per-rule violation columns rule_00..rule_{K-1}. Column index j maps to the
    # j-th rule in configs/retained_predicates.yaml order (frozen), so Tasks 6/9
    # can name them. Absent for spatial-only modes (no concept branch).
    viol = predictions.get('violations')
    if viol is not None:
        v = _np(viol)                                    # (B, K)
        if v.ndim == 2:
            for j in range(v.shape[1]):
                out[f'rule_{j:02d}'] = v[:, j]
    return out


def _rule_cols(extras: dict) -> list:
    """Sorted rule_NN columns present in an extras dict (may be empty)."""
    return sorted(k for k in extras if k.startswith('rule_'))


def concat_batches(batches: list) -> dict:
    """Concatenate a list of per-batch column dicts into full-length arrays.

    Concatenates every column present (the fixed _NUMERIC set plus any dynamic
    rule_NN columns), so per-rule violations survive to the CSV writers.
    """
    if not batches:
        return {c: np.array([]) for c in _NUMERIC}
    keys = list(batches[0].keys())
    return {c: np.concatenate([b[c] for b in batches]) for c in keys}


def _fmt(x):
    """Blank cell for NaN, else 6-dp float."""
    return '' if (isinstance(x, float) and np.isnan(x)) else f'{float(x):.6f}'


def _parse_name(name: str):
    parts = name.replace('\\', '/').split('/')
    video_id = parts[-2] if len(parts) >= 2 else parts[-1]
    frame_idx = os.path.splitext(parts[-1])[0]
    return video_id, frame_idx


def _method_from_path(name: str) -> str:
    """FF++ manipulation / source type from the image path, '' if not FF++.

    FF++ layout: .../{manipulated_sequences|original_sequences}/<METHOD>/c23/
    frames/<vid>/<frame>. Returns the <METHOD> token (Deepfakes, Face2Face,
    FaceSwap, NeuralTextures, youtube, actors, ...). Other datasets (Celeb-DF,
    DFDC, ...) lack these anchors and yield '' — which is what Tasks 5/6 want
    (FF++-only manipulation breakdown / leave-one-manipulation-out).
    """
    parts = name.replace('\\', '/').split('/')
    for anchor in ('manipulated_sequences', 'original_sequences'):
        if anchor in parts:
            k = parts.index(anchor)
            if k + 1 < len(parts):
                return parts[k + 1]
    return ''


def _spe(label_spe, i):
    """Specific-method label at row i as int, or '' if unavailable."""
    if label_spe is None:
        return ''
    try:
        return int(label_spe[i])
    except (TypeError, ValueError, IndexError):
        return ''


def write_per_sample_csv(path: str, img_names, labels, extras: dict,
                         label_spe=None) -> None:
    """Write the per-frame CSV (one row per frame).

    When *label_spe* is given, a `label_spe` column (the dataset's specific-
    method label — e.g. FF++ manipulation type) is written after `label`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rule_cols = _rule_cols(extras)
    numeric = _NUMERIC + rule_cols
    has_spe = label_spe is not None
    spe_col = ['label_spe'] if has_spe else []
    n = len(img_names)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(FRAME_COLUMNS[:3] + ['method'] + spe_col
                   + FRAME_COLUMNS[3:] + rule_cols)
        for i in range(n):
            vid, frame_idx = _parse_name(img_names[i])
            row = [vid, frame_idx, int(labels[i]), _method_from_path(img_names[i])]
            if has_spe:
                row.append(_spe(label_spe, i))
            row += [_fmt(extras[c][i]) for c in numeric]
            w.writerow(row)


def write_video_aggregate_csv(path: str, img_names, labels, extras: dict,
                              label_spe=None) -> None:
    """Write the video-level aggregate CSV (mean over frames per video)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rule_cols = _rule_cols(extras)
    numeric = _NUMERIC + rule_cols
    has_spe = label_spe is not None
    spe_col = ['label_spe'] if has_spe else []
    agg = defaultdict(lambda: {'label': None, 'method': '', 'label_spe': '',
                               'n': 0, **{c: [] for c in numeric}})
    for i in range(len(img_names)):
        vid, _ = _parse_name(img_names[i])
        rec = agg[vid]
        rec['label'] = int(labels[i])
        rec['method'] = _method_from_path(img_names[i])
        if has_spe:
            rec['label_spe'] = _spe(label_spe, i)
        rec['n'] += 1
        for c in numeric:
            v = extras[c][i]
            if not (isinstance(v, float) and np.isnan(v)):
                rec[c].append(float(v))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video_id', 'label', 'method'] + spe_col
                   + ['num_frames'] + numeric)
        for vid, rec in agg.items():
            row = [vid, rec['label'], rec['method']]
            if has_spe:
                row.append(rec['label_spe'])
            row.append(rec['n'])
            row += [_fmt(np.mean(rec[c]) if rec[c] else _NAN) for c in numeric]
            w.writerow(row)


def write_logs(out_dir: str, dataset_name: str, img_names, labels, extras: dict,
               label_spe=None):
    """Write both per-frame and video-aggregate CSVs; returns the two paths."""
    frame_path = os.path.join(out_dir, f'per_sample_{dataset_name}.csv')
    video_path = os.path.join(out_dir, f'per_sample_video_{dataset_name}.csv')
    write_per_sample_csv(frame_path, img_names, labels, extras, label_spe)
    write_video_aggregate_csv(video_path, img_names, labels, extras, label_spe)
    return frame_path, video_path
