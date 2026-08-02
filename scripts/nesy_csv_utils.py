"""
scripts/nesy_csv_utils.py
=========================
Shared read/feature helpers for the Phase-2 post-hoc analysis scripts
(risk_head, audit, error_analysis, evidence_record, temporal_ladder).

These operate ONLY on the per-sample CSVs written by training/per_sample_logger.py
(`logs/test/<run>/per_sample_<dataset>.csv`, one row per frame). Nothing here
touches the model, the datasets, or any OOD data during fitting — the risk/ladder
FITTING guards live in the individual scripts.

Frame CSV columns consumed:
  video_id, label, method, p_fake, uncertainty,
  S_spatial/S_concept/S_causal            (Dirichlet strengths, blank if absent)
  p_fake_spatial/p_fake_concept/p_fake_causal   (per-branch Dirichlet means)
  disagreement_d, rule_00..rule_{K-1}     (per-rule violations, optional)

Branch commitment / vacuity use the Dirichlet strength S_b: for a K-class
Dirichlet, vacuity V_b = K / S_b and commitment q_b = 1 - V_b.
"""
import os

import numpy as np
import pandas as pd

BRANCHES = ['spatial', 'concept', 'causal']
K_CLASSES = 2
_EPS = 1e-8


def frame_csv_path(run_dir, dataset):
    return os.path.join(run_dir, f'per_sample_{dataset}.csv')


def load_frames(run_dir, dataset):
    """Load a per-frame CSV as a DataFrame, or None if absent."""
    p = frame_csv_path(run_dir, dataset)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def rule_columns(df):
    """Sorted rule_NN columns present in the frame DataFrame."""
    return sorted(c for c in df.columns if c.startswith('rule_'))


def present_branches(df):
    """Branches whose per-fake-prob column has any finite value."""
    out = []
    for b in BRANCHES:
        col = f'p_fake_{b}'
        if col in df.columns and df[col].notna().any():
            out.append(b)
    return out


def _js_2class(p, q):
    """Jensen-Shannon divergence (base-2) between fake-probabilities p, q (arrays).

    Each branch prob is the 2-vector [1-pf, pf]; JS is symmetric in [0, 1].
    """
    p = np.clip(p, _EPS, 1 - _EPS)
    q = np.clip(q, _EPS, 1 - _EPS)
    P = np.stack([1 - p, p], axis=1)
    Q = np.stack([1 - q, q], axis=1)
    M = 0.5 * (P + Q)
    kl_pm = (P * (np.log2(P) - np.log2(M))).sum(axis=1)
    kl_qm = (Q * (np.log2(Q) - np.log2(M))).sum(axis=1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def add_frame_uncertainty_features(df):
    """Add per-frame commitment-weighted vacuity V and q-weighted branch JS C.

    Returns df with new columns 'V_frame' and 'C_frame'. Uses whatever branches
    are present (>=1 for V, >=2 for C; C=0 when fewer than two branches).
    """
    df = df.copy()
    branches = present_branches(df)

    # Per-branch commitment q_b = 1 - K/S_b and fake-prob.
    q, pf = {}, {}
    for b in branches:
        S = df[f'S_{b}'].to_numpy(dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            V = np.where(S > 0, K_CLASSES / S, np.nan)
        q[b] = np.clip(1.0 - V, 0.0, 1.0)
        pf[b] = df[f'p_fake_{b}'].to_numpy(dtype=float)

    n = len(df)
    if branches:
        qs = np.stack([q[b] for b in branches], axis=1)          # (n, B)
        Vs = np.stack([K_CLASSES / df[f'S_{b}'].to_numpy(dtype=float)
                       for b in branches], axis=1)
        wsum = np.nansum(qs, axis=1)
        V_frame = np.nansum(qs * Vs, axis=1) / np.clip(wsum, _EPS, None)
    else:
        V_frame = np.full(n, np.nan)

    C_num = np.zeros(n)
    C_den = np.zeros(n)
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            w = q[branches[i]] * q[branches[j]]
            js = _js_2class(pf[branches[i]], pf[branches[j]])
            C_num += w * js
            C_den += w
    C_frame = np.where(C_den > 0, C_num / np.clip(C_den, _EPS, None), 0.0)

    df['V_frame'] = V_frame
    df['C_frame'] = C_frame
    return df


def per_video_features(df):
    """Aggregate frame rows into one row per video with risk features.

    Columns: video_id, label, method, num_frames, p_fake (mean), pred, error,
             V, C, Q, T  (the four dual-uncertainty risk features).
      V = mean commitment-weighted vacuity
      C = mean q-weighted pairwise branch JS divergence (reasoning conflict)
      Q = frame-score variance + mean |p_fake - 0.5|   (ambiguity proxy)
      T = std of p_fake across frames                  (temporal instability)
    """
    df = add_frame_uncertainty_features(df)
    if 'method' not in df.columns:
        df['method'] = ''
    df['method'] = df['method'].fillna('').astype(str)
    rows = []
    # Group by (method, video_id): in FF++ the same source_target pair (video_id)
    # appears under every manipulation, so video_id alone would merge distinct
    # fakes. For OOD (method='') video_id is already unique.
    for (method, vid), g in df.groupby(['method', 'video_id'], sort=False):
        p = g['p_fake'].to_numpy(dtype=float)
        p = p[np.isfinite(p)]
        if len(p) == 0:
            continue
        label = int(g['label'].iloc[0])
        p_mean = float(p.mean())
        pred = int(p_mean >= 0.5)
        rows.append({
            'video_id': vid,
            'label': label,
            'method': method,
            'num_frames': len(p),
            'p_fake': p_mean,
            'pred': pred,
            'error': int(pred != label),
            'V': float(np.nanmean(g['V_frame'].to_numpy(dtype=float))),
            'C': float(np.nanmean(g['C_frame'].to_numpy(dtype=float))),
            'Q': float(np.var(p) + np.mean(np.abs(p - 0.5))),
            'T': float(np.std(p)),
        })
    return pd.DataFrame(rows)


def rule_names(yaml_path='configs/retained_predicates.yaml'):
    """rule_NN → predicate name map, in the model's violation-column order.

    RetainedConsistencyRules emits columns in the YAML `retained_predicates`
    LIST order (rank order), so rule_00 = retained_predicates[0]['name'], etc.
    Returns {'rule_00': 'cr_angry_au7', ...}; empty dict if the YAML is absent.
    """
    import yaml
    if not os.path.exists(yaml_path):
        return {}
    spec = yaml.safe_load(open(yaml_path))
    names = [str(e['name']) for e in spec.get('retained_predicates', [])]
    return {f'rule_{j:02d}': nm for j, nm in enumerate(names)}


def per_video_rule_means(df):
    """Per-(method, video_id) mean of each rule_NN violation column.

    Returns a DataFrame indexed to match per_video_features' (method, video_id)
    ordering, with one column per rule. Empty DataFrame if no rule columns.
    """
    rc = rule_columns(df)
    if not rc:
        return pd.DataFrame()
    d = df.copy()
    if 'method' not in d.columns:
        d['method'] = ''
    d['method'] = d['method'].fillna('').astype(str)
    g = d.groupby(['method', 'video_id'], sort=False)[rc].mean()
    return g.reset_index()


def risk_coverage(risk, error):
    """Selective-risk curve when abstaining by descending risk.

    Keep the lowest-risk samples first; at each coverage report the error rate
    of the kept set. Returns dict with coverage, selective_risk arrays, aurc,
    eaurc (excess over the oracle ordering by error), and coverage@risk hooks.
    """
    risk = np.asarray(risk, dtype=float)
    error = np.asarray(error, dtype=float)
    n = len(risk)
    order = np.argsort(risk, kind='mergesort')          # ascending risk
    sorted_err = error[order]
    cum_err = np.cumsum(sorted_err)
    coverage = np.arange(1, n + 1) / n
    selective_risk = cum_err / np.arange(1, n + 1)
    aurc = float(np.trapz(selective_risk, coverage))

    oracle = np.sort(error)                               # errors last
    oracle_risk = np.cumsum(oracle) / np.arange(1, n + 1)
    aurc_oracle = float(np.trapz(oracle_risk, coverage))
    return {
        'coverage': coverage,
        'selective_risk': selective_risk,
        'aurc': aurc,
        'eaurc': aurc - aurc_oracle,
    }


def coverage_at_risk(risk, error, target_risk):
    """Max coverage whose selective risk stays <= target_risk (0 if none)."""
    rc = risk_coverage(risk, error)
    ok = rc['selective_risk'] <= target_risk
    return float(rc['coverage'][ok].max()) if ok.any() else 0.0
