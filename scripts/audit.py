"""
scripts/audit.py  —  Phase-2 Task 6: intervention audit.

Quantifies what the neuro-symbolic fusion does to the spatial (neural) baseline,
per dataset and (for FF++) per manipulation, at the VIDEO level:

  benefit rate  = P(spatial wrong AND fused right)      — fusion fixes neural errors
  harm rate     = P(spatial right AND fused wrong)      — fusion breaks neural correct
  net correction= benefit - harm

Both decisions use thresholds calibrated on FF++ (video EER of each stream) and
FROZEN for every dataset — so the comparison is consistent and no OOD data drives
threshold choice (CRITICAL RULE).

Per-rule breakdown: mean per-rule violation on benefited vs harmed videos, ranked
by |benefit - harm| separation, so you can see which symbolic rules co-occur with
useful corrections. (Signed per-class `rule_contributions` are only produced by a
rule_linear concept head; when the CSV lacks them we fall back to the rule_NN
violation columns, noted in the report.)

Pure analysis: reads only per-sample CSVs, writes results/audit_report.md.

Usage (repo root):
    python scripts/audit.py --run logs/test/p4b_full_ccv_reeval
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nesy_csv_utils as U

FF = 'FaceForensics++'
FF_MANIPS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
ALL_DS = [FF, 'Celeb-DF-v2', 'Celeb-DF-v3', 'DeepFakeDetection', 'DFDC', 'DFDCP']


def eer_threshold(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores)
    m = np.isfinite(scores)
    fpr, tpr, thr = roc_curve(labels[m], scores[m])
    return float(thr[int(np.nanargmin(np.abs((1 - tpr) - fpr)))])


def video_scores(df):
    """Per-video mean p_fake (fused) and mean p_fake_spatial, with label/method."""
    d = U.per_video_features(df)                       # gives p_fake (fused), label, method
    # spatial per-video mean
    dd = df.copy()
    if 'method' not in dd.columns:
        dd['method'] = ''
    dd['method'] = dd['method'].fillna('').astype(str)
    sp = (dd.groupby(['method', 'video_id'], sort=False)['p_fake_spatial']
          .mean().reset_index().rename(columns={'p_fake_spatial': 'p_spatial'}))
    d = d.merge(sp, on=['method', 'video_id'], how='left')
    return d


def benefit_harm(vf, tau_sp, tau_fu):
    """benefit/harm/net rates for a video table at frozen thresholds."""
    sp_pred = (vf['p_spatial'].to_numpy() >= tau_sp).astype(int)
    fu_pred = (vf['p_fake'].to_numpy() >= tau_fu).astype(int)
    y = vf['label'].to_numpy()
    sp_right = sp_pred == y
    fu_right = fu_pred == y
    benefit = (~sp_right) & fu_right
    harm = sp_right & (~fu_right)
    n = len(vf)
    return {
        'n': n,
        'spatial_acc': float(sp_right.mean()),
        'fused_acc': float(fu_right.mean()),
        'benefit': float(benefit.mean()),
        'harm': float(harm.mean()),
        'net': float(benefit.mean() - harm.mean()),
        '_benefit_mask': benefit,
        '_harm_mask': harm,
    }


def rule_breakdown(df, vf, benefit_mask, harm_mask, top=8):
    """Mean per-rule violation on benefited vs harmed videos; ranked by |diff|."""
    rm = U.per_video_rule_means(df)
    if rm.empty:
        return None
    merged = vf.merge(rm, on=['method', 'video_id'], how='left')
    rule_cols = [c for c in merged.columns if c.startswith('rule_')]
    ben = merged.loc[benefit_mask, rule_cols].mean()
    har = merged.loc[harm_mask, rule_cols].mean()
    diff = (ben - har).abs().sort_values(ascending=False)
    rows = []
    for rc in diff.index[:top]:
        rows.append((rc, float(ben[rc]), float(har[rc]), float(ben[rc] - har[rc])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='logs/test/p4b_full_ccv_reeval')
    ap.add_argument('--datasets', nargs='+', default=ALL_DS)
    ap.add_argument('--out', default='results/audit_report.md')
    args = ap.parse_args()

    ff = U.load_frames(args.run, FF)
    if ff is None:
        raise SystemExit(f"missing FF++ CSV under {args.run}")
    vf_ff = video_scores(ff)
    tau_sp = eer_threshold(vf_ff['label'], vf_ff['p_spatial'])
    tau_fu = eer_threshold(vf_ff['label'], vf_ff['p_fake'])

    L = ['# Intervention audit — spatial → fused correction (Task 6)\n']
    L.append(f'- Run: `{args.run}`  |  thresholds frozen from FF++ video EER: '
             f'spatial={tau_sp:.3f}, fused={tau_fu:.3f}')
    L.append('')
    L.append('## By dataset (video level)\n')
    L.append('| dataset | n | spatial acc | fused acc | benefit | harm | **net** |')
    L.append('|---|---|---|---|---|---|---|')

    per_ds = {}
    for ds in args.datasets:
        df = U.load_frames(args.run, ds)
        if df is None:
            continue
        vf = video_scores(df)
        bh = benefit_harm(vf, tau_sp, tau_fu)
        per_ds[ds] = (df, vf, bh)
        L.append(f"| {ds} | {bh['n']} | {bh['spatial_acc']:.3f} | "
                 f"{bh['fused_acc']:.3f} | {bh['benefit']:.3f} | {bh['harm']:.3f} | "
                 f"**{bh['net']:+.3f}** |")

    # FF++ per-manipulation breakdown
    if FF in per_ds:
        df, vf, _ = per_ds[FF]
        L.append('')
        L.append('## FF++ by manipulation (video level)\n')
        L.append('| manipulation | n | spatial acc | fused acc | benefit | harm | **net** |')
        L.append('|---|---|---|---|---|---|---|')
        for manip in FF_MANIPS + ['youtube']:
            sub = vf[vf['method'] == manip]
            if len(sub) == 0:
                continue
            bh = benefit_harm(sub, tau_sp, tau_fu)
            L.append(f"| {manip} | {bh['n']} | {bh['spatial_acc']:.3f} | "
                     f"{bh['fused_acc']:.3f} | {bh['benefit']:.3f} | "
                     f"{bh['harm']:.3f} | **{bh['net']:+.3f}** |")

    # Per-rule breakdown (benefited vs harmed) on each dataset that has rules
    L.append('')
    L.append('## Per-rule breakdown — mean violation on benefited vs harmed videos\n')
    any_rules = False
    for ds, (df, vf, bh) in per_ds.items():
        rows = rule_breakdown(df, vf, bh['_benefit_mask'], bh['_harm_mask'])
        if rows is None:
            continue
        any_rules = True
        L.append(f'### {ds}  (benefited={int(bh["_benefit_mask"].sum())}, '
                 f'harmed={int(bh["_harm_mask"].sum())})\n')
        L.append('| rule | mean(benefited) | mean(harmed) | diff |')
        L.append('|---|---|---|---|')
        for rc, b, h, d in rows:
            L.append(f'| {rc} | {b:.3f} | {h:.3f} | {d:+.3f} |')
        L.append('')
    if not any_rules:
        L.append('_No rule columns in these CSVs (spatial-only run); '
                 'rule breakdown skipped._')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    open(args.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nWritten {args.out}')


if __name__ == '__main__':
    main()
