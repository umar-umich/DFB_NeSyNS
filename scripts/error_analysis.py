"""
scripts/error_analysis.py  —  Phase-2 Task 6 (error analysis).

For the MISCLASSIFIED videos of each OOD dataset, characterises the failures:
distributions of the dual-uncertainty signals V (vacuity) and C (branch conflict),
the post-hoc risk R, and the top firing symbolic rules — separated into
false-positives (real called fake) and false-negatives (fake called real).

Decisions use the FF++-calibrated video-EER threshold; the risk head R is fit on
FF++ only (both imported from risk_head). OOD data is read for analysis only.

Pure analysis. Writes results/error_analysis.md. NOTHING in the model/mechanism
imports this script.

Usage (repo root):
    python scripts/error_analysis.py --run logs/test/p4b_full_ccv_reeval
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nesy_csv_utils as U
import risk_head as RH


def top_rules(df, vf, mask, k=5):
    """Top-k rules by mean violation over the masked videos."""
    rm = U.per_video_rule_means(df)
    if rm.empty or mask.sum() == 0:
        return []
    merged = vf.merge(rm, on=['method', 'video_id'], how='left')
    rule_cols = [c for c in merged.columns if c.startswith('rule_')]
    means = merged.loc[mask, rule_cols].mean().sort_values(ascending=False)
    return [(r, float(means[r])) for r in means.index[:k]]


def _stats(vf, mask, col):
    v = vf.loc[mask, col].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return (float(v.mean()), float(v.std())) if len(v) else (float('nan'),) * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='logs/test/p4b_full_ccv_reeval')
    ap.add_argument('--ood', nargs='+', default=RH.ALL_OOD)
    ap.add_argument('--out', default='results/error_analysis.md')
    args = ap.parse_args()

    # Fit the frozen risk head + threshold on FF++ (guarded inside risk_head).
    ff = U.load_frames(args.run, RH.FF)
    if ff is None:
        raise SystemExit(f"missing FF++ CSV under {args.run}")
    vf_ff = U.per_video_features(ff)
    tau = RH.eer_threshold(vf_ff['label'].to_numpy(), vf_ff['p_fake'].to_numpy())
    scaler, clf = RH.fit_final_head(vf_ff, tau)

    L = ['# Error analysis — misclassified OOD videos (Task 6)\n']
    L.append(f'- Run: `{args.run}`  |  threshold (FF++ video EER) = {tau:.3f}  |  '
             'risk head R fit on FF++ only')
    L.append('')

    for ds in args.ood:
        df = U.load_frames(args.run, ds)
        if df is None:
            continue
        vf = U.per_video_features(df)
        vf = vf.copy()
        vf['R'] = RH.risk_of(vf, scaler, clf)
        pred = (vf['p_fake'].to_numpy() >= tau).astype(int)
        y = vf['label'].to_numpy()
        fp = (y == 0) & (pred == 1)
        fn = (y == 1) & (pred == 0)
        n_err = int(fp.sum() + fn.sum())

        L.append(f'## {ds}  —  {n_err} errors '
                 f'(FP={int(fp.sum())}, FN={int(fn.sum())}) of {len(vf)} videos\n')
        L.append('| group | n | V mean±sd | C mean±sd | R mean±sd |')
        L.append('|---|---|---|---|---|')
        for name, mask in [('false-positive (real→fake)', fp),
                           ('false-negative (fake→real)', fn)]:
            vM, vS = _stats(vf, mask, 'V')
            cM, cS = _stats(vf, mask, 'C')
            rM, rS = _stats(vf, mask, 'R')
            L.append(f'| {name} | {int(mask.sum())} | {vM:.3f}±{vS:.3f} | '
                     f'{cM:.3f}±{cS:.3f} | {rM:.3f}±{rS:.3f} |')
        # reference: correctly-classified
        correct = pred == y
        vM, vS = _stats(vf, correct, 'V'); cM, cS = _stats(vf, correct, 'C')
        rM, rS = _stats(vf, correct, 'R')
        L.append(f'| correct (reference) | {int(correct.sum())} | {vM:.3f}±{vS:.3f} | '
                 f'{cM:.3f}±{cS:.3f} | {rM:.3f}±{rS:.3f} |')
        L.append('')

        for name, mask in [('FP top rules', fp), ('FN top rules', fn)]:
            tr = top_rules(df, vf, mask)
            if tr:
                L.append(f'- **{name}**: '
                         + ', '.join(f'{r}={v:.3f}' for r, v in tr))
        L.append('')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    open(args.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nWritten {args.out}')


if __name__ == '__main__':
    main()
