"""
scripts/risk_head.py  —  Phase-2 Task 5 (S4): dual uncertainty + post-hoc risk head.

Reads the per-sample CSVs of a trained NeSy run and, per VIDEO, computes four
dual-uncertainty features:

  V  mean commitment-weighted Dirichlet vacuity  (how uninformative the branches are)
  C  mean q-weighted pairwise branch JS divergence (reasoning conflict between branches)
  Q  frame-score variance + mean |p-0.5|          (prediction ambiguity)
  T  std of p_fake across frames                  (temporal instability)

Fits a logistic risk head  R = sigma(w·[V,C,Q,T] + b)  to predict the frozen
detector's ERRORS, using ONLY FaceForensics++ (leave-one-manipulation-out CV for
the generalization estimate; final head refit on all FF++). The head + the
FF++-calibrated decision threshold are then FROZEN and evaluated on the OOD
datasets: error-prediction AUROC, risk-coverage curves, AURC / E-AURC,
coverage@target-risk.

CRITICAL RULE (enforced): fitting touches FF++ only. `--fit_dataset` must be
FaceForensics++; the threshold and scaler are calibrated on FF++ and reused
unchanged on OOD. OOD data is read for EVALUATION only.

Usage (repo root):
    python scripts/risk_head.py \
        --run logs/test/p4b_full_ccv_reeval \
        [--target_risk 0.10] [--out results/risk_report.md]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nesy_csv_utils as U

FEATURES = ['V', 'C', 'Q', 'T']
FF = 'FaceForensics++'
FF_MANIPS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
ALL_OOD = ['Celeb-DF-v2', 'Celeb-DF-v3', 'DeepFakeDetection', 'DFDC', 'DFDCP']


def eer_threshold(labels, scores):
    """Equal-error-rate threshold from an ROC curve."""
    fpr, tpr, thr = roc_curve(labels, scores)
    i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    return float(thr[i])


def errors_at(vf, tau):
    """Video error labels at decision threshold tau."""
    pred = (vf['p_fake'].to_numpy() >= tau).astype(int)
    return (pred != vf['label'].to_numpy()).astype(int)


def real_fold_id(video_ids, n_folds):
    """Deterministic fold assignment for real videos (stable hash)."""
    return np.array([hash(('rf', v)) % n_folds for v in video_ids])


def lomo_cv(vf, tau):
    """Leave-one-manipulation-out CV of the risk head on FF++.

    Fold k holds out manipulation k's fakes + a fixed 1/4 slice of the reals;
    trains on the rest. Returns per-fold error-prediction AUROC.
    """
    err = errors_at(vf, tau)
    vf = vf.copy()
    vf['_err'] = err
    is_real = vf['method'].to_numpy() == 'youtube'
    real_folds = real_fold_id(vf.loc[is_real, 'video_id'], len(FF_MANIPS))
    real_fold_map = dict(zip(vf.loc[is_real, 'video_id'], real_folds))

    aucs = {}
    for k, manip in enumerate(FF_MANIPS):
        val_mask = np.zeros(len(vf), dtype=bool)
        val_mask |= (vf['method'].to_numpy() == manip)
        val_mask |= np.array([real_fold_map.get(v, -1) == k
                              for v in vf['video_id']])
        train_mask = ~val_mask
        Xtr = vf.loc[train_mask, FEATURES].to_numpy()
        ytr = vf.loc[train_mask, '_err'].to_numpy()
        Xva = vf.loc[val_mask, FEATURES].to_numpy()
        yva = vf.loc[val_mask, '_err'].to_numpy()
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            aucs[manip] = float('nan')
            continue
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(sc.transform(Xtr), ytr)
        risk = clf.predict_proba(sc.transform(Xva))[:, 1]
        aucs[manip] = float(roc_auc_score(yva, risk))
    return aucs


def fit_final_head(vf, tau):
    """Fit scaler + logistic head on ALL FF++ videos. Returns (scaler, clf)."""
    X = vf[FEATURES].to_numpy()
    y = errors_at(vf, tau)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(sc.transform(X), y)
    return sc, clf


def risk_of(vf, scaler, clf):
    return clf.predict_proba(scaler.transform(vf[FEATURES].to_numpy()))[:, 1]


def evaluate(vf, risk, tau, target_risk):
    """Error-prediction AUROC + risk-coverage metrics for one dataset."""
    err = errors_at(vf, tau)
    out = {'n': len(vf), 'error_rate': float(err.mean())}
    out['error_auroc'] = (float(roc_auc_score(err, risk))
                          if len(np.unique(err)) == 2 else float('nan'))
    rc = U.risk_coverage(risk, err)
    out['aurc'] = rc['aurc']
    out['eaurc'] = rc['eaurc']
    out['cov@risk'] = U.coverage_at_risk(risk, err, target_risk)
    out['_rc'] = rc
    return out


def plot_rc(results, figdir):
    os.makedirs(figdir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for ds, r in results.items():
        rc = r['_rc']
        plt.plot(rc['coverage'], rc['selective_risk'], label=f"{ds} (E-AURC {r['eaurc']:.3f})")
    plt.xlabel('coverage'); plt.ylabel('selective risk (error rate)')
    plt.title('Risk-coverage by dataset (risk head R)')
    plt.legend(fontsize=7); plt.grid(alpha=0.3); plt.tight_layout()
    path = os.path.join(figdir, 'risk_coverage.png')
    plt.savefig(path, dpi=130); plt.close()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='logs/test/p4b_full_ccv_reeval')
    ap.add_argument('--fit_dataset', default=FF)
    ap.add_argument('--ood', nargs='+', default=ALL_OOD)
    ap.add_argument('--target_risk', type=float, default=0.10)
    ap.add_argument('--out', default='results/risk_report.md')
    ap.add_argument('--figdir', default='results/risk_figs')
    args = ap.parse_args()

    # ── FF++-only-fitting guard ───────────────────────────────────────────
    if args.fit_dataset != FF:
        raise SystemExit(
            f"[GUARD] risk head must be fit on {FF} only; got "
            f"--fit_dataset={args.fit_dataset!r}. OOD is evaluation-only.")

    ff = U.load_frames(args.run, FF)
    if ff is None:
        raise SystemExit(f"missing FF++ CSV under {args.run}")
    vf_ff = U.per_video_features(ff)

    # Decision threshold calibrated on FF++ (video EER), frozen for OOD.
    tau = eer_threshold(vf_ff['label'].to_numpy(), vf_ff['p_fake'].to_numpy())

    # Generalization estimate: leave-one-manipulation-out CV.
    cv = lomo_cv(vf_ff, tau)
    cv_vals = [v for v in cv.values() if v == v]
    cv_mean = float(np.mean(cv_vals)) if cv_vals else float('nan')

    # Final frozen head on all FF++.
    scaler, clf = fit_final_head(vf_ff, tau)

    # Evaluate on FF++ (in-domain reference) + each OOD dataset.
    results = {}
    results[FF] = evaluate(vf_ff, risk_of(vf_ff, scaler, clf), tau, args.target_risk)
    for ds in args.ood:
        df = U.load_frames(args.run, ds)
        if df is None:
            continue
        vf = U.per_video_features(df)
        results[ds] = evaluate(vf, risk_of(vf, scaler, clf), tau, args.target_risk)

    figpath = plot_rc(results, args.figdir)

    # ── Report ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    coef = dict(zip(FEATURES, clf.coef_[0]))
    L = []
    L.append('# Post-hoc risk head — dual uncertainty (Task 5 / S4)\n')
    L.append(f'- Run: `{args.run}`  |  fit: **{FF} only**  |  '
             f'decision threshold (FF++ video EER) tau = **{tau:.3f}**')
    L.append(f'- Risk features: {FEATURES}  |  head = logistic regression '
             '(class-balanced), standardized features')
    L.append(f'- Frozen head weights (standardized): '
             + ', '.join(f'{k}={v:+.3f}' for k, v in coef.items())
             + f', bias={clf.intercept_[0]:+.3f}')
    L.append('')
    L.append('## FF++ leave-one-manipulation-out (error-prediction AUROC)\n')
    L.append('| held-out manip | ' + ' | '.join(FF_MANIPS) + ' | **mean** |')
    L.append('|' + '---|' * (len(FF_MANIPS) + 2))
    L.append('| val AUROC | ' + ' | '.join(f'{cv[m]:.3f}' for m in FF_MANIPS)
             + f' | **{cv_mean:.3f}** |')
    L.append('')
    L.append(f'## Frozen head on OOD (threshold + head fixed from FF++)\n')
    L.append('| dataset | n | error rate | error-pred AUROC | AURC | E-AURC | '
             f'cov@risk≤{args.target_risk:.2f} |')
    L.append('|---|---|---|---|---|---|---|')
    for ds, r in results.items():
        tag = ' *(in-domain)*' if ds == FF else ''
        L.append(f"| {ds}{tag} | {r['n']} | {r['error_rate']:.3f} | "
                 f"{r['error_auroc']:.3f} | {r['aurc']:.4f} | {r['eaurc']:.4f} | "
                 f"{r['cov@risk']:.3f} |")
    L.append('')
    L.append(f'Risk-coverage figure: `{figpath}`')
    open(args.out, 'w').write('\n'.join(L) + '\n')

    print('\n'.join(L))
    print(f'\nWritten {args.out} and {figpath}')


if __name__ == '__main__':
    main()
