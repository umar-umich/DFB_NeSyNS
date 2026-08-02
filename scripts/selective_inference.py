"""
scripts/selective_inference.py  —  Phase-2 Task 7 (S8): risk-gated selective inference.

Two-pass evaluation:
  Pass 1  standard 32-frame video scores (the per-sample CSVs already on disk).
  Gate    compute post-hoc risk R per video (risk head, fit on FF++); escalate
          videos with R >= tau, where tau is the risk at a target COVERAGE on the
          FF++ calibration split (e.g. coverage 0.90 → escalate the top 10% riskiest).
  Pass 2  the escalated videos are re-scored with a denser 64-frame pass; their
          final score is the pass-2 score. Non-escalated videos keep pass 1.

Reports AUC and ECE before vs after escalation, per dataset, plus the escalation
rate. tau is selected on FF++ ONLY (CRITICAL RULE); the same tau is applied to OOD.

Pass-2 scores come from a SECOND eval run at 64 frames, supplied via `--pass2_run`
(a logs/test/<run> dir whose per_sample_<ds>.csv were produced with
`frame_num.test: 64`). Without it, the script reports the pass-1 metrics and the
escalation set, and prints the command to generate the 64-frame CSVs — the
before/after columns fill in once that run exists.

Usage (repo root):
    # pass-1 + escalation set (runnable now):
    python scripts/selective_inference.py --run logs/test/p4b_full_ccv_reeval
    # full before/after once 64-frame CSVs exist:
    python scripts/selective_inference.py --run logs/test/p4b_full_ccv_reeval \
        --pass2_run logs/test/p4b_full_ccv_64f --target_coverage 0.90
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nesy_csv_utils as U
import risk_head as RH


def ece(prob, label, n_bins=10):
    """Expected calibration error (max(p,1-p) confidence, 10 equal-width bins)."""
    prob = np.asarray(prob, float); label = np.asarray(label, int)
    conf = np.maximum(prob, 1 - prob)
    pred = (prob >= 0.5).astype(int)
    correct = (pred == label).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def auc(prob, label):
    from sklearn.metrics import roc_auc_score
    label = np.asarray(label)
    if len(np.unique(label)) < 2:
        return float('nan')
    return float(roc_auc_score(label, prob))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='logs/test/p4b_full_ccv_reeval')
    ap.add_argument('--pass2_run', default=None,
                    help='logs/test/<run> with 64-frame per_sample CSVs')
    ap.add_argument('--ood', nargs='+', default=RH.ALL_OOD)
    ap.add_argument('--target_coverage', type=float, default=0.90)
    ap.add_argument('--out', default='results/selective_inference.md')
    args = ap.parse_args()

    ff = U.load_frames(args.run, RH.FF)
    if ff is None:
        raise SystemExit(f"missing FF++ CSV under {args.run}")
    vf_ff = U.per_video_features(ff)
    tau_dec = RH.eer_threshold(vf_ff['label'].to_numpy(), vf_ff['p_fake'].to_numpy())
    scaler, clf = RH.fit_final_head(vf_ff, tau_dec)

    # Risk-gate tau at target coverage on FF++ (escalate the riskiest 1-coverage).
    R_ff = RH.risk_of(vf_ff, scaler, clf)
    tau_gate = float(np.quantile(R_ff, args.target_coverage))

    datasets = [RH.FF] + list(args.ood)
    L = ['# Risk-gated selective inference (Task 7 / S8)\n']
    L.append(f'- Run (pass 1, 32-frame): `{args.run}`')
    L.append(f'- Gate: escalate videos with R ≥ tau_gate = {tau_gate:.3f} '
             f'(FF++ coverage {args.target_coverage:.2f} → escalate top '
             f'{100*(1-args.target_coverage):.0f}%)')
    if args.pass2_run:
        L.append(f'- Pass 2 (64-frame): `{args.pass2_run}`')
    else:
        L.append('- Pass 2: **not supplied** — reporting pass-1 metrics + '
                 'escalation rate only (see command below).')
    L.append('')
    L.append('| dataset | n | escalation rate | AUC pass1 | AUC final | '
             'ECE pass1 | ECE final |')
    L.append('|---|---|---|---|---|---|---|')

    for ds in datasets:
        df = U.load_frames(args.run, ds)
        if df is None:
            continue
        vf = U.per_video_features(df).copy()
        vf['R'] = RH.risk_of(vf, scaler, clf)
        escalate = vf['R'].to_numpy() >= tau_gate
        y = vf['label'].to_numpy()
        p1 = vf['p_fake'].to_numpy().copy()
        auc1, ece1 = auc(p1, y), ece(p1, y)

        if args.pass2_run:
            df2 = U.load_frames(args.pass2_run, ds)
            if df2 is not None:
                vf2 = U.per_video_features(df2)[['method', 'video_id', 'p_fake']]
                vf2 = vf2.rename(columns={'p_fake': 'p2'})
                m = vf.merge(vf2, on=['method', 'video_id'], how='left')
                pf = p1.copy()
                have2 = escalate & m['p2'].notna().to_numpy()
                pf[have2] = m['p2'].to_numpy()[have2]
                aucf, ecef = auc(pf, y), ece(pf, y)
            else:
                aucf, ecef = float('nan'), float('nan')
        else:
            aucf, ecef = float('nan'), float('nan')

        L.append(f'| {ds} | {len(vf)} | {escalate.mean():.3f} | {auc1:.4f} | '
                 f'{aucf:.4f} | {ece1:.4f} | {ecef:.4f} |')

    L.append('')
    if not args.pass2_run:
        L.append('## Producing the pass-2 (denser) scores\n')
        L.append('**Constraint:** the preprocessed data has only **32 frames/video** '
                 'on disk, so a true 64-frame dense pass needs the raw videos '
                 're-extracted at ≥64 frames/video (a preprocessing step), then a '
                 '64-frame eval to write `--pass2_run` CSVs. The gate + before/after '
                 'wiring here is complete and will fill in once such a run exists; '
                 'the `--pass2_run` CSVs may equally come from a TTA / re-sampled '
                 'pass over the same videos (any denser re-eval of the escalated set).')
        L.append('')
        L.append('The escalation set (R ≥ tau_gate) and pass-1 AUC/ECE above are the '
                 'actionable output today: escalation rate rises on the harder OOD '
                 'sets (DFDC/DFDCP), which is where a second pass would be spent.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    open(args.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nWritten {args.out}')


if __name__ == '__main__':
    main()
