"""
Compile ablation-probe results into paper-ready tables.

Reads each probe's latest test run under logs/test/<probe>_<timestamp>/ and emits:
  1. results_auc.md / .csv      — probe x dataset, frame & video AUC (+ EER)
  2. results_headline.md        — the README table (FF++ frame, CDFv2 frame/video)
  3. results_diagnostics.csv    — per-branch EDL diagnostics from the per_sample
                                  CSVs (mean uncertainty / disagreement / gates,
                                  and per-branch p_fake split by real/fake).

Run from the repo root:
    python training/config/detector/probes/compile_results.py \
        [--test_dir logs/test] [--out logs/probes_summary]
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

# Logical probe order (matches the manifest).
PROBES = [
    'p1_spatial_ce', 'p2_spatial_edl', 'p3_concept', 'p3a_rules_only',
    'p3b_substrate', 'p4_full_scm', 'p4a_causal_only', 'p4b_full_ccv',
]
# Dataset display order; only those present are shown.
DATASETS = ['FaceForensics++', 'Celeb-DF-v2', 'Celeb-DF-v3',
            'DeepFakeDetection', 'DFDC', 'DFDCP']

_DIAG = ['uncertainty', 'disagreement_d', 'concept_gate', 'causal_gate',
         'p_fake_spatial', 'p_fake_concept', 'p_fake_causal']

# Per-(probe, dataset) run-dir overrides. A dataset that was re-evaluated in a
# separate run supersedes the probe's main run for that dataset only. The CCV
# numerical-hardening fix re-ran DFDC + DeepFakeDetection (which were NaN /
# missing under the original p4b_full_ccv run) as `p4b_fix_eval`. Provenance
# stays visible: the overridden rows carry run_dir=p4b_fix_eval.
RUN_OVERRIDES = {
    ('p4b_full_ccv', 'DFDC'): 'p4b_fix_eval',
    ('p4b_full_ccv', 'DeepFakeDetection'): 'p4b_fix_eval',
}


def find_run(test_dir, probe):
    """Latest logs/test/<probe>_<timestamp>/ for a probe (by sorted name)."""
    hits = sorted(glob.glob(os.path.join(test_dir, f'{probe}_*')))
    # Exact-name runs (no timestamp, e.g. an override dir) also count.
    exact = os.path.join(test_dir, probe)
    if not hits and os.path.isdir(exact):
        return exact
    return hits[-1] if hits else None


def run_for(test_dir, runs, probe, dataset):
    """Run dir to read (probe, dataset) from — an override if one exists,
    else the probe's main run."""
    override = RUN_OVERRIDES.get((probe, dataset))
    if override:
        hits = sorted(glob.glob(os.path.join(test_dir, f'{override}*')))
        if hits:
            return hits[-1]
    return runs.get(probe)


def read_metrics(run_dir, dataset):
    path = os.path.join(run_dir, dataset, 'metrics.csv')
    if not os.path.exists(path):
        return None
    return next(csv.DictReader(open(path)))


def _fmt(v, nd=4):
    try:
        return f'{float(v):.{nd}f}'
    except (TypeError, ValueError):
        return ''


def _mean(vals):
    vals = [v for v in vals if v != '' and v is not None]
    return sum(map(float, vals)) / len(vals) if vals else None


def diagnostics(run_dir, dataset):
    """Mean per-branch EDL diagnostics from per_sample_<dataset>.csv, by class."""
    path = os.path.join(run_dir, f'per_sample_{dataset}.csv')
    if not os.path.exists(path):
        return None
    by_class = {0: defaultdict(list), 1: defaultdict(list), 'all': defaultdict(list)}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                lab = int(row['label'])
            except (KeyError, ValueError):
                continue
            for c in _DIAG:
                v = row.get(c, '')
                if v != '':
                    by_class[lab][c].append(v)
                    by_class['all'][c].append(v)
    out = {}
    for scope in (0, 1, 'all'):
        for c in _DIAG:
            out[f'{c}[{scope}]'] = _mean(by_class[scope][c])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', default='logs/test')
    ap.add_argument('--out', default='logs/probes_summary')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = {p: find_run(args.test_dir, p) for p in PROBES}
    present_probes = [p for p in PROBES if runs[p]]
    # datasets that appear in at least one run (honoring per-dataset overrides)
    datasets = [d for d in DATASETS
                if any(run_for(args.test_dir, runs, p, d)
                       and os.path.isdir(os.path.join(run_for(args.test_dir, runs, p, d), d))
                       for p in present_probes)]

    # ---- 0. COMPLETE metrics dump: one row per (probe, dataset), ALL metrics ----
    # This is the full table for downstream analysis — every column of every
    # metrics.csv, plus probe / dataset / run_dir provenance.
    all_rows = []
    metric_keys = []
    for p in present_probes:
        for d in datasets:
            rd = run_for(args.test_dir, runs, p, d)
            m = read_metrics(rd, d)
            if m is None:
                continue
            for k in m:
                if k not in metric_keys:
                    metric_keys.append(k)
            all_rows.append({'probe': p, 'dataset': d,
                             'run_dir': os.path.basename(rd), **m})
    metric_keys = sorted(metric_keys)
    all_cols = ['probe', 'dataset', 'run_dir'] + metric_keys
    with open(os.path.join(args.out, 'all_metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)

    # ---- 1. AUC matrix (frame + video) ----
    auc_rows = []
    for p in present_probes:
        row = {'probe': p}
        for d in datasets:
            m = read_metrics(run_for(args.test_dir, runs, p, d), d)
            row[f'{d}|frame_auc'] = _fmt(m['auroc_frame']) if m else ''
            row[f'{d}|video_auc'] = _fmt(m['auroc_video']) if m else ''
            row[f'{d}|frame_eer'] = _fmt(m['eer_frame']) if m else ''
        auc_rows.append(row)

    auc_cols = ['probe'] + [f'{d}|{k}' for d in datasets
                            for k in ('frame_auc', 'video_auc', 'frame_eer')]
    with open(os.path.join(args.out, 'results_auc.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=auc_cols)
        w.writeheader()
        w.writerows(auc_rows)

    # markdown AUC table (frame + video AUC only, compact)
    md = ['# Ablation probe results — AUC', '',
          '| probe | ' + ' | '.join(f'{d} F/V' for d in datasets) + ' |',
          '|' + '---|' * (len(datasets) + 1)]
    for r in auc_rows:
        cells = [f"{r[f'{d}|frame_auc'] or '-'} / {r[f'{d}|video_auc'] or '-'}"
                 for d in datasets]
        md.append(f"| {r['probe']} | " + ' | '.join(cells) + ' |')
    open(os.path.join(args.out, 'results_auc.md'), 'w').write('\n'.join(md) + '\n')

    # ---- 2. headline table (README shape) ----
    hl = ['# Headline results', '',
          '| probe | FF++ frame AUC | CDFv2 frame AUC | CDFv2 video AUC |',
          '|---|---|---|---|']
    for r in auc_rows:
        ff = r.get('FaceForensics++|frame_auc', '') or '-'
        cf = r.get('Celeb-DF-v2|frame_auc', '') or '-'
        cv = r.get('Celeb-DF-v2|video_auc', '') or '-'
        hl.append(f'| {r["probe"]} | {ff} | {cf} | {cv} |')
    open(os.path.join(args.out, 'results_headline.md'), 'w').write('\n'.join(hl) + '\n')

    # ---- 3. per-branch diagnostics ----
    diag_cols = ['probe', 'dataset'] + [f'{c}[{s}]' for s in (0, 1, 'all') for c in _DIAG]
    with open(os.path.join(args.out, 'results_diagnostics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=diag_cols)
        w.writeheader()
        for p in present_probes:
            for d in datasets:
                dg = diagnostics(run_for(args.test_dir, runs, p, d), d)
                if dg is None:
                    continue
                w.writerow({'probe': p, 'dataset': d,
                            **{k: _fmt(v, 4) for k, v in dg.items()}})

    # ---- console summary ----
    print(f'Compiled {len(present_probes)}/{len(PROBES)} probes over {len(datasets)} datasets')
    for p in PROBES:
        print(f'  {p:18} {"→ " + os.path.basename(runs[p]) if runs[p] else "MISSING"}')
    print('\n' + '\n'.join(hl))
    print(f'\nComplete metrics: {len(all_rows)} (probe x dataset) rows x '
          f'{len(metric_keys)} metrics -> all_metrics.csv')
    print(f'\nWritten to {args.out}/:')
    for fn in ('all_metrics.csv', 'results_auc.md', 'results_auc.csv',
               'results_headline.md', 'results_diagnostics.csv'):
        print(f'  {fn}')


if __name__ == '__main__':
    main()
