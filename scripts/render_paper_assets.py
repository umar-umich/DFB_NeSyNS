"""
scripts/render_paper_assets.py
==============================
Render paper-quality assets from the predicate-gap analysis:

  - results/tab_predicate_gaps.tex
        LaTeX table with columns: predicate | category | mean_real |
        mean_fake | gap | retained?

  - results/predicate_gap_barplot.png  (overwritten)
        Single-column-width, sans-serif, retained predicates highlighted
        in a contrasting colour, grayscale-readable hatch on dropped bars.

Inputs:
    results/predicate_gaps_sorted.csv
    configs/retained_predicates.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


# Human-readable display names for each candidate predicate. Used in
# the LaTeX table and in the bar-plot y-tick labels so the paper reads
# like English rather than variable identifiers. Order matches
# CANDIDATE_RULE_NAMES_V8.
PRETTY_NAMES = {
    # v7 inherited
    'cr_happy_au6':            'Happy $\\leftrightarrow$ AU6 (cheek raise)',
    'cr_happy_au12':           'Happy $\\leftrightarrow$ AU12 (lip corner)',
    'cr_surprise_au1au2':      'Surprise $\\leftrightarrow$ AU1\\,+\\,AU2',
    'cr_sad_au15':             'Sad $\\leftrightarrow$ AU15 (lip depressor)',
    'cr_angry_au4':            'Angry $\\leftrightarrow$ AU4 (brow lower)',
    'cr_fear_au1au5':          'Fear $\\leftrightarrow$ AU1\\,+\\,AU5',
    'cr_disgust_au9':          'Disgust $\\leftrightarrow$ AU9 (nose wrinkle)',
    'cr_contempt_au14':        'Contempt $\\leftrightarrow$ AU14 (dimpler)',
    'cr_neutral_any_au':       'Neutral $\\times$ max-AU',
    'cr_gaze_lr_divergence':   'Gaze L--R divergence',
    'cr_eye_asymmetry':        'Eye L--R symmetry',
    'cr_jaw_asymmetry':        'Jaw symmetry',
    # README §5 faceswap-specific
    'cr_yaw_gaze_misalign':    'Head yaw $\\leftrightarrow$ gaze',
    'cr_pose_facewidth':       'Yaw $\\leftrightarrow$ face W/H',
    'cr_lip_geometry_au12':    'Mouth aspect $\\leftrightarrow$ AU12',
    'cr_brow_eye_couple':      'Brow height $\\leftrightarrow$ AU1$-$AU4',
    # FACS additions
    'cr_sad_au4':              'Sad $\\leftrightarrow$ AU4 (brow lower)',
    'cr_angry_au7':            'Angry $\\leftrightarrow$ AU7 (lid tighten)',
    'cr_disgust_au17':         'Disgust $\\leftrightarrow$ AU17 (chin raise)',
    'cr_surprise_au2':         'Surprise $\\leftrightarrow$ AU2 (outer brow)',
    'cr_sad_au1':              'Sad $\\leftrightarrow$ AU1 (inner brow)',
    'cr_pitch_gaze_y':         'Pitch $\\leftrightarrow$ vertical gaze',
    'cr_roll_eye_sym':         'Head roll $\\leftrightarrow$ eye sym.',
    'cr_ipd_facewidth':        'Inter-pupillary $\\leftrightarrow$ face W',
    'cr_jaw_cheekbone':        'Jaw width $\\leftrightarrow$ face W/H',
    'cr_mouth_symmetry':       'Mouth symmetry',
    'cr_duchenne_smile':       'Duchenne smile (AU6$\\,\\&\\,$AU12)',
    'cr_genuine_surprise':     'Genuine surprise (AU1$\\,\\&\\,$AU2)',
}

# Human-readable category names. Used in both the LaTeX table and the
# bar plot so the paper reads like English.
PRETTY_CATEGORIES = {
    'expr-au':    'Expression–AU coherence',
    'pose-gaze':  'Pose–gaze coherence',
    'symmetry':   'Bilateral symmetry',
    'geometry':   'Facial geometry',
    'au-pair':    'AU co-firing',
    'other':      'Other',
}

# LaTeX variant (en-dash escaped for safety even though most engines
# render the unicode fine).
PRETTY_CATEGORIES_TEX = {
    'expr-au':    'Expression--AU coherence',
    'pose-gaze':  'Pose--gaze coherence',
    'symmetry':   'Bilateral symmetry',
    'geometry':   'Facial geometry',
    'au-pair':    'AU co-firing',
    'other':      'Other',
}

# Plain-text variant used by the bar plot (matplotlib mathtext can't
# parse the LaTeX-only commands above)
PRETTY_NAMES_PLAIN = {
    'cr_happy_au6':          'Happy ↔ AU6 (cheek raise)',
    'cr_happy_au12':         'Happy ↔ AU12 (lip corner)',
    'cr_surprise_au1au2':    'Surprise ↔ AU1+AU2',
    'cr_sad_au15':           'Sad ↔ AU15 (lip depressor)',
    'cr_angry_au4':          'Angry ↔ AU4 (brow lower)',
    'cr_fear_au1au5':        'Fear ↔ AU1+AU5',
    'cr_disgust_au9':        'Disgust ↔ AU9 (nose wrinkle)',
    'cr_contempt_au14':      'Contempt ↔ AU14 (dimpler)',
    'cr_neutral_any_au':     'Neutral × max-AU',
    'cr_gaze_lr_divergence': 'Gaze L–R divergence',
    'cr_eye_asymmetry':      'Eye L–R symmetry',
    'cr_jaw_asymmetry':      'Jaw symmetry',
    'cr_yaw_gaze_misalign':  'Head yaw ↔ gaze',
    'cr_pose_facewidth':     'Yaw ↔ face W/H',
    'cr_lip_geometry_au12':  'Mouth aspect ↔ AU12',
    'cr_brow_eye_couple':    'Brow height ↔ AU1−AU4',
    'cr_sad_au4':            'Sad ↔ AU4 (brow lower)',
    'cr_angry_au7':          'Angry ↔ AU7 (lid tighten)',
    'cr_disgust_au17':       'Disgust ↔ AU17 (chin raise)',
    'cr_surprise_au2':       'Surprise ↔ AU2 (outer brow)',
    'cr_sad_au1':            'Sad ↔ AU1 (inner brow)',
    'cr_pitch_gaze_y':       'Pitch ↔ vertical gaze',
    'cr_roll_eye_sym':       'Head roll ↔ eye sym.',
    'cr_ipd_facewidth':      'Inter-pupillary ↔ face W',
    'cr_jaw_cheekbone':      'Jaw width ↔ face W/H',
    'cr_mouth_symmetry':     'Mouth symmetry',
    'cr_duchenne_smile':     'Duchenne smile (AU6 & AU12)',
    'cr_genuine_surprise':   'Genuine surprise (AU1 & AU2)',
}


def _safe_tex(s: str) -> str:
    """Make a string safe for LaTeX (escape underscores in predicate names)."""
    return s.replace('_', r'\_')


def load_rows(csv_path: str):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def load_retained_names(yaml_path: str):
    import yaml
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)
    return [e['name'] for e in spec['retained_predicates']]


def write_latex_table(rows, retained_set, out_path: str) -> None:
    """8.5cm-wide booktabs table; predicates sorted by gap, retained
    column ticks whichever made it through selection."""
    lines = []
    lines.append(r'% Auto-generated by scripts/render_paper_assets.py')
    lines.append(r'% Source: results/predicate_gaps_sorted.csv')
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \small')
    lines.append(r'  \setlength{\tabcolsep}{4pt}')
    lines.append(r'  \begin{tabular}{@{}rllrrrrc@{}}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Rank & Predicate & Category & '
        r'$\bar v_{\rm real}$ & $\bar v_{\rm fake}$ & Gap & AUC & Retained \\')
    lines.append(r'    \midrule')
    for i, r in enumerate(rows, 1):
        ret = r['predicate'] in retained_set
        check = r'\checkmark' if ret else ''
        pretty = PRETTY_NAMES.get(r['predicate'], _safe_tex(r['predicate']))
        cat = PRETTY_CATEGORIES_TEX.get(r['category'], r['category'])
        lines.append(
            f'    {i} & '
            f'{pretty} & '
            f'{cat} & '
            f'{float(r["mean_real"]):.3f} & '
            f'{float(r["mean_fake"]):.3f} & '
            f'{float(r["gap"]):.3f} & '
            f'{float(r["auc"]):.3f} & '
            f'{check} \\\\'
        )
    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(
        r'  \caption{Per-predicate discriminative gap on the FF++ '
        r'training-fold validation slice (10\%, video-level split, '
        r'seed=42). 28 candidates ranked by '
        r'$|\bar v_{\rm fake}-\bar v_{\rm real}|$; the top 18 are '
        r'retained for the symbolic stream. AUC is the single-predicate '
        r'AUROC of the violation score (after polarity correction).}')
    lines.append(r'  \label{tab:predicate-gaps}')
    lines.append(r'\end{table}')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def restyle_barplot(rows, retained_set, out_path: str) -> None:
    rows_sorted = sorted(rows, key=lambda r: -float(r['gap']))
    names = [r['predicate'] for r in rows_sorted]
    gaps = np.array([float(r['gap']) for r in rows_sorted])
    is_retained = np.array(
        [n in retained_set for n in names], dtype=bool)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 7.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # Single-column width (~3.4 in for NeurIPS), tall enough for 28 labels
    fig, ax = plt.subplots(figsize=(3.4, 4.6))
    bars = ax.barh(
        np.arange(len(names))[::-1], gaps,
        color=np.where(is_retained, '#1f77b4', '#cccccc'),
        edgecolor='black', linewidth=0.4,
        height=0.78,
    )
    # Hatch dropped bars so the figure stays legible in greyscale
    for b, ret in zip(bars, is_retained):
        if not ret:
            b.set_hatch('////')

    ax.set_yticks(np.arange(len(names))[::-1])
    pretty_y = [PRETTY_NAMES_PLAIN.get(n, n) for n in names]
    ax.set_yticklabels(pretty_y, fontsize=6.5)
    ax.set_xlabel(
        r'$|\overline{v}_{\mathrm{fake}}-\overline{v}_{\mathrm{real}}|$',
        fontsize=8)
    ax.set_xlim(0, gaps.max() * 1.12 + 1e-6)

    # Numeric annotations
    for y, g in zip(np.arange(len(names))[::-1], gaps):
        ax.text(g + 0.0008, y, f'{g:.3f}',
                va='center', ha='left', fontsize=6.0,
                color='#222')

    # Cliff-line at the gap = 0.005 retention threshold
    ax.axvline(0.005, color='#d62728', ls='--', lw=0.8, alpha=0.85)
    ax.text(0.0055, len(names) - 1.5, r'gap $> 0.005$',
            color='#d62728', fontsize=6.5, va='center')

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='retained (k=18)'),
        Patch(facecolor='#cccccc', edgecolor='black', hatch='////',
              label='dropped'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=6.5,
              frameon=True, framealpha=0.92, borderpad=0.3)
    ax.set_title('Predicate selection by discriminative gap (FF++ val)',
                 fontsize=8.5, pad=4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',  default=os.path.join(
        REPO_ROOT, 'results', 'predicate_gaps_sorted.csv'))
    ap.add_argument('--yaml', default=os.path.join(
        REPO_ROOT, 'configs', 'retained_predicates.yaml'))
    ap.add_argument('--tex_out', default=os.path.join(
        REPO_ROOT, 'results', 'tab_predicate_gaps.tex'))
    ap.add_argument('--png_out', default=os.path.join(
        REPO_ROOT, 'results', 'predicate_gap_barplot.png'))
    args = ap.parse_args()

    rows = load_rows(args.csv)
    retained = set(load_retained_names(args.yaml))
    write_latex_table(rows, retained, args.tex_out)
    restyle_barplot(rows, retained, args.png_out)
    print(f'Wrote {args.tex_out}')
    print(f'Wrote {args.png_out}')


if __name__ == '__main__':
    main()
