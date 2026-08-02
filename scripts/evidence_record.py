"""
scripts/evidence_record.py  —  Phase-2 Task 8 (S7): structured evidence record.

For a single video, emit a JSON + markdown record summarising WHY the detector
reached its verdict:

  verdict, p_fake, V (vacuity), C (branch conflict), R (risk),
  decision (decide / escalate / defer),
  top-5 firing rules (named from retained_predicates.yaml),
  per-branch evidence bars (spatial / concept / causal),
  CCV forensic anomaly-group scores, counterfactual residual,
  temporal profile (p_fake and disagreement d per frame).

Thresholds (decision tau, escalate/defer risk cut-offs) come from FF++ only.

Optional --render: turn the JSON into 4 plain sentences via an external LLM
(`--llm_cmd`, reads JSON on stdin → prose on stdout). A faithfulness check then
verifies EVERY number in the prose appears in the JSON; on failure (or when no
LLM is configured) it falls back to a template that is faithful by construction.

Usage (repo root):
    python scripts/evidence_record.py --run logs/test/p4b_full_ccv_reeval \
        --dataset FaceForensics++ --video_id 158_379 --method Face2Face --render
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nesy_csv_utils as U
import risk_head as RH

BRANCHES = ['spatial', 'concept', 'causal']


def _num(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None


def build_record(run, dataset, video_id, method, tau, tau_esc, tau_def,
                 scaler, clf, names):
    df = U.load_frames(run, dataset)
    if df is None:
        raise SystemExit(f"missing {dataset} CSV under {run}")
    if 'method' not in df.columns:
        df['method'] = ''
    df['method'] = df['method'].fillna('').astype(str)
    sel = df['video_id'].astype(str) == str(video_id)
    if method is not None:
        sel &= df['method'] == method
    g = df[sel].copy()
    if len(g) == 0:
        raise SystemExit(f"no frames for video_id={video_id} "
                         f"(method={method}) in {dataset}")

    # order frames
    def _fkey(s):
        try:
            return int(re.sub(r'\D', '', str(s)) or 0)
        except ValueError:
            return 0
    g = g.sort_values('frame_idx', key=lambda s: s.map(_fkey))

    vf = U.per_video_features(g).iloc[0]
    R = float(RH.risk_of(vf.to_frame().T, scaler, clf)[0])
    p_fake = float(vf['p_fake'])
    verdict = 'fake' if p_fake >= tau else 'real'
    if R >= tau_def:
        decision = 'defer'
    elif R >= tau_esc:
        decision = 'escalate'
    else:
        decision = 'decide'

    # top-5 firing rules (mean violation over frames), named
    rule_cols = U.rule_columns(g)
    top_rules = []
    if rule_cols:
        means = g[rule_cols].mean().sort_values(ascending=False)
        for rc in means.index[:5]:
            top_rules.append({'rule': rc, 'name': names.get(rc, rc),
                              'mean_violation': _num(means[rc])})

    # per-branch evidence bars
    branch_bars = {}
    for b in BRANCHES:
        pc, sc = f'p_fake_{b}', f'S_{b}'
        if pc in g.columns and g[pc].notna().any():
            branch_bars[b] = {'p_fake': _num(g[pc].mean()),
                              'S': _num(g[sc].mean()) if sc in g.columns else None}

    # CCV diagnostics
    anomaly = {c: _num(g[c].mean()) for c in sorted(g.columns)
               if c.startswith('anomaly_') and g[c].notna().any()}
    cf_residual = _num(g['cf_residual'].mean()) if (
        'cf_residual' in g.columns and g['cf_residual'].notna().any()) else None

    # temporal profile
    temporal = {
        'frame_idx': [str(x) for x in g['frame_idx'].tolist()],
        'p_fake': [_num(x) for x in g['p_fake'].tolist()],
        'disagreement_d': [_num(x) for x in g['disagreement_d'].tolist()]
                          if 'disagreement_d' in g.columns else [],
    }

    return {
        'dataset': dataset, 'video_id': str(video_id),
        'method': method or (g['method'].iloc[0] or ''),
        'label': int(vf['label']), 'num_frames': int(vf['num_frames']),
        'verdict': verdict, 'p_fake': _num(p_fake),
        'decision': decision, 'decision_thresholds': {
            'tau_decision': _num(tau), 'tau_escalate': _num(tau_esc),
            'tau_defer': _num(tau_def)},
        'V': _num(vf['V']), 'C': _num(vf['C']),
        'Q': _num(vf['Q']), 'T': _num(vf['T']), 'R': _num(R),
        'top_rules': top_rules, 'branch_evidence': branch_bars,
        'ccv_anomaly_groups': anomaly, 'counterfactual_residual': cf_residual,
        'temporal_profile': temporal,
    }


# ── prose rendering + faithfulness check ───────────────────────────────────

def _record_numbers_and_literals(rec):
    """Numeric values (floats) + digit-groups from string fields (identifiers).

    A prose number is faithful if it matches a numeric value at the prose's own
    decimal precision, OR appears verbatim as a digit-group in a record string
    (e.g. the video_id '376_381' → literals {'376', '381'}).
    """
    nums, literals = [], set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            nums.append(float(x))
        elif isinstance(x, str):
            literals.update(re.findall(r'\d+', x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(rec)
    return nums, literals


def faithfulness_ok(prose, rec):
    """Every number in the prose must match a record value at the number's own
    precision (or be an identifier digit-group)."""
    nums, literals = _record_numbers_and_literals(rec)
    for tok in re.findall(r'-?\d+\.?\d*', prose):
        core = tok.strip('.')
        if core in ('', '-'):
            continue
        if core in literals or tok in literals:
            continue                                   # identifier, not a claim
        val = float(tok)
        d = len(tok.split('.')[1]) if '.' in tok else 0
        if not any(round(r, d) == round(val, d) for r in nums):
            return False, tok
    return True, None


def template_prose(rec):
    """A faithful 4-sentence summary built only from record fields."""
    tr = rec['top_rules'][0]['name'] if rec['top_rules'] else 'n/a'
    return (
        f"The detector judges {rec['dataset']} video {rec['video_id']} to be "
        f"{rec['verdict']} with fake-probability {rec['p_fake']:.2f}. "
        f"Branch conflict C is {rec['C']:.3f} and vacuity V is {rec['V']:.3f}, "
        f"giving a post-hoc risk R of {rec['R']:.3f}. "
        f"The strongest firing rule is {tr}, and the counterfactual residual is "
        f"{rec['counterfactual_residual']}. "
        f"On this evidence the recommended action is to {rec['decision']}."
    )


def render_prose(rec, llm_cmd):
    """Render 4 sentences; try the LLM if configured, else template. Returns
    (prose, source, faithful)."""
    if llm_cmd:
        instr = ("Write exactly 4 sentences describing this deepfake evidence "
                 "record. Reference ONLY the provided fields and numbers; invent "
                 "no numbers.\n\n" + json.dumps(rec))
        try:
            p = subprocess.run(llm_cmd, shell=True, input=instr, text=True,
                               capture_output=True, timeout=120)
            prose = p.stdout.strip()
            if prose:
                ok, bad = faithfulness_ok(prose, rec)
                if ok:
                    return prose, 'llm', True
                return template_prose(rec), f'template (llm number {bad} not in record)', True
        except Exception as e:
            return template_prose(rec), f'template (llm error: {e})', True
    return template_prose(rec), 'template', True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='logs/test/p4b_full_ccv_reeval')
    ap.add_argument('--dataset', default=RH.FF)
    ap.add_argument('--video_id', required=True)
    ap.add_argument('--method', default=None)
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--llm_cmd', default=None,
                    help='shell cmd reading JSON on stdin, writing prose on stdout')
    ap.add_argument('--escalate_coverage', type=float, default=0.90)
    ap.add_argument('--defer_coverage', type=float, default=0.98)
    ap.add_argument('--outdir', default='results/evidence_records')
    args = ap.parse_args()

    # FF++-only calibration of decision tau + risk head + risk cut-offs.
    ff = U.load_frames(args.run, RH.FF)
    if ff is None:
        raise SystemExit(f"missing FF++ CSV under {args.run}")
    vf_ff = U.per_video_features(ff)
    tau = RH.eer_threshold(vf_ff['label'].to_numpy(), vf_ff['p_fake'].to_numpy())
    scaler, clf = RH.fit_final_head(vf_ff, tau)
    R_ff = RH.risk_of(vf_ff, scaler, clf)
    tau_esc = float(np.quantile(R_ff, args.escalate_coverage))
    tau_def = float(np.quantile(R_ff, args.defer_coverage))
    names = U.rule_names()

    rec = build_record(args.run, args.dataset, args.video_id, args.method,
                       tau, tau_esc, tau_def, scaler, clf, names)

    os.makedirs(args.outdir, exist_ok=True)
    stem = f"{args.dataset}_{rec['method'] or 'na'}_{args.video_id}".replace('/', '_')
    json_path = os.path.join(args.outdir, stem + '.json')
    md_path = os.path.join(args.outdir, stem + '.md')
    json.dump(rec, open(json_path, 'w'), indent=2)

    L = [f"# Evidence record — {rec['dataset']} / {rec['video_id']} "
         f"({rec['method'] or 'n/a'})\n"]
    L.append(f"- **Verdict:** {rec['verdict']}  (p_fake={rec['p_fake']}, "
             f"label={rec['label']})  |  **Decision:** {rec['decision']}")
    L.append(f"- V={rec['V']}  C={rec['C']}  Q={rec['Q']}  T={rec['T']}  "
             f"**R={rec['R']}**  (frames={rec['num_frames']})")
    L.append(f"- Risk cut-offs: escalate≥{_num(tau_esc)}, defer≥{_num(tau_def)}")
    L.append('\n**Top firing rules:**')
    for r in rec['top_rules']:
        L.append(f"  - {r['name']} ({r['rule']}): {r['mean_violation']}")
    L.append('\n**Per-branch evidence:**')
    for b, d in rec['branch_evidence'].items():
        L.append(f"  - {b}: p_fake={d['p_fake']}, S={d['S']}")
    if rec['ccv_anomaly_groups']:
        L.append(f"\n**CCV anomaly groups:** {rec['ccv_anomaly_groups']}")
    L.append(f"**Counterfactual residual:** {rec['counterfactual_residual']}")

    if args.render:
        prose, source, _ = render_prose(rec, args.llm_cmd)
        L.append(f"\n## Narrative ({source})\n\n{prose}")
        rec['narrative'] = {'text': prose, 'source': source}
        json.dump(rec, open(json_path, 'w'), indent=2)

    open(md_path, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f"\nWritten {json_path}\n        {md_path}")


if __name__ == '__main__':
    main()
