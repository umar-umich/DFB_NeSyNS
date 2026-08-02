"""
scripts/feature_shift_audit.py  —  Phase-3 Task 3.

Shift-aware audit of the cached symbolic inputs. Measures how far each OOD set's
REAL-face feature distribution drifts from FF++ — reals only, so the shift is
pipeline/domain drift uncontaminated by any manipulation signal.

Loads the cached per-video features directly (no image decode, no model):
  fast_semantic  58-d  (grouped by name prefix from refined_attributes.FAST_FEATURE_NAMES)
  forensic       83-d  (grouped by ccv_branch.FORENSIC_GROUPS)

For each feature: Wasserstein-1 distance FF++ vs each OOD set, on features
STANDARDIZED by FF++-train stats. Computed twice — raw, and under per-video
z-normalization (removes each video's own offset, isolating within-video drift).
Aggregated per group (mean + max).

Report `results/feature_shift_report.md`:
  (a) group × dataset shift matrix (raw and per-video-normalized);
  (b) top individual shifting features, named;
  (c) overlap with the p4b first-layer |weights| (concept MLP + CCV
      constraint/anomaly heads) — do the high-shift dims coincide with the
      dims the model actually leans on?

Pure analysis. No existing code touched. FF++ reference is FF++-train only.

Usage (repo root, dfb_nesy):
    python scripts/feature_shift_audit.py [--max_videos 200]
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'training'))
from dataset.nesy_defake_dataset import NeSyDeFakeDataset
from networks.nesy_defake.semantic.refined_attributes import FAST_FEATURE_NAMES
from networks.nesy_defake.ccv_branch import FORENSIC_GROUPS

FF = 'FaceForensics++'
OOD = ['Celeb-DF-v2', 'Celeb-DF-v3', 'DeepFakeDetection', 'DFDC', 'DFDCP']
CKPT = 'logs/train/p4b_full_ccv_2026-07-30-12-41-00/best_avg.pth'


# ── feature grouping ────────────────────────────────────────────────────────

def semantic_group(name):
    if name.startswith('fs_expr_'):
        return 'expressions'
    if name.startswith('fs_AU'):
        return 'AUs'
    if name.startswith('fs_pose_'):
        return 'pose'
    if name.startswith('fs_gaze_'):
        return 'gaze'
    if name in ('fs_gender_score', 'fs_age_score', 'fs_ethnicity_entropy'):
        return 'demographic'
    if name in ('fs_det_confidence', 'fs_landmark_confidence', 'fs_blur_laplacian',
                'fs_eye_lr_symmetry', 'fs_mouth_symmetry', 'fs_jaw_symmetry',
                'fs_face_area_ratio'):
        return 'quality-symmetry'
    return 'geometry'


SEM_GROUPS = {}
for i, nm in enumerate(FAST_FEATURE_NAMES):
    SEM_GROUPS.setdefault(semantic_group(nm), []).append(i)

FORE_NAMES = [f'{g}_{i - s}' for (g, s, e) in FORENSIC_GROUPS for i in range(s, e)]
FORE_GROUPS = {g: list(range(s, e)) for (g, s, e) in FORENSIC_GROUPS}


# ── cached-feature loading (reals only) ─────────────────────────────────────

def _cache_paths(frame_path):
    parts = frame_path.replace('\\', '/').split('/')
    fi = None
    for i, p in enumerate(parts):
        if p == 'frames' or p.startswith('frames_aug_'):
            fi = i
            break
    if fi is None:
        return None, None, None
    video_name = parts[fi + 1]
    base = '/'.join(parts[:fi])
    return base, video_name, parts[fi + 1]


def load_real_features(config, dataset_name, mode, max_videos):
    """(attrs (N,58), forensic (N,83), video_ids (N,)) for REAL videos."""
    cfg = dict(config)
    if mode == 'test':
        cfg['test_dataset'] = dataset_name
    else:
        cfg['train_dataset'] = [dataset_name]
    ds = NeSyDeFakeDataset(cfg, mode=mode)
    images = ds.data_dict['image']
    labels = ds.data_dict['label']
    fast_sub = cfg.get('fast_semantic', {}).get('precomputed_dir', 'fast_semantic')
    fore_sub = cfg.get('forensic_features', {}).get('precomputed_dir', 'forensic_features')

    seen, A, Fo, V = set(), [], [], []
    for path, lab in zip(images, labels):
        if int(lab) != 0:                          # reals only
            continue
        base, vname, _ = _cache_paths(path)
        if base is None or (base, vname) in seen:
            continue
        seen.add((base, vname))
        fp_a = os.path.join(base, fast_sub, f'{vname}.pt')
        fp_f = os.path.join(base, fore_sub, f'{vname}.pt')
        if not (os.path.exists(fp_a) and os.path.exists(fp_f)):
            continue
        try:
            a = torch.load(fp_a, map_location='cpu', weights_only=False)['features']
            f = torch.load(fp_f, map_location='cpu', weights_only=False)['features']
        except Exception:
            continue
        a = np.asarray(a, dtype=float); f = np.asarray(f, dtype=float)
        n = min(len(a), len(f))
        if n == 0:
            continue
        A.append(a[:n]); Fo.append(f[:n]); V.extend([vname] * n)
        if len(seen) >= max_videos:
            break
    if not A:
        return None
    return np.vstack(A), np.vstack(Fo), np.asarray(V)


def per_video_z(X, vids):
    """Z-normalize each column within each video (removes per-video offset)."""
    out = np.empty_like(X)
    for v in np.unique(vids):
        m = vids == v
        seg = X[m]
        out[m] = (seg - seg.mean(0)) / (seg.std(0) + 1e-8)
    return out


def col_w1(ref, oth):
    """Per-column Wasserstein-1 between two (n,F) matrices."""
    return np.array([wasserstein_distance(ref[:, j], oth[:, j])
                     for j in range(ref.shape[1])])


# ── p4b first-layer weight importance ───────────────────────────────────────

def load_weights():
    if not os.path.exists(CKPT):
        return None, None
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = sd.get('model_state_dict', sd.get('state_dict', sd)) if isinstance(sd, dict) else sd

    def absum(key, axis=0):
        return np.asarray(sd[key]).__abs__().sum(axis=axis) if key in sd else None

    # attrs (58): concept MLP first-Linear over [58 attrs || 18 rules] → take 58;
    # plus CCV learned-constraints first-Linear over 58 attrs.
    attrs_w = np.zeros(58)
    cm = absum('concept_branch.concept_mlp.1.weight')      # (76,)
    if cm is not None:
        attrs_w += cm[:58] / (np.linalg.norm(cm[:58]) + 1e-9)
    lc = absum('causal_branch.learned_constraints.trunk.1.weight')  # (58,)
    if lc is not None:
        attrs_w += lc / (np.linalg.norm(lc) + 1e-9)

    # forensic (83): CCV anomaly encoders, per group.
    fore_w = np.zeros(83)
    for gi, (_, s, e) in enumerate(FORENSIC_GROUPS):
        w = absum(f'causal_branch.forensic_anomaly.encoders.{gi}.1.weight')  # (e-s,)
        if w is not None and len(w) == e - s:
            fore_w[s:e] = w / (np.linalg.norm(w) + 1e-9)
    return attrs_w, fore_w


def overlap_report(shift, weight, names, k=10):
    """Correlation + top-k overlap between shift and weight importance."""
    if weight is None or not np.any(weight):
        return None
    corr = float(np.corrcoef(shift, weight)[0, 1])
    top_shift = set(np.argsort(-shift)[:k])
    top_weight = set(np.argsort(-weight)[:k])
    inter = sorted(top_shift & top_weight)
    return {'corr': corr, 'n_overlap': len(inter),
            'overlap_names': [names[i] for i in inter]}


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector',
                    default='training/config/detector/full/f2_full.yaml')
    ap.add_argument('--max_videos', type=int, default=200)
    ap.add_argument('--out', default='results/feature_shift_report.md')
    args = ap.parse_args()

    base = yaml.safe_load(open(args.detector))
    tc = 'training/config/test_config.yaml'
    if os.path.exists(tc):
        base.update(yaml.safe_load(open(tc)))

    print('Loading FF++ train reals...', flush=True)
    ff = load_real_features(base, FF, 'train', args.max_videos)
    if ff is None:
        raise SystemExit('no FF++ train real features found')
    ff_a, ff_f, ff_va = ff
    a_mean, a_std = ff_a.mean(0), ff_a.std(0) + 1e-8
    f_mean, f_std = ff_f.mean(0), ff_f.std(0) + 1e-8

    # Clip standardized features to ±10σ so a handful of pathological outliers
    # (e.g. DFDC patch-noise reals — the same extreme forensic values behind the
    # CCV NaN) can't dominate the W1; the shift is then a distribution measure,
    # not an outlier magnitude. A group that saturates the clip is itself a
    # finding (flagged in the report).
    _CLIP = 10.0

    def std_a(x): return np.clip((x - a_mean) / a_std, -_CLIP, _CLIP)
    def std_f(x): return np.clip((x - f_mean) / f_std, -_CLIP, _CLIP)

    per_ds = {}      # ds -> {'attrs_raw','attrs_z','fore_raw','fore_z'} column W1
    for ds in OOD:
        print(f'Loading {ds} reals...', flush=True)
        r = load_real_features(base, ds, 'test', args.max_videos)
        if r is None:
            continue
        a, f, va = r
        per_ds[ds] = {
            'attrs_raw': col_w1(std_a(ff_a), std_a(a)),
            'fore_raw':  col_w1(std_f(ff_f), std_f(f)),
            'attrs_z':   col_w1(per_video_z(ff_a, ff_va), per_video_z(a, va)),
            'fore_z':    col_w1(per_video_z(ff_f, ff_va), per_video_z(f, va)),
            'n': len(a),
        }

    attrs_w, fore_w = load_weights()

    # ── report ──────────────────────────────────────────────────────────
    L = ['# Feature-shift audit — FF++ vs OOD (reals only, Task 3)\n']
    L.append(f'- Reference: **{FF} train** reals (features standardized by FF++-train '
             f'stats)  |  ≤{args.max_videos} videos/dataset  |  checkpoint `{os.path.basename(CKPT)}`')
    L.append('- Per-video z-normalization removes each video\'s own offset '
             '(isolates within-video drift). video_id present in the cache → true '
             'per-video (not a per-column proxy).')
    L.append('- Standardized features clipped to ±10σ before W1 (robust to a few '
             'extreme OOD outliers; a group near this ceiling — e.g. DFDC '
             'patch_noise — is itself out-of-distribution).')
    L.append('')

    def group_matrix(kind_raw, kind_z, groups, title):
        L.append(f'## (a) {title} — group × dataset shift (mean W1 | max W1)\n')
        dss = [d for d in OOD if d in per_ds]
        L.append('| group | ' + ' | '.join(dss) + ' |')
        L.append('|' + '---|' * (len(dss) + 1))
        for g, idx in groups.items():
            cells = []
            for d in dss:
                raw = per_ds[d][kind_raw][idx]
                cells.append(f'{raw.mean():.3f}|{raw.max():.3f}')
            L.append(f'| {g} | ' + ' | '.join(cells) + ' |')
        L.append('\n*Per-video z-normalized (mean W1):*\n')
        L.append('| group | ' + ' | '.join(dss) + ' |')
        L.append('|' + '---|' * (len(dss) + 1))
        for g, idx in groups.items():
            cells = [f'{per_ds[d][kind_z][idx].mean():.3f}' for d in dss]
            L.append(f'| {g} | ' + ' | '.join(cells) + ' |')
        L.append('')

    group_matrix('attrs_raw', 'attrs_z', SEM_GROUPS, 'Semantic (58-d)')
    group_matrix('fore_raw', 'fore_z', FORE_GROUPS, 'Forensic (83-d)')

    # (b) top individual features (mean raw W1 across OOD)
    dss = [d for d in OOD if d in per_ds]
    attrs_mean = np.mean([per_ds[d]['attrs_raw'] for d in dss], axis=0)
    fore_mean = np.mean([per_ds[d]['fore_raw'] for d in dss], axis=0)
    L.append('## (b) Highest-shift individual features (mean raw W1 over OOD)\n')
    L.append('| rank | semantic feature | W1 | forensic feature | W1 |')
    L.append('|---|---|---|---|---|')
    sa = np.argsort(-attrs_mean); sf = np.argsort(-fore_mean)
    for r in range(10):
        L.append(f'| {r+1} | {FAST_FEATURE_NAMES[sa[r]]} | {attrs_mean[sa[r]]:.3f} '
                 f'| {FORE_NAMES[sf[r]]} | {fore_mean[sf[r]]:.3f} |')
    L.append('')

    # (c) overlap with model weights
    L.append('## (c) Do high-shift dims coincide with high-|weight| dims?\n')
    ov_a = overlap_report(attrs_mean, attrs_w, FAST_FEATURE_NAMES)
    ov_f = overlap_report(fore_mean, fore_w, FORE_NAMES)
    if ov_a:
        L.append(f'- **Semantic (concept MLP + CCV constraint head):** '
                 f"corr(shift, |weight|) = {ov_a['corr']:+.3f}; "
                 f"top-10 overlap = {ov_a['n_overlap']}/10 "
                 f"({', '.join(ov_a['overlap_names']) or 'none'}).")
    if ov_f:
        L.append(f'- **Forensic (CCV anomaly encoders):** '
                 f"corr(shift, |weight|) = {ov_f['corr']:+.3f}; "
                 f"top-10 overlap = {ov_f['n_overlap']}/10 "
                 f"({', '.join(ov_f['overlap_names']) or 'none'}).")
    if attrs_w is None:
        L.append('- checkpoint not found → weight overlap skipped.')
    L.append('\nHigh positive corr / large overlap ⇒ the model leans on exactly the '
             'dims that drift most OOD (a generalization liability).')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    open(args.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L[:40]))
    print(f'\n... Written {args.out}')


if __name__ == '__main__':
    main()
