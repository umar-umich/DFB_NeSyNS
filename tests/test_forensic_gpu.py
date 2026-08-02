"""
Unit test for the differentiable forensic bank (Phase-3 Task 4).

Loads ~100 clean (real) FF++ frames and their cached CPU forensic features, runs
the torch ForensicBankGPU on the same images (at native 256×256 — the size the
cache used), and checks per-GROUP cosine similarity > 0.95 for the three
LANDMARK-FREE groups (patch_noise, srm_noise, fft_spectral).

The two region-based groups (boundary_texture, symmetry_color) are DOCUMENTED as
parity-impossible from pixels alone (they need the SegFormer parsing map; the
quality_* dims additionally need the anti-spoof / detector models) and are not
asserted.

NOT wired into training — this only exercises the standalone module.

Run (repo root, dfb_nesy):
    python -m pytest tests/test_forensic_gpu.py -q
    python tests/test_forensic_gpu.py        # standalone, prints per-group cosine
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'training'))
from networks.nesy_defake.forensic_gpu import ForensicBankGPU, GROUPS, REPRODUCIBLE

FF_FORENSIC_GLOB = ('/data/umar/Datasets/preprocessed/FaceForensics++/'
                    'original_sequences/youtube/c23/forensic_features/*.pt')
N_FRAMES = 100


def _load_pairs(n_frames=N_FRAMES):
    """Return (imgs (N,3,H,W) in [0,1] RGB, cached (N,83))."""
    import cv2
    caches = sorted(glob.glob(FF_FORENSIC_GLOB))
    imgs, cached = [], []
    for pt in caches:
        d = torch.load(pt, map_location='cpu', weights_only=False)
        feats = np.asarray(d['features'], dtype=np.float32)
        paths = d.get('frame_paths', [])
        # cache stores bare filenames; frames live at <base>/frames/<video>/<fn>
        base = os.path.dirname(os.path.dirname(pt))     # .../c23
        video = os.path.splitext(os.path.basename(pt))[0]
        for i, fn in enumerate(paths):
            fp = os.path.join(base, 'frames', video, os.path.basename(fn))
            if not os.path.exists(fp):
                continue
            bgr = cv2.imread(fp)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            imgs.append(torch.from_numpy(rgb).permute(2, 0, 1))
            cached.append(feats[i])
            if len(imgs) >= n_frames:
                break
        if len(imgs) >= n_frames:
            break
    if not imgs:
        return None, None
    return torch.stack(imgs), np.stack(cached)


def _group_cosine(a, b, sl):
    """Flattened cosine over a group slice across all frames (nan-safe)."""
    x = a[:, sl[0]:sl[1]].reshape(-1)
    y = b[:, sl[0]:sl[1]].reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    dn = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / dn) if dn > 1e-12 else float('nan')


def compute_group_cosines():
    imgs, cached = _load_pairs()
    if imgs is None:
        return None
    with torch.no_grad():
        out = ForensicBankGPU()(imgs).numpy()
    return {g: _group_cosine(out, cached, sl) for g, sl in GROUPS.items()}, len(imgs)


def test_reproducible_groups_parity():
    res = compute_group_cosines()
    if res is None:
        import pytest
        pytest.skip('FF++ forensic caches not available on this machine')
    cos, n = res
    for g in REPRODUCIBLE:
        assert cos[g] > 0.95, f'{g} cosine {cos[g]:.4f} <= 0.95 over {n} frames'


def test_region_groups_are_nan():
    """Region groups must be returned as NaN (documented parity-impossible)."""
    out = ForensicBankGPU()(torch.rand(2, 3, 256, 256))
    for g in ('boundary_texture', 'symmetry_color'):
        s = GROUPS[g]
        assert torch.isnan(out[:, s[0]:s[1]]).all()


def _run_all():
    res = compute_group_cosines()
    if res is None:
        print('SKIP: FF++ forensic caches not found'); return
    cos, n = res
    print(f'Per-group cosine (torch vs cached CPU) over {n} FF++ frames:')
    for g, (s, e) in GROUPS.items():
        tag = 'reproducible' if g in REPRODUCIBLE else 'PARITY-IMPOSSIBLE (region/model)'
        c = cos[g]
        mark = 'nan' if c != c else f'{c:.4f}'
        print(f'  {g:16} [{s:2}:{e:2}]  cosine={mark:>7}   {tag}')
    ok = all(cos[g] > 0.95 for g in REPRODUCIBLE)
    print('\nReproducible groups all > 0.95:', ok)
    # also exercise the assert-based tests (no pytest needed)
    for g in REPRODUCIBLE:
        assert cos[g] > 0.95, f'{g} parity {cos[g]:.4f} <= 0.95'
    test_region_groups_are_nan()
    print('assertions passed (parity + region-NaN)')


if __name__ == '__main__':
    _run_all()
