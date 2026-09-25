#!/usr/bin/env python3
"""Head inputs for the I-JEPA residual views: quality standardisation and cross-block agreement.

Both operate on the cached residuals, so neither costs GPU time.

QUALITY STANDARDISATION (brief, "Quality standardization"). Per block, log eps_k is regressed on the
label-free quality descriptors q with linear and squared terms, fitted on REAL FF++ c23 training frames
only under the B mixture. sigma_k is the standard deviation of that fit's residuals. Then
a_k = (log eps_k - mu_k(q)) / sigma_k. mu_k and sigma_k are frozen and applied unchanged everywhere.

CROSS-BLOCK AGREEMENT, row R5 (amendment 1, section B). A manipulated region should disagree with its
neighbour: the direction in which the predictor is wrong over the lower face should stop resembling the
direction it is wrong over the upper face. The head cannot form this for itself -- its inputs are the
per-block r_bar_k pushed through ONE SHARED LINEAR MAP and concatenated, so every feature it sees is
linear in the r_bar_k, while a cosine is a ratio of a bilinear form to a product of norms.

Only the two DISJOINT pairs are used, (upper, lower) and (left cheek, right cheek). The centre block
overlaps all four, so any cosine involving it would be partly a block against itself. Disjointness is
asserted, not assumed.

A near-zero r_bar_k has an unstable direction, so each block carries a frozen 5th-percentile norm
floor tau_k measured on real FF++ training frames; below it the cosine is suppressed to 0 and a
validity flag is raised, which lets the head learn how much to trust the cosine instead of being fed
a direction that is numerical noise.

Deliberately NOT added: signed differences of a_k, which are linear in inputs the head already has.
"""
from __future__ import annotations

import numpy as np

BLOCK_NAMES = ["upper", "lower", "left_cheek", "right_cheek", "centre"]
BLOCKS = {"upper": (2, 9, 4, 12), "lower": (9, 16, 4, 12), "left_cheek": (5, 12, 0, 8),
          "right_cheek": (5, 12, 8, 16), "centre": (4, 11, 4, 12)}          # mirrors the extractor
PAIRS = [(0, 1), (2, 3)]                                                    # upper/lower, left/right cheek
PAIR_NAMES = [f"{BLOCK_NAMES[i]}_{BLOCK_NAMES[j]}" for i, j in PAIRS]
G = 16
COS_EPS = 1e-6


def patch_set(name):
    r0, r1, c0, c1 = BLOCKS[name]
    return {r * G + c for r in range(r0, r1) for c in range(c0, c1)}


def assert_pairs_disjoint():
    """The amendment requires the paired blocks to share no patch. Checked, not assumed."""
    for i, j in PAIRS:
        a, b = patch_set(BLOCK_NAMES[i]), patch_set(BLOCK_NAMES[j])
        overlap = a & b
        if overlap:
            raise AssertionError(f"{BLOCK_NAMES[i]} and {BLOCK_NAMES[j]} share {len(overlap)} patches")
    return True


assert_pairs_disjoint()


# --------------------------------------------------------------------- quality standardisation
def design(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q, q ** 2, np.ones((len(q), 1))], axis=1)


def fit_standardizer(eps, q):
    """Fit mu_k(q) and sigma_k on REAL FF++ train frames only. eps (N,5), q (N,D).

    q is cast to float64 BEFORE any reduction. The cache stores it as fp16, and numpy reduces in the
    input dtype: summing 115,198 fp16 values overflows the accumulator well before the true total,
    which silently shifts the frozen mu_q the whole study then applies. No cached value is anywhere
    near the fp16 ceiling -- the accumulator is the problem, not the data.
    """
    eps = np.asarray(eps, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if eps.min() <= 0:
        raise ValueError("eps must be positive to take a log")
    mu_q, sd_q = q.mean(0), q.std(0)                  # z-scored for conditioning; frozen with the fit
    sd_q = np.where(sd_q < 1e-12, 1.0, sd_q)
    X = design((q - mu_q) / sd_q)
    coef, sigma = [], []
    for k in range(eps.shape[1]):
        y = np.log(eps[:, k])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        coef.append(c)
        sigma.append(float((y - X @ c).std()))
    sigma = np.array(sigma)
    if (sigma < 1e-8).any():
        raise ValueError("a block has near-zero residual spread; standardisation would explode")
    return {"coef": np.stack(coef), "sigma": sigma, "mu_q": mu_q, "sd_q": sd_q}


def apply_standardizer(eps, q, st):
    X = design((np.asarray(q, dtype=np.float64) - st["mu_q"]) / st["sd_q"])
    return ((np.log(np.asarray(eps, dtype=np.float64)) - X @ st["coef"].T) / st["sigma"]).astype(np.float32)


# --------------------------------------------------------------------- R5 cross-block agreement
# NOTE ON THE THIRD QUALITY DESCRIPTOR. The brief asks for blur, a blockiness score and the face box
# size. Every stored crop in this corpus is 256x256 -- checked on FFpp_train, CDFv2 and IDForge -- so
# the size descriptor is CONSTANT and carries no information, whether measured before or after the
# 224 resize. Its z-scored column and its square are identically zero and the least-squares fit
# simply ignores them. Standardisation therefore runs on blur and blockiness, which are the two that
# bear on the compression confound G1 tests. Recorded rather than quietly dropped.


def fit_tau(rbar_real_train, pct=5.0):
    """tau_k = the 5th percentile of ||r_bar_k|| over real FF++ c23 train frames, B mixture. Frozen."""
    n = np.linalg.norm(np.asarray(rbar_real_train, dtype=np.float32), axis=2)   # (N,5)
    return np.percentile(n, pct, axis=0).astype(np.float32)


def frame_agreement(rbar, tau):
    """Per frame: c_ij for the two disjoint pairs and their validity flags.

    Returns c (N,2) and v (N,2). v_ij = 1 means the pair was suppressed because a block's residual
    direction was too small to trust; c_ij is then 0.
    """
    r = np.asarray(rbar, dtype=np.float32)
    if r.ndim != 3 or r.shape[1] != len(BLOCK_NAMES):
        raise ValueError(f"expected (N, {len(BLOCK_NAMES)}, D), got {r.shape}")
    tau = np.asarray(tau, dtype=np.float32)
    nrm = np.linalg.norm(r, axis=2)                                             # (N,5)
    C = np.zeros((len(r), len(PAIRS)), dtype=np.float32)
    V = np.zeros((len(r), len(PAIRS)), dtype=np.float32)
    for p, (i, j) in enumerate(PAIRS):
        dot = (r[:, i] * r[:, j]).sum(1)
        c = dot / (nrm[:, i] * nrm[:, j] + COS_EPS)
        bad = (nrm[:, i] < tau[i]) | (nrm[:, j] < tau[j])
        C[:, p] = np.where(bad, 0.0, c)
        V[:, p] = bad.astype(np.float32)
    return C, V


def video_agreement(rbar, keys, tau):
    """Video-level inputs: the frame means of c_ij and v_ij for both pairs, so 4 per video.

    Returns (videos, X) with X of shape (n_videos, 4), columns
    [c_upper_lower, c_left_right, v_upper_lower, v_left_right].
    """
    C, V = frame_agreement(rbar, tau)
    from data import video_of                              # THE rule, never a local shortcut
    vid = np.array([video_of(k) for k in keys])
    order, inv = np.unique(vid, return_inverse=True)
    out = np.zeros((len(order), 2 * len(PAIRS)), dtype=np.float32)
    cnt = np.bincount(inv, minlength=len(order)).astype(np.float32)
    for p in range(len(PAIRS)):
        out[:, p] = np.bincount(inv, weights=C[:, p], minlength=len(order)) / cnt
        out[:, len(PAIRS) + p] = np.bincount(inv, weights=V[:, p], minlength=len(order)) / cnt
    return order, out


VIDEO_FEATURE_NAMES = [f"c_{n}" for n in PAIR_NAMES] + [f"v_{n}" for n in PAIR_NAMES]


def _selftest():
    rng = np.random.default_rng(0)
    N, D, B = 500, 1280, len(BLOCK_NAMES)

    assert assert_pairs_disjoint()
    # and the centre block is correctly excluded: it DOES overlap, which is why it is not paired
    assert patch_set("centre") & patch_set("upper")
    assert len(patch_set("upper")) == 56 and len(patch_set("left_cheek")) == 56

    r = rng.normal(size=(N, B, D)).astype(np.float32)
    tau = fit_tau(r)
    assert tau.shape == (B,)
    C, V = frame_agreement(r, tau)
    assert C.shape == (N, 2) and V.shape == (N, 2)
    # ~5% of frames fall below a 5th-percentile floor on each block, so a little more than 5% of pairs
    assert 0.02 < V.mean() < 0.25, V.mean()

    # cosine matches a direct reference wherever the pair is valid
    ok = V[:, 0] == 0
    ref = np.array([r[n, 0] @ r[n, 1] / (np.linalg.norm(r[n, 0]) * np.linalg.norm(r[n, 1]) + COS_EPS)
                    for n in range(N)], dtype=np.float32)
    assert np.abs(C[ok, 0] - ref[ok]).max() < 1e-5

    # a tiny block is suppressed, not turned into noise
    r2 = rng.normal(size=(3, B, D)).astype(np.float32)
    r2[1, 0] *= 1e-6
    C2, V2 = frame_agreement(r2, fit_tau(r))
    assert V2[1, 0] == 1.0 and C2[1, 0] == 0.0
    assert np.isfinite(C2).all()

    # an exactly zero block never produces a NaN
    r3 = np.zeros((1, B, D), dtype=np.float32); r3[0, 1] = 1.0
    C3, V3 = frame_agreement(r3, np.zeros(B, dtype=np.float32))
    assert np.isfinite(C3).all()

    # duplicated and opposed directions give +1 and -1
    r4 = np.zeros((2, B, D), dtype=np.float32)
    base = rng.normal(size=D).astype(np.float32)
    r4[0, 0] = base; r4[0, 1] = base
    r4[1, 0] = base; r4[1, 1] = -base
    C4, _ = frame_agreement(r4, np.zeros(B, dtype=np.float32))
    assert abs(C4[0, 0] - 1.0) < 1e-4 and abs(C4[1, 0] + 1.0) < 1e-4

    # video aggregation averages the right frames, under THE video rule (full directory path)
    import sys as _sys
    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from data import video_of
    keys = [f"ds/src/vid{n // 10}/frame{n}.png" for n in range(N)]
    order, X = video_agreement(r, keys, tau)
    assert X.shape == (len(order), 4) and len(order) == N // 10, (X.shape, len(order))
    sel = [i for i, k in enumerate(keys) if video_of(k) == order[0]]
    assert sel, "the aggregation key must match the video rule"
    assert abs(X[0, 0] - C[sel, 0].mean()) < 1e-5

    # standardiser removes a planted quality effect and its frozen coefficients transfer
    q = rng.normal(size=(N, 3))
    le = 0.7 * q[:, [0]] - 0.3 * (q ** 2)[:, [1]] + rng.normal(scale=0.25, size=(N, 1))
    eps = np.exp(np.repeat(le, B, axis=1) + rng.normal(scale=0.05, size=(N, B)))
    st = fit_standardizer(eps, q)
    a = apply_standardizer(eps, q, st)
    assert abs(a.std() - 1.0) < 0.15 and abs(a.mean()) < 0.05
    for d in range(q.shape[1]):
        assert abs(np.corrcoef(a[:, 0], q[:, d])[0, 1]) < 0.15
    q2 = rng.normal(size=(N, 3))
    le2 = 0.7 * q2[:, [0]] - 0.3 * (q2 ** 2)[:, [1]] + rng.normal(scale=0.25, size=(N, 1))
    eps2 = np.exp(np.repeat(le2, B, axis=1) + rng.normal(scale=0.05, size=(N, B)))
    assert abs(apply_standardizer(eps2, q2, st).std() - 1.0) < 0.25

    print("features.py self-test OK")
    print(f"  pairs: {PAIR_NAMES} (disjoint, centre excluded)")
    print(f"  R5 video-level inputs: {VIDEO_FEATURE_NAMES}")
    print(f"  R4 head input 5 + 5*64 = 325 dims; R5 head input 325 + 4 = 329 dims")


if __name__ == "__main__":
    _selftest()
