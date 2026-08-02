"""
Unit tests for the W1 separation / confidence metrics (Phase-2 Task 10).

Locks in the properties that explain the "W1-sep byte-identical to W1-conf"
observation, and — the crux of the task — a synthetic case where the two metrics
MUST differ, so a regression that accidentally aliases the columns is caught.

Run (repo root, dfb_nesy):
    python -m pytest tests/test_wasserstein_metrics.py -q
    # or standalone:
    python tests/test_wasserstein_metrics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'training'))
from metrics_wasserstein import compute_wasserstein1


def _probs(p1):
    """Complementary 2-class prob array from fake-probabilities p1."""
    p1 = np.asarray(p1, float)
    return np.stack([1 - p1, p1], axis=1)


def _make(real_scores, fake_scores):
    p1 = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([np.zeros(len(real_scores)),
                             np.ones(len(fake_scores))]).astype(int)
    return _probs(p1), labels


def test_sep_real_equals_sep_fake_always():
    """Reflection invariance: W1-sep-real ≡ W1-sep-fake for complementary probs."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        real = np.clip(rng.normal(rng.uniform(0.1, 0.6), 0.15, 200), 0, 1)
        fake = np.clip(rng.normal(rng.uniform(0.4, 0.9), 0.15, 200), 0, 1)
        m = compute_wasserstein1(*_make(real, fake))
        assert abs(m['W1-sep-real'] - m['W1-sep-fake']) < 1e-12


def test_sep_differs_from_conf_when_miscalibrated():
    """CRUX: both classes on the same side of 0.5 → W1-sep ≠ W1-conf."""
    rng = np.random.default_rng(1)
    real = np.clip(rng.normal(0.60, 0.05, 400), 0, 1)   # both well above 0.5
    fake = np.clip(rng.normal(0.90, 0.05, 400), 0, 1)
    m = compute_wasserstein1(*_make(real, fake))
    assert abs(m['W1-sep'] - m['W1-conf']) > 1e-3, \
        f"sep {m['W1-sep']} should differ from conf {m['W1-conf']} when miscalibrated"


def test_sep_equals_conf_when_calibrated():
    """The historical coincidence: reflection-symmetric classes → W1-sep ≡ W1-conf."""
    rng = np.random.default_rng(2)
    real = np.clip(rng.normal(0.30, 0.10, 400), 0, 1)   # mirror images about 0.5
    fake = np.clip(rng.normal(0.70, 0.10, 400), 0, 1)
    m = compute_wasserstein1(*_make(real, fake))
    assert abs(m['W1-sep'] - m['W1-conf']) < 1e-9


def test_empty_class_returns_empty():
    probs = _probs(np.array([0.2, 0.3, 0.4]))
    assert compute_wasserstein1(probs, np.zeros(3, int)) == {}


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f'  PASS {fn.__name__}')
    print(f'{len(fns)} tests passed')


if __name__ == '__main__':
    _run_all()
