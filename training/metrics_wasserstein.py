"""
metrics_wasserstein.py
======================
Wasserstein-1 separation / confidence metrics for 2-class detector outputs,
extracted from test.py so they can be unit-tested in isolation (test.py parses
CLI args at import time and pulls in the whole detector stack).

Root cause of the "W1-sep byte-identical to W1-conf" observation (Task 10)
--------------------------------------------------------------------------
The input `probs` is a **complementary** 2-class vector: `probs[:,0] = 1 - probs[:,1]`.
W1 (1-D earth-mover distance) is invariant under any isometry of the line, and
x → 1-x is an isometry (a reflection). Two consequences follow:

  1. `W1-sep-real ≡ W1-sep-fake` ALWAYS.  W1-sep-real compares column 0 across
     classes, W1-sep-fake compares column 1; but column 0 = 1 - column 1, so
     W1(1-R, 1-F) = W1(R, F). The `-real` / `-fake` split of the separation is
     therefore redundant — they are provably equal, not merely close.

  2. `W1-sep ≡ W1-conf` whenever the two class score-distributions are reflections
     of each other about 0.5 — i.e. the WELL-CALIBRATED regime where real scores
     sit as far below 0.5 as fake scores sit above it. In that regime the
     cross-class separation and the within-class spread of the complementary
     columns coincide exactly, so the two columns print byte-identical. This is
     what "pre-fix" runs were in.

They DIVERGE once both class distributions fall on the SAME side of 0.5
(miscalibration) — e.g. the current model, where real≈0.63 and fake≈0.89 both
exceed 0.5. That is why "post-fix" runs show W1-sep ≠ W1-conf.

Conclusion: **not a code bug.** W1-sep (cross-class separation) and W1-conf
(within-class confidence spread) are genuinely different quantities that happen
to coincide for complementary probabilities in the calibrated regime. The
metrics are kept as-is; see test_wasserstein_metrics.py for the locked-in
properties (sep-real ≡ sep-fake always; sep ≠ conf under miscalibration).
"""
import numpy as np
from scipy.stats import wasserstein_distance


def compute_wasserstein1(probs, labels):
    """W1 separation / confidence metrics for a complementary 2-class prob array.

    Args:
        probs:  (N, 2) array with probs[:,0] = P(real), probs[:,1] = P(fake).
        labels: (N,) array, 0 = real, 1 = fake.
    Returns:
        dict of W1-sep-* (cross-class separation) and W1-conf-* (within-class
        confidence spread), or {} if a class is missing.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    is_real = labels == 0
    is_fake = labels == 1
    if not (is_real.any() and is_fake.any()):
        return {}
    return {
        'W1-sep-real': wasserstein_distance(probs[is_real, 0], probs[is_fake, 0]),
        'W1-sep-fake': wasserstein_distance(probs[is_real, 1], probs[is_fake, 1]),
        'W1-sep':      (wasserstein_distance(probs[is_real, 0], probs[is_fake, 0]) +
                        wasserstein_distance(probs[is_real, 1], probs[is_fake, 1])) / 2,
        'W1-conf-real': wasserstein_distance(probs[is_real, 0], probs[is_real, 1]),
        'W1-conf-fake': wasserstein_distance(probs[is_fake, 0], probs[is_fake, 1]),
        'W1-conf':     (wasserstein_distance(probs[is_real, 0], probs[is_real, 1]) +
                        wasserstein_distance(probs[is_fake, 0], probs[is_fake, 1])) / 2,
    }
