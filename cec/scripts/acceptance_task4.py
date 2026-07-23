"""TASK 4 ACCEPTANCE GATE — mandatory. No downstream work until this passes.

Reproduce the rev3 pilot's effort numbers on the acceptance sample, using the
ported repair + control code (cec/repair/) and the ported pairing + masks
(cec/data/, cec/masks/).

Acceptance sample: FF++ Deepfakes / 257_420 / 762 (the frame the v1 acceptance
named; retargeted to the rev3 Poisson numbers — see docs/cec/LOG.md CONFLICT 2).
Target (from results/pilot_rev3/effort/deltas.json):
    drop_gt   = 0.648    (Poisson repair of the GT mask)
    drop_blur = +0.001   (LPIPS-matched blur of the same region)
    p_orig    = 0.9875
    p_real    = 0.3943

Determinism note (Task 2): orig and every variant are scored in ONE batch so the
~1e-4 batch-composition float offset cancels in each drop.

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/umar/miniconda3/envs/GenD/bin/python cec/scripts/acceptance_task4.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cec.data.pairing import build_pair  # noqa: E402
from cec.repair import Lpips, build_variants  # noqa: E402

METHOD, VID, FRAME = "Deepfakes", "257_420", "762"
DETECTOR = "effort"

# Targets from results/pilot_rev3/effort/deltas.json, sample 257_420/762.
TARGET = {"drop_gt": 0.648, "drop_blur": 0.0013, "p_orig": 0.9875, "p_real": 0.3943}
# Tolerances: p(fake) carries ~1e-4 batch-composition noise; LPIPS matching picks
# a blur kernel from a discrete sweep, so drop_gt can move a little more. These
# are "reproduce within measurement noise", not "bit-exact".
TOL_DROP = 0.02
TOL_P = 0.01


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pair = build_pair(METHOD, VID, FRAME)
    if pair is None:
        print(f"FAIL: acceptance sample {METHOD}/{VID}/{FRAME} did not pass pairing QC.")
        return 1
    print(f"sample {METHOD}/{VID}/{FRAME}  fg={pair.fg:.3f}  bg_mse={pair.bg_mse:.2f}")

    # Build repair + controls (LPIPS on the same device).
    lp = Lpips(device)
    variants = build_variants(pair, lp)
    print(f"variants: {sorted(variants)}")

    # Load the frozen instrument and score every variant in ONE batch.
    from cec.instruments.spatial import SpatialInstrument
    inst = SpatialInstrument(DETECTOR, device=str(device))
    names = list(variants)
    probs = inst.p_fake_batch([variants[n] for n in names])
    p = dict(zip(names, probs))

    drop_gt = float(p["orig"] - p["gt_repair"])
    drop_blur = float(p["orig"] - p["match_blur"])
    drop_shift = float(p["orig"] - p["match_shift"])
    drop_wrong = float(p["orig"] - p.get("wrong_region", p["orig"]))
    p_orig, p_real = float(p["orig"]), float(p["real_orig"])

    rows = [
        ("p_orig", p_orig, TARGET["p_orig"], TOL_P),
        ("p_real", p_real, TARGET["p_real"], TOL_P),
        ("drop_gt", drop_gt, TARGET["drop_gt"], TOL_DROP),
        ("drop_blur", drop_blur, TARGET["drop_blur"], TOL_DROP),
    ]
    print(f"\n{'quantity':<12} {'measured':>10} {'target':>10} {'|Δ|':>8} {'tol':>7}  result")
    failures = []
    for name, meas, tgt, tol in rows:
        d = abs(meas - tgt)
        ok = d <= tol
        print(f"{name:<12} {meas:10.4f} {tgt:10.4f} {d:8.4f} {tol:7.3f}  {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{name}: {meas:.4f} vs target {tgt:.4f} (|Δ|={d:.4f} > {tol})")

    # Report the other controls for the record (not gated, but must be inert).
    print(f"\ncontrols (for the record): drop_shift={drop_shift:+.4f}  drop_wrong={drop_wrong:+.4f}")
    if "real_offset" in p:
        print(f"real_offset |Δ| = {abs(p['real_offset'] - p['real_orig']):.4f}")

    print()
    if failures:
        print("ACCEPTANCE FAILED — the port does not reproduce the rev3 effort numbers:")
        for f in failures:
            print("  -", f)
        print("Do NOT proceed to downstream tasks.")
        return 1
    print("ACCEPTANCE PASSED — ported repair + controls reproduce effort "
          f"Δgt={drop_gt:.3f} / Δblur={drop_blur:+.3f} on {VID}/{FRAME}, "
          "within measurement tolerance. Task 4 gate cleared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
