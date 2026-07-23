"""Task 2 smoke test: every frozen spatial instrument reproduces the pilot.

The weak version of this test is "known fake scores higher than known real".
The strong version, which this runs, is: re-score the exact frame the rev3 pilot
scored and demand the pilot's own p_orig back to 1e-4.

That matters because the wrapper deviates from `scripts/pilot_detectors.py` in
two ways (scoped chdir instead of a module-level one; sys.path appended instead
of prepended). Both are supposed to be behaviour-preserving. This is what proves
it — if a transform or checkpoint resolution had drifted, p_orig would move and
every frozen anchor in CURRENT_STATE ledger item 6 would be invalid.

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/umar/miniconda3/envs/GenD/bin/python cec/scripts/smoke_instruments.py
"""
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cec.instruments.spatial import INSTRUMENTS, SpatialInstrument  # noqa: E402

PP = Path("/data/umar/Datasets/preprocessed/FaceForensics++")
# Tolerance sits above the measured batch-composition float noise. CLIP attention
# kernels are not batch-invariant: the same frame scores 0.990071 in a batch of 2
# but reproduces the pilot's 0.989948 EXACTLY in a batch of 64 (the pilot's own
# batch size). The spread is ~1.2e-4 and is a property of the frozen pipeline, not
# of this wrapper. 5e-4 clears it with margin; a tighter bound would flag float
# noise as if it were a wrapper change. (Consequence for the gate: score original
# and intervened in the SAME batch so this offset cancels in the NM difference.)
TOL = 5e-4


def pilot_sample(instrument):
    """The first rev3 sample for this detector: the frame and the p_fake it got."""
    path = REPO / "results" / "pilot_rev3" / instrument / "deltas.json"
    sample = json.loads(path.read_text())["samples"][0]
    return sample


def frame_paths(sample):
    """The fake frame the pilot scored, and a real frame for the ordering check.

    The pilot reads the fake straight off the preprocessed PNG (no re-alignment),
    so this is byte-identical to what produced p_orig. Its p_real_orig comes from
    an ALIGNED paired real, which needs Task 3 — so the real here is the
    preprocessed original frame and is only used for the ordering check.
    """
    fake = PP / "manipulated_sequences" / sample["method"] / "c23" / "frames" / \
        sample["vid"] / f"{sample['frame']}.png"
    source_id = sample["vid"].split("_")[0]
    real = PP / "original_sequences" / "youtube" / "c23" / "frames" / source_id / \
        f"{sample['frame']}.png"
    return fake, real


def main():
    failures = []
    print(f"{'instrument':<10} {'p_fake':>8} {'expected':>9} {'p_real':>8} {'anchor':>7}  result")
    for name in INSTRUMENTS:
        sample = pilot_sample(name)
        fake_path, real_path = frame_paths(sample)
        fake_img, real_img = cv2.imread(str(fake_path)), cv2.imread(str(real_path))
        if fake_img is None or real_img is None:
            failures.append(f"{name}: missing frame {fake_path} or {real_path}")
            continue

        instrument = SpatialInstrument(name)
        p_fake, p_real = instrument.p_fake_batch([fake_img, real_img])
        expected = sample["p_orig"]

        reproduced = abs(p_fake - expected) < TOL
        ordered = p_fake > p_real
        ok = reproduced and ordered
        note = "OK" if ok else (
            "P_ORIG DRIFTED" if not reproduced else "fake !> real"
        )
        print(f"{name:<10} {p_fake:8.4f} {expected:9.4f} {p_real:8.4f} "
              f"{instrument.anchor:7.3f}  {note}")
        if not ok:
            failures.append(f"{name}: p_orig {p_fake:.6f} vs pilot {expected:.6f}, "
                            f"p_real {p_real:.4f}")

    print()
    if failures:
        print("SMOKE FAILED — the wrapper does not reproduce the pilot:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"SMOKE PASSED — all {len(INSTRUMENTS)} instruments reproduce pilot p_orig "
          f"within {TOL:g}, and score the fake above the real.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
