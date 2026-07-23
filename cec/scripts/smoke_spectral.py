"""Task 5 smoke test (FreqNet arm; NPR-independent).

Checks the spectral instrument end-to-end WITHOUT the Pilot-S data run:
  1. FreqNet loads frozen through the GenD wrapper and scores.
  2. It separates a known fake above a known real on our crops.
  3. Each Study-1B intervention (notch, checkerboard, residual) produces a
     well-formed crop and a finite necessity margin.
This is a wiring check, not the gate — Pilot S (does the freq detector separate
at AUC and respond where CLIP ≈ 0) is a data run that also needs NPR.

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/umar/miniconda3/envs/GenD/bin/python cec/scripts/smoke_spectral.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cec.data.pairing import build_pair  # noqa: E402
from cec.instruments.spectral import SpectralInstrument  # noqa: E402

INTERVENTIONS = ["spectral_notch", "checkerboard_suppress", "residual_renormalize"]


def find_sample():
    from cec.data.pairing import fake_dir
    fdir = fake_dir("Deepfakes") / "frames"
    for vid in sorted(p.name for p in fdir.iterdir())[:40]:
        for frame in sorted(p.stem for p in (fdir / vid).glob("*.png"))[:5]:
            pair = build_pair("Deepfakes", vid, frame)
            if pair is not None:
                return pair
    return None


def main():
    failures = []
    pair = find_sample()
    if pair is None:
        print("no QC-passing sample found")
        return 1
    print(f"sample: {pair.method}/{pair.vid}/{pair.frame}")

    inst = SpectralInstrument("freqnet")

    # 1 + 2: loads and separates
    p_fake, p_real = inst.p_fake_batch([pair.fake, pair.real])
    print(f"[1/2] FreqNet loaded. p_fake={p_fake:.4f}  p_real={p_real:.4f}  "
          f"{'OK (fake>real)' if p_fake > p_real else 'NOTE: fake !> real'}")
    if not np.isfinite(p_fake) or not np.isfinite(p_real):
        failures.append("FreqNet produced a non-finite probability")
    # FreqNet is GAN-trained (ForenSynths); it may NOT separate on FF++ face
    # swaps. That is exactly what Pilot S measures, so a low margin here is a
    # finding to report, not a smoke failure.

    # 3: interventions produce well-formed crops and finite margins
    print("[3] interventions:")
    for it in INTERVENTIONS:
        ref = pair.real if it == "residual_renormalize" else None
        out = inst.intervene(it, pair.fake, reference=ref)
        shape_ok = out.shape == pair.fake.shape and out.dtype == pair.fake.dtype
        nm = inst.necessity_margin(pair.fake, out)
        ok = shape_ok and np.isfinite(nm)
        print(f"    {it:24} NM={nm:+.4f}  crop {out.shape} {out.dtype}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"intervention '{it}': shape_ok={shape_ok}, nm={nm}")

    print()
    if failures:
        print("SMOKE FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("SMOKE PASSED — FreqNet wraps and scores, all three interventions build "
          "well-formed crops with finite margins. (Separation/AUC is Pilot S, not this.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
