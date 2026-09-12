#!/usr/bin/env python3
"""Model-independent quality indicators, computed from pixels alone.

    python analysis/tbiom/quality_indicators.py --dataset FFpp_val --process jpeg:50

WHY THESE EXIST SEPARATELY FROM THE MODEL'S MEASUREMENTS. `recon_err` responds strongly to JPEG
severity, which makes it tempting as a severity signal. It is also computed by a branch that was
trained to detect manipulation, so it can depend on manipulation, identity and content as well
as on quality. A "severity estimator" built on it could quietly become a second class predictor,
and the failure mode that creates is specific and bad: genuine manipulation evidence gets
reinterpreted as poor quality and discounted.

These four indicators are computed from the pixels with no network involved, so they cannot
carry label information through a trained path:

    blur          variance of the Laplacian. Falls as the image is softened.
    blocking      mean gradient energy ON the 8-pixel JPEG block grid minus the same quantity
                  off-grid. Positive when a DCT blocking lattice is present, near zero when not.
                  This is the indicator that targets compression specifically.
    hf_energy     fraction of spectral energy above half-Nyquist. A downscale-upscale round trip
                  removes it; it is the effective-resolution proxy.
    contrast      standard deviation of luminance, as a content control -- a scene can be
                  low-detail without being low-quality, and without this the other three would
                  confound the two.

They are deployment-available: nothing here needs a label, a source video, or knowledge of what
processing was applied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def indicators(img: np.ndarray) -> dict:
    """img: HxWx3 uint8."""
    x = img.astype(np.float32)
    lum = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

    lap = (lum[:-2, 1:-1] + lum[2:, 1:-1] + lum[1:-1, :-2] + lum[1:-1, 2:]
           - 4 * lum[1:-1, 1:-1])
    blur = float(lap.var())

    # blocking: gradient energy on the 8-px grid against the average off-grid column/row
    gx = np.abs(np.diff(lum, axis=1))
    gy = np.abs(np.diff(lum, axis=0))
    on_x = gx[:, 7::8].mean() if gx.shape[1] > 8 else 0.0
    on_y = gy[7::8, :].mean() if gy.shape[0] > 8 else 0.0
    off_x = np.delete(gx, np.arange(7, gx.shape[1], 8), axis=1).mean()
    off_y = np.delete(gy, np.arange(7, gy.shape[0], 8), axis=0).mean()
    blocking = float(((on_x - off_x) + (on_y - off_y)) / 2)

    F = np.fft.fftshift(np.abs(np.fft.fft2(lum)))
    h, w = F.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    tot = F.sum() + 1e-9
    hf_energy = float(F[r > 0.5].sum() / tot)

    return {"blur": blur, "blocking": blocking, "hf_energy": hf_energy,
            "contrast": float(lum.std())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h5", default=None, help="defaults to the dataset's eval h5")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--process", default=None, help="jpeg:<q> or resize:<scale>")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import h5py
    import pandas as pd
    h5 = args.h5 or (f"/data/umar/Repos/DISCERN_Ext/eval_adaptation/data/h5/"
                     f"{args.dataset}.h5")
    tag = (args.process or "base").replace(":", "").replace(".", "")
    out = args.out or Path(f"logs/tbiom/quality/q_{args.dataset}_{tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    def degrade(a: np.ndarray) -> np.ndarray:
        if not args.process:
            return a
        kind, _, val = args.process.partition(":")
        from PIL import Image
        im = Image.fromarray(a)
        if kind == "resize":
            s = float(val); h, w = a.shape[:2]
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
            im = im.resize((w, h), Image.BILINEAR)
        elif kind == "jpeg":
            import io
            b = io.BytesIO(); im.save(b, format="JPEG", quality=int(val)); b.seek(0)
            im = Image.open(b).convert("RGB")
        return np.asarray(im)

    rows, n = [], 0
    with h5py.File(h5, "r") as f:
        def walk(g, pref=""):
            for k in g:
                node = g[k]
                if isinstance(node, h5py.Group):
                    yield from walk(node, f"{pref}/{k}")
                else:
                    yield f"{pref}/{k}", node
        for key, node in walk(f):
            a = np.asarray(node)
            if a.ndim != 3:
                continue
            rows.append({"key": key, **indicators(degrade(a))})
            n += 1
            if args.limit and n >= args.limit:
                break
            if n % 2000 == 0:
                print(f"  {n}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
