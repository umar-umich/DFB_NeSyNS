#!/usr/bin/env python3
"""The two-case diagnostic, reported whatever the gates show (brief, Gate 0).

    OMP_NUM_THREADS=4 <discern_ext python> training/joint_ibdc/diagnostics.py

On the development pool, how often is the video view right when both spatial views are wrong, and
the reverse? Each view is thresholded at its OWN FF++ validation EER, as the brief specifies, so a
view is not penalised for being differently calibrated from the others.

This is the complementarity the fusion exists to exploit, counted directly rather than inferred from
an AUROC difference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal")]

import confirm as C          # noqa: E402
from health import eer       # noqa: E402

SPATIAL, BRANCH = "warmgate_e19", "vjepa2_lora_B_seed42"
VIEWS = ("T", "APP", "OM")
OUT = ROOT / "analysis/joint_ibdc"


def main():
    dev = C.dev_table(SPATIAL, BRANCH)
    y = dev.y.to_numpy()
    ff = np.where(dev.source.to_numpy() == "FFpp_val")[0]
    print(f"development pool {len(dev)} videos, FF++ val rows {len(ff)}")

    tau, correct = {}, {}
    for v in VIEWS:
        p = dev[f"p_{v}"].to_numpy()
        tau[v] = float(eer(y[ff], p[ff])[1])
        correct[v] = ((p >= tau[v]).astype(int) == y)
        print(f"  {v:4s} tau {tau[v]:.4f}  accuracy {correct[v].mean():.4f}")

    t_only = correct["T"] & ~correct["APP"] & ~correct["OM"]
    sp_only = ~correct["T"] & correct["APP"] & correct["OM"]
    res = {"n": int(len(dev)), "tau": tau,
           "accuracy": {v: float(correct[v].mean()) for v in VIEWS},
           "video_right_both_spatial_wrong": int(t_only.sum()),
           "both_spatial_right_video_wrong": int(sp_only.sum()),
           "video_right_both_spatial_wrong_frac": float(t_only.mean()),
           "both_spatial_right_video_wrong_frac": float(sp_only.mean()),
           "by_source": {}}

    src = dev.source.to_numpy()
    for s in sorted(set(src)):
        m = src == s
        res["by_source"][s] = {"n": int(m.sum()),
                               "video_only": int((t_only & m).sum()),
                               "spatial_only": int((sp_only & m).sum())}

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "two_case_diagnostic.json", "w"), indent=1)

    print(f"\n  video view right, BOTH spatial views wrong : {t_only.sum():5d} "
          f"({t_only.mean() * 100:.2f}%)")
    print(f"  both spatial views right, video view wrong : {sp_only.sum():5d} "
          f"({sp_only.mean() * 100:.2f}%)")
    print(f"\n  {'source':14s} {'n':>6} {'video only':>11} {'spatial only':>13}")
    for s, d in sorted(res["by_source"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {s:14s} {d['n']:6d} {d['video_only']:11d} {d['spatial_only']:13d}")
    print(f"\n-> {OUT / 'two_case_diagnostic.json'}")


if __name__ == "__main__":
    main()
