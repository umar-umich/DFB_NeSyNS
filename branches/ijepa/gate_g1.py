#!/usr/bin/env python3
"""Gate G1: is the regional prediction error driven by compression rather than by manipulation?

    python branches/ijepa/gate_g1.py

On FF++ c23 validation, per block k:
  d_comp  the shift in mean log eps_k of REAL videos from clean to JPEG q50, and again to resize 0.50
  d_fake  the difference in mean log eps_k between fake and real videos, both clean
  AUROC   real against fake on the standardised a_k, within each quality level

PASS if d_comp < d_fake for at least 3 of the 5 blocks, for BOTH JPEG q50 and resize 0.50.
All numbers are reported whether it passes or not, as the brief requires.

The unit is the video: frames are averaged within a video first, then across videos, because the
video is the unit of analysis everywhere else in this system. Frame-level figures are reported
alongside so the choice is visible rather than buried.

The standardiser is the frozen one fitted on REAL FF++ c23 TRAINING frames under the B mixture. It is
never refitted here -- refitting on validation would hide exactly the confound this gate looks for.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D          # noqa: E402
import features as FX     # noqa: E402
from v31 import fast_auc  # noqa: E402

OUT = Path("/data/umar/Repos/DFB_NeSyNS/analysis/ijepa")
BLOCKS = FX.BLOCK_NAMES
LEVELS = {"clean": "FFpp_val", "jpeg50": "FFpp_val_jpeg50", "jpeg30": "FFpp_val_jpeg30",
          "rs050": "FFpp_val_rs050", "rs035": "FFpp_val_rs035"}


def video_mean_log_eps(d):
    """(videos, 5) mean of log eps_k within each video, plus the video labels."""
    le = np.log(d["eps"].astype(np.float64))
    order, m = D.video_mean(le, d["video"])
    _, yv = D.video_mean(d["y"].astype(float), d["video"])
    return order, m, (yv > 0.5).astype(int)


def main():
    missing = [s for s in LEVELS.values() if not D.shard_files(s)]
    if missing:
        print(f"G1 cannot run yet; not cached: {missing}")
        return 2
    if not D.shard_files("FFpp_train_B"):
        print("G1 needs FFpp_train_B for the frozen standardiser; not cached yet")
        return 2

    # The standardiser is fitted on FFpp_train_B224: the same per-clip arm-B draw, applied AFTER the
    # 224 resize, where the evaluation severities are applied. Fitted at the native-256 placement its
    # blockiness never exceeds 1.37 while evaluation JPEG reaches 1.62, so the quadratic extrapolated
    # 8 to 13 sd out of range and d_comp on a_k exploded to 1.0-2.7. At the matched placement the
    # evaluation cells sit within 2.3 sd and the fit interpolates.
    STD_STEM = "FFpp_train_B224" if D.shard_files("FFpp_train_B224") else "FFpp_train_B"
    tr = D.load(STD_STEM, arrays=("eps", "q"))
    real = tr["y"] == 0
    st = FX.fit_standardizer(tr["eps"][real], tr["q"][real])
    print(f"standardiser fitted on {int(real.sum())} real FF++ train frames from {STD_STEM}")

    per_level, frame_level = {}, {}
    for lvl, stem in LEVELS.items():
        d = D.load(stem, arrays=("eps", "q"))
        _, m, yv = video_mean_log_eps(d)
        a = FX.apply_standardizer(d["eps"], d["q"], st)
        _, av = D.video_mean(a, d["video"])
        per_level.setdefault("_a", {})[lvl] = (av, yv)
        per_level[lvl] = {
            "n_videos": int(len(yv)), "n_real": int((yv == 0).sum()), "n_fake": int((yv == 1).sum()),
            "mean_log_eps_real": m[yv == 0].mean(0).tolist(),
            "mean_log_eps_fake": m[yv == 1].mean(0).tolist(),
            "auroc_a": [fast_auc(av[yv == 0, k], av[yv == 1, k]) for k in range(len(BLOCKS))],
        }
        le = np.log(d["eps"].astype(np.float64))
        frame_level[lvl] = {"mean_log_eps_real": le[d["y"] == 0].mean(0).tolist()}

    # PRIMARY: measured on the standardised a_k, which is the quantity R4 actually consumes and the
    # one the brief's "standardized against what real faces of the same quality produce" refers to.
    A = per_level.pop("_a")
    ac, ay = A["clean"]
    clean = ac[ay == 0].mean(0)
    d_fake = ac[ay == 1].mean(0) - clean
    for lvl in LEVELS:
        av, yv = A[lvl]
        per_level[lvl]["mean_a_real"] = av[yv == 0].mean(0).tolist()
        per_level[lvl]["mean_a_fake"] = av[yv == 1].mean(0).tolist()
    # THE RULE IS AMBIGUOUS AND THE TWO READINGS DISAGREE, so both are computed and reported.
    #   signed:    d_comp < d_fake, the brief's literal wording
    #   magnitude: |d_comp| < |d_fake|, "is the quality effect smaller than the manipulation effect"
    # They differ here because every resize-0.50 shift is NEGATIVE -- downsample-then-upsample makes a
    # face MORE predictable, against the brief's premise that "compression and blur also raise the
    # error". A signed test passes automatically whenever processing lowers eps, however large the
    # shift, which is why the magnitude reading is reported as the primary one.
    verdicts = {}
    for lvl in ("jpeg50", "rs050"):
        d_comp = np.array(per_level[lvl]["mean_a_real"]) - clean
        magn = [bool(abs(c) < abs(f)) for c, f in zip(d_comp, d_fake)]
        sign = [bool(c < f) for c, f in zip(d_comp, d_fake)]
        verdicts[lvl] = {"d_comp": d_comp.tolist(),
                         "n_blocks_passing": int(sum(magn)), "per_block": magn,
                         "blocks_passing": [b for b, w in zip(BLOCKS, magn) if w],
                         "n_blocks_passing_signed": int(sum(sign)), "per_block_signed": sign}
    ok = all(v["n_blocks_passing"] >= 3 for v in verdicts.values())
    ok_signed = all(v["n_blocks_passing_signed"] >= 3 for v in verdicts.values())

    res = {"gate": "G1", "verdict": "PASS" if ok else "FAIL",
           "verdict_signed_reading": "PASS" if ok_signed else "FAIL",
           "reading": "primary is |d_comp| < |d_fake|; the signed reading is reported beside it",
           "premise_check": "every resize-0.50 shift is NEGATIVE, so resizing LOWERS the prediction "
                            "error; the brief assumes compression and blur raise it",
           "measured_on": "standardised a_k (the quantity R4 consumes); raw log eps reported per level",
           "standardiser": STD_STEM,
           "rule":
           "d_comp < d_fake for >= 3 of 5 blocks, for both JPEG q50 and resize 0.50",
           "d_fake": d_fake.tolist(), "blocks": BLOCKS, "verdicts": verdicts,
           "per_level": per_level, "frame_level_real_mean_log_eps": frame_level}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "gate_g1.json", "w"), indent=1)

    w = max(len(b) for b in BLOCKS)
    print(f"\n{'block':{w}}  {'d_fake':>8} {'d_comp q50':>11} {'d_comp rs.50':>13}   pass")
    for k, b in enumerate(BLOCKS):
        j, r = verdicts["jpeg50"]["d_comp"][k], verdicts["rs050"]["d_comp"][k]
        mark = ("y" if verdicts["jpeg50"]["per_block"][k] else "n") + \
               ("y" if verdicts["rs050"]["per_block"][k] else "n")
        print(f"{b:{w}}  {d_fake[k]:8.4f} {j:11.4f} {r:13.4f}   {mark}")
    print(f"\nmagnitude reading: JPEG q50 {verdicts['jpeg50']['n_blocks_passing']}/5, "
          f"resize 0.50 {verdicts['rs050']['n_blocks_passing']}/5 -> {'PASS' if ok else 'FAIL'}")
    print(f"signed reading   : JPEG q50 {verdicts['jpeg50']['n_blocks_passing_signed']}/5, "
          f"resize 0.50 {verdicts['rs050']['n_blocks_passing_signed']}/5 -> "
          f"{'PASS' if ok_signed else 'FAIL'}")
    print(f"\n{'block':{w}}  " + "  ".join(f"{l:>7}" for l in LEVELS))
    for k, b in enumerate(BLOCKS):
        print(f"{b:{w}}  " + "  ".join(f"{per_level[l]['auroc_a'][k]:7.4f}" for l in LEVELS))
    print(f"\nG1 {res['verdict']} -> {OUT / 'gate_g1.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
