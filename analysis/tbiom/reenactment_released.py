#!/usr/bin/env python3
"""Reenactment-bearing benchmarks: our framework against DiCoME's released weights and our own DiCoME training.

    OMP_NUM_THREADS=8 python analysis/tbiom/reenactment_released.py

All three systems are scored on the SAME videos with the SAME aggregation -- per-frame opinions averaged per video,
then each system's own fusion. Ours is PCR6+-R over three views; both DiCoME variants use DiCoME's own two-view
Dempster fusion, replayed identically, so the only difference between the two DiCoME rows is the checkpoint.

The documented CDFv3 exclusion (IP_LAP) is applied to the released-weight export too: it is present in that export
and absent from the cached tables, so leaving it in would compare different video sets.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "pcr6p"), str(HERE / "grouped"), str(HERE / "temporal")]
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import arms as AR  # noqa: E402
import ops_pcr6p as P  # noqa: E402
import symbolic_overlay as SO  # noqa: E402
from video_id import video_id  # noqa: E402

R, TS = Path("tbiom/reports"), Path("logs/tsne")
S23 = Path("logs/tbiom/stage23")
EXCLUDE = ("IP_LAP",)                     # documented CDFv3 exclusion, applied to every system
REEN = "FaceReenact|TalkingFace"


def dicome_fuse(frames: pd.DataFrame):
    """DiCoME's own two-view fusion, on per-video mean opinions. Identical replay for both checkpoints."""
    g = frames.groupby("v", as_index=False).agg(
        p_semantic=("p_semantic", "mean"), u_semantic=("u_semantic", "mean"),
        p_artifact=("p_artifact", "mean"), u_artifact=("u_artifact", "mean"), y=("label", "max"),
        subset=("subset", "first"))
    SO.BRANCHES[:] = ["semantic", "artifact"]
    s = SO.fuse(g)
    SO.BRANCHES[:] = ["semantic", "artifact", "fsvfm"]
    return g, s


def load_released(path: Path):
    d = pd.read_csv(path, usecols=["key", "label", "p_semantic", "u_semantic", "p_artifact", "u_artifact"])
    tail = d.key.str.split("::").str[-1]
    d["subset"] = tail.str.split("/").str[1]
    d["v"] = video_id(d["key"])
    return d[~d.subset.str.contains("|".join(EXCLUDE))].reset_index(drop=True)


def load_ourtrained(path: Path):
    d = A.load_frames(path)
    return d[~d.subset.str.contains("|".join(EXCLUDE))].reset_index(drop=True)


def auc(s, y):
    return A.fast_auc(s[y == 0], s[y == 1]) if (y == 0).any() and (y == 1).any() else float("nan")


def main():
    dev, _dp, tests = D.load("vjepa2_lora_B_seed42")
    ws = []
    for _k, Fd, _Sd in D.splits(dev):
        ws.append(AR.fit_w(Fd, "PCR6P-R", A.cells(Fd)))
    out = {}

    for ds in ("CDFv3", "KoDF"):
        # both DiCoME exports come from the same script and the same run of it; only the checkpoint differs
        rel_p, own_p = TS / f"dicome_released_{ds}_scores.csv", TS / f"dicome_{ds}_scores.csv"
        if not (rel_p.exists() and own_p.exists()):
            print(f"skip {ds}: missing {'released' if not rel_p.exists() else 'our-trained'} export"); continue
        rel, own = load_released(rel_p), load_released(own_p)
        T = tests[ds]
        T = T[~T.subset.str.contains("|".join(EXCLUDE))].reset_index(drop=True)
        ours = np.mean([P.p_of(AR.fuse(T, "PCR6P-R", w)) for w in ws], 0)
        gr, sr = dicome_fuse(rel)
        go, so = dicome_fuse(own)
        # align the three systems on the intersection of video ids
        common = sorted(set(T.v) & set(gr.v) & set(go.v))
        iT = T.set_index("v").loc[common]
        ir = gr.set_index("v").loc[common]; sr = pd.Series(sr, index=gr.v).loc[common].to_numpy()
        io = go.set_index("v").loc[common]; so = pd.Series(so, index=go.v).loc[common].to_numpy()
        po = pd.Series(ours, index=T.v).loc[common].to_numpy()
        y = iT.y.to_numpy()
        assert (ir.y.to_numpy() == y).all() and (io.y.to_numpy() == y).all(), "label mismatch across systems"
        sub = iT.subset.to_numpy()
        slices = {"all": np.ones(len(y), bool),
                  "reenactment only": (y == 0) | pd.Series(sub).str.contains(REEN).to_numpy(),
                  "swap only": (y == 0) | ~pd.Series(sub).str.contains(REEN).to_numpy()}
        if ds == "KoDF":
            slices = {"all": np.ones(len(y), bool),
                      "first-order motion only": (y == 0) | (sub == "fake-fo")}
        res = {"n_videos": len(common), "n_real": int((y == 0).sum()), "slices": {}}
        for nm, m in slices.items():
            res["slices"][nm] = {"n_fake": int(((y == 1) & m).sum()),
                                 "ours": auc(po[m], y[m]), "dicome_released": auc(sr[m], y[m]),
                                 "dicome_ourtrain": auc(so[m], y[m])}
        # per reenactment generator
        res["per_generator"] = {}
        for g_ in sorted(set(sub[pd.Series(sub).str.contains(REEN).to_numpy()])):
            m = (y == 0) | (sub == g_)
            res["per_generator"][g_] = {"n_fake": int((sub == g_).sum()),
                                        "ours": auc(po[m], y[m]), "dicome_released": auc(sr[m], y[m]),
                                        "dicome_ourtrain": auc(so[m], y[m])}
        out[ds] = res
        print(f"{ds}: {len(common)} videos | " + " | ".join(
            f"{nm} ours {v['ours']:.4f} rel {v['dicome_released']:.4f} own {v['dicome_ourtrain']:.4f}"
            for nm, v in res["slices"].items()), flush=True)

    json.dump(out, open(R / "EXP_REENACTMENT_RELEASED.json", "w"), indent=1, default=float)
    print("wrote", R / "EXP_REENACTMENT_RELEASED.json")


if __name__ == "__main__":
    main()
