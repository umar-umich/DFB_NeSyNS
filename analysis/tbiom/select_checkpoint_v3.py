#!/usr/bin/env python3
"""Checkpoint selection, protocol v3 (tbiom/docs/evaluation-protocol-v3.md §3-4).

    python analysis/tbiom/select_checkpoint_v3.py --arm sbiarmA

Cells = (source, forgery type): AUROC of that source's fakes of that type vs that source's genuine videos
(DF40train vs FF++ val genuine). Type score = mean of cells; selection = mean of five type scores.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoint as V2  # noqa: E402  (shared: checkpoint lookup, epoch-from-file, sha256)
from health import auroc, eer  # noqa: E402
import symbolic_overlay as SO  # noqa: E402
from video_id import video_id  # noqa: E402

RULE_VERSION = "v3-2026-09-14"
OUT, TIE_TOL, BR = V2.OUT, 1e-6, SO.BRANCHES
TYPES = {
    "swap": {"CDFv2val": "*", "DFDCPval": "*",
             "CDFv3valG": ["BlendFace", "Celeb-DF-v2", "GHOST", "HifiFace", "InSwapper", "MobileFaceSwap", "SimSwap", "UniFace"],
             "DS11val": ["facefusion", "facefusion_gan", "facefusion_live"], "DS20val": ["facefusion"]},
    "reenactment": {"CDFv3valG": ["DaGAN", "FSRT", "HyperReenact", "LIA", "LivePortrait", "MCNET", "TPSMM"],
                    "DS20val": ["liveportrait", "hellomeme"],
                    "DF40train": ["danet", "facevid2vid", "fomm", "hyperreenact", "lia", "mcnet", "MRAA", "one_shot_free", "pirender", "tpsm"]},
    "talking_face": {"CDFv3valG": ["AniTalker", "EchoMimic", "EDTalk", "FLOAT", "IP_LAP", "R3DPortrait", "SadTalker"],
                     "DS11val": ["retalking", "wav2lip"], "DS20val": ["diff2lip", "latentsync", "memo"],
                     "DF40train": ["sadtalker", "wav2lip"]},
    "entire_face_synthesis": {"DF40train": ["ddim", "DiT", "pixart", "RDDM", "sd2.1", "SiT", "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN"]},
    "in_the_wild": {"DFEval24val": "*", "FFIWval": "*"},
}


def table(path):
    if not path.is_file(): return None
    d = pd.read_csv(path)
    need = [f"{k}_{b}" for b in BR for k in ("p", "u")]
    d["v"] = video_id(d["key"]); raw = d["key"].astype(str).str.split("::").str[-1]
    d["subset"] = raw.str.split("/").str[-3]
    d["source"] = raw.map(lambda k: next((s for s in ("CDFv2val", "DFDCPval", "DFEval24val") if s in k), k.split("/")[-4]))
    g = d.groupby("v", as_index=False).agg(**{c: (c, "mean") for c in need}, y=("label", "max"),
                                          subset=("subset", "first"), source=("source", "first"))
    g["s"] = SO.fuse(g); return g


def score(tag):
    vm, v2, ff = (table(OUT / f"{tag}_{n}.csv") for n in ("VALmix", "VALmixV2", "FFpp_val"))
    if vm is None or v2 is None or ff is None: return None
    allv = pd.concat([vm, v2], ignore_index=True); ffreal = ff[ff.y == 0]
    types, cells = {}, {}
    for t, srcs in TYPES.items():
        vals = []
        for src, subs in srcs.items():
            s = allv[allv.source == src]
            fake = s[s.y == 1] if subs == "*" else s[(s.y == 1) & s.subset.isin(subs)]
            real = ffreal if src == "DF40train" else s[s.y == 0]
            if len(fake) == 0 or len(real) == 0: raise ValueError(f"{tag}: empty cell {t}/{src}")
            x = pd.concat([real, fake]); a = float(auroc(x.y.to_numpy(), x.s.to_numpy()))
            cells[f"{t}/{src}"] = {"auroc": a, "n_real": int(len(real)), "n_fake": int(len(fake))}; vals.append(a)
        types[t] = float(np.mean(vals))
    return {"score": float(np.mean(list(types.values()))), "types": types, "cells": cells,
            "ffpp_val_auroc_diagnostic": float(auroc(ff.y.to_numpy(), ff.s.to_numpy()))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=sorted(V2.RUN_DIR))
    args = ap.parse_args()
    tags = sorted({p.name[: -len("_VALmixV2.csv")] for p in OUT.glob(f"{args.arm}_*_VALmixV2.csv")
                   if re.fullmatch(rf"{args.arm}_(e\d+|last)", p.name[: -len("_VALmixV2.csv")])})
    cands = []
    for tag in tags:
        r, ck = score(tag), V2.checkpoint_for(args.arm, tag)
        if r is None or ck is None: print(f"  {tag}: incomplete, skipped"); continue
        cands.append({"tag": tag, "checkpoint": str(ck), "epoch": V2.stored_epoch(ck), **r})
    if not cands: print("no complete candidates"); return 1
    best = max(c["score"] for c in cands); tied = [c for c in cands if best - c["score"] <= TIE_TOL]
    sel = min(tied, key=lambda c: c["epoch"])
    ff = table(OUT / f"{sel['tag']}_FFpp_val.csv"); _, tau = eer(ff.y.to_numpy(), ff.s.to_numpy())
    man = {p.name: V2.sha256(p) for p in sorted((V2.EXT / "eval_adaptation/data/splits/VALmixV2").glob("*.txt"))}
    man.update({f"VALmix/{k}": v["sha256"] for k, v in json.loads(json.dumps(
        {f.name: {"sha256": V2.sha256(f)} for f in sorted(V2.VALMIX_SPLITS.glob("*.txt"))})).items()})
    res = {"rule_version": RULE_VERSION, "arm": args.arm, "tie_tol": TIE_TOL,
           "selected": {**sel, "sha256": V2.sha256(Path(sel["checkpoint"])), "tau_ffpp_val": float(tau),
                        "tied_with": [c["tag"] for c in tied if c is not sel]},
           "candidates": sorted(cands, key=lambda c: c["epoch"]), "split_hashes": man}
    (OUT / f"{args.arm}_selection_v3.json").write_text(json.dumps(res, indent=2))
    print(f"{'tag':16} {'ep':>3} {'score':>11} " + " ".join(f"{t[:10]:>10}" for t in TYPES))
    for c in res["candidates"]:
        print(f"{c['tag']:16} {c['epoch']:3d} {c['score']:11.6f} " + " ".join(f"{c['types'][t]:10.4f}" for t in TYPES)
              + ("  <- selected" if c["tag"] == sel["tag"] else ""))
    print(f"tau {tau:.6f}; wrote {OUT}/{args.arm}_selection_v3.json"); return 0


if __name__ == "__main__":
    sys.exit(main())
