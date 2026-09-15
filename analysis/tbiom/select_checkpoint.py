#!/usr/bin/env python3
"""Checkpoint selection, rule v2 — reproducible and dev-only.

    python analysis/tbiom/select_checkpoint.py --arm sbiarmA

Specification (tbiom/docs/evaluation-protocol-v2.md), fixed before any v2 selection was run:

  candidates   every checkpoint saved for the run, including last.ckpt; epoch read from the
               checkpoint file itself, not its name
  score        VALmix macro AUROC: the unweighted mean of video-level AUROC over the three VALmix
               domains (CDFv2val, DFDCPval, DFEval24val). Video score = DS replay of the
               per-branch opinions averaged over the video's frames -- the same convention as
               every results table. Unrounded.
  ties         candidates within TIE_TOL = 1e-6 of the best score are tied; the earliest epoch wins
  FF++ val     AUROC logged as a DIAGNOSTIC ONLY. It never excludes or selects a checkpoint.
  threshold    tau = EER of the same video-level DS-replay score on FF++ val, recomputed for the
               selected checkpoint. Never carried over from another checkpoint.

The historical selector (stage23_select.py, exported p_fused, used for Stages 2/3/5) is kept
unchanged so earlier selections stay reproducible.
"""
from __future__ import annotations

import argparse, hashlib, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health import auroc, eer  # noqa: E402
import symbolic_overlay as SO  # noqa: E402
from video_id import video_id  # noqa: E402

RULE_VERSION = "v2-2026-09-13"
OUT = Path("logs/tbiom/stage23")
EXT = Path("/data/umar/Repos/DISCERN_Ext")
VALMIX_SPLITS = EXT / "eval_adaptation/data/splits/VALmix"
DOMAINS = ("CDFv2val", "DFDCPval", "DFEval24val")
TIE_TOL = 1e-6
RUN_DIR = {"dicomeours": "../_baseline_dicome/dicome_train/ffpp_reproduce_seed42", "run1auxedl": "run1_auxedl_seed42", "sbiarmA": "sbi_armA_seed42",
           "sbiarmB": "sbi_armB_seed42", "sbiarmC": "sbi_armC_seed42",
           "sbiarmAs1337": "sbi_armA_seed1337", "sbiarmAs7": "sbi_armA_seed7",
           "sbiarmBs1337": "sbi_armB_seed1337", "sbiarmBs7": "sbi_armB_seed7", "sbiarmBs42r": "sbi_armB_seed42r",
           "warmgate": "warm_dicome_gate_seed42"}
BR = SO.BRANCHES


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def video_table(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    d = pd.read_csv(path)
    need = [f"{k}_{b}" for b in BR for k in ("p", "u")]
    if any(c not in d for c in need):
        return None
    d["v"] = video_id(d["key"])
    g = d.groupby("v", as_index=False).agg(**{c: (c, "mean") for c in need},
                                          y=("label", "max"), raw=("key", "first"))
    g["domain"] = "other"
    for name in DOMAINS:
        g.loc[g["raw"].astype(str).str.contains(name, regex=False), "domain"] = name
    g["s"] = SO.fuse(g)
    return g


def checkpoint_for(arm: str, tag: str) -> Path | None:
    ck = EXT / "runs/discern_ext_train" / RUN_DIR[arm] / "checkpoints"
    m = re.search(r"_e(\d+)$", tag)
    hits = sorted(ck.glob(f"*epoch={m.group(1)}-*.ckpt")) if m else [ck / "last.ckpt"]
    return hits[0] if hits and hits[0].is_file() else None


def stored_epoch(path: Path) -> int:
    # Lightning checkpoints pickle the project Config; unpickling needs DISCERN_Ext importable.
    if str(EXT) not in sys.path:
        sys.path.insert(0, str(EXT))
    import torch
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return int(ck["epoch"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(RUN_DIR))
    args = ap.parse_args()

    tags = sorted({p.name[: -len("_FFpp_val.csv")] for p in OUT.glob(f"{args.arm}_*_FFpp_val.csv")
                   if re.fullmatch(rf"{args.arm}_(e\d+|last)", p.name[: -len("_FFpp_val.csv")])})
    manifest = {}
    for f in sorted(VALMIX_SPLITS.glob("*.txt")):
        keys = f.read_text().split()
        manifest[f.name] = {"sha256": sha256(f), "frames": len(keys),
                            "videos": len({k.rsplit("/", 1)[0] for k in keys})}

    cands = []
    for tag in tags:
        ff, vm, ck = (video_table(OUT / f"{tag}_FFpp_val.csv"), video_table(OUT / f"{tag}_VALmix.csv"),
                      checkpoint_for(args.arm, tag))
        if ff is None or vm is None or ck is None:
            print(f"  {tag}: incomplete, skipped"); continue
        per = {}
        for dom in DOMAINS:
            s = vm[vm["domain"] == dom]
            per[dom] = {"videos": int(len(s)), "auroc": float(auroc(s["y"].to_numpy(), s["s"].to_numpy()))}
        cands.append({"tag": tag, "checkpoint": str(ck), "epoch": stored_epoch(ck),
                      "score": float(np.mean([per[d]["auroc"] for d in DOMAINS])),
                      "per_domain": per,
                      "ffpp_val_auroc_diagnostic": float(auroc(ff["y"].to_numpy(), ff["s"].to_numpy()))})
    if not cands:
        print(f"no complete candidates for {args.arm}"); return 1

    best = max(c["score"] for c in cands)
    tied = [c for c in cands if best - c["score"] <= TIE_TOL]
    sel = min(tied, key=lambda c: c["epoch"])
    ff = video_table(OUT / f"{sel['tag']}_FFpp_val.csv")
    _, tau = eer(ff["y"].to_numpy(), ff["s"].to_numpy())
    result = {"rule_version": RULE_VERSION, "arm": args.arm, "tie_tol": TIE_TOL,
              "selected": {**sel, "sha256": sha256(Path(sel["checkpoint"])), "tau_ffpp_val": float(tau),
                           "tied_with": [c["tag"] for c in tied if c is not sel]},
              "candidates": sorted(cands, key=lambda c: c["epoch"]), "valmix_manifest": manifest}
    out = OUT / f"{args.arm}_selection_v2.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"{'tag':16} {'epoch':>5} {'VALmix macro AUROC':>19} {'FF++ val (diag)':>16}")
    for c in result["candidates"]:
        mark = "  <- selected" if c is sel else ("  (tied)" if c in tied else "")
        print(f"{c['tag']:16} {c['epoch']:5d} {c['score']:19.8f} {c['ffpp_val_auroc_diagnostic']:16.6f}{mark}")
    print(f"tau (FF++ val EER, DS replay) = {tau:.6f}; sha256 {result['selected']['sha256'][:16]}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
