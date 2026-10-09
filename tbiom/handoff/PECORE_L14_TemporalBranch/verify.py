#!/usr/bin/env python3
"""Check this copy of the V-JEPA 2 temporal branch, without trusting anything in the folder.

    <temporal_vjepa2 python> verify.py --check                 # deps, data, weights. no GPU, seconds
    CUDA_VISIBLE_DEVICES=N <...> verify.py --rerun CDFv2       # re-export one set and diff vs reference

`--check` proves the folder can run. `--rerun` proves it produces OUR numbers: it loads the shipped
checkpoint, re-exports opinions for one dataset through the shipped code, and compares them against
the reference CSV video by video. That is the check that matters -- a folder that imports cleanly
but scores differently is the failure worth catching.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE / "code"
EXT = Path("/data/umar/Repos/DISCERN_Ext")          # h5 data, splits, dataset configs, two DISCERN modules
REF = HERE / "reference" / "opinions"

# AUROC of the shipped opinions. Regenerating a set must land on its number; the tolerance is for
# GPU non-determinism, not for a different model.
EXPECTED = {"FFpp_val": 0.9904, "VALmix": 0.7420, "CDFv2": 0.7383, "CDFv3": 0.9263, "DFD": 0.7967,
            "DFDC": 0.7009, "DFDCP": 0.7408, "DFEval24": 0.6732, "FFIW": 0.8011, "KoDF": 0.9656,
            "IDForge": 0.8899, "DFDM": 0.9798, "DF40_tpsm_cdf": 0.9915, "DF40_danet_cdf": 0.9522}
TOL = 0.002

# Named slices of a test set. The subset is the second-to-last component of the video id, so a slice
# is derivable from the shipped opinions alone -- no re-export needed. `CDFv3 reenactment` is the
# slice the fusion study reports, and its definition here is the one in dsr_system.py: every real,
# plus the fakes whose subset is FaceReenact or TalkingFace.
SLICES = {"CDFv3-reenactment": ("CDFv3", "FaceReenact|TalkingFace", 0.9741, 3378),
          "CDFv3-facereenact": ("CDFv3", "FaceReenact", 0.9705, 1578),
          "CDFv3-talkingface": ("CDFv3", "TalkingFace", 0.9769, 1978),
          "CDFv3-faceswap": ("CDFv3", "FaceSwap", 0.8383, 1918)}


def slice_of(d, pattern):
    """Every real, plus the fakes whose subset matches. Reals are never filtered: a slice changes
    which manipulations are asked about, not which genuine videos the detector must not flag."""
    import pandas as pd
    d = d.copy()
    d["subset"] = d.v.str.split("/").str[-2]
    return pd.concat([d[d.y == 0], d[(d.y == 1) & d.subset.str.contains(pattern)]], ignore_index=True)


def slices(names=None) -> int:
    """Score the named slices from the shipped opinions, with the frame-order control beside them."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    bad = 0
    print(f"{'slice':22s} {'videos':>7s} {'clean':>8s} {'expected':>9s} {'shuffled':>9s} {'delta':>8s}")
    for name in (names or SLICES):
        ds, pat, want, n_want = SLICES[name]
        d = pd.read_csv(REF / f"{ds}.csv")
        a = slice_of(d, pat)
        got = roc_auc_score(a.y, a.p)
        sh = REF / f"{ds}_shuffled.csv"
        if sh.is_file():
            b = slice_of(pd.read_csv(sh), pat)
            shv = roc_auc_score(b.y, b.p)
            sht, dl = f"{shv:9.4f}", f"{shv - got:+8.4f}"
        else:
            sht, dl = f"{'--':>9s}", f"{'--':>8s}"
        flag = "" if abs(got - want) < 1e-3 and len(a) == n_want else "   <-- MISMATCH"
        bad += bool(flag)
        print(f"{name:22s} {len(a):7d} {got:8.4f} {want:9.4f} {sht} {dl}{flag}")
    print("\nShuffled = the same weights with frames in a fixed shuffled order. A branch that reads")
    print("motion should lose more than this when the order is destroyed; see README.")
    return 1 if bad else 0


def ok(label, good, detail=""):
    print(f"  [{'PASS' if good else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return good


def check() -> int:
    fails = []
    print("environment")
    try:
        import torch, transformers
        fails.append(not ok("torch / transformers import",
                            True, f"torch {torch.__version__}, transformers {transformers.__version__}"))
        fails.append(not ok("CUDA visible", torch.cuda.is_available(),
                            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"))
    except Exception as exc:                                        # noqa: BLE001
        fails.append(not ok("torch / transformers import", False, repr(exc)))
    try:
        from transformers import VJEPA2Model  # noqa: F401
        fails.append(not ok("VJEPA2Model available", True))
    except Exception as exc:                                        # noqa: BLE001
        fails.append(not ok("VJEPA2Model available", False,
                            f"{exc!r} -- transformers too old; use the temporal_vjepa2 env"))

    print("\nexternal paths (data is NOT copied into this folder; it is read in place)")
    fails.append(not ok("DISCERN_Ext readable", (EXT / "eval_adaptation/data/h5").is_dir(), str(EXT)))
    for ds, rel in (("FFpp_train", "eval_adaptation/data/h5/FFpp_train.h5"),
                    ("CDFv2", "eval_adaptation/data/h5/CDFv2.h5")):
        p = EXT / rel
        fails.append(not ok(f"{ds} h5", p.is_file(), f"{p} ({p.stat().st_size / 2**30:.1f} GiB)" if p.is_file() else str(p)))
    for mod in ("src/losses/evidential_loss.py", "src/dataset/training_arms.py"):
        fails.append(not ok(f"DISCERN_Ext {mod}", (EXT / mod).is_file()))

    print("\nshipped artefacts")
    w = HERE / "weights/best.pt"
    fails.append(not ok("weights/best.pt", w.is_file(), f"{w.stat().st_size / 2**20:.0f} MiB" if w.is_file() else "missing"))
    man = HERE / "MANIFEST.sha256"
    if man.is_file():
        bad = []
        for line in man.read_text().split("\n"):
            if not line.strip() or line.startswith("#"):
                continue
            digest, rel = line.split(maxsplit=1)
            f = HERE / rel.strip()
            if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
                bad.append(rel.strip())
        fails.append(not ok("MANIFEST.sha256 matches", not bad, f"{len(bad)} mismatched: {bad[:3]}" if bad else "all files"))
    rec = json.loads((HERE / "weights/train_record.json").read_text())
    fails.append(not ok("checkpoint is the selected epoch", rec["best_epoch"] == 6,
                        f"epoch {rec['best_epoch']}, dev v3.1 {rec['best_dev_v31']:.4f}"))

    print("\nshipped code imports (the real modules, laid out so their own sys.path inserts resolve)")
    sys.path[:0] = [str(CODE / "tbiom"), str(CODE / "tbiom/xclip"), str(CODE / "tbiom/temporal"), str(EXT)]
    try:
        import train_branch as TB
        fails.append(not ok("train_branch imports", True,
                            f"{len(TB.MODELS)} model defs, batch {TB.BATCH}, {TB.EPOCHS} epochs"))
        h5, lists = TB.dataset_spec("CDFv2")
        fails.append(not ok("dataset_spec resolves", (EXT / h5).is_file(), f"{h5}, {len(lists)} split lists"))
    except Exception as exc:                                        # noqa: BLE001
        fails.append(not ok("train_branch imports", False, repr(exc)))

    print("\nreference opinions")
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    for ds, want in list(EXPECTED.items())[:4]:
        p = REF / f"{ds}.csv"
        if not p.is_file():
            fails.append(not ok(f"{ds}.csv", False, "missing")); continue
        d = pd.read_csv(p)
        got = roc_auc_score(d.y, d.p)
        fails.append(not ok(f"{ds}.csv AUROC", abs(got - want) < 1e-3, f"{got:.4f} (expected {want:.4f}, {len(d)} videos)"))

    n = sum(fails)
    print(f"\n{'ALL CHECKS PASSED' if not n else str(n) + ' CHECK(S) FAILED'}")
    print("`--check` does not run the model. Run `--rerun CDFv2` to confirm the weights reproduce our numbers.")
    return 1 if n else 0


def rerun(ds: str, workers: int) -> int:
    import numpy as np, pandas as pd, torch
    from sklearn.metrics import roc_auc_score
    sys.path[:0] = [str(CODE / "tbiom"), str(CODE / "tbiom/xclip"), str(CODE / "tbiom/temporal"), str(EXT)]
    import train_branch as TB

    dev = "cuda"
    model = TB.Branch("vjepa2", lora=True).to(dev)
    ck = torch.load(HERE / "weights/best.pt", map_location=dev)
    model.trainable_blocks.load_state_dict(ck["trainable_blocks"]); model.head.load_state_dict(ck["head"])
    print(f"loaded epoch {ck['epoch']}; re-exporting {ds}", flush=True)
    got = TB.opinions(model, ds, dev, workers=workers)

    ref = REF / f"{ds}.csv"
    a = roc_auc_score(got.y, got.p)
    print(f"\n  re-exported AUROC {a:.4f} over {len(got)} videos")
    if ds in EXPECTED:
        print(f"  expected          {EXPECTED[ds]:.4f}   -> {'MATCH' if abs(a - EXPECTED[ds]) < TOL else 'MISMATCH'}")
    if ref.is_file():
        m = pd.read_csv(ref).merge(got, on="v", suffixes=("_ref", "_new"))
        if len(m) != len(got):
            print(f"  !! video sets differ: {len(got)} re-exported, {len(m)} matched to the reference")
        d = np.abs(m.p_ref - m.p_new)
        print(f"  per-video |p_ref - p_new|: max {d.max():.2e}, mean {d.mean():.2e}, "
              f"{(d > 1e-3).sum()} of {len(m)} above 1e-3")
        return 0 if (ds not in EXPECTED or abs(a - EXPECTED[ds]) < TOL) else 1
    print(f"  (no reference CSV for {ds})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rerun", metavar="DATASET")
    ap.add_argument("--slices", nargs="*", metavar="NAME",
                    help=f"score named slices from the shipped opinions (no GPU). known: {', '.join(SLICES)}")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.slices is not None:
        sys.exit(slices(a.slices or None))
    if a.rerun:
        sys.exit(rerun(a.rerun, a.workers))
    sys.exit(check())
