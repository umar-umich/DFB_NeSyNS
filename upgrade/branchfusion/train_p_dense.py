#!/usr/bin/env python3
"""End-to-end PE-Core-L/14 temporal branch on DENSE stride-1 clips (the P-dense variant).

    CUDA_VISIBLE_DEVICES=0 <discern_ext_timm python> upgrade/branchfusion/train_p_dense.py --smoke
    CUDA_VISIBLE_DEVICES=0 <discern_ext_timm python> upgrade/branchfusion/train_p_dense.py --train

WHAT THIS ANSWERS. Branch P -- Dr Kutub's trained PE-Core-L/14 temporal branch -- is order-invariant
to three decimal places: permuting its 16 frames moves its probability by 0.0004-0.0021 and its AUROC
by under 0.0002. It WAS trained on ordered frames, through the same `Clips` class as our V-JEPA
branch, so that is not a training-recipe artefact. The measured cause candidate is FRAME SPACING: the
stored crops it samples from are 10-22 source frames apart, so a "clip" spans several seconds and
holds almost no motion.

This retrains the identical architecture on 32 CONSECUTIVE frames (median span 1.28 s) from
`/data/umar/Datasets/tbiom_dense`, sampling 16 of them, which lands at stride 2 -- the configuration
section 2.2 pre-registers for the dense pilot. If the dense branch becomes order-sensitive, the
inertness was the sampling. If it stays inert with real motion in front of it, the inertness is
architectural, and no amount of retraining P will make it spatio-temporal.

HOW IT REUSES KUTUB'S TRAINER RATHER THAN FORKING IT. The handoff trainer is imported and exactly one
function is replaced: `videos_of`, which resolves a dataset name to (h5 path, items). The dense
version returns the dense h5 and the dense key list. Three details make that safe:

  * `Clips` opens `EXT / self.h5rel`, and pathlib returns an ABSOLUTE right-hand side unchanged, so
    passing an absolute dense path works without touching `Clips`.
  * VIDEO IDS ARE COMPUTED FROM THE SPARSE h5rel, not the dense one, so they match the ids in the
    selection CSVs exactly. Computing them from the dense path would silently produce a disjoint id
    space and the development score would be measured over the wrong units -- a failure that has
    reversed conclusions in this project twice.
  * The scope filter and the label rule are the trainer's own, applied to the same path components.

SCOPE. Dense crops exist for FF++ train and the development pool only, so this is a
DEVELOPMENT-ONLY pilot by construction. It can be trained and selected, never scored on the test
partitions, and `--train` refuses to export test opinions rather than silently reading sparse crops.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                      # noqa: E402

# A PATCHED COPY of the handoff trainer, not the handoff itself: the original writes its run into
# `/data/kutub/.../runs/` and reads its selection CSVs from his `reference/` folder. The copy redirects
# both into this repository and exposes `--epochs`. Writing into another user's directory would be
# wrong whatever the result.
HANDOFF = C.ROOT / "logs/branchfusion/p_dense_code"
DENSE = Path("/data/umar/Datasets/tbiom_dense")
PARTS = {"FFpp_train": ("ffpp_train_dense.h5", "ffpp_train_dense_keys.txt"),
         "FFpp_val": ("dev2_dense.h5", "dev2_dense_keys.txt"),
         "VALmix": ("dev2_dense.h5", "dev2_dense_keys.txt"),
         "VALmixV2": ("dev2_dense.h5", "dev2_dense_keys.txt")}

C.thread_caps()


def _trainer():
    """Import the handoff trainer with its own layout on sys.path, as its modules expect."""
    for p in (HANDOFF / "tbiom/temporal", HANDOFF / "tbiom",
              C.ROOT / "analysis/tbiom/xclip", str(C.EXT)):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    import train_branch as TB
    return TB


def dense_videos_of(ds: str, TB):
    """The dense replacement for `train_branch.videos_of`.

    Returns `(absolute dense h5 path, [(frame keys, label, video id), ...])` restricted to `ds`.
    """
    h5name, keyname = PARTS[ds]
    keys = [k.strip() for k in (DENSE / keyname).read_text().splitlines() if k.strip()]
    keys = [k for k in keys if k.split("/", 1)[0] == ds]
    # the trainer's own scope filter, on the same path components
    keys = [k for k in keys
            if not (k.split("/")[-3] in TB.OUT_OF_SCOPE
                    or k.split("/")[-3].split("-")[-1] in TB.OUT_OF_SCOPE)]
    vids: dict[str, list[str]] = {}
    for k in keys:
        vids.setdefault(k.rsplit("/", 1)[0], []).append(k)
    # Ids from the SPARSE h5rel so they match the selection CSVs.
    sparse_h5rel, _lists = TB.dataset_spec(ds)
    from video_id import video_id
    items = []
    for v in sorted(vids):
        y = 0 if "real" in v.split("/")[-2].lower() else 1
        vid = video_id(pd.Series([f"{sparse_h5rel}::{v}/x.png"])).iloc[0]
        items.append((sorted(vids[v]), y, vid))
    return str(DENSE / h5name), items


def install(TB):
    """Replace `videos_of` and restrict the dataset ledger to the dense partitions.

    The ledger restriction is not cosmetic. Dense crops exist for FF++ train and the development
    pool ONLY, so the trainer's post-selection export loops -- `TESTS`, the processed test
    severities, and the shuffled control's `DEV + [CDFv3, KoDF]` -- would hand `dense_videos_of` a
    dataset that is not in `PARTS` and raise KeyError. That happens AFTER the ten training epochs,
    so the cost of leaving it unpatched is the whole run. Emptying the ledgers is also what this
    module's docstring promises: refuse to export test opinions rather than quietly fall back to the
    sparse crops and report a dense branch scored on sparse clips.
    """
    TB.videos_of = lambda ds: dense_videos_of(ds, TB)
    TB.TESTS = []
    TB.PROCESSED_TESTS = ()

    def dense_shuffled(model, dev, out_dir, workers):
        """The order control, development sets only."""
        for n in TB.DEV:
            TB.opinions(model, n, dev, workers=workers, shuffle=True).to_csv(
                out_dir / "opinions" / f"{n}_shuffled.csv", index=False)
        (out_dir / "SHUFFLED_DONE").touch()

    TB.export_shuffled = dense_shuffled
    return TB


def smoke() -> dict:
    """Check the dense loader against the sparse one before spending GPU hours on it."""
    TB = install(_trainer())
    out = {}
    for ds in ("FFpp_train", "VALmix"):
        h5, items = dense_videos_of(ds, TB)
        strides = []
        for ks, _y, _v in items[:200]:
            idx = sorted(int(k.rsplit("/", 1)[-1].split(".")[0]) for k in ks)
            if len(idx) > 1:
                strides.append(float(np.median(np.diff(idx))))
        # the ids must match the sparse run's ids for the same videos
        sparse_h5, sparse_items = None, None
        import importlib
        mod = importlib.import_module("train_branch")
        orig = mod.__dict__.get("_orig_videos_of")
        out[ds] = {"h5": h5, "videos": len(items),
                   "frames_per_video": int(np.median([len(k) for k, _y, _v in items])),
                   "median_stride_source_frames": float(np.median(strides)) if strides else None,
                   "labels": {"real": sum(1 for _k, y, _v in items if y == 0),
                              "fake": sum(1 for _k, y, _v in items if y == 1)},
                   "example_id": items[0][2], "example_video": items[0][0][0].rsplit("/", 1)[0]}
    # id agreement with the reference selection CSVs
    ref = TB.S23 if (TB.S23 / "warmgate_e19_VALmix.csv").exists() else None
    if ref is None:
        ref = C.S23
        csv = ref / "warmgate_e19_VALmix.csv"
    else:
        csv = ref / "warmgate_e19_VALmix.csv"
    if csv.exists():
        import exp1_fusion_audit as A
        t = A.video_table(A.load_frames(csv), [1.0])
        ids_ref = set(str(v) for v in t.v)
        _h5, items = dense_videos_of("VALmix", TB)
        ids_dense = set(str(v) for _k, _y, v in items)
        inter = ids_ref & ids_dense
        out["id_agreement_VALmix"] = {
            "reference_ids": len(ids_ref), "dense_ids": len(ids_dense),
            "overlap": len(inter),
            "fraction_of_dense_matched": round(len(inter) / max(len(ids_dense), 1), 4),
            "verdict": ("ids match the selection CSVs" if len(inter) >= 0.95 * len(ids_dense)
                        else "ID MISMATCH -- do not train; the development score would be measured "
                             "over the wrong units")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    if a.smoke or not a.train:
        r = smoke()
        print(json.dumps(r, indent=1, default=str))
        json.dump(r, open(C.OUT / "p_dense_smoke.json", "w"), indent=1, default=str)
        ok = r.get("id_agreement_VALmix", {}).get("fraction_of_dense_matched", 0) >= 0.95
        print(f"\nsmoke: {'PASS' if ok else 'FAIL'} -- "
              f"{r.get('id_agreement_VALmix', {}).get('verdict', 'no reference CSV found')}")
        return 0 if ok else 3

    TB = install(_trainer())
    # Train with the trainer's own argv, restricted to the dense development partitions.
    out_dir = C.ROOT / f"logs/branchfusion/p_dense/pecore_l14_dense_seed{a.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = ["train_branch.py", "--model", "pecore_l14", "--lora", "--augment", "B",
                "--seed", str(a.seed), "--epochs", str(a.epochs),
                "--workers", str(a.workers), "--run-suffix", "_dense"]
    # The trainer resolves its own output directory from these flags; the patched copy points it at
    # logs/branchfusion/p_dense/.
    # Resolve the selection gate's imports BEFORE the first batch. The gate runs at the end of each
    # epoch, so a path problem there is invisible until 21 minutes of training have already been spent.
    TB.preflight()
    print(f"training P-dense, seed {a.seed}, {a.epochs} epochs -> {out_dir}", flush=True)
    print("DEVELOPMENT-ONLY PILOT: dense crops exist for FF++ train and the development pool only, "
          "so no test opinions are exported.", flush=True)
    return TB.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
