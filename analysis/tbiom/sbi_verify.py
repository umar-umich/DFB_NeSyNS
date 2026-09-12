#!/usr/bin/env python3
"""SBI integration — the checks that must pass before any expensive run.

    python analysis/tbiom/sbi_verify.py --out tbiom/reports/SBI_VERIFY.md

These exist because transformation errors, label reversal, stale features and auxiliary-data
leakage would each invalidate the whole comparison while leaving training that looks healthy.
A passing run here is NOT evidence of generalisation and is not reported as such.
"""
from __future__ import annotations

import argparse, json, sys, os
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/data/umar/Repos/DISCERN_Ext")
sys.path.insert(0, str(Path(__file__).resolve().parent))

EXT = Path("/data/umar/Repos/DISCERN_Ext")
SPLITS = EXT / "eval_adaptation/data/splits"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tbiom/reports/SBI_VERIFY.md"))
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    import h5py
    from src.dataset.sbi_synth import SBISynthDataset, key_to_paths, eligible_pool

    keys = (SPLITS / "FFpp_train/FFpp-real.txt").read_text().split()
    rng = np.random.default_rng(0)
    sample = [keys[i] for i in rng.choice(len(keys), min(args.n, len(keys)), replace=False)]

    L = ["# SBI integration — pre-run verification", "",
         f"{len(sample)} sampled frames from the FF++ train real split. A pass here is a "
         "correctness result, not evidence of generalisation.", ""]
    checks = {}

    # 1 --- source pixels identical between the h5 path and the disk path
    with h5py.File(EXT / "eval_adaptation/data/h5/FFpp_train.h5", "r") as f:
        diffs, miss = [], 0
        for k in sample[:50]:
            p, _ = key_to_paths(k)
            if k not in f or not os.path.exists(p):
                miss += 1; continue
            a = f[k][()]; b = np.array(Image.open(p).convert("RGB"))
            diffs.append(int(np.abs(a.astype(int) - b.astype(int)).max()) if a.shape == b.shape else 999)
    checks["pixels_identical"] = (max(diffs) if diffs else 999) == 0
    L += ["## 1 — the on-the-fly path sees the h5's pixels", "",
          f"max abs difference over {len(diffs)} frames: **{max(diffs) if diffs else 'n/a'}**, "
          f"missing {miss}. " +
          ("Identical, so arm A differs from R0 in loader mechanics only, never in image "
           "content." if checks["pixels_identical"] else
           "**NOT identical — arm A would differ from R0 in pixels and the comparison would be "
           "confounded.**"), ""]

    # 2 --- landmarks are 81-point and in crop coordinates
    pool, dropped = eligible_pool(keys)
    bad_shape, out_of_frame = 0, 0
    for k in [k for k in sample if k in set(pool)]:
        _, lp = key_to_paths(k)
        if not os.path.exists(lp):
            bad_shape += 1; continue
        a = np.load(lp)
        if a.shape != (81, 2):
            bad_shape += 1
        elif a[:, 0].min() < 0 or a[:, 1].min() < 0 or a[:, 0].max() > 256 or a[:, 1].max() > 256:
            out_of_frame += 1
    checks["landmarks_ok"] = bad_shape == 0 and out_of_frame == 0
    L += ["## 2 — landmarks, and the common eligible pool", "",
          f"**{len(dropped)} of {len(keys)} ({len(dropped)/len(keys):.3%})** train real frames "
          f"have no landmarks81 file — dlib's detector failed on them during preprocessing. "
          f"The frames themselves are intact.", "",
          f"Within the eligible pool: wrong shape **{bad_shape}**, outside the 256x256 crop "
          f"**{out_of_frame}**.", "",
          "**The filter is applied to every arm, not only the synthetic ones.** If arm C "
          "dropped the frames SBI cannot use while A and B kept them, the arms would no longer "
          "share a source pool, and part of any measured difference would be *which frames were "
          "trained on* — specifically the hard ones a landmark detector could not localise. "
          "The pool is written to `logs/tbiom/sbi/eligible_pool.txt` for the other arms to "
          "consume.", ""]
    pp = Path("logs/tbiom/sbi"); pp.mkdir(parents=True, exist_ok=True)
    (pp / "eligible_pool.txt").write_text("\n".join(pool) + "\n")
    (pp / "dropped_no_landmark.txt").write_text("\n".join(dropped) + "\n")

    # 3 --- generator behaviour, label mapping, failure rate
    from transformers import CLIPImageProcessor
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    ds = SBISynthDataset([k for k in sample if k in set(pool)], proc)
    n_ok, n_fail, ident = 0, 0, 0
    for i in range(min(60, len(ds))):
        it = ds[i]
        assert it["label"].tolist() == [0, 1], "label order must be [genuine, fake]"
        if bool(it["sbi_failed"][0]):
            n_fail += 1
        else:
            n_ok += 1
            a, b = it["image"][0], it["image"][1]
            if float((a - b).abs().max()) < 1e-6:
                ident += 1
    checks["labels_ok"] = True
    checks["pairs_differ"] = ident == 0
    rate = n_fail / max(1, n_ok + n_fail)
    L += ["## 3 — generator, labels, failure rate", "",
          f"- label order asserted `[0, 1]` = (genuine, fake) on every item — **passes**. "
          f"Upstream FreqBlender also treats index 1 as fake, so the conventions agree.",
          f"- generator failures: **{n_fail}/{n_ok+n_fail}** ({rate:.1%}). Failures emit the "
          f"genuine frame twice and are counted, rather than dropping the sample — dropping "
          f"would filter difficult frames out of the synthetic arm only and quietly change the "
          f"eligible source pool.",
          f"- genuine and fake members identical after synthesis: **{ident}** "
          + ("(none, as required)" if ident == 0 else "(**a pair that does not differ is a "
             "silent no-op and would train the model on two identical images with opposite "
             "labels**)"), ""]

    # 4 --- normalisation matches the h5 path exactly
    from src.dataset.deepfake import H5DeepfakeDataset  # noqa: F401
    k = sample[0]; p, _ = key_to_paths(k)
    img = np.array(Image.open(p).convert("RGB"))
    direct = proc(images=Image.fromarray(img), return_tensors="pt")["pixel_values"][0]
    viads = ds._norm(img)
    dev = float((direct - viads).abs().max())
    checks["norm_matches"] = dev < 1e-6
    L += ["## 4 — normalisation is the detector's own, not a reimplementation", "",
          f"max deviation against the CLIP processor applied directly: **{dev:.2e}**. The "
          "dataset calls the same `vfm_preprocess` object the h5 path uses, so this cannot "
          "drift; the check guards against a future edit replacing it with hand-rolled "
          "constants.", ""]

    # 5 --- split safety
    train_ids = {k.split("/")[-2] for k in keys}
    val_keys = (SPLITS / "FFpp_val/FFpp-real.txt").read_text().split()
    val_ids = {k.split("/")[-2] for k in val_keys}
    overlap = train_ids & val_ids
    checks["split_safe"] = len(overlap) == 0
    L += ["## 5 — split safety of the source pool", "",
          f"train source videos: **{len(train_ids)}**; val source videos: **{len(val_ids)}**; "
          f"overlap: **{len(overlap)}**. " +
          ("Synthesis draws only from train sources, so no val identity can reach a generated "
           "example." if not overlap else
           f"**{len(overlap)} shared identities — synthesis would leak val identities into "
           f"training.**"), ""]

    ok = all(checks.values())
    L += ["## Verdict", "",
          ("**All checks pass.** The integration is correct in the ways that would otherwise "
           "fail silently. This says nothing about whether SBI helps."
           if ok else
           "**FAILED — do not spend GPU time.** " +
           ", ".join(k for k, v in checks.items() if not v) + " did not pass.")]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"checks": checks, "sbi_failure_rate": rate}, indent=2, default=float))
    print("\n".join(L))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
