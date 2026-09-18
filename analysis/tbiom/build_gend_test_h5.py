#!/usr/bin/env python3
"""Build h5, split files and export configs for new OOD test sets from the GenD release.

    python analysis/tbiom/build_gend_test_h5.py --dataset FFIW

Source: /data/kutub/datasets/GEND/extracted/test — 256x256 face crops, <=32 frames per video.
GenD's crop convention (x1.3 margin, detector threshold 0.5) differs from ours; recorded, not corrected.

Key = <DATASET>/<subset>/<video>/<frame>. The H5 loader labels a frame genuine iff its subset name
contains "real", so every genuine subset name contains "real" and no fake subset name does.

Choices, fixed before any result on these sets:
  KoDF     clean subsets: real + fake-{fsgan,dfl,dffs,fo,audio-driven}; the adversarially perturbed
           real-adv / fake-adv are separate splits (adv-real, adv-fake), never pooled into clean KoDF.
  IDForge  the three visual face-swap subsets (infoswap, roop, simswap; audio is TTS, ignored). Genuine
           videos: 2,987 drawn from 18,834 with numpy default_rng(0), matching the fake count.
  DFDM     no genuine videos ship with DFDM; following GenD, Celeb-real and YouTube-real from Celeb-DF v2
           test are used. This is the same genuine pool as CDFv2/CDFv3, so results are not independent.
  FFIW     test-real / test-fake as released.
"""
import argparse, json, os, re, sys
from pathlib import Path
import numpy as np, h5py
from PIL import Image

EXT = Path("/data/umar/Repos/DISCERN_Ext")
G = Path("/data/kutub/datasets/GEND/extracted/test")
H5DIR, SPL, CFG = EXT / "eval_adaptation/data/h5", EXT / "eval_adaptation/data/splits", EXT / "eval_adaptation/configs"
ARMS = ["run1auxedl", "sbiarmA", "sbiarmB", "sbiarmC", "warmgate"]


def videos(d: Path):
    return sorted(x for x in os.listdir(d) if (d / x).is_dir())


def plan(ds):
    if ds == "FFIW":
        return [("FFIW-real", G / "FFIW/test-real", None), ("FFIW-fake", G / "FFIW/test-fake", None)]
    if ds == "KoDF":
        k = G / "KoDF"
        out = [("real", k / "real", None), ("adv-real", k / "real-adv", None), ("adv-fake", k / "fake-adv", None)]
        out += [(s, k / s, None) for s in ("fake-fsgan", "fake-dfl", "fake-dffs", "fake-fo", "fake-audio-driven")]
        return out
    if ds == "IDForge":
        k = G / "IDForge-v1"
        fakes = [("infoswap", k / "test-face_tts_infoswap"), ("roop", k / "test-face_tts_roop"),
                 ("simswap", k / "test-face_tts_simswap")]
        n_fake = sum(len(videos(p)) for _, p in fakes)
        reals = videos(k / "test-real")
        pick = sorted(np.random.default_rng(0).choice(reals, size=n_fake, replace=False).tolist())
        return [("real", k / "test-real", pick)] + [(s, p, None) for s, p in fakes]
    if ds == "FaceShifter":
        # SEALED until the recipe is fixed (tbiom/docs/SEALED_BENCHMARKS.md). The official release manipulates the
        # FF++ videos, and the audit of 2026-09-17 established that this release covers exactly the FF++ TEST split
        # ids, disjoint from our train (720) and validation (140) source ids -- unseen manipulation on held-out
        # source videos, which is what the report must say. Real and fake sit in one directory each.
        k = G / "FSh"
        return [("FaceShifter-real", k / "real", None), ("FaceShifter-fake", k / "fake", None)]
    if ds == "DFDM":
        k = G / "DFDM"
        return [("Celeb-real", G / "CDFv2/Celeb-real", None), ("YouTube-real", G / "CDFv2/YouTube-real", None)] + \
               [(s, k / s, None) for s in ("dfaker", "dfl", "iae", "lightweight")]
    raise ValueError(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["FFIW", "KoDF", "IDForge", "DFDM", "FaceShifter"])
    ap.add_argument("--unseal", action="store_true",
                    help="required for FaceShifter: confirms the recipe is fixed and the seal has been lifted")
    args = ap.parse_args(); ds = args.dataset
    if ds == "FaceShifter" and not args.unseal:
        print("FaceShifter is sealed (tbiom/docs/SEALED_BENCHMARKS.md). Build it only once the final recipe is fixed, "
              "and then pass --unseal; it is scored once, with no tuning afterwards.")
        return 2
    items = plan(ds)
    for sub, _, _ in items:
        assert ("real" in sub.lower()) == (sub in ("FFIW-real", "real", "adv-real", "Celeb-real", "YouTube-real",
                                                   "FaceShifter-real")), sub
    h5p = H5DIR / f"{ds}.h5"
    if h5p.exists():
        print(f"{h5p} exists; refusing to overwrite"); return 1
    (SPL / ds).mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": ds, "source": str(G), "subsets": {}}
    tmp = h5p.with_suffix(".h5.partial")
    with h5py.File(tmp, "w") as h:
        for sub, src, pick in items:
            vs = pick if pick is not None else videos(src)
            keys = []
            for v in vs:
                for fr in sorted(os.listdir(src / v)):
                    if not fr.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    key = f"{ds}/{sub}/{v}/{fr}"
                    h.create_dataset(key, data=np.asarray(Image.open(src / v / fr).convert("RGB")), compression="gzip")
                    keys.append(key)
            (SPL / ds / f"{sub}.txt").write_text("\n".join(keys) + "\n")
            manifest["subsets"][sub] = {"label": 0 if "real" in sub.lower() else 1, "videos": len(vs),
                                        "frames": len(keys), "source_dir": str(src)}
            print(f"  {ds}/{sub}: {len(vs)} videos, {len(keys)} frames", flush=True)
    tmp.rename(h5p)
    (SPL / ds / "manifest.json").write_text(json.dumps(manifest, indent=2))
    files = "\n".join(f"  - eval_adaptation/data/splits/{ds}/{s}.txt" for s, _, _ in items)
    for arm in ARMS:
        tmpl = (CFG / f"{arm}_CDFv2.yaml").read_text()
        out = re.sub(r"^run_name: .*$", f"run_name: {arm}_{ds}", tmpl, flags=re.M)
        out = re.sub(r"^tst_h5_path: .*$", f"tst_h5_path: eval_adaptation/data/h5/{ds}.h5", out, flags=re.M)
        out = re.sub(r"^tst_files:\n(?:  \w+:\n)(?:  - .*\n)+", f"tst_files:\n  {ds}:\n{files}\n", out, flags=re.M)
        (CFG / f"{arm}_{ds}.yaml").write_text(out)
    print(f"done {ds}: {sum(v['frames'] for v in manifest['subsets'].values())} frames -> {h5p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
