#!/usr/bin/env python3
"""Build VALmixV2.h5 per tbiom/docs/evaluation-protocol-v3.md (sections 2, 5, 6)."""
import glob, json, os, re, sys
from pathlib import Path
import numpy as np, h5py
from PIL import Image

EXT = Path("/data/umar/Repos/DISCERN_Ext")
H5, SPL, CFG = EXT / "eval_adaptation/data/h5/VALmixV2.h5", EXT / "eval_adaptation/data/splits/VALmixV2", EXT / "eval_adaptation/configs"
GV = Path("/data/kutub/datasets/GEND/extracted/val")
DF40 = Path("/data/saad/datasets/video/df40/train")
ARMS = ["run1auxedl", "sbiarmA", "sbiarmB", "sbiarmC"]
FAKE_CAP, FRAMES = 64, 16
REEN = ["danet", "facevid2vid", "fomm", "hyperreenact", "lia", "mcnet", "MRAA", "one_shot_free", "pirender", "tpsm"]
TALK = ["sadtalker", "wav2lip"]
EFS = ["ddim", "DiT", "pixart", "RDDM", "sd2.1", "SiT", "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN"]


def dirs(p): return sorted(x for x in os.listdir(p) if (Path(p) / x).is_dir())


def frames_of(d):
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if len(fs) <= FRAMES: return fs
    return [fs[i] for i in np.linspace(0, len(fs) - 1, FRAMES).round().astype(int)]


def sample(vs, cap, tag):
    if len(vs) <= cap: return vs
    return sorted(np.random.default_rng(0).choice(vs, size=cap, replace=False).tolist())


def test_names():
    names = set()
    for ds in ("CDFv2", "CDFv3"):
        for f in glob.glob(str(EXT / f"eval_adaptation/data/splits/{ds}/*.txt")):
            names |= {k.rsplit("/", 1)[0].split("/")[-1] for k in open(f).read().split()}
    return names


def plan():
    leak = test_names(); items = []; removed = []
    c = GV / "CDFv3-x1.3-th0.5-all"
    for sub in dirs(c):
        vs = dirs(c / sub)
        bad = [v for v in vs if v in leak]; removed += [f"{sub}/{v}" for v in bad]
        vs = [v for v in vs if v not in leak]
        cap = 128 if "real" in sub.lower() else FAKE_CAP
        name = "R3DPortrait" if sub == "Real3DPortrait" else sub
        items.append(("CDFv3valG", name, c / sub, sample(vs, cap, sub)))
    for src, d in (("DS11val", GV / "DeepSpeak-1.1"), ("DS20val", GV / "DeepSpeak-2.0")):
        for sub in dirs(d):
            name = sub.replace("train-", "")
            cap = 256 if name == "real" else FAKE_CAP
            items.append((src, name, d / sub, sample(dirs(d / sub), cap, sub)))
    for sub in dirs(GV / "FFIW"):
        items.append(("FFIWval", sub.replace("val-", ""), GV / "FFIW" / sub, dirs(GV / "FFIW" / sub)))
    for m in REEN + TALK:
        items.append(("DF40train", m, DF40 / m / "frames", sample(dirs(DF40 / m / "frames"), FAKE_CAP, m)))
    for m in EFS:
        items.append(("DF40train", m, DF40 / m, sample(dirs(DF40 / m), FAKE_CAP, m)))
    return items, removed


def main():
    if H5.exists():
        print("VALmixV2.h5 exists; refusing to overwrite"); return 1
    items, removed = plan()
    for _, sub, _, _ in items:
        assert "real" not in sub.lower() or sub in ("Celeb-real", "YouTube-real", "real"), sub
    SPL.mkdir(parents=True, exist_ok=True)
    man = {"protocol": "v3", "removed_for_leakage": removed, "subsets": {}}
    tmp = H5.with_suffix(".h5.partial"); splits = []
    with h5py.File(tmp, "w") as h:
        for src, sub, d, vs in items:
            keys = []
            for v in vs:
                for fr in frames_of(d / v):
                    key = f"VALmixV2/{src}/{sub}/{v}/{fr}"
                    h.create_dataset(key, data=np.asarray(Image.open(d / v / fr).convert("RGB")), compression="gzip")
                    keys.append(key)
            fn = SPL / f"{src}__{sub}.txt"; fn.write_text("\n".join(keys) + "\n"); splits.append(fn.name)
            man["subsets"][f"{src}/{sub}"] = {"label": 0 if "real" in sub.lower() else 1, "videos": len(vs),
                                              "frames": len(keys), "source_dir": str(d)}
            print(f"  {src}/{sub}: {len(vs)} videos, {len(keys)} frames", flush=True)
    tmp.rename(H5)
    (SPL / "manifest.json").write_text(json.dumps(man, indent=2))
    files = "\n".join(f"  - eval_adaptation/data/splits/VALmixV2/{s}" for s in splits)
    for arm in ARMS:
        t = (CFG / f"{arm}_CDFv2.yaml").read_text()
        t = re.sub(r"^run_name: .*$", f"run_name: {arm}_VALmixV2", t, flags=re.M)
        t = re.sub(r"^tst_h5_path: .*$", "tst_h5_path: eval_adaptation/data/h5/VALmixV2.h5", t, flags=re.M)
        t = re.sub(r"^tst_files:\n(?:  \w+:\n)(?:  - .*\n)+", f"tst_files:\n  VALmixV2:\n{files}\n", t, flags=re.M)
        (CFG / f"{arm}_VALmixV2.yaml").write_text(t)
    print(f"done: {sum(s['videos'] for s in man['subsets'].values())} videos, removed for leakage {len(removed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
