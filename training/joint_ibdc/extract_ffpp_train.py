#!/usr/bin/env python3
"""FF++ c23 TRAIN feature cache with independent augmentation draws (Task 2).

    <discern_ext_timm python> training/joint_ibdc/extract_ffpp_train.py --verify
    CUDA_VISIBLE_DEVICES=3 <python> training/joint_ibdc/extract_ffpp_train.py --draws 3 --arm B

Caches the exact per-frame inputs of the Appearance and Off-manifold evidential heads
(`semantic_feature` and the rebuilt `artifact_feature`, both before their frozen LayerNorms), for
each independent augmentation draw, together with the drawn setting per sample so the realisation is
auditable rather than merely asserted.

WHY THIS IS NOT export_views.py --process. Arm B applies to the NATIVE crop BEFORE the CLIP
processor, which is where training applies it (`src/dataset/deepfake.py:__getitem__` runs
`self.augmentations` on the PIL image, then `_preprocess_image`). `--process` applies to the
already-normalised batch, which is the EVALUATION severity path and a different operation. The
processing grid itself is imported from `src/dataset/training_arms.py:SharedProcessing` rather than
copied, so it cannot drift from the one training used.

THE DRAW IS PER SAMPLE AND PER DRAW INDEX, seeded from (seed, draw, crc32(key)), so draws are
independent, reproducible, and independent of batch order or worker count.

READ THIS BEFORE USING THE OUTPUT. The DEPLOYED SPATIAL VIEWS TRAIN WITH NO AUGMENTATION.
`warm_dicome_gate_seed42.yaml` carries no `training_arm` key at all; arm B lives in the separate
`warm_dicome_gate_B_seed42.yaml`, and the `_B_` in `vjepa2_lora_B_seed42` is the VIDEO view's
augmentation. So the brief's "that view's deployed training augmentation" is NONE for Appearance and
Off-manifold, and `--arm none` reproduces deployment while `--arm B` deliberately does not. Both are
provided; whichever is used must be stated in the report.
"""
from __future__ import annotations

import argparse, json, os, sys, time, zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EXT = Path("/data/umar/Repos/DISCERN_Ext")
ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path.insert(0, str(EXT))
os.chdir(EXT)          # the configs name split files relative to DISCERN_Ext

from src.config import Config, load_config                      # noqa: E402
from src.dataset.deepfake import H5DeepfakeDataModule           # noqa: E402
from src.dataset.training_arms import SharedProcessing          # noqa: E402  the grid training used
from src.model.discern_ext_module import DiscernExtModule       # noqa: E402

CKPT = EXT / "runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt"
CONFIG = EXT / "eval_adaptation/configs/warmgate_FFpp_train.yaml"
DATASET = "FFpp_train"
OUT = ROOT / "logs/joint_ibdc/cache"
SEED = 42


def setting_for(key: str, draw: int, arm: str, proc: SharedProcessing, seed: int = SEED):
    """One processing setting per (sample, draw). Reproducible from the key alone."""
    if arm != "B":
        return ("none", None)
    rng = np.random.default_rng([seed, draw, zlib.crc32(str(key).encode())])
    return proc.sample(rng)


class DrawDataset(torch.utils.data.Dataset):
    """The base H5 dataset with arm-B applied to the native crop, before CLIP preprocessing."""

    def __init__(self, base, draw: int, arm: str):
        self.base, self.draw, self.arm = base, draw, arm
        self.proc = SharedProcessing(enabled=(arm == "B"))

    def __len__(self):
        return len(self.base.files)

    def __getitem__(self, i):
        key = self.base.files[i]
        self.base._ensure_h5_open()
        img = self.base._read_image(key)
        st = setting_for(key, self.draw, self.arm, self.proc)
        if st[0] != "none":
            img = Image.fromarray(self.proc.apply(np.asarray(img), st))
        return {"image": self.base._preprocess_image(img),
                "label": int(self.base.labels[i]),
                "key": f"{self.base.h5_path}::{key}",
                "setting": f"{st[0]}:{st[1]}" if st[1] is not None else "none"}


def build(device):
    cfg = load_config(CONFIG)
    cfg = Config(**cfg.model_copy(update={"batch_size": 64, "mini_batch_size": 64}).model_dump())
    dm = H5DeepfakeDataModule(config=cfg, vfm_model_name=cfg.backbone)
    dm.setup("test")
    loader = dm.test_dataloader()
    if isinstance(loader, dict):
        loader = loader[DATASET]
    elif isinstance(loader, (list, tuple)):
        loader = loader[list(cfg.tst_files.keys()).index(DATASET)]
    base = loader.dataset
    module = DiscernExtModule.load_from_checkpoint(str(CKPT), config=cfg, map_location=device)
    return base, module.model.to(device).eval()


@torch.no_grad()
def extract(base, model, draw, arm, device, bs=64, workers=4):
    ds = DrawDataset(base, draw, arm)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                                     collate_fn=torch.utils.data.default_collate)
    sem, art, lab, keys, sets = [], [], [], [], []
    t0 = time.time()
    for bi, b in enumerate(dl):
        out = model(b["image"].to(device))
        fs = out[3]
        fa = model._geometric_view_purification(out[3], out[4])   # the model's own function
        sem.append(fs.float().cpu().numpy()); art.append(fa.float().cpu().numpy())
        lab.append(np.asarray(b["label"])); keys += list(b["key"]); sets += list(b["setting"])
        if bi % 100 == 0:
            n = (bi + 1) * bs
            print(f"    draw {draw} batch {bi} ~{n} frames, {n / max(time.time() - t0, 1e-9):.0f} f/s",
                  flush=True)
    return (np.concatenate(sem), np.concatenate(art), np.concatenate(lab),
            np.array(keys, dtype=object), np.array(sets, dtype=object), time.time() - t0)


def verify(base):
    """Cheap checks that do not need the GPU: draws differ, reproduce, and are inert under 'none'."""
    proc = SharedProcessing(enabled=True)
    keys = [base.files[i] for i in range(0, len(base.files), max(1, len(base.files) // 2000))]
    d0 = [setting_for(k, 0, "B", proc) for k in keys]
    d1 = [setting_for(k, 1, "B", proc) for k in keys]
    d2 = [setting_for(k, 2, "B", proc) for k in keys]
    again = [setting_for(k, 0, "B", proc) for k in keys]
    diff01 = np.mean([a != b for a, b in zip(d0, d1)])
    print(f"  sampled {len(keys)} keys")
    print(f"  draw 0 reproduces exactly: {d0 == again}")
    print(f"  draws 0 and 1 differ on {diff01:.1%} of samples")
    for i, d in enumerate((d0, d1, d2)):
        from collections import Counter
        c = Counter(f"{a}:{b}" for a, b in d)
        tot = sum(c.values())
        print(f"  draw {i} mix: " + ", ".join(f"{k} {v / tot:.2f}" for k, v in sorted(c.items())))
    none_draws = [setting_for(k, j, "none", proc) for k in keys for j in range(3)]
    print(f"  --arm none gives a single realisation: {set(none_draws) == {('none', None)}}")
    # the expected arm-B mix is 40% none and 10% each of six severities
    from collections import Counter
    frac_none = Counter(a for a, _ in d0)["none"] / len(d0)
    assert 0.34 < frac_none < 0.46, f"arm-B 'none' fraction {frac_none:.2f} is not near 0.40"
    assert d0 == again and diff01 > 0.4
    print("verification OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--arm", default="B", choices=["B", "none"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verify", action="store_true", help="check the draws without touching the GPU")
    a = ap.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if a.verify:
        cfg = load_config(CONFIG)
        cfg = Config(**cfg.model_copy(update={"batch_size": 8, "mini_batch_size": 8}).model_dump())
        dm = H5DeepfakeDataModule(config=cfg, vfm_model_name=cfg.backbone)
        dm.setup("test")
        loader = dm.test_dataloader()
        if isinstance(loader, dict):
            loader = loader[DATASET]
        elif isinstance(loader, (list, tuple)):
            loader = loader[list(cfg.tst_files.keys()).index(DATASET)]
        return verify(loader.dataset)

    OUT.mkdir(parents=True, exist_ok=True)
    base, model = build(device)
    print(f"FF++ train: {len(base.files)} frames, arm {a.arm}, {a.draws} draw(s), device {device}",
          flush=True)
    meta = {"dataset": DATASET, "arm": a.arm, "draws": a.draws, "seed": SEED,
            "checkpoint": str(CKPT), "config": str(CONFIG),
            "realizations": ("several cached variants" if a.arm == "B" and a.draws > 1
                             else "one fixed realization"),
            "deployed_spatial_training_augmentation": "none",
            "note": ("arm B is NOT the deployed spatial training augmentation; "
                     "warm_dicome_gate_seed42.yaml has no training_arm key"),
            "per_draw": []}
    for d in range(a.draws):
        f = OUT / f"FFpp_train_{a.arm}_d{d}.npz"
        if f.exists():
            print(f"  skip draw {d} (exists)"); continue
        sem, art, lab, keys, sets, secs = extract(base, model, d, a.arm, device,
                                                  a.batch_size, a.workers)
        tmp = f.with_suffix(".partial.npz")          # np.savez appends .npz unless it already ends in it
        np.savez_compressed(tmp, semantic=sem, artifact=art, label=lab, key=keys, setting=sets)
        tmp.replace(f)
        from collections import Counter
        mix = {k: v / len(sets) for k, v in Counter(sets).items()}
        meta["per_draw"].append({"draw": d, "frames": int(len(lab)), "seconds": round(secs, 1),
                                 "file": str(f), "size_bytes": f.stat().st_size, "mix": mix})
        print(f"  draw {d}: {len(lab)} frames in {secs / 60:.1f} min -> {f.name}", flush=True)
    json.dump(meta, open(ROOT / "analysis/joint_ibdc/ffpp_train_cache.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
