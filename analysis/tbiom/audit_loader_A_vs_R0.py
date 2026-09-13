#!/usr/bin/env python3
"""Loader audit — does the arm-A wrapper change what the model sees?

    python analysis/tbiom/audit_loader_A_vs_R0.py --out tbiom/reports/AUDIT_LOADER.md

A differs from R0 by -0.0101 mean OOD AUROC. "Loader only" says where the code changed, not what
changed scientifically. This separates the possibilities without retraining:

  1. identical sample IDs through both loaders, augmentation OFF: key, label, identity, source,
     idx and the decoded, normalised image tensor must match exactly
  2. same again with augmentation ON and the global RNG reseeded identically before each item:
     the wrapper must not consume or perturb the random stream the augmentations draw from
  3. one frozen checkpoint (R0, epoch 5) run on both loaders' tensors: fused and per-branch
     evidence must match
"""
from __future__ import annotations
import argparse, os, sys, random
from pathlib import Path
import numpy as np
import torch

EXT = "/data/umar/Repos/DISCERN_Ext"
sys.path.insert(0, EXT)
OUT_DEFAULT = Path("/data/umar/Repos/DFB_NeSyNS/tbiom/reports/AUDIT_LOADER.md")


def build(cfg_path):
    from src.config import load_config
    from src.dataset.deepfake import H5DeepfakeDataModule
    cfg = load_config(cfg_path)
    dm = H5DeepfakeDataModule(config=cfg, vfm_model_name=cfg.backbone)
    dm.setup("fit")
    return cfg, dm


def base_of(ds):
    return ds.base if hasattr(ds, "arm") else ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    os.chdir(EXT)

    cfg_r, dm_r = build("eval_adaptation/configs/run1_auxedl.yaml")
    cfg_a, dm_a = build("eval_adaptation/configs/sbi_armA.yaml")
    dr, da = dm_r.train_dataset, dm_a.train_dataset
    L = ["# Loader audit — arm A wrapper vs R0 path", "",
         f"R0 train dataset: `{type(dr).__name__}`; arm A train dataset: `{type(da).__name__}` "
         f"wrapping `{type(base_of(da)).__name__}`.", "",
         f"- lengths: R0 {len(dr):,} / A {len(da):,}",
         f"- files lists identical: {list(base_of(dr).files) == list(base_of(da).files)}",
         f"- labels identical: {list(base_of(dr).labels) == list(base_of(da).labels)}", ""]

    idx = np.random.default_rng(0).choice(len(dr), args.n, replace=False)

    # 1 -- augmentation off
    saved = (base_of(dr).augmentations, base_of(da).augmentations)
    base_of(dr).augmentations = None; base_of(da).augmentations = None
    mism = {"key": 0, "label": 0, "image": 0, "other": 0}; maxdiff = 0.0
    for i in idx:
        a, b = dr[int(i)], da[int(i)]
        if a.get("path") != b.get("path"): mism["key"] += 1
        if int(a["label"]) != int(b["label"]): mism["label"] += 1
        d = float((a["image"] - b["image"]).abs().max()); maxdiff = max(maxdiff, d)
        if d > 0: mism["image"] += 1
        for k in set(a) | set(b):
            if k in ("image", "label", "path"): continue
            va, vb = a.get(k), b.get(k)
            try:
                if (torch.as_tensor(va) != torch.as_tensor(vb)).any(): mism["other"] += 1
            except Exception:
                if va != vb: mism["other"] += 1
    L += ["## 1 — identical IDs, augmentation off", "",
          f"{args.n} sampled indices. Mismatches — key {mism['key']}, label {mism['label']}, "
          f"image {mism['image']} (max abs diff {maxdiff:.2e}), other fields {mism['other']}.", ""]
    ok1 = sum(mism.values()) == 0

    # 2 -- augmentation on, RNG reseeded identically
    base_of(dr).augmentations, base_of(da).augmentations = saved
    m2 = 0; md2 = 0.0
    for i in idx[:64]:
        torch.manual_seed(int(i)); random.seed(int(i)); np.random.seed(int(i) % 2**32)
        a = dr[int(i)]
        torch.manual_seed(int(i)); random.seed(int(i)); np.random.seed(int(i) % 2**32)
        b = da[int(i)]
        d = float((a["image"] - b["image"]).abs().max()); md2 = max(md2, d)
        if d > 0: m2 += 1
    L += ["## 2 — augmentation on, random state reseeded identically", "",
          f"64 indices: image mismatches {m2} (max abs diff {md2:.2e}). A mismatch here would mean "
          f"the wrapper consumes or perturbs the random stream the augmentations use.", ""]
    ok2 = m2 == 0

    # 3 -- same frozen checkpoint on both loaders' tensors
    base_of(dr).augmentations = None; base_of(da).augmentations = None
    from src.model.discern_ext_module import DiscernExtModule
    ck = ("runs/discern_ext_train/run1_auxedl_seed42/checkpoints/"
          "discern_ext-best-epoch=05-val_auroc_video=0.9955.ckpt")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = DiscernExtModule.load_from_checkpoint(ck, config=cfg_r, map_location=dev).eval()
    sel = [int(i) for i in idx[:32]]
    xr = torch.stack([dr[i]["image"] for i in sel]).to(dev)
    xa = torch.stack([da[i]["image"] for i in sel]).to(dev)
    with torch.no_grad():
        orr, oa = model(xr), model(xa)
    diffs = {"fused": float((orr[0] - oa[0]).abs().max()),
             "semantic": float((orr[1] - oa[1]).abs().max()),
             "artifact": float((orr[2] - oa[2]).abs().max())}
    auxr, auxa = (orr[8] if len(orr) > 8 else {}) or {}, (oa[8] if len(oa) > 8 else {}) or {}
    if "fsvfm" in auxr and "fsvfm" in auxa:
        diffs["fsvfm"] = float((auxr["fsvfm"]["evidence"] - auxa["fsvfm"]["evidence"]).abs().max())
    L += ["## 3 — one frozen checkpoint (R0 epoch 5) on both loaders' inputs", "",
          "| output | max abs difference |", "|---|---:|"] + \
         [f"| {k} evidence | {v:.2e} |" for k, v in diffs.items()] + [""]
    ok3 = all(v == 0 for v in diffs.values())

    L += ["## Verdict", "",
          ("**The wrapper is inert for arm A.** Same samples, same labels, same metadata, "
           "bit-identical tensors with and without augmentation, bit-identical evidence from a "
           "fixed checkpoint. The A − R0 gap is not an inference or preprocessing difference."
           if ok1 and ok2 and ok3 else
           "**The loaders differ.** See the non-zero rows above; that difference is a candidate "
           "cause of A − R0 and must be understood before any arm comparison is read.")]
    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
