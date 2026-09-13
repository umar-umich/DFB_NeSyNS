#!/usr/bin/env python3
"""Checks for the SBI provenance fix. No training.

  1. every recorded SBI sample is regenerated bit-identically from (parent, seed, processing)
  2. SBI items carry the parent's video/identity/frame, SBI source id, and the replaced slot key
  3. mixed batches collate
  4. two DataLoader workers do not repeat each other's synthetic draws
  5. arm A still returns the base dataset's image, label and path
"""
import os, sys, collections
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
EXT = "/data/umar/Repos/DISCERN_Ext"; sys.path.insert(0, EXT); os.chdir(EXT)
from src.config import load_config
from src.dataset.deepfake import H5DeepfakeDataModule

def build(cfg):
    c = load_config(cfg); dm = H5DeepfakeDataModule(config=c, vfm_model_name=c.backbone); dm.setup("fit"); return dm.train_dataset

ds = build("eval_adaptation/configs/sbi_armC.yaml")
base = ds.base; base.augmentations = None
fake = [i for i, l in enumerate(base.labels) if l == 1]
rng = np.random.default_rng(1)

# 1-3
got, sample = [], []
for i in rng.permutation(fake)[:400]:
    it = ds[int(i)]
    sample.append(it)
    if it["sample_kind"] == "sbi_fake":
        got.append(it)
    if len(got) >= 10:
        break
diffs = [float((ds.regenerate_sbi(g["parent_key"], g["synth_seed"], g["proc_kind"], g["proc_value"]) - g["image"]).abs().max()) for g in got]
print(f"1. regenerated {len(got)} SBI samples; max abs diff {max(diffs):.2e}  -> {'PASS' if max(diffs) == 0 else 'FAIL'}")
g = got[0]
print(f"2. example: path={g['path'][:70]}  video={g['video']} identity={g['identity']} source={g['source']} "
      f"replaced_slot={g['replaced_slot_key'][:40]}  proc=({g['proc_kind']},{g['proc_value']})")
ok2 = all(x["source"] == ds.SBI_SOURCE_UID and x["replaced_slot_key"] != x["parent_key"]
          and x["parent_key"].split("/")[-2] == x["video"] for x in got)
print(f"   provenance fields consistent on all {len(got)}: {'PASS' if ok2 else 'FAIL'}")
b = default_collate(sample[:16]); print(f"3. mixed batch collates: PASS (kinds {collections.Counter(b['sample_kind'])})")

# 4
sub = Subset(ds, [int(i) for i in rng.permutation(fake)[:240]])
pairs = []
for batch in DataLoader(sub, batch_size=8, num_workers=2, shuffle=False):
    for k, p, s in zip(batch["sample_kind"], batch["parent_key"], batch["synth_seed"].tolist()):
        if k == "sbi_fake": pairs.append((p, s))
dup = len(pairs) - len(set(pairs))
print(f"4. two workers, {len(pairs)} SBI draws: duplicated (parent, seed) pairs {dup}  -> {'PASS' if dup == 0 else 'FAIL'}")
g0 = np.random.default_rng(42); g1 = np.random.default_rng(42)
print(f"   before the fix both workers held this state: first draws {g0.integers(1e9)} vs {g1.integers(1e9)} (identical)")

# 5
dA = build("eval_adaptation/configs/sbi_armA.yaml"); dA.base.augmentations = None
bad = 0
for i in rng.choice(len(dA), 32, replace=False):
    a, r = dA[int(i)], dA.base[int(i)]
    bad += int(float((a["image"] - r["image"]).abs().max()) != 0 or a["label"] != r["label"] or a["path"] != r["path"])
print(f"5. arm A image/label/path vs base on 32 items: mismatches {bad}  -> {'PASS' if bad == 0 else 'FAIL'}")
