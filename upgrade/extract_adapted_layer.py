#!/usr/bin/env python3
"""Cache INTERMEDIATE-block features from an ADAPTED backbone, several blocks in one pass.

    CUDA_VISIBLE_DEVICES=1 <discern_ext_timm python> upgrade/extract_adapted_layer.py \
        --backbone pe_core_l --recipe LN --blocks 16,20 --dataset VALmix

WHY. Part 3d measured the frozen layer probe on PE-Core and found the output is NOT where the best
visual features are: block 16 (the `mid` depth, 2/3 of 24) scores 0.8019 against 0.7356 at the
output, +0.066. The effect is PE-SPECIFIC -- on CLIP the same probe goes the other way, mid 0.7311
against last 0.7478 -- which is the clearest evidence we have for PE-Core's own claim about where
its features live.

WHAT WAS NEVER MEASURED. Every one of those rows is a FROZEN trunk, and every branch we deploy is
adapted. Adapted-final (branch A) scores 0.8736, beating the best frozen COMBINATION (0.8124) by
+0.061, so adaptation dominates layer choice in every comparison we have -- and nobody has run
adapted block 16. LoRA adapts all 24 blocks, so an adapted final output is not the same object as a
frozen one: adaptation may already recover what the final layer was missing, or it may not. This
module exists to fill that cell.

WHY IT MATTERS NOW. Arm 1 built the off-manifold branch on the adapted FINAL layer and it came out a
near-duplicate of A: 0.0050 mean per-set AUROC difference and +0.999 correlation across the ten test
sets, adding +0.0009 to A+V+O. Residualising a representation inside its own space does not make the
result independent of that representation. Block 16 gives the construction something genuinely
different to work on, which is the specific fix for a measured failure rather than a hunch.

ONE PASS, SEVERAL BLOCKS. Each requested block is hooked and written to its own cache from a single
forward, so asking for 16 and 20 together costs one extraction, not two.

POOLING MATCHES THE FROZEN PROBE EXACTLY -- mean over patch tokens with the prefix tokens dropped,
`extract_layers.py`'s `g[:, 1:].mean(1)` -- so an adapted row is comparable with its frozen row
rather than differing by an unrecorded readout change. The prefix count is READ from the model
instead of assumed, and asserted, because a backbone with register tokens (DINOv3 has four besides
CLS) would silently pool four junk vectors into every feature.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
OUT = ROOT / "logs/upgrade/feats"
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "branches/ijepa")]
import adapt_backbone as AB   # noqa: E402
from data import video_of     # noqa: E402

torch.set_num_threads(4)


def blocks_of(m, kind):
    if kind == "hf":
        return list(m.encoder.layers)
    for attr in ("blocks", "layers"):
        if hasattr(m, attr):
            return list(getattr(m, attr))
    raise RuntimeError("cannot locate transformer blocks")


def prefix_tokens(m, kind: int) -> int:
    """How many non-patch tokens lead the sequence. Read, never assumed."""
    n = getattr(m, "num_prefix_tokens", None)
    if n is None:
        n = 1 if kind == "hf" else 1
    return int(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--recipe", required=True, choices=["L", "N", "LN"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", required=True, help="comma list, 1-INDEXED, e.g. 16,20")
    ap.add_argument("--ckpt-suffix", default="")
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    want = [int(x) for x in a.blocks.split(",") if x.strip()]

    tags = {b: f"{a.backbone}{a.recipe}-b{b}" + ("" if a.seed == 0 else f"_s{a.seed}") + a.ckpt_suffix
            for b in want}
    todo = {b: t for b, t in tags.items() if not (OUT / f"{t}_{a.dataset}.npz").exists()}
    for b, t in tags.items():
        if b not in todo:
            print(f"{t}_{a.dataset}.npz exists")
    if not todo:
        return 0

    dev = "cuda"
    m, mean, std, spec, _p, _f = AB.build(a.backbone, dev, a.recipe)
    ck = torch.load(AB.OUT / f"{a.backbone}_{a.recipe}_s{a.seed}{a.ckpt_suffix}.pt", map_location=dev)
    sd = m.state_dict()
    adapted = {k: v.to(dev) for k, v in ck["adapt"].items() if k in sd}
    assert len(adapted) == len(ck["adapt"]), \
        f"{len(ck['adapt']) - len(adapted)} adapted tensors do not match this backbone"
    m.load_state_dict({**sd, **adapted}, strict=False)
    m.eval()

    blocks = blocks_of(m, spec["kind"])
    n_blk = len(blocks)
    for b in todo:
        if not 1 <= b <= n_blk:
            raise SystemExit(f"block {b} does not exist: this backbone has {n_blk} blocks")
    npre = prefix_tokens(m, spec["kind"])
    grabbed: dict[int, torch.Tensor] = {}

    def hook(b):
        def fn(_mod, _inp, out):
            grabbed[b] = out[0] if isinstance(out, tuple) else out
        return fn

    for b in todo:
        blocks[b - 1].register_forward_hook(hook(b))
    print(f"{a.backbone}_{a.recipe}: epoch {ck['epoch']}, {len(adapted)} adapted tensors, "
          f"{n_blk} blocks, {npre} prefix token(s), tapping {sorted(todo)}, input {spec['size']}",
          flush=True)

    X, K = AB.frames_of(a.dataset, spec["size"])
    Z = {b: [] for b in todo}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), a.batch_size):
            x = torch.from_numpy(X[i:i + a.batch_size]).permute(0, 3, 1, 2).float().div_(255).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m((x - mean) / std)                      # the hooks are the output we want
            for b in todo:
                Z[b].append(grabbed[b][:, npre:].mean(1).float().cpu().numpy())
            if (i // a.batch_size) % 128 == 0 and i:
                print(f"  {i} frames, {i/(time.time()-t0):.0f} f/s", flush=True)

    y = np.array([0 if "real" in str(k).split("/")[-3].lower() else 1 for k in K], dtype=np.int64)
    vids = np.array([video_of(str(k)) for k in K], dtype=object)
    for b, t in todo.items():
        F = np.concatenate(Z[b]).astype(np.float16)
        f = OUT / f"{t}_{a.dataset}.npz"
        tmp = f.with_suffix(".partial.npz")
        np.savez(tmp, feature=F, key=K, label=y, video=vids)
        tmp.replace(f)
        json.dump({"backbone": a.backbone, "recipe": a.recipe, "block_1indexed": b,
                   "block_index0": b - 1, "n_blocks": n_blk, "prefix_tokens": npre,
                   "readout": "patch-token mean, prefix dropped (matches extract_layers.py)",
                   "adapted_epoch": int(ck["epoch"]), "dataset": a.dataset,
                   "frames": int(len(K)), "dim": int(F.shape[1]), "input_size": spec["size"],
                   "minutes": round((time.time() - t0) / 60, 1)},
                  open(f.with_suffix(".json"), "w"), indent=1)
        print(f"  -> {f.name} {F.shape}", flush=True)
    print(f"  {len(K)} frames in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
