#!/usr/bin/env python3
"""Section 2.3: R's patch-covariance features, and the block-16 patch mean S-frozen needs.

    CUDA_VISIBLE_DEVICES=1 <discern_ext_timm python> upgrade/branchfusion/extract_patch.py \
        --dataset CDFv2 --depth mid

One forward pass emits BOTH outputs, so R's covariance feature and its own mean-pooled control -- and
S-frozen's input -- cost one extraction rather than three. `upgrade/extract_layers.py` already proved
the multi-depth hook pattern; this adds the second-order summary beside the first-order one.

THE SEVEN STEPS SECTION 2.3 FIXES, implemented in order:

  1. read block 16 of the verified 24-block PE-L encoder     -> `--depth mid` resolves to index 15
  2. strip every prefix token, count from model metadata     -> `num_prefix_tokens`, NOT a hardcoded 1
  3. fixed seeded orthonormal 32-dimensional projection      -> QR of a seeded Gaussian, saved
  4. per frame, centre the patches, C_t = Z_t' Z_t / (N_t-1)
  5. vectorise the symmetric covariance, off-diagonals x sqrt(2)  -> 32*33/2 = 528 values
  6. average over frames, standardise with TRAINING statistics    -> done in `train_patch_head.py`
  7. keep a log-norm feature so magnitude is not discarded        -> appended, giving 529

WHY THE PREFIX COUNT IS READ RATHER THAN ASSUMED. PE-Core-L has one prefix token, so `z[:, 1:]` is
correct for it. DINOv3 ViT-L has FIVE (one class token plus four register tokens), so the same slice
would fold register tokens into the patch statistics. Section 12 makes the correct strip an acceptance
check, and `--backbone` is free to change, so the count comes from `num_prefix_tokens`.

The covariance is computed on the PROJECTED channels, not the raw 1024, because a 1024x1024
covariance is 524,800 numbers per frame and the projection is what makes this a head-sized feature.
The projection is orthonormal and seeded, so it is a rotation-and-truncation fixed before any data is
seen -- not a learned compression that could absorb label information.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                      # noqa: E402

sys.path[:0] = [str(C.ROOT / "upgrade"), str(C.ROOT / "branches/ijepa")]
from extract_spatial import SPECS                       # noqa: E402
from extract_layers import blocks_of, degrade, DEPTHS    # noqa: E402

C.thread_caps()

PROJ_DIM = 32
PROJ_SEED = 20261008            # fixed before any data is seen, recorded in every output
TRI = PROJ_DIM * (PROJ_DIM + 1) // 2          # 528


def projection(d_in: int, d_out: int = PROJ_DIM, seed: int = PROJ_SEED) -> np.ndarray:
    """A fixed seeded orthonormal (d_in, d_out) projection.

    QR of a seeded Gaussian gives orthonormal columns, so this is a rotation into a 32-dimensional
    subspace chosen without reference to the data or the labels. Returned as float32 and saved with
    the features so a later run cannot silently use a different one.
    """
    g = np.random.default_rng(seed)
    q, _r = np.linalg.qr(g.standard_normal((d_in, d_out)))
    return np.ascontiguousarray(q[:, :d_out], dtype=np.float32)


def tri_indices(n: int = PROJ_DIM):
    return np.triu_indices(n)


def vectorise_cov(Cmat: torch.Tensor) -> torch.Tensor:
    """Upper triangle of a symmetric covariance with off-diagonals scaled by sqrt(2).

    The scaling makes the vectorisation an isometry: the Euclidean norm of the 528-vector equals the
    Frobenius norm of the matrix, so a distance in feature space is a distance between covariances
    rather than one that counts the diagonal differently from the rest.
    """
    n = Cmat.shape[-1]
    iu, ju = torch.triu_indices(n, n, device=Cmat.device)
    vals = Cmat[..., iu, ju]
    scale = torch.where(iu == ju, torch.ones_like(vals[0]), torch.full_like(vals[0], 2.0 ** 0.5))
    return vals * scale


@torch.no_grad()
def frame_features(tokens: torch.Tensor, P: torch.Tensor) -> tuple:
    """(batch, N_tokens, d) patch tokens -> (covariance vector + log norm, mean pool).

    `tokens` must already have its prefix stripped. Returns float32 on CPU.
    """
    Z = tokens.float() @ P                                 # (B, N, 32)
    N = Z.shape[1]
    Zc = Z - Z.mean(1, keepdim=True)                       # centre the patches, step 4
    Cm = Zc.transpose(1, 2) @ Zc / max(N - 1, 1)           # (B, 32, 32)
    cov = vectorise_cov(Cm)                                # (B, 528), step 5
    # step 7: magnitude is discarded by centring, so carry it explicitly
    lognorm = torch.log1p(Z.pow(2).sum(-1).mean(1, keepdim=True).clamp_min(1e-12))
    return (torch.cat([cov, lognorm], 1).cpu().numpy().astype(np.float32),
            tokens.float().mean(1).cpu().numpy().astype(np.float32))


def register(backbone: str = "pe_core_l", depth: str = "mid") -> dict:
    """Symlink the patch caches into `logs/upgrade/feats` under the names a head trainer expects.

    `upgrade/train_head.py` reads `{backbone}_{dataset}.npz` from that one directory, so registering
    the caches there means R and its control train through the SAME head, loss, schedule and
    selection rule as every other branch -- rather than through a second trainer that could drift.
    """
    out = {}
    for src_prefix, name in (("R", "Rcov"), ("MEANPOOL", "Rmean")):
        for f in sorted(C.PATCH.glob(f"{src_prefix}_{backbone}-{depth}_*.npz")):
            ds = f.name[len(f"{src_prefix}_{backbone}-{depth}_"):-len(".npz")]
            dst = C.FEATS / f"{name}_{ds}.npz"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(f)
            out[dst.name] = str(f.relative_to(C.ROOT))
    return out


def gate(n_boot: int = 2000) -> dict:
    """Section 2.3's keep rule: R enters the subset search only if it beats its mean-pooled control.

    The control is the mean-pooled feature from the SAME block, so the comparison separates the
    second-order covariance summary from the depth it was read at. The interval is a paired
    source-group bootstrap and must lie above zero; otherwise R is reported as a side result and
    never enters a subset.
    """
    import branches as B
    rows = {}
    for b in ("R", "Rmean"):
        try:
            t = B.dev_table([b], require_all=True)
        except Exception as e:
            return {"status": "unavailable", "branch": b, "error": f"{type(e).__name__}: {e}"}
        rows[b] = t
    shared = set(rows["R"].v) & set(rows["Rmean"].v)
    a = rows["R"][rows["R"].v.isin(shared)].sort_values("v").reset_index(drop=True)
    c = rows["Rmean"][rows["Rmean"].v.isin(shared)].sort_values("v").reset_index(drop=True)
    y = a.y.to_numpy()
    g = C.groups_of(a)
    d = C.paired_delta(y, a.p_R.to_numpy(), c.p_Rmean.to_numpy(), g, n_rep=n_boot)
    macro_r = C.v31_macro(a, a.p_R.to_numpy())[0]
    macro_c = C.v31_macro(c, c.p_Rmean.to_numpy())[0]
    passed = bool(d["lo2.5"] > 0)
    return {"status": "ok", "n_videos": int(len(a)),
            "R_auroc": C.auroc(y, a.p_R.to_numpy()),
            "control_auroc": C.auroc(y, c.p_Rmean.to_numpy()),
            "R_v31_macro": macro_r, "control_v31_macro": macro_c,
            "paired_delta": d, "passed": passed,
            "rule": ("R enters the subset search only if it beats its mean-pooled control on "
                     "development with a source-grouped interval above zero (section 2.3)"),
            "consequence": ("R enters the subset search" if passed else
                            "R is reported as a SIDE RESULT and does not enter any subset"),
            "naming": "R is a patch-STATISTICS branch, not a localization method"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true",
                    help="symlink the patch caches into logs/upgrade/feats as Rcov_* and Rmean_*")
    ap.add_argument("--gate", action="store_true", help="run section 2.3's keep rule")
    ap.add_argument("--backbone", default="pe_core_l", choices=list(SPECS))
    ap.add_argument("--dataset", default="")
    ap.add_argument("--depth", default="mid", choices=sorted(DEPTHS),
                    help="mid = two-thirds depth = block 16 of 24, as section 2.3 specifies")
    ap.add_argument("--process", default=None, help="e.g. jpeg:40 or resize:0.60")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.register:
        r = register(a.backbone, a.depth)
        print(f"registered {len(r)} caches into {C.FEATS}")
        for k in sorted(r):
            print(f"  {k} -> {r[k]}")
        json.dump(r, open(C.OUT / "r_register.json", "w"), indent=1)
        return 0
    if a.gate:
        g = gate()
        print(json.dumps(g, indent=1, default=str))
        json.dump(g, open(C.OUT / "r_gate.json", "w"), indent=1, default=str)
        return 0 if g.get("passed") else 3

    tag = f"{a.backbone}-{a.depth}_{a.dataset}" + (f"_{a.process.replace(':', '')}" if a.process else "")
    cov_f = C.PATCH / f"R_{tag}.npz"
    mean_f = C.PATCH / f"MEANPOOL_{tag}.npz"
    if cov_f.exists() and mean_f.exists() and not a.force:
        print(f"{cov_f.name} and {mean_f.name} exist")
        return 0
    C.PATCH.mkdir(parents=True, exist_ok=True)

    spec = SPECS[a.backbone]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import timm
    if spec["kind"] == "hf":
        from transformers import CLIPVisionModel
        m = CLIPVisionModel.from_pretrained(spec["id"]).vision_model.to(dev).eval()
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(dev)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(dev)
        n_prefix = 1                                        # CLIP: the CLS token
    else:
        m = timm.create_model(spec["id"], pretrained=True, num_classes=0).to(dev).eval()
        cfg = m.pretrained_cfg
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(dev)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(dev)
        # step 2: the count comes from metadata. PE-Core-L = 1, DINOv3 ViT-L = 5.
        n_prefix = int(getattr(m, "num_prefix_tokens", 1))
    for p in m.parameters():
        p.requires_grad_(False)

    blocks = blocks_of(m, spec["kind"])
    n_blk = len(blocks)
    depth_idx = max(0, min(n_blk - 1, int(round(n_blk * DEPTHS[a.depth])) - 1))
    grabbed: dict = {}

    def hook(_mod, _inp, out):
        grabbed["z"] = (out[0] if isinstance(out, tuple) else out).detach()

    blocks[depth_idx].register_forward_hook(hook)

    d_model = int(getattr(m, "embed_dim", None) or blocks[0].norm1.weight.shape[0])
    Pnp = projection(d_model)
    P = torch.from_numpy(Pnp).to(dev)
    print(f"{a.backbone}: {n_blk} blocks, depth '{a.depth}' -> block index {depth_idx} "
          f"(1-based block {depth_idx + 1}), width {d_model}, prefix tokens {n_prefix}, "
          f"input {spec['size']}" + (f", process {a.process}" if a.process else ""), flush=True)

    if not a.dataset:
        raise SystemExit("--dataset is required unless --register or --gate is given")
    man = sorted((C.SPLITS / a.dataset).glob("*.txt"))
    if not man:
        raise FileNotFoundError(f"no split manifests under {C.SPLITS / a.dataset}")
    lines = [l.strip() for mm in man for l in mm.read_text().splitlines() if l.strip()]

    covs, means, keys, t0 = [], [], [], time.time()
    with h5py.File(C.H5 / f"{a.dataset}.h5", "r") as h5:
        buf, bk = [], []

        def flush():
            if not buf:
                return
            x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().div_(255).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                m((x - mean) / std)
            z = grabbed["z"][:, n_prefix:]                   # step 2
            c, mp = frame_features(z, P)
            covs.append(c)
            means.append(mp)
            keys.extend(bk)
            buf.clear()
            bk.clear()

        for k in lines:
            if k not in h5:
                continue
            arr = h5[k][()]
            im = (Image.open(io.BytesIO(arr.tobytes())) if arr.dtype == np.uint8 and arr.ndim == 1
                  else Image.fromarray(np.asarray(arr).astype(np.uint8))).convert("RGB")
            im = degrade(im, a.process)
            if im.size != (spec["size"],) * 2:
                im = im.resize((spec["size"],) * 2, Image.BICUBIC)
            buf.append(np.asarray(im, dtype=np.uint8))
            bk.append(k)
            if len(buf) == a.batch_size:
                flush()
                if len(keys) % 8192 < a.batch_size:
                    print(f"  {len(keys)} frames, {len(keys) / (time.time() - t0):.0f} f/s",
                          flush=True)
        flush()

    if not keys:
        raise RuntimeError(f"{a.dataset}: no frames read from {C.H5 / f'{a.dataset}.h5'}")
    cov = np.concatenate(covs)
    mp = np.concatenate(means)
    from data import video_of
    vids = np.array([video_of(k) for k in keys], dtype=object)
    # The ESTABLISHED label rule, copied from `upgrade/extract_spatial.py:141` rather than
    # re-derived from manifest filenames: the class directory is the third path component from the
    # end, and a real one has "real" in its name. Every other cache in this repository was built
    # with exactly this expression, so R's labels cannot disagree with A's.
    ys = np.array([0 if "real" in str(k).split("/")[-3].lower() else 1 for k in keys],
                  dtype=np.int64)
    meta = {"backbone": a.backbone, "timm_id": spec["id"], "dataset": a.dataset,
            "depth": a.depth, "block_index_0based": depth_idx, "n_blocks": n_blk,
            "width": d_model, "prefix_tokens": n_prefix, "input_size": spec["size"],
            "crop_resolution": 256, "process": a.process, "frames": len(keys),
            "proj_dim": PROJ_DIM, "proj_seed": PROJ_SEED, "cov_dim": TRI,
            "feature_dim": TRI + 1, "offdiag_scale": "sqrt(2)",
            "readout": f"block {depth_idx + 1} patch tokens, prefix stripped",
            "adapted": False,
            "minutes": round((time.time() - t0) / 60, 1)}
    meta["config_hash"] = C.config_hash(meta)

    np.savez_compressed(cov_f, feature=cov.astype(np.float16), key=np.array(keys, dtype=object),
                        video=vids, label=ys, projection=Pnp)
    json.dump(meta, open(cov_f.with_suffix(".json"), "w"), indent=1)
    mmeta = dict(meta); mmeta.update({"feature_dim": d_model, "readout": meta["readout"] + " mean",
                                      "proj_dim": None, "cov_dim": None})
    np.savez_compressed(mean_f, feature=mp.astype(np.float16), key=np.array(keys, dtype=object),
                        video=vids, label=ys)
    json.dump(mmeta, open(mean_f.with_suffix(".json"), "w"), indent=1)
    print(f"{len(keys)} frames in {meta['minutes']} min\n-> {cov_f}\n-> {mean_f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
