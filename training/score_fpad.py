#!/usr/bin/env python3
"""Score a frozen Stage-A student and dump its depth-resolved adaptation profile.

    🔴 UMAR-RUNS (GPU):

    # FF++ val — the in-domain side of the Stage-2 audit
    python training/score_fpad.py --student logs/fpad/studentA_ordinary_seed42 \
        --datasets FaceForensics++ --split val \
        --output logs/fpad/score/ordinary_ffppval --device cuda:1

    # DF40-Dev — the OOD side. `--df40` treats --datasets as per-method names.
    python training/score_fpad.py --student logs/fpad/studentA_ordinary_seed42 \
        --datasets danet_cdf mcnet_cdf tpsm_cdf facevid2vid_cdf --df40 \
        --output logs/fpad/score/ordinary_df40dev --device cuda:1

Writes one parquet row per frame with the full profile `D(x)` (one column per configured layer),
the CLS diagnostic, patch-map summaries and the direct head's probability. Stage 2 reads it; no
stage recomputes a forward pass it could have read.

Provenance of the OOD real half is recorded per frame
-----------------------------------------------------
DF40 borrows its authentic halves from other corpora: a `*_cdf` method's reals are Celeb-DF-v2
frames, a `*_ff` method's reals are FF++ frames. The Stage-2 gate needs that distinction, because
FF++-sourced reals overlap FF++-only training and would understate domain separability. So
`real_source` is derived from each frame's resolved path and written out, rather than inferred
later from the method name — the same method can supply both halves and the manifest is the only
thing that knows which frame came from where.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

from networks.fpad import DirectEvidenceHead, FPADTeacherStudent  # noqa: E402
from train_fpad import FpadViews  # noqa: E402
from train_v1 import load_split, prepare_dataset_config, video_of  # noqa: E402


def real_source_of(path: str) -> str:
    """Which corpus this frame's pixels actually came from, read off the resolved path."""
    p = str(path)
    for marker, name in (("/Celeb-DF-v2/", "Celeb-DF-v2"), ("/Celeb-DF-v3/", "Celeb-DF-v3"),
                         ("/FaceForensics++/", "FaceForensics++"), ("/DFDC", "DFDC"),
                         ("/UADFV/", "UADFV"), ("/Deepfake-Eval-2024/", "Deepfake-Eval-2024")):
        if marker in p:
            return name
    if "/df40/test/" in p or "/df40/real/" in p:
        return "DF40-generated"
    return "unknown"


@torch.no_grad()
def score_dataset(model, direct, loader, views, device: str, name: str,
                  max_batches: int = 0, traj=None, spatial=None, rate=None,
                  rate_probe=None) -> pd.DataFrame:
    from tqdm import tqdm

    model.eval()
    direct.eval()
    rows = []
    layers_1x = [i + 1 for i in model.layers]
    for i, batch in enumerate(tqdm(loader, desc=f"  {name}", leave=False)):
        if max_batches and i >= max_batches:
            break
        pixels = views(batch, device)
        out = model(pixels)
        head = direct(out["h_student"])
        record = {
            "dataset": name,
            "key": [str(k) for k in batch["name"]],
            "video_id": [video_of(p) for p in batch["name"]],
            "label": torch.where(batch["label"] != 0, 1, 0).numpy(),
            "real_source": [real_source_of(p) for p in batch["name"]],
            "p_direct": head["prob"].cpu().numpy(),
            "u_direct": head["u"].cpu().numpy(),
        }
        D = out["D"].cpu().numpy()
        D_cls = out["D_cls"].cpu().numpy()
        for j, layer in enumerate(layers_1x):
            record[f"d_l{layer}"] = D[:, j]
            record[f"d_cls_l{layer}"] = D_cls[:, j]
        record["d_mean"] = D.mean(axis=1)
        record["d_early"] = D[:, : max(1, len(layers_1x) // 2)].mean(axis=1)
        record["d_late"] = D[:, len(layers_1x) // 2:].mean(axis=1)
        if rate is not None:
            # h_teacher is the FROZEN FS-VFM pooled feature — exactly the space the operator was
            # fit on. Reusing it means the rate expert rides this pass for free.
            R = rate(out["h_teacher"])
            for j in range(R.shape[1]):
                record[f"rate_r_{j}"] = R[:, j].cpu().numpy()
            if rate_probe is not None:
                z = (R.cpu().numpy() - rate_probe["mu"]) / rate_probe["sigma"]
                logit = z @ rate_probe["coef"] + rate_probe["intercept"]
                p_rate = 1.0 / (1.0 + np.exp(-logit))
                record["p_rate"] = p_rate
                record["u_rate"] = np.full(len(p_rate), np.nan)
        if spatial is not None:
            sp = spatial(out["patch_delta"], out["patch_tokens"])
            record["p_spatial"] = sp["prob"].cpu().numpy()
            record["u_spatial"] = sp["u"].cpu().numpy()
        if traj is not None:
            # The rung's prediction for B2/B3. Read from the SAME `D` the profile records, so the
            # table and the diagnostics cannot describe different quantities.
            t = traj(out["D"])
            record["p_traj"] = t["prob"].cpu().numpy()
            record["u_traj"] = t["u"].cpu().numpy()
        pm = out["patch_map"].flatten(1).cpu().numpy()
        record["patch_max"] = pm.max(axis=1)
        record["patch_std"] = pm.std(axis=1)
        # concentration: how much of the total adaptation sits in the top 10% of patches. A
        # localized manipulation should concentrate; a global domain shift should not.
        k = max(1, pm.shape[1] // 10)
        top = np.sort(pm, axis=1)[:, -k:].sum(axis=1)
        record["patch_top10_share"] = top / np.clip(pm.sum(axis=1), 1e-12, None)
        rows.append(pd.DataFrame(record))
    if not rows:
        raise RuntimeError(f"{name}: no batches produced predictions")
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", type=Path, required=True, help="a Stage-A run directory")
    ap.add_argument("--epoch", type=int, default=None,
                    help="which Stage-A epoch to score (default: the latest on disk)")
    ap.add_argument("--config", type=Path, default=REPO / "training/config/fpad/FPAD_CONFIG.yaml")
    ap.add_argument("--detector-config", type=Path,
                    default=REPO / "training/config/detector/nesy_defake_d1_v.yaml")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--df40", action="store_true",
                    help="treat --datasets as DF40 per-method names")
    ap.add_argument("--traj-head", type=Path, default=None,
                    help="a Stage-B run directory. With it the scorer also emits `p_traj`, the "
                         "trajectory readout that rungs B2/B3 are scored on. Without it only the "
                         "direct readout `p_direct` is emitted, which is what B0/B1 use.")
    ap.add_argument("--rate-operator", type=Path, default=None,
                    help="a frozen MR-VAE artifact. Emits the rate response `rate_r_*` from the "
                         "FROZEN-teacher FS-VFM feature this pass already computes, so the rate "
                         "expert costs no extra forward pass. With --rate-probe it also emits "
                         "`p_rate`.")
    ap.add_argument("--rate-probe", type=Path, default=None,
                    help="a logistic probe over the standardized rate response, fit on FF++ train "
                         "(see analysis/tbiom/fit_rate_probe.py). Stage 1 uses this as a cheap, "
                         "honest stand-in for a trained EDL head: it is a LOWER BOUND, so an "
                         "expert that shows no complementarity here would not gain it from a "
                         "bigger head.")
    ap.add_argument("--spatial-head", type=Path, default=None,
                    help="rung B5: emit `p_spatial` from a spatial readout head")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--jpeg-quality", type=int, default=None,
                    help="Stage 6 compression probe: JPEG-encode and decode every frame at this "
                         "quality before the encoder sees it. NOTE our frames are PNG crops "
                         "already derived from H.264 c23 video, so this is a SECOND, additional "
                         "compression — a JPEG probe, not a substitute for c40.")
    ap.add_argument("--seed", type=int, default=42,
                    help="seeds the dataset shuffle so two scoring runs cover "
                         "the SAME frames — required for the JPEG probe, which "
                         "compares one frame with and without compression")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # `abstract_dataset.py:346` shuffles the collected frame list with `random.shuffle` on the
    # GLOBAL module RNG, so an unseeded scoring process covers a DIFFERENT subset of frames than
    # the next one. With --max-batches that is not cosmetic: the JPEG probe compares the same
    # frame with and without compression, and two unseeded runs shared only 132 of 1,280 frames,
    # silently shrinking the audit to a tenth of its intended power.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoints = sorted(args.student.glob("epoch_*.pth"))
    if not checkpoints:
        raise SystemExit(f"no Stage-A checkpoints in {args.student}")
    chosen = (args.student / f"epoch_{args.epoch:03d}.pth") if args.epoch is not None \
        else checkpoints[-1]
    if not chosen.is_file():
        raise SystemExit(f"{chosen} not found; available: {[c.name for c in checkpoints]}")
    blob = torch.load(str(chosen), map_location="cpu", weights_only=False)
    student_meta = blob["run_meta"]
    print(f"student {chosen} — arm `{student_meta.get('arm')}`, "
          f"lambda_preserve {student_meta.get('lambda_preserve')}, epoch {blob['epoch']}")

    cfg = yaml.safe_load(args.config.read_text())
    model = FPADTeacherStudent(
        checkpoint=cfg["encoder"].get("checkpoint") or (REPO / "weights/FS-VFM/checkpoint-599.pth"),
        layers=tuple(student_meta["layers"]), lora=student_meta["lora"],
        img_size=int(cfg["encoder"].get("img_size", 224))).to(args.device)
    model.load_state_dict(blob["lora"], strict=False)
    direct = DirectEvidenceHead(feature_dim=model.embed_dim,
                                hidden_dim=int(cfg["heads"]["hidden_dim"])).to(args.device)
    direct.load_state_dict(blob["direct_head"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    traj = None
    if args.traj_head:
        from networks.fpad import TrajectoryEvidenceHead
        heads = sorted(args.traj_head.glob("epoch_*.pth"))
        if not heads:
            raise SystemExit(f"no Stage-B checkpoints in {args.traj_head}")
        hb = torch.load(str(heads[-1]), map_location="cpu", weights_only=False)
        desc = hb["head_describe"]
        if desc["n_layers"] != len(model.layers):
            raise SystemExit(
                f"the trajectory head expects {desc['n_layers']} layers but this student has "
                f"{len(model.layers)}. A head applied to a different layer set reads a different "
                f"signal.")
        student_of_head = (hb["run_meta"].get("student") or {}).get("run")
        if student_of_head and Path(student_of_head).resolve() != args.student.resolve():
            raise SystemExit(
                f"this trajectory head was trained on the student at {student_of_head}, not on "
                f"{args.student}. H_traj is fit to one frozen adaptation; applying it to another "
                f"student's trajectory is not that rung.")
        traj = TrajectoryEvidenceHead(
            n_layers=desc["n_layers"], hidden_dim=int(cfg["heads"]["hidden_dim"]),
            use_slopes=desc["use_slopes"], use_calibrator=desc["use_calibrator"])
        traj.load_state_dict(hb["traj_head"])
        traj = traj.to(args.device).eval()
        print(f"  trajectory readout from {heads[-1].name}: {desc}")

    spatial = None
    if args.spatial_head:
        from networks.fpad import SpatialEvidenceHead
        heads = sorted(args.spatial_head.glob("epoch_*.pth"))
        if not heads:
            raise SystemExit(f"no Stage-B checkpoints in {args.spatial_head}")
        hb = torch.load(str(heads[-1]), map_location="cpu", weights_only=False)
        d = hb["head_describe"]
        spatial = SpatialEvidenceHead(n_layers=d["n_layers"], feature_dim=model.embed_dim,
                                      hidden_dim=int(cfg["heads"]["hidden_dim"]),
                                      mode=d["mode"], top_k_fraction=d["top_k_fraction"])
        spatial.load_state_dict(hb["spatial_head"])
        spatial = spatial.to(args.device).eval()
        print(f"  B5 spatial readout: {d}")

    rate = rate_probe = None
    if args.rate_operator:
        from networks.discern_v2.rate_branch import FrozenRateOperator
        rate = FrozenRateOperator(args.rate_operator).to(args.device)
        rate.assert_frozen()
        print(f"  rate operator: beta grid {rate.beta_grid}, hidden {rate.hidden_dim}")
        if args.rate_probe:
            rate_probe = json.loads(args.rate_probe.read_text())
            rate_probe = {k: np.asarray(v) if isinstance(v, list) else v
                          for k, v in rate_probe.items()}
            print(f"  rate probe: FF++-train logistic, train AUROC "
                  f"{rate_probe.get('train_auroc')}")

    data_cfg = prepare_dataset_config(args.detector_config, args.batch_size, args.workers)
    views = FpadViews(augment=False, matched=False, jpeg_quality=args.jpeg_quality)
    if args.jpeg_quality is not None:
        print(f"  JPEG probe: re-encoding every frame at quality {args.jpeg_quality}")

    df40 = None
    resolution = {}
    if args.df40:
        from dataset import df40_paths as df40
        json_dir = df40.DF40_JSON_DIR
        data_cfg = {**data_cfg, "dataset_json_folder": str(json_dir),
                    "label_dict": {**data_cfg.get("label_dict", {}),
                                   **df40.label_dict_for(args.datasets, json_dir)}}
        print(f"DF40 mode: {json_dir}")

    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in args.datasets:
        dataset = load_split(data_cfg, name, args.split)
        if df40 is not None:
            resolution[name] = df40.remap_dataset(dataset)
            print(f"  {name}: resolved {resolution[name]['resolution_rate']:.3f}")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=int(data_cfg["workers"]), collate_fn=dataset.collate_fn)
        frames.append(score_dataset(model, direct, loader, views, args.device, name,
                                    args.max_batches, traj=traj, spatial=spatial,
                                    rate=rate, rate_probe=rate_probe))
        got = frames[-1]
        print(f"  {name}: {len(got)} frames · real_source "
              f"{got.groupby('real_source').size().to_dict()} · "
              f"mean d {got['d_mean'].mean():.5f}")

    table = pd.concat(frames, ignore_index=True)
    dest = args.output / f"profile_epoch_{blob['epoch']:03d}.parquet"
    table.to_parquet(dest, index=False)
    (args.output / "score_meta.json").write_text(json.dumps({
        "student_run": str(args.student), "checkpoint": str(chosen),
        "student_meta": student_meta, "epoch": int(blob["epoch"]),
        "datasets": args.datasets, "split": args.split, "df40": bool(args.df40),
        "df40_resolution": resolution or None, "n_frames": int(len(table)),
        "layers_one_indexed": [i + 1 for i in model.layers],
        "traj_head": str(args.traj_head) if args.traj_head else None,
        "jpeg_quality": args.jpeg_quality,
        "compression_note": (None if args.jpeg_quality is None else
                             f"additional JPEG at quality {args.jpeg_quality} applied on top of "
                             f"c23-derived PNG crops; NOT equivalent to H.264 c40"),
        "readouts": (["p_direct"] + (["p_traj"] if traj is not None else [])
                     + (["p_spatial"] if spatial is not None else [])),
        "spatial_head": str(args.spatial_head) if args.spatial_head else None,
    }, indent=2, default=str))
    print(f"\nwrote {dest} ({len(table)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
