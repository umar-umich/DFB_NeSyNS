#!/usr/bin/env python3
"""Stage 5 — patch-level localization and faithfulness.

    🔴 UMAR-RUNS (GPU):

    python training/localize_fpad.py --student logs/fpad/studentA_preserve_seed42 \
        --manipulations Deepfakes Face2Face FaceSwap NeuralTextures \
        --traj-head logs/fpad/B3_seed42 \
        --output logs/fpad/localization/B3 --device cuda:1

This is the axis that makes the work a standalone paper rather than an AUC wrapper: FS-VFM claims
detection, not faithful region-localization.

Ground-truth tiers, kept strictly apart
---------------------------------------
**Tier 1, genuine masks.** FF++ ships per-manipulation masks and they ARE staged here for
Deepfakes, Face2Face, FaceSwap, NeuralTextures and DeepFakeDetection — already warped through the
same crop transform as the frames, so no re-alignment is needed. Quantitative localization is
reported only for these. `FaceShifter` has no masks and is EXCLUDED rather than pseudo-masked in.

**Pseudo-masks are not computed here at all.** Source-paired real-minus-fake difference maps carry
alignment, compression, colour and rendering changes unrelated to the manipulated region, so the
brief permits them for qualitative sanity only. Rather than emit a number that could be mistaken
for localization accuracy, this script does not produce one.

**Faithfulness needs no mask** and is therefore always available: deletion of the highest-adaptation
patches versus random patches. It answers "does the map explain the decision", which is a different
question from "does the map match the manipulated region", and both belong in the paper.

AUPRC is the headline, IoU is reported with its resolution
---------------------------------------------------------
The patch grid is 14x14 = 196 at 224 input, so each patch covers ~18px of a 256x256 mask. That is
adequate for AUPRC (~43 of 196 patches positive at a typical 22% mask fraction) but coarse for IoU,
where discretisation is a large fraction of the region. Masks are downsampled to the patch grid by
AREA, so a patch's label is the fraction of it that is manipulated, thresholded once at 0.5 and
recorded.
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
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

from networks.fpad import DirectEvidenceHead, FPADTeacherStudent  # noqa: E402

MASK_ROOT = Path("/data/umar/Datasets/preprocessed/FaceForensics++/manipulated_sequences")
WITH_MASKS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "DeepFakeDetection")
WITHOUT_MASKS = ("FaceShifter",)
PATCH_LABEL_THRESHOLD = 0.5


def collect_pairs(manipulation: str, compression: str = "c23",
                  limit_videos: int = 0, per_video: int = 4) -> list[tuple[Path, Path]]:
    """(frame, mask) pairs that both exist. A frame without its mask is skipped and counted."""
    root = MASK_ROOT / manipulation / compression
    frames_dir, masks_dir = root / "frames", root / "masks"
    if not masks_dir.is_dir():
        return []
    pairs = []
    videos = sorted(p.name for p in frames_dir.iterdir() if p.is_dir())
    if limit_videos:
        videos = videos[:limit_videos]
    for video in videos:
        fs = sorted((frames_dir / video).glob("*.png"))
        step = max(1, len(fs) // per_video) if per_video else 1
        for f in fs[::step][:per_video or None]:
            m = masks_dir / video / f.name
            if m.is_file():
                pairs.append((f, m))
    return pairs


def load_batch(pairs: list[tuple[Path, Path]], size: int = 224):
    """Frames as [0,1] tensors; masks as float in [0,1] at their native resolution."""
    from PIL import Image

    imgs, masks = [], []
    for fpath, mpath in pairs:
        img = Image.open(fpath).convert("RGB").resize((size, size), Image.BILINEAR)
        imgs.append(torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1))
        m = Image.open(mpath).convert("L")
        masks.append(torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0))
    return torch.stack(imgs), masks


def mask_to_patch_labels(mask: torch.Tensor, grid: int) -> torch.Tensor:
    """Downsample a mask to the patch grid BY AREA: each cell is the manipulated fraction."""
    m = mask.unsqueeze(0).unsqueeze(0)
    return F.adaptive_avg_pool2d(m, (grid, grid)).squeeze(0).squeeze(0)


@torch.no_grad()
def faithfulness(model, direct, traj, spatial, pixels: torch.Tensor, patch_map: torch.Tensor,
                 device: str, fractions=(0.1, 0.2, 0.3), seed: int = 42) -> dict:
    """Deletion: suppress the highest-adaptation patches vs random ones, watch fake evidence fall.

    Suppression replaces the patch's pixels with the batch mean colour — a neutral value rather
    than black, which would itself be an out-of-distribution artifact the encoder reacts to.
    Needs no mask, so it runs regardless of tier.
    """
    g = patch_map.shape[-1]
    side = pixels.shape[-1] // g
    rng = np.random.default_rng(seed)
    fill = pixels.mean(dim=(2, 3), keepdim=True)

    def score(x):
        out = model(x)
        if spatial is not None:
            return spatial(out["patch_delta"], out["patch_tokens"])["prob"]
        return (traj(out["D"])["prob"] if traj is not None
                else direct(out["h_student"])["prob"])

    def logodds(p_fake: torch.Tensor) -> torch.Tensor:
        """log p/(1-p). Reported instead of the probability because the probability SATURATES.

        The first run of this test measured probability drops and returned 0.0000 for every
        deletion fraction, at a baseline p(fake) of 0.99. That is not a faithfulness failure, it
        is a ceiling: at p = 0.99 removing a third of the patches cannot move the number much even
        if it moves the evidence a great deal. Log-odds has headroom where probability does not,
        so a real effect becomes visible and a real absence stays visible too.
        """
        q = p_fake.clamp(1e-6, 1 - 1e-6)
        return torch.log(q / (1 - q))

    base = score(pixels)
    base_lo = logodds(base)
    results = {"baseline_p_fake": float(base.mean()),
               "baseline_logodds": float(base_lo.mean()),
               "metric": "log-odds drop; probability saturates at this confidence"}
    flat = patch_map.flatten(1)
    for frac in fractions:
        k = max(1, int(round(frac * flat.shape[1])))
        top = flat.topk(k, dim=1).indices
        rand = torch.stack([torch.from_numpy(
            rng.choice(flat.shape[1], size=k, replace=False)) for _ in range(flat.shape[0])])
        for tag, idx in (("top", top), ("random", rand.to(top.device))):
            x = pixels.clone()
            for b in range(x.shape[0]):
                for cell in idx[b].tolist():
                    r, c = divmod(int(cell), g)
                    x[b, :, r * side:(r + 1) * side, c * side:(c + 1) * side] = fill[b]
            s_ = score(x)
            pct = int(frac * 100)
            results[f"p_fake_delete_{tag}_{pct}pct"] = float(s_.mean())
            results[f"logodds_delete_{tag}_{pct}pct"] = float(logodds(s_).mean())
    for frac in fractions:
        pct = int(frac * 100)
        # the faithfulness quantity: deleting the map's OWN top patches should cost more evidence
        # than deleting the same number at random
        results[f"faithfulness_gain_{pct}pct"] = (
            results[f"p_fake_delete_random_{pct}pct"] - results[f"p_fake_delete_top_{pct}pct"])
        results[f"faithfulness_logodds_gain_{pct}pct"] = (
            results[f"logodds_delete_random_{pct}pct"] - results[f"logodds_delete_top_{pct}pct"])
        results[f"logodds_drop_top_{pct}pct"] = (
            results["baseline_logodds"] - results[f"logodds_delete_top_{pct}pct"])
    return results


@torch.no_grad()
def insertion(model, direct, traj, spatial, pixels: torch.Tensor, patch_map: torch.Tensor,
              device: str, fractions=(0.1, 0.2, 0.3), seed: int = 42) -> dict:
    """Insertion: start from a blank image and ADD the map's top patches back.

    The complement of deletion, and more sensitive on a saturated model. Deletion asks "does
    removing this hurt", which at p(fake) = 0.99 leaves the model plenty of remaining evidence to
    stay confident on. Insertion asks "is this ENOUGH on its own", which starts from no evidence
    and so has the whole range to move through.

    Sign convention is the mirror of deletion: here the map's top patches should produce MORE
    evidence than the same number of random patches, so gain = top - random.
    """
    g = patch_map.shape[-1]
    side = pixels.shape[-1] // g
    rng = np.random.default_rng(seed)
    fill = pixels.mean(dim=(2, 3), keepdim=True)

    def score(x):
        out = model(x)
        if spatial is not None:
            return spatial(out["patch_delta"], out["patch_tokens"])["prob"]
        return (traj(out["D"])["prob"] if traj is not None
                else direct(out["h_student"])["prob"])

    def logodds(p_fake):
        q = p_fake.clamp(1e-6, 1 - 1e-6)
        return torch.log(q / (1 - q))

    blank = fill.expand_as(pixels).clone()
    results = {"blank_p_fake": float(score(blank).mean()),
               "blank_logodds": float(logodds(score(blank)).mean()),
               "full_logodds": float(logodds(score(pixels)).mean())}
    flat = patch_map.flatten(1)
    for frac in fractions:
        k = max(1, int(round(frac * flat.shape[1])))
        top = flat.topk(k, dim=1).indices
        rand = torch.stack([torch.from_numpy(
            rng.choice(flat.shape[1], size=k, replace=False)) for _ in range(flat.shape[0])]
        ).to(top.device)
        pct = int(frac * 100)
        for tag, idx in (("top", top), ("random", rand)):
            x = blank.clone()
            for b in range(x.shape[0]):
                for cell in idx[b].tolist():
                    r, c = divmod(int(cell), g)
                    x[b, :, r * side:(r + 1) * side, c * side:(c + 1) * side] = \
                        pixels[b, :, r * side:(r + 1) * side, c * side:(c + 1) * side]
            s_ = score(x)
            results[f"p_fake_insert_{tag}_{pct}pct"] = float(s_.mean())
            results[f"logodds_insert_{tag}_{pct}pct"] = float(logodds(s_).mean())
        results[f"insertion_logodds_gain_{pct}pct"] = (
            results[f"logodds_insert_top_{pct}pct"]
            - results[f"logodds_insert_random_{pct}pct"])
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--student", type=Path, required=True)
    ap.add_argument("--traj-head", type=Path, default=None)
    ap.add_argument("--spatial-head", type=Path, default=None,
                    help="rung B5. Localization is then scored on the head's OWN "
                         "learned-weight adaptation map, not the equal-weight mean — comparing "
                         "B5's decision to a map it does not use would measure nothing.")
    ap.add_argument("--config", type=Path, default=REPO / "training/config/fpad/FPAD_CONFIG.yaml")
    ap.add_argument("--manipulations", nargs="+", default=list(WITH_MASKS))
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--limit-videos", type=int, default=100)
    ap.add_argument("--per-video", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", type=Path, required=True)
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

    refused = [m for m in args.manipulations if m in WITHOUT_MASKS]
    if refused:
        raise SystemExit(
            f"{refused} have NO masks staged. Quantitative localization is reported only where "
            f"genuine masks back it; pseudo-masks carry alignment, compression and rendering "
            f"differences unrelated to the manipulated region and must never be reported as "
            f"localization accuracy. Drop them from --manipulations.")

    checkpoints = sorted(args.student.glob("epoch_*.pth"))
    if not checkpoints:
        raise SystemExit(f"no Stage-A checkpoints in {args.student}")
    blob = torch.load(str(checkpoints[-1]), map_location="cpu", weights_only=False)
    cfg = yaml.safe_load(args.config.read_text())
    model = FPADTeacherStudent(
        checkpoint=cfg["encoder"].get("checkpoint") or (REPO / "weights/FS-VFM/checkpoint-599.pth"),
        layers=tuple(blob["run_meta"]["layers"]), lora=blob["run_meta"]["lora"]).to(args.device)
    model.load_state_dict(blob["lora"], strict=False)
    direct = DirectEvidenceHead(model.embed_dim, int(cfg["heads"]["hidden_dim"])).to(args.device)
    direct.load_state_dict(blob["direct_head"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    direct.eval()

    spatial = None
    if args.spatial_head:
        from networks.fpad import SpatialEvidenceHead
        heads = sorted(args.spatial_head.glob("epoch_*.pth"))
        if not heads:
            raise SystemExit(f"no Stage-B checkpoints in {args.spatial_head}")
        hb = torch.load(str(heads[-1]), map_location="cpu", weights_only=False)
        d = hb["head_describe"]
        student_of_head = (hb["run_meta"].get("student") or {}).get("run")
        if student_of_head and Path(student_of_head).resolve() != args.student.resolve():
            raise SystemExit(
                f"this spatial head was trained on the student at {student_of_head}, not on "
                f"{args.student}. B5's head is fit to one frozen adaptation.")
        spatial = SpatialEvidenceHead(
            n_layers=d["n_layers"], feature_dim=model.embed_dim,
            hidden_dim=int(cfg["heads"]["hidden_dim"]), mode=d["mode"],
            top_k_fraction=d["top_k_fraction"])
        spatial.load_state_dict(hb["spatial_head"])
        spatial = spatial.to(args.device).eval()
        print(f"  B5 spatial readout: {d}")

    traj = None
    if args.traj_head:
        from networks.fpad import TrajectoryEvidenceHead
        heads = sorted(args.traj_head.glob("epoch_*.pth"))
        hb = torch.load(str(heads[-1]), map_location="cpu", weights_only=False)
        d = hb["head_describe"]
        traj = TrajectoryEvidenceHead(n_layers=d["n_layers"],
                                      hidden_dim=int(cfg["heads"]["hidden_dim"]),
                                      use_slopes=d["use_slopes"],
                                      use_calibrator=d["use_calibrator"])
        traj.load_state_dict(hb["traj_head"])
        traj = traj.to(args.device).eval()

    from sklearn.metrics import average_precision_score, roc_auc_score

    args.output.mkdir(parents=True, exist_ok=True)
    per_manip, rows, faith_all = {}, [], {}
    for manip in args.manipulations:
        pairs = collect_pairs(manip, args.compression, args.limit_videos, args.per_video)
        if not pairs:
            per_manip[manip] = {"status": "no (frame, mask) pairs found"}
            continue
        scores, labels, ious = [], [], []
        # per-LAYER patch maps, so "which depth localizes" is measured rather than assumed. Three
        # prior analyses put the usable signal at layer 20; this either confirms or refutes it,
        # and it decides what a spatial readout should aggregate over.
        layer_scores: dict[str, list] = {}
        for i in range(0, len(pairs), args.batch_size):
            chunk = pairs[i:i + args.batch_size]
            pixels, masks = load_batch(chunk)
            pixels = pixels.to(args.device)
            with torch.no_grad():
                out = model(pixels)
            if spatial is not None:
                # B5's own learned-weight map, reshaped to the grid. Scoring B5's localization on
                # the equal-weight mean would evaluate a map its decision never uses.
                # no_grad: the head's parameters require grad, and the map is only being read.
                with torch.no_grad():
                    a = spatial.adaptation_map(out["patch_delta"])
                g_ = int(round(a.shape[-1] ** 0.5))
                pm = a.reshape(-1, g_, g_).cpu()
            else:
                pm = out["patch_map"].cpu()
            pd_layers = out["patch_delta"].cpu()          # (B, L, N)
            layer_names = [f"d_l{l + 1}" for l in model.layers]
            g = pm.shape[-1]
            for b, mask in enumerate(masks):
                lab = mask_to_patch_labels(mask, g).flatten().numpy()
                s = pm[b].flatten().numpy()
                binary = (lab >= PATCH_LABEL_THRESHOLD).astype(int)
                if binary.sum() == 0 or binary.sum() == len(binary):
                    continue
                scores.append(s)
                labels.append(binary)
                for li, lname in enumerate(layer_names):
                    layer_scores.setdefault(lname, []).append(
                        (pd_layers[b, li].numpy(), binary))
                pred = s >= np.quantile(s, 1 - binary.mean())
                inter = np.logical_and(pred, binary).sum()
                union = np.logical_or(pred, binary).sum()
                ious.append(inter / union if union else np.nan)
            if i == 0:
                faith_all[manip] = faithfulness(model, direct, traj, spatial, pixels,
                                                pm.to(args.device), args.device)
                faith_all[manip].update(
                    insertion(model, direct, traj, spatial, pixels, pm.to(args.device),
                              args.device))
        if not scores:
            per_manip[manip] = {"status": "no frame had a mixed patch mask"}
            continue
        flat_s = np.concatenate(scores)
        flat_y = np.concatenate(labels)
        per_manip[manip] = {
            "status": "ok", "tier": "genuine FF++ masks",
            "n_frames": len(scores), "grid": int(g),
            "patch_auprc": float(average_precision_score(flat_y, flat_s)),
            "patch_auroc": float(roc_auc_score(flat_y, flat_s)),
            "patch_iou_at_oracle_k": float(np.nanmean(ious)),
            "positive_patch_fraction": float(flat_y.mean()),
            "chance_auprc": float(flat_y.mean()),
        }
        per_manip[manip]["per_layer"] = {}
        for lname, pairs in layer_scores.items():
            ls = np.concatenate([a for a, _ in pairs])
            ly = np.concatenate([b for _, b in pairs])
            per_manip[manip]["per_layer"][lname] = {
                "patch_auprc": float(average_precision_score(ly, ls)),
                "patch_auroc": float(roc_auc_score(ly, ls))}
        m = per_manip[manip]
        best = max(m["per_layer"], key=lambda k: m["per_layer"][k]["patch_auprc"])
        print(f"    per-layer AUPRC: " + " ".join(
            f"{k.replace('d_l','L')}={v['patch_auprc']:.3f}" for k, v in m["per_layer"].items())
            + f"   best={best.replace('d_l','L')} · aggregate={m['patch_auprc']:.3f}")
        print(f"  {manip:18s} AUPRC {m['patch_auprc']:.4f} (chance {m['chance_auprc']:.4f}) · "
              f"AUROC {m['patch_auroc']:.4f} · IoU {m['patch_iou_at_oracle_k']:.4f} · "
              f"{m['n_frames']} frames")
        rows.append({"manipulation": manip, **m})

    payload = {
        "student": str(args.student), "traj_head": str(args.traj_head) if args.traj_head else None,
        "readout": ("spatial(B5)" if spatial is not None else
                    "trajectory" if traj is not None else "direct"),
        "spatial_head": str(args.spatial_head) if args.spatial_head else None,
        "epoch": int(blob["epoch"]), "grid_note":
            "14x14 patches at 224 input; masks downsampled BY AREA and thresholded at "
            f"{PATCH_LABEL_THRESHOLD}. AUPRC is the headline; IoU is coarse at this resolution.",
        "excluded_no_masks": list(WITHOUT_MASKS),
        "localization": per_manip, "faithfulness": faith_all,
    }
    (args.output / "localization.json").write_text(json.dumps(payload, indent=2, default=str))
    if rows:
        pd.DataFrame(rows).to_csv(args.output / "localization.csv", index=False)
    print("\nfaithfulness (deletion; positive = deleting the map's own top patches costs more "
          "evidence than deleting at random):")
    for manip, f in faith_all.items():
        lg = {k: v for k, v in f.items() if k.startswith("faithfulness_logodds_gain")}
        dr = {k: v for k, v in f.items() if k.startswith("logodds_drop_top")}
        print(f"  {manip:18s} baseline p {f['baseline_p_fake']:.4f} "
              f"(log-odds {f['baseline_logodds']:+.2f})")
        print(f"      top-vs-random log-odds gain  "
              + " · ".join(f"{k.split('_')[-1]} {v:+.3f}" for k, v in lg.items()))
        print(f"      log-odds lost to deletion    "
              + " · ".join(f"{k.split('_')[-1]} {v:+.3f}" for k, v in dr.items()))
        ig = {k: v for k, v in f.items() if k.startswith("insertion_logodds_gain")}
        if ig:
            print(f"      INSERTION top-vs-random gain "
                  + " · ".join(f"{k.split('_')[-1]} {v:+.3f}" for k, v in ig.items())
                  + f"   (blank log-odds {f['blank_logodds']:+.2f} -> full "
                    f"{f['full_logodds']:+.2f})")
    print(f"\nwrote {args.output}/localization.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
