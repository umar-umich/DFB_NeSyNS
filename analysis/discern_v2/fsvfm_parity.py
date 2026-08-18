#!/usr/bin/env python3
"""§8 parity control + §6 numerical parity — what our FS-VFM path costs, measured.

Two independent questions, both required by the spec and both answered here.

**§6 — are we running the authors' pipeline at all?** A unit-style check that our
`FrozenFSVFM` reproduces the official FSFM-CVPR25 downstream feature to numerical tolerance,
matching the *full* input pipeline: crop convention, FS-VFM normalization, pooling rule
(global-pool vs CLS) and final norm — not normalization alone. If this fails, nothing else in
Branch B means what it claims.

**§8 — what does the aligned crop cost?** FS-VFM was released for DLIB face detection with 30%
additional cropping (`face_scale = 1.3`, `datasets/finetune/preprocess/config/default.py:14`),
and the downstream path uses the NON-aligned variant (`extract_and_save_face`), so no landmark
predictor is involved. Per Umar's 2026-08-18 decision the V1 default is our existing aligned
crop, to avoid a second crop cache across every split; this control measures what that choice
costs so the decision rests on a number.

The comparison, on one fixed FF++ subset:

    (a) native  DLIB + 30% crop, re-extracted from the SOURCE video at the same frame indices
    (b) aligned our existing preprocessed crop
    both -> FS-VFM normalization -> frozen FS-VFM -> same-capacity probe

Reported: paired feature agreement (cosine, norm ratio) and the probe-performance gap. Only FF++,
CDF v1-v3, DFD and Deepfake-Eval-2024 have source videos locally — DFDC, DFDCP and UADFV exist
only as 256x256 crops — so a large native-crop gap would also be a coverage problem, not just a
preprocessing one. That is stated in the report rather than discovered later.

🔴 UMAR-RUNS:

    python analysis/discern_v2/fsvfm_parity.py --videos 40 --frames 8 \
        --out analysis/discern_v2/V1_parity

Use at least ~40 videos: below roughly 8 video groups per class the probe separates identities
rather than manipulations and the gap is not interpretable (the script says so when that holds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))

from networks.discern_v2.fsvfm_encoder import FrozenFSVFM, load_normalization  # noqa: E402

FFPP_ROOT = Path("/data/umar/Datasets/FaceForensics++")
PREPROCESSED = Path("/data/umar/Datasets/preprocessed/FaceForensics++")
FACE_SCALE = 1.3          # FSFM-CVPR25 datasets/*/preprocess/config/default.py:14
OFFICIAL_REPO = Path("/data/umar/Repos/FSFM-CVPR25")


# ---------------------------------------------------------------------------
# the official DLIB + 30% crop, ported
# ---------------------------------------------------------------------------


def get_boundingbox(face, width: int, height: int, scale: float = FACE_SCALE):
    """Ported verbatim in behaviour from FSFM-CVPR25 `datasets/*/preprocess/tools/util.py`.

    Square box around the detection, enlarged by `scale`, clipped to the frame. Kept identical
    rather than "improved": the frozen encoder was pretrained and released on exactly this
    convention, and a tighter or looser crop is a different input distribution.
    """
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def native_crops(video_path: Path, frame_indices: list[int], detector) -> dict[int, np.ndarray]:
    """DLIB+30% crops from the SOURCE video at the given frame indices.

    Indexed by frame number so a crop is paired with the aligned crop of the *same* frame: our
    preprocessed files are named `{frame_index:03d}.png`, so the correspondence is exact rather
    than positional.
    """
    import cv2

    wanted = set(frame_indices)
    cap = cv2.VideoCapture(str(video_path))
    out: dict[int, np.ndarray] = {}
    idx = 0
    while cap.isOpened() and len(out) < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector(rgb, 1)
            if faces:
                # the official code takes faces[0]; kept, and failures are counted rather than
                # silently replaced by a different face
                x, y, size = get_boundingbox(faces[0], frame.shape[1], frame.shape[0])
                crop = rgb[y:y + size, x:x + size]
                if crop.size:
                    out[idx] = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        idx += 1
    cap.release()
    return out


def aligned_crop(video: str, subset_dir: Path, frame_index: int) -> np.ndarray | None:
    import cv2

    path = subset_dir / "frames" / video / f"{frame_index:03d}.png"
    if not path.is_file():
        return None
    img = cv2.imread(str(path))
    return cv2.cvtColor(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC),
                        cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# §6 — numerical parity against the official downstream path
# ---------------------------------------------------------------------------


def official_parity(images: np.ndarray, pooling: str = "global_pool") -> dict:
    """Our FrozenFSVFM vs the official models_vit built and loaded by the official recipe.

    The official path is constructed here from the cloned repo, so this compares implementations
    rather than comparing our code with itself.
    """
    if not (OFFICIAL_REPO / "fsvfm" / "models_vit.py").is_file():
        return {"status": "skipped", "reason": f"no official clone at {OFFICIAL_REPO}"}

    sys.path.insert(0, str(OFFICIAL_REPO / "fsvfm"))
    import models_vit as official_models          # noqa: E402
    from util.pos_embed import interpolate_pos_embed  # noqa: E402

    ours = FrozenFSVFM(pooling=pooling)
    official = official_models.vit_large_patch16(
        num_classes=2, drop_path_rate=0.0, global_pool=(pooling == "global_pool"))
    blob = torch.load(ours.checkpoint_path, map_location="cpu", weights_only=False)
    state = blob["model"]
    model_state = official.state_dict()
    for k in [k for k in list(state) if k in model_state
              and state[k].shape != model_state[k].shape]:
        del state[k]
    interpolate_pos_embed(official, state)
    official.load_state_dict(state, strict=False)
    official.eval()

    mean, std = load_normalization()
    x = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)

    with torch.no_grad():
        f_ours = ours(x)
        f_official = official.forward_features(x)

    diff = (f_ours - f_official).abs()
    cos = torch.nn.functional.cosine_similarity(f_ours, f_official, dim=1)
    return {
        "status": "ok",
        "pooling": pooling,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "min_cosine": float(cos.min()),
        "matches": bool(diff.max() < 1e-4),
        "n": int(x.shape[0]),
    }


# ---------------------------------------------------------------------------
# §8 — the crop parity control
# ---------------------------------------------------------------------------


def features_for(encoder: FrozenFSVFM, images: np.ndarray, device: str,
                 batch: int = 16) -> np.ndarray:
    mean = torch.tensor(encoder.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(encoder.std, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch]).float().permute(0, 3, 1, 2).to(device) / 255.0
        out.append(encoder((x - mean) / std).cpu().numpy())
    return np.concatenate(out)


def probe_gap(f_native: np.ndarray, f_aligned: np.ndarray, labels: np.ndarray,
              groups: np.ndarray, seed: int = 42) -> dict:
    """Same-capacity probe on each crop variant, scored with grouped CV on the same subset.

    Identical estimator, grid and folds for both arms — the gap must describe the crops, not the
    effort spent on each probe.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    if len(np.unique(labels)) < 2:
        return {"status": "skipped", "reason": "the subset is single-class"}
    n_splits = min(4, len(np.unique(groups)))
    if n_splits < 2:
        return {"status": "skipped", "reason": "too few groups for grouped CV"}
    # StratifiedGroupKFold, not GroupKFold: videos are grouped real-then-fake, so plain grouped
    # folds land single-class and every AUROC comes back nan. Stratified folds keep whole videos
    # intact AND both classes present.
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(f_native, labels, groups))
    usable = [(tr, te) for tr, te in splits if len(np.unique(labels[te])) > 1]
    if not usable:
        return {"status": "skipped", "n": int(len(labels)),
                "reason": (f"no fold of {n_splits} contains both classes over "
                           f"{len(np.unique(groups))} video groups — enlarge the subset "
                           f"(--videos) before reading a probe gap")}

    def score(x):
        aucs = []
        for tr, te in usable:
            model = LogisticRegression(max_iter=2000, C=0.01, random_state=seed)
            model.fit(x[tr], labels[tr])
            aucs.append(float(roc_auc_score(labels[te], model.decision_function(x[te]))))
        return float(np.mean(aucs)), aucs

    native, native_folds = score(f_native)
    aligned, aligned_folds = score(f_aligned)

    # Minimum size for the number to mean anything. Below this the probe separates IDENTITIES
    # rather than manipulations — the held-out videos are different people — and the fold AUROCs
    # swing between 0 and 1 rather than estimating anything. Reporting a bare number there would
    # be worse than reporting nothing, so the caveat travels with the result.
    n_groups = int(len(np.unique(groups)))
    per_class = min(len(np.unique(groups[labels == c])) for c in np.unique(labels))
    spread = float(np.std(native_folds + aligned_folds))
    caveat = None
    if max(native, aligned) < 0.55:
        # The decisive symptom, and it is NOT "worse than chance by noise": both arms sit near
        # zero across folds, i.e. the probe ranks held-out videos consistently backwards. With
        # few groups it fits the training identities and every held-out identity falls on the
        # wrong side. No gap between two such numbers describes crop quality.
        caveat = (f"NOT INTERPRETABLE: neither arm exceeds chance (native {native:.3f}, aligned "
                  f"{aligned:.3f}) over {per_class} groups per class. The probe is separating "
                  f"identities, not manipulations — rerun with --videos 40 or more.")
    elif per_class < 12:
        caveat = (f"only {per_class} video groups in the smallest class; treat the gap as "
                  f"indicative and confirm with --videos 40 or more")
    elif spread > 0.2:
        caveat = (f"fold AUROC spread is {spread:.2f} — the gap is within sampling noise at this "
                  f"subset size; enlarge --videos before drawing a conclusion")

    return {"status": "ok", "native_auroc": native, "aligned_auroc": aligned,
            "gap_native_minus_aligned": native - aligned,
            "native_auroc_per_fold": native_folds, "aligned_auroc_per_fold": aligned_folds,
            "fold_auroc_std": spread, "n": int(len(labels)),
            "n_groups": n_groups, "min_groups_per_class": per_class,
            "cv_folds": len(usable), "caveat": caveat}


def feature_agreement(f_native: np.ndarray, f_aligned: np.ndarray) -> dict:
    """How far the two crops move the frozen representation for the SAME frame."""
    a = torch.from_numpy(f_native)
    b = torch.from_numpy(f_aligned)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return {
        "mean_cosine": float(cos.mean()), "min_cosine": float(cos.min()),
        "mean_one_minus_cosine": float((1 - cos).mean()),
        "norm_ratio_mean": float((b.norm(dim=1) / a.norm(dim=1).clamp_min(1e-8)).mean()),
        "n_pairs": int(len(cos)),
    }


def collect_subset(n_videos: int, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                          np.ndarray, dict]:
    """One fixed FF++ subset, paired native/aligned crops of the same frames."""
    import dlib

    detector = dlib.get_frontal_face_detector()
    sources = [
        ("real", 0, FFPP_ROOT / "original_sequences/youtube/c23/videos",
         PREPROCESSED / "original_sequences/youtube/c23"),
        ("Deepfakes", 1, FFPP_ROOT / "manipulated_sequences/Deepfakes/c23/videos",
         PREPROCESSED / "manipulated_sequences/Deepfakes/c23"),
    ]
    native, aligned, labels, groups = [], [], [], []
    stats = {"videos": 0, "frames_requested": 0, "dlib_failures": 0, "missing_aligned": 0}

    for _name, label, video_dir, prep_dir in sources:
        videos = sorted(p for p in video_dir.glob("*.mp4"))[: n_videos // len(sources)]
        for vp in videos:
            stem = vp.stem
            frame_dir = prep_dir / "frames" / stem
            if not frame_dir.is_dir():
                continue
            indices = sorted(int(p.stem) for p in frame_dir.glob("*.png"))[:n_frames]
            if not indices:
                continue
            stats["videos"] += 1
            stats["frames_requested"] += len(indices)
            crops = native_crops(vp, indices, detector)
            for idx in indices:
                if idx not in crops:
                    stats["dlib_failures"] += 1
                    continue
                al = aligned_crop(stem, prep_dir, idx)
                if al is None:
                    stats["missing_aligned"] += 1
                    continue
                native.append(crops[idx])
                aligned.append(al)
                labels.append(label)
                groups.append(stem)
    if not native:
        raise SystemExit("collected no paired crops; check the FF++ video and preprocessed paths")
    return (np.stack(native), np.stack(aligned), np.array(labels),
            np.array(groups), stats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", type=int, default=40)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--pooling", default="global_pool", choices=("global_pool", "cls"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "V1_parity")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"collecting a fixed subset: {args.videos} videos x {args.frames} frames")
    native, aligned, labels, groups, stats = collect_subset(args.videos, args.frames)
    print(f"  paired crops: {len(native)} ({stats})")

    print("§6 numerical parity against the official FS-VFM path")
    parity = official_parity(aligned[: min(4, len(aligned))], args.pooling)
    print(f"  {parity}")

    print("§8 crop parity control")
    encoder = FrozenFSVFM(pooling=args.pooling).to(args.device)
    f_native = features_for(encoder, native, args.device)
    f_aligned = features_for(encoder, aligned, args.device)
    agreement = feature_agreement(f_native, f_aligned)
    gap = probe_gap(f_native, f_aligned, labels, groups)
    print(f"  feature agreement: {agreement}")
    print(f"  probe: {gap}")

    result = {
        "spec": "V1 §6 numerical parity + §8 crop parity control",
        "subset": {"n_pairs": int(len(native)), "n_real": int((labels == 0).sum()),
                   "n_fake": int((labels == 1).sum()), **stats},
        "encoder": {"pooling": args.pooling, "fingerprint": encoder.fingerprint(),
                    "checkpoint": encoder.checkpoint_path},
        "numerical_parity_vs_official": parity,
        "feature_agreement_native_vs_aligned": agreement,
        "probe_gap": gap,
        "v1_default": "aligned crop (Umar, 2026-08-18) — recorded deviation from §8's native "
                      "default; this control measures its cost",
        "coverage_note": "native DLIB crops are only reachable for FF++, CDF v1-v3, DFD and "
                         "Deepfake-Eval-2024; DFDC, DFDCP and UADFV exist locally as 256x256 "
                         "crops with no source frames",
    }
    (args.out / "parity.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {args.out}/parity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
