#!/usr/bin/env python3
"""Frame-to-frame consistency features on our stored face crops (temporal panel), dfb_nesy env (insightface/onnxruntime).

    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python analysis/tbiom/temporal/extract_consistency.py --dataset CDFv2

Per stored frame (all <= 32, in temporal order), batched per video: 106-point landmarks (buffalo_l 2d106det) with the face
box taken as the crop's central square (our crops carry a fixed 1.3x margin, so no detector is run), then the ArcFace
identity embedding (w600k_r50) on the 112x112 face aligned from five landmark-derived points (eye centres, nose tip,
mouth corners). Preprocessing constants come from insightface's own model wrappers. Per video, 'ordered' features use the true
frame order and 'shuffled' features the same statistics after a fixed random permutation (seed 0), so only
ORDER-DEPENDENT statistics can differ between the two:
  identity   mean/min cosine to the video's mean embedding (order-free); mean/max cosine distance between consecutive
             frames (order-dependent)
  motion     mean/max consecutive landmark displacement normalised by inter-ocular distance: whole face, mouth, eyes
             (order-dependent); std of mouth-opening and eye-opening ratios (order-free)
A video with < 2 stored frames gets NaN features (the probe imputes the training median).
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import h5py, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "xclip"))
from video_id import video_id  # noqa: E402

EXT = Path("/data/umar/Repos/DISCERN_Ext")
OUT = Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/temporal/feats/consistency")
OUT_OF_SCOPE = {"IP_LAP", "fake-audio-driven", "adv-real", "adv-fake", "retalking", "wav2lip", "diff2lip", "latentsync",
                "memo", "ddim", "DiT", "pixart", "RDDM", "sd2.1", "SiT", "StyleGAN2", "StyleGAN3", "StyleGANXL", "VQGAN"}
MOUTH, LEYE, REYE = list(range(52, 72)), list(range(33, 43)), list(range(87, 97))   # 2d106 index groups


def dataset_spec(ds):
    import yaml
    cfg = EXT / "eval_adaptation/configs" / ("run1_auxedl.yaml" if ds == "FFpp_train" else f"run1auxedl_{ds}.yaml")
    c = yaml.safe_load(open(cfg))
    return (c["trn_h5_path"], list(c["trn_files"])) if ds == "FFpp_train" else \
        (c["tst_h5_path"], [f for fs in c["tst_files"].values() for f in fs])


def process_frame(img, spec):
    """Same pixel operations as export_views.apply_processing, on the stored 8-bit crop."""
    import io
    from PIL import Image
    kind, _, val = spec.partition(":")
    im = Image.fromarray(img)
    if kind == "jpeg":
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=int(val)); buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"))
    w, h = im.size; s = float(val)
    small = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    return np.asarray(small.resize((w, h), Image.BILINEAR))


def stats(E, L, perm):
    """E: (n,512) unit embeddings, L: (n,106,2) landmarks, perm: order to use."""
    E, L = E[perm], L[perm]
    iod = np.linalg.norm(L[:, LEYE].mean(1) - L[:, REYE].mean(1), axis=1).mean() + 1e-6
    cons = 1 - (E[1:] * E[:-1]).sum(1)
    disp = lambda idx: np.linalg.norm(L[1:, idx] - L[:-1, idx], axis=2).mean(1) / iod
    face, mouth, eyes = disp(slice(None)), disp(MOUTH), disp(LEYE + REYE)
    mu = E.mean(0); mu /= np.linalg.norm(mu) + 1e-9; tomean = E @ mu
    mouth_open = np.ptp(L[:, MOUTH, 1], axis=1) / iod; eye_open = np.ptp(L[:, LEYE, 1], axis=1) / iod
    return [tomean.mean(), tomean.min(), cons.mean(), cons.max(), face.mean(), face.max(), mouth.mean(), mouth.max(),
            eyes.mean(), eyes.max(), mouth_open.std(), eye_open.std()]


NAMES = ["id_cos_to_mean", "id_cos_to_mean_min", "id_consec_dist", "id_consec_dist_max", "lmk_face_disp", "lmk_face_disp_max",
         "lmk_mouth_disp", "lmk_mouth_disp_max", "lmk_eye_disp", "lmk_eye_disp_max", "mouth_open_std", "eye_open_std"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--process", default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit-videos", type=int, default=0); ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    import cv2
    from insightface.model_zoo import get_model
    from insightface.utils import face_align
    root = Path.home() / ".insightface/models/buffalo_l"
    import onnxruntime as ort
    lmk = get_model(str(root / "2d106det.onnx"), providers=["CPUExecutionProvider"]); lmk.prepare(ctx_id=-1)
    rec = get_model(str(root / "w600k_r50.onnx"), providers=["CPUExecutionProvider"]); rec.prepare(ctx_id=-1)
    # insightface's wrappers only supply preprocessing constants and I/O names. The sessions are rebuilt explicitly:
    # onnxruntime ignores OMP_NUM_THREADS and silently falls back to CPU when cuDNN 9 is not on LD_LIBRARY_PATH,
    # which on this shared box turned four workers into ~275 cores. Fail loudly instead.
    so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.inter_op_num_threads = 1
    for m in (lmk, rec):
        m.session = ort.InferenceSession(m.model_file, so, providers=["CUDAExecutionProvider"])
        assert m.session.get_providers()[0] == "CUDAExecutionProvider", \
            "onnxruntime CUDA provider unavailable (set LD_LIBRARY_PATH to the env's nvidia/*/lib)"
    LI = lmk.input_size[0]

    def landmarks(imgs):
        aimgs, Ms = [], []
        for img in imgs:
            hh, ww = img.shape[:2]; side = min(hh, ww) / 1.3; center = (ww / 2, hh / 2)
            aimg, M = face_align.transform(img, center, LI, LI / (side * 1.5), 0); aimgs.append(aimg); Ms.append(M)
        blob = cv2.dnn.blobFromImages(aimgs, 1.0 / lmk.input_std, (LI, LI), (lmk.input_mean,) * 3, swapRB=True)
        pred = lmk.session.run(lmk.output_names, {lmk.input_name: blob})[0].reshape(len(imgs), -1, 2)[:, -106:, :]
        pred = (pred + 1) * (LI // 2)
        return np.stack([face_align.trans_points(pr, cv2.invertAffineTransform(M)) for pr, M in zip(pred, Ms)])

    def identity(imgs, L):
        kps = np.stack([L[:, 33:43].mean(1), L[:, 87:97].mean(1), L[:, 86], L[:, 52], L[:, 61]], 1)
        faces = [face_align.norm_crop(img, k, 112) for img, k in zip(imgs, kps)]
        blob = cv2.dnn.blobFromImages(faces, 1.0 / rec.input_std, (112, 112), (rec.input_mean,) * 3, swapRB=True)
        e = rec.session.run(rec.output_names, {rec.input_name: blob})[0]
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    h5rel, lists = dataset_spec(a.dataset)
    keys = [k for f in lists for k in (EXT / f).read_text().split("\n") if k]
    keys = [k for k in keys if not (k.split("/")[-3] in OUT_OF_SCOPE or k.split("/")[-3].split("-")[-1] in OUT_OF_SCOPE)]
    vids: dict[str, list[str]] = {}
    for k in keys:
        vids.setdefault(k.rsplit("/", 1)[0], []).append(k)
    order = sorted(vids)[a.shard::a.nshards]
    if a.limit_videos:
        order = order[:a.limit_videos]
    tag = a.dataset + (f"_{a.process.replace(':', '').replace('.', '')}" if a.process else "") + \
        (f".s{a.shard}of{a.nshards}" if a.nshards > 1 else "")
    out = a.out or OUT / f"{tag}.npz"; out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = {"vid": [], "label": [], "subset": [], "n_stored": []}; F = {"ordered": [], "shuffled": []}
    t0 = time.time()
    with h5py.File(EXT / h5rel, "r") as h:
        for i, v in enumerate(order):
            ks = sorted(vids[v]); imgs = []
            for k in ks:
                img = np.asarray(h[k])
                img = process_frame(img, a.process) if a.process else img
                imgs.append(np.ascontiguousarray(img[:, :, ::-1]))            # insightface works in BGR
            rows["vid"].append(video_id(pd.Series([f"{h5rel}::{ks[0]}"])).iloc[0])
            rows["label"].append(0 if "real" in ks[0].split("/")[-3].lower() else 1)
            rows["subset"].append(ks[0].split("/")[-3]); rows["n_stored"].append(len(ks))
            if len(imgs) >= 2:
                L = landmarks(imgs); E = identity(imgs, L); n = len(E)
                F["ordered"].append(stats(E, L, np.arange(n))); F["shuffled"].append(stats(E, L, rng.permutation(n)))
            else:
                F["ordered"].append([np.nan] * len(NAMES)); F["shuffled"].append([np.nan] * len(NAMES))
            if i % 200 == 0:
                print(f"  {i + 1}/{len(order)} videos  {time.time() - t0:.0f}s", flush=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in rows.items()}, **{k: np.asarray(v, dtype=np.float32) for k, v in F.items()},
             names=np.asarray(NAMES), meta=np.asarray(["buffalo_l", a.process or "none", h5rel]))
    tmp.rename(out)
    print(f"wrote {out}: {len(rows['vid'])} videos in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
