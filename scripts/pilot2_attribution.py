"""
Pilot 2 part one, attribution signal on DF40 (CLAUDE_CODE_PILOTS Pilot 2 part one).

The GATE. Decides whether frozen features plus predicate signatures separate generator
families, whether the predicates carry that signal, and whether a predicate-count
baseline beats raw DCT. All UNVERIFIED here, the verification interaction (part two) runs
only on GO.

Families per docs/DF40_PREREGISTRATION.md (Option B, edit methods folded into EFS):
  face_swap, face_reenactment, entire_face_synthesis
Seen generators are those present in DF40 train/. Held-out generators are the nine
test-only ones, unseen generators within known families. Heads train on seen-generator
train frames, evaluate on test frames of seen generators (seen) and of held-out
generators (held-out transfer).

Baselines, each with an attribution head, three seeds:
  A  linear probe on frozen CLIP ViT-L/14 CLS features
  B  raw DCT log-magnitude radial profile
  C  predicate COUNT vector (how many of each predicate type fire, over 5 sub-regions)
  D  MLP over the full unverified predicate attribute vector (deterministic probes)

Run:
  HF_HOME=~/.cache/huggingface \
  /data/umar/miniconda3/envs/GenD/bin/python scripts/pilot2_attribution.py --gpu 0
"""
import os, sys, json, random, argparse, glob
import numpy as np
import cv2

REPO = "/data/umar/Repos/DFB_NeSyNS"
DF = "/data/umar/Datasets/df40"
CACHE = f"{REPO}/results/pilot2/features.npz"

FAMILY = {
    # face_swap
    "blendface": "face_swap", "faceswap": "face_swap", "fsgan": "face_swap",
    "simswap": "face_swap", "inswap": "face_swap", "uniface": "face_swap",
    "mobileswap": "face_swap", "facedancer": "face_swap", "e4s": "face_swap",
    "deepfacelab": "face_swap",
    # face_reenactment
    "fomm": "face_reenactment", "facevid2vid": "face_reenactment", "MRAA": "face_reenactment",
    "one_shot_free": "face_reenactment", "pirender": "face_reenactment", "tpsm": "face_reenactment",
    "lia": "face_reenactment", "danet": "face_reenactment", "sadtalker": "face_reenactment",
    "mcnet": "face_reenactment", "hyperreenact": "face_reenactment", "wav2lip": "face_reenactment",
    "heygen": "face_reenactment",
    # entire_face_synthesis (+ folded edits per Option B)
    "StyleGAN2": "entire_face_synthesis", "StyleGAN3": "entire_face_synthesis",
    "StyleGANXL": "entire_face_synthesis", "VQGAN": "entire_face_synthesis",
    "sd2.1": "entire_face_synthesis", "ddim": "entire_face_synthesis",
    "pixart": "entire_face_synthesis", "DiT": "entire_face_synthesis",
    "SiT": "entire_face_synthesis", "RDDM": "entire_face_synthesis",
    "CollabDiff": "entire_face_synthesis", "MidJourney": "entire_face_synthesis",
    "whichfaceisreal": "entire_face_synthesis",
    "e4e": "entire_face_synthesis", "stargan": "entire_face_synthesis",
    "starganv2": "entire_face_synthesis", "styleclip": "entire_face_synthesis",
}
FAMS = ["face_swap", "face_reenactment", "entire_face_synthesis"]
HELD_OUT = ["deepfacelab", "heygen", "CollabDiff", "MidJourney", "whichfaceisreal",
            "e4e", "stargan", "starganv2", "styleclip"]     # the 9 test-only generators
SEED = 0
N_TRAIN = 80          # frames per seen generator from train split
N_EVAL = 50           # frames per generator from test split


# --------------------------------------------------------------------------- frame sampling
def list_frames(gen, split):
    root = f"{DF}/{split}/{gen}"
    if not os.path.isdir(root):
        return []
    fs = glob.glob(f"{root}/**/*.png", recursive=True) + glob.glob(f"{root}/**/*.jpg", recursive=True)
    return sorted(fs)


def sample_frames(gen, split, n, rng):
    fs = list_frames(gen, split)
    if len(fs) > n:
        fs = rng.sample(fs, n)
    return fs


# --------------------------------------------------------------------------- predicate probes
def dct_profile(gray):
    """Radial log-magnitude DCT profile, 32 bins. Baseline B feature."""
    g = cv2.resize(gray, (256, 256)).astype(np.float32)
    d = np.abs(cv2.dct(g))
    d = np.log1p(d)
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(xx ** 2 + yy ** 2)
    rb = (r / r.max() * 31).astype(int)
    prof = np.array([d[rb == i].mean() if (rb == i).any() else 0.0 for i in range(32)])
    return prof


def _fft_mag(gray):
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    return np.abs(F)


def predicate_attributes(bgr):
    """Deterministic codebook attribute vector for a face crop (whole + region probes).
    Returns (attr_vec, count_vec). Attributes follow CODEBOOK section 3 quantitative fields."""
    img = cv2.resize(bgr, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    mag = _fft_mag(gray)
    cy, cx = H / 2, W / 2
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (0.5 * H)
    total = mag.sum() + 1e-6

    # frequency_anomaly
    hf_energy_ratio = mag[rr > 0.5].sum() / total
    radial = np.array([mag[(rr >= i / 16) & (rr < (i + 1) / 16)].mean() for i in range(16)])
    radial = np.nan_to_num(radial)
    dominant_band = float(np.argmax(radial)) / 16.0
    band_concentration = float(radial.max() / (radial.mean() + 1e-6))
    # checkerboard: energy at the extreme corners (upsampling harmonics)
    corner = mag[rr > 0.9].mean()
    checkerboard_score = float(corner / (mag[rr < 0.3].mean() + 1e-6))
    # radial slope (log-log)
    xs = np.arange(1, 17); ys = np.log1p(radial)
    radial_slope = float(np.polyfit(np.log(xs), ys, 1)[0])

    # noise_inconsistency (SRM-ish residual)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    res = gray - blur
    residual_std = float(res.std())
    center = res[64:192, 64:192]; border = res.copy(); border[64:192, 64:192] = 0
    cross_region_noise_gap = float(abs(center.std() - border[border != 0].std()))
    noise_periodicity = float(_fft_mag(res)[rr > 0.7].mean() / (np.abs(res).mean() + 1e-6))

    # geometry (symmetry via left-right flip)
    flip = cv2.flip(gray, 1)
    symmetry_deviation = float(np.abs(gray - flip).mean() / 255.0)

    # texture
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    skin = img[96:160, 96:160]
    skin_gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY).astype(np.float32)
    skin_texture_regularity = float(cv2.Laplacian(skin_gray, cv2.CV_32F).var())
    pore_detail_loss = float(-np.log1p(lap.var()))          # low variance -> more loss

    # lighting / color
    b, g, r = cv2.split(img.astype(np.float32) + 1e-6)
    color_temperature_gap = float((r.mean() - b.mean()) / 255.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    shadow_direction = float(np.arctan2(gy.mean(), gx.mean()))

    attrs = dict(
        hf_energy_ratio=hf_energy_ratio, dominant_band=dominant_band,
        band_concentration=band_concentration, checkerboard_score=checkerboard_score,
        radial_slope=radial_slope, residual_std=residual_std,
        cross_region_noise_gap=cross_region_noise_gap, noise_periodicity=noise_periodicity,
        symmetry_deviation=symmetry_deviation, skin_texture_regularity=skin_texture_regularity,
        pore_detail_loss=pore_detail_loss, color_temperature_gap=color_temperature_gap,
        shadow_direction=shadow_direction,
    )
    vec = np.array([attrs[k] for k in sorted(attrs)], np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, sorted(attrs)


def count_vector(attr_matrix):
    """Predicate COUNT vector: per column, does it fire (above global median). Grouped
    into the codebook predicate families by column ranges. Returns per-sample counts."""
    med = np.median(attr_matrix, axis=0)
    fires = (attr_matrix > med).astype(np.float32)
    # group columns (sorted attr names) into codebook predicate types by keyword
    return fires


# --------------------------------------------------------------------------- feature build
def build_features(gpu):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    dev = f"cuda:{gpu}"
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(dev)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    rng = random.Random(SEED)

    rows = []      # (gen, family, split_role, frame_path)
    for gen in FAMILY:
        if gen not in HELD_OUT:                       # seen: train frames for training
            for f in sample_frames(gen, "train", N_TRAIN, rng):
                rows.append((gen, FAMILY[gen], "train", f))
        for f in sample_frames(gen, "test", N_EVAL, rng):   # eval frames for all
            role = "eval_heldout" if gen in HELD_OUT else "eval_seen"
            rows.append((gen, FAMILY[gen], role, f))
    print(f"[data] {len(rows)} frames across {len(FAMILY)} generators")

    from PIL import Image
    clip_feats, dct_feats, pred_feats = [], [], []
    B = 64
    buf_img, buf_idx = [], []

    @torch.no_grad()
    def flush(paths):
        pil = [Image.open(p).convert("RGB") for p in paths]
        inp = proc(images=pil, return_tensors="pt").to(dev)
        f = clip.get_image_features(**inp)
        return f.cpu().numpy()

    valid = []
    batch_paths = []
    for gen, fam, role, f in rows:
        img = cv2.imread(f)
        if img is None:
            continue
        dct_feats.append(dct_profile(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        pv, _ = predicate_attributes(img)
        pred_feats.append(pv)
        valid.append((gen, fam, role))
        batch_paths.append(f)
        if len(batch_paths) == B:
            clip_feats.append(flush(batch_paths)); batch_paths = []
            if len(valid) % 640 == 0:
                print(f"  featurized {len(valid)}")
    if batch_paths:
        clip_feats.append(flush(batch_paths))

    clip_feats = np.concatenate(clip_feats).astype(np.float32)
    dct_feats = np.stack(dct_feats).astype(np.float32)
    pred_feats = np.stack(pred_feats).astype(np.float32)
    gens = np.array([v[0] for v in valid]); fams = np.array([v[1] for v in valid])
    roles = np.array([v[2] for v in valid])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, clip=clip_feats, dct=dct_feats, pred=pred_feats,
                        gens=gens, fams=fams, roles=roles)
    print(f"[cache] wrote {CACHE}")
    return dict(clip=clip_feats, dct=dct_feats, pred=pred_feats, gens=gens, fams=fams, roles=roles)


# --------------------------------------------------------------------------- evaluation
def evaluate(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    y = np.array([FAMS.index(f) for f in data["fams"]])
    roles = data["roles"]
    tr = roles == "train"; ev_s = roles == "eval_seen"; ev_h = roles == "eval_heldout"

    # predicate count vector (fires above train-median), computed from train stats
    med = np.median(data["pred"][tr], axis=0)
    cnt = (data["pred"] > med).astype(np.float32)

    feats = {"A_clip": data["clip"], "B_dct": data["dct"], "C_count": cnt, "D_pred": data["pred"]}
    results = {}
    for name, X in feats.items():
        accs_seen, accs_held = [], []
        for seed in [0, 1, 2]:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xs, Xh = sc.transform(X[tr]), sc.transform(X[ev_s]), sc.transform(X[ev_h])
            if name == "D_pred":
                clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                                    random_state=seed)
            else:
                clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            clf.fit(Xtr, y[tr])
            accs_seen.append(float((clf.predict(Xs) == y[ev_s]).mean()))
            accs_held.append(float((clf.predict(Xh) == y[ev_h]).mean()))
        results[name] = dict(
            seen_mean=float(np.mean(accs_seen)), seen_std=float(np.std(accs_seen)),
            held_mean=float(np.mean(accs_held)), held_std=float(np.std(accs_held)),
            gap=float(np.mean(accs_seen) - np.mean(accs_held)),
        )
    # chance = majority family fraction among eval_seen
    vals, cnts = np.unique(y[ev_s], return_counts=True)
    chance = float(cnts.max() / cnts.sum())
    return results, chance, dict(n_train=int(tr.sum()), n_eval_seen=int(ev_s.sum()),
                                 n_eval_heldout=int(ev_h.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if os.path.exists(CACHE) and not args.rebuild:
        print(f"[cache] loading {CACHE}")
        d = np.load(CACHE, allow_pickle=True); data = {k: d[k] for k in d.files}
    else:
        data = build_features(args.gpu)

    results, chance, counts = evaluate(data)
    outd = f"{REPO}/results/pilot2"; os.makedirs(outd, exist_ok=True)
    S = dict(results=results, chance=chance, counts=counts, families=FAMS, held_out=HELD_OUT)
    json.dump(S, open(f"{outd}/attribution.json", "w"), indent=2)

    lines = [
        "Pilot 2 part one -- DF40 family attribution GATE",
        "=" * 64,
        f"families: {FAMS}",
        f"held-out generators (unseen, known family): {HELD_OUT}",
        f"frames: train {counts['n_train']}, eval-seen {counts['n_eval_seen']}, "
        f"eval-heldout {counts['n_eval_heldout']}",
        f"chance (majority family among eval-seen): {chance:.3f}",
        "",
        f"{'baseline':<10} {'seen acc':>16} {'held-out acc':>16} {'gap':>8}",
        "-" * 54,
    ]
    label = {"A_clip": "A CLIP", "B_dct": "B DCT", "C_count": "C pred-count", "D_pred": "D pred-MLP"}
    for k in ["A_clip", "B_dct", "C_count", "D_pred"]:
        r = results[k]
        lines.append(f"{label[k]:<10} {r['seen_mean']:.3f} +/- {r['seen_std']:.3f}   "
                     f"{r['held_mean']:.3f} +/- {r['held_std']:.3f}   {r['gap']:+.3f}")
    # gate reading
    C, B, D = results["C_count"], results["B_dct"], results["D_pred"]
    lines += ["", "Gate reading:"]
    lines.append(f"  predicate-count (C) beats raw DCT (B) on seen: "
                 f"{'YES' if C['seen_mean'] > B['seen_mean'] else 'NO'} "
                 f"({C['seen_mean']:.3f} vs {B['seen_mean']:.3f})")
    lines.append(f"  predicate model separates seen above chance: "
                 f"{'YES' if D['seen_mean'] > chance + 0.10 else 'NO'} "
                 f"({D['seen_mean']:.3f} vs chance {chance:.3f})")
    go = (C['seen_mean'] > B['seen_mean']) and (D['seen_mean'] > chance + 0.10)
    lines.append(f"  => {'GO -- run part two (verified refit).' if go else 'INSPECT -- predicate signal weak vs features; see numbers.'}")
    open(f"{outd}/attribution.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {outd}/attribution.json  {outd}/attribution.txt")


if __name__ == "__main__":
    main()
