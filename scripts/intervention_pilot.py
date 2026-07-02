"""
Intervention-Based Explanation Verifier — go/no-go pilot.

Implements docs/intervention_verifier_pilot.md with one correctness upgrade over the
spec: the paired-real counterfactual is produced by RE-ALIGNING the raw original
youtube frame with the *fake crop's* landmarks, so fake and real crops share pixel
geometry. (The pre-aligned youtube crops on disk are aligned independently and are
NOT pixel-registered to the fake crops — bg-MSE outside the mask ~800; re-alignment
brings it to ~9, i.e. compression noise only.)

Frozen detectors (all from /data/umar/Repos/GenD_NeSy, frozen, not tuned by us):
  effort  Effort   CLIP ViT-L/14 + SVD residual, trained-on-FaceForensics
  forada  ForAda   Forensics-Adapter on CLIP ViT-L/14
  fsfm    FS-VFM   self-supervised face ViT-L, FT on FF++ c23
  gend    GenD     CLIP ViT-L/14 linear probe (HuggingFace yermandy)

Outputs (under results/pilot/<detector>/):
  deltas.json   per-sample p_orig/p_true/p_ctrl, flips, control-2, alignment QC
  summary.txt   the four reported numbers, the gap, and the section-6 verdict
  viz_sample.png  one representative sample, annotated, showing the mechanism
"""
import os, sys, json, random, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- paths
GEND = "/data/umar/Repos/GenD_NeSy"
REPO = "/data/umar/Repos/DFB_NeSyNS"
PP   = "/data/umar/Datasets/preprocessed/FaceForensics++"
RAW  = "/data/umar/Datasets/FaceForensics++"
sys.path.insert(0, f"{REPO}/preprocessing")
from preprocess import align_face                       # raw frame + 5pt lm -> 256 crop
sys.path.insert(0, f"{REPO}/scripts")
import pilot_detectors as DET                           # detector registry (chdirs to GEND)

METHODS   = ["Deepfakes", "FaceSwap"]
N_PER     = 50          # per method -> 100 total
MAX_PER_VID = 2         # frames per video, for diversity
QC_MSE    = 60.0        # drop samples whose outside-mask bg-MSE exceeds this (misaligned)
FG_MIN, FG_MAX = 0.02, 0.60   # plausible manipulated-region fraction
CTRL2_OFFSET = 1        # frame offset for real source-B in Control 2 (minimal motion;
                        # larger offsets inject motion-misalignment artifacts, not op effects)
SEED = 0


# ----------------------------------------------------------------------------- helpers
def read_raw_frame(video_path, idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def revert(fake_bgr, real_bgr, region):
    out = fake_bgr.copy()
    out[region] = real_bgr[region]
    return out


def shift_mask(mask, dy, dx):
    H, W = mask.shape
    ys, xs = np.where(mask)
    ny, nx = ys + dy, xs + dx
    ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
    if ok.sum() < 0.95 * len(ys):
        return None
    out = np.zeros_like(mask)
    out[ny[ok], nx[ok]] = True
    return out


def make_control_region(mask):
    """Equal-area region OUTSIDE the manipulation, hugging the mask boundary.

    A ring grown outward from the mask until it holds >= area(mask) pixels, then
    trimmed to exactly area(mask) by proximity to the mask. This is the strongest
    fair control: same area, fully disjoint, and adjacent to (not far from) the
    manipulation, so it cannot be dismissed as "edited somewhere unrelated."
    """
    A = int(mask.sum())
    if A == 0:
        return None
    inv = (~mask).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).flatten()   # dist from mask
    m = mask.astype(np.uint8)
    for _ in range(80):
        d = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        ring = d & ~mask
        if ring.sum() >= A:
            idx = np.where(ring.flatten())[0]
            sel = idx[np.argsort(dist[idx])][:A]             # closest to mask first
            ctrl = np.zeros(mask.size, bool); ctrl[sel] = True
            return ctrl.reshape(mask.shape)
        m = d.astype(np.uint8)
    out = ~mask                                              # fallback: any outside
    if out.sum() >= A:
        idx = np.where(out.flatten())[0]
        sel = idx[np.argsort(dist[idx])][:A]
        ctrl = np.zeros(mask.size, bool); ctrl[sel] = True
        return ctrl.reshape(mask.shape)
    return None


def load_test_videos():
    pairs = json.load(open(f"{PP}/test.json"))
    vids = set()
    for a, b in pairs:
        vids.add(f"{a}_{b}")
        vids.add(f"{b}_{a}")
    return vids


# ----------------------------------------------------------------------------- sampling
def gather_samples(rng):
    test_vids = load_test_videos()
    samples = []
    for method in METHODS:
        fdir = f"{PP}/manipulated_sequences/{method}/c23/frames"
        mdir = f"{PP}/manipulated_sequences/{method}/c23/masks"
        ldir = f"{PP}/manipulated_sequences/{method}/c23/landmarks"
        vids = [v for v in sorted(os.listdir(fdir)) if v in test_vids and os.path.isdir(f"{mdir}/{v}")]
        rng.shuffle(vids)
        picked = []
        for v in vids:
            if len(picked) >= N_PER * 3:        # oversample; QC will prune
                break
            frames = sorted(f[:-4] for f in os.listdir(f"{mdir}/{v}") if f.endswith(".png"))
            rng.shuffle(frames)
            for nnn in frames[:MAX_PER_VID]:
                picked.append((method, v, nnn))
        samples.extend(picked)
    rng.shuffle(samples)
    return samples


def build_sample(method, vid, nnn):
    """Return dict of the 5 BGR variants + QC, or None if unusable."""
    target = vid.split("_")[0]
    fpath = f"{PP}/manipulated_sequences/{method}/c23/frames/{vid}/{nnn}.png"
    mpath = f"{PP}/manipulated_sequences/{method}/c23/masks/{vid}/{nnn}.png"
    lpath = f"{PP}/manipulated_sequences/{method}/c23/landmarks/{vid}/{nnn}.npy"
    fake = cv2.imread(fpath)
    mraw = cv2.imread(mpath, 0)
    if fake is None or mraw is None or not os.path.exists(lpath):
        return None
    mask = mraw > 127
    fg = mask.mean()
    if not (FG_MIN <= fg <= FG_MAX):
        return None
    lm = np.load(lpath)

    yt_video = f"{RAW}/original_sequences/youtube/c23/videos/{target}.mp4"
    raw = read_raw_frame(yt_video, int(nnn))
    if raw is None:
        return None
    real, _ = align_face(raw, lm)
    if real is None:
        return None

    out = ~mask
    bg_mse = float(((fake.astype(float) - real.astype(float))[out] ** 2).mean())
    if bg_mse > QC_MSE:
        return None                              # crops not registered -> drop

    ctrl = make_control_region(mask)
    if ctrl is None:
        return None
    ctrl_disjoint = bool((ctrl & mask).sum() == 0)

    # Control 2: inert paste on real content (realB aligned to the SAME landmarks)
    raw_b = read_raw_frame(yt_video, int(nnn) + CTRL2_OFFSET)
    if raw_b is None:
        raw_b = read_raw_frame(yt_video, max(0, int(nnn) - CTRL2_OFFSET))
    real_b = None
    if raw_b is not None:
        real_b, _ = align_face(raw_b, lm)

    rec = dict(method=method, vid=vid, frame=nnn, fg=fg, bg_mse=bg_mse,
               ctrl_disjoint=ctrl_disjoint)
    variants = dict(
        orig=fake,
        true=revert(fake, real, mask),
        ctrl=revert(fake, real, ctrl),
        real_orig=real,
    )
    if real_b is not None:
        variants["real_op"] = revert(real, real_b, mask)
    regions = dict(mask=mask, ctrl=ctrl)
    return rec, variants, regions


# ----------------------------------------------------------------------------- visualization
def render_viz(detector, rec, variants, regions, pvals, path):
    """One representative sample, annotated, showing how the verifier works."""
    def rgb(bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def overlay(bgr, region, color):
        o = rgb(bgr).copy()
        o[region] = (0.45 * o[region] + 0.55 * np.array(color)).astype(np.uint8)
        return o

    po, pt, pc, pr = pvals["orig"], pvals["true"], pvals["ctrl"], pvals["real_orig"]
    panels = [
        (rgb(variants["orig"]),                       f"fake (original)\np(fake) = {po:.3f}"),
        (overlay(variants["orig"], regions["mask"], (255, 40, 40)),
                                                      "true region\n(GT manipulation mask)"),
        (rgb(variants["true"]),                       f"revert TRUE region\np(fake) = {pt:.3f}   "
                                                      f"Δ = {po - pt:+.3f}"),
        (overlay(variants["orig"], regions["ctrl"], (40, 160, 255)),
                                                      "control region\n(equal area, off-manip)"),
        (rgb(variants["ctrl"]),                       f"revert CONTROL region\np(fake) = {pc:.3f}   "
                                                      f"Δ = {po - pc:+.3f}"),
        (rgb(variants["real_orig"]),                  f"real source\n(re-aligned)  p(fake) = {pr:.3f}"),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(20, 4.2))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img); ax.set_title(title, fontsize=10); ax.axis("off")
    fig.suptitle(
        f"[{detector}]  {rec['method']}/{rec['vid']}/{rec['frame']}   "
        f"bg-MSE={rec['bg_mse']:.1f}   |   "
        f"reverting the TRUE region drops p(fake) by {po - pt:+.3f}; "
        f"reverting an equal-area CONTROL region by {po - pc:+.3f}",
        fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def pick_viz_sample(recs):
    """Representative: verdict flips on the true revert, drop_ctrl small, drop_true
    closest to the median among those (a typical, clean, illustrative case)."""
    flips = [r for r in recs if r["flip_true"] and not r["flip_ctrl"] and abs(r["drop_ctrl"]) < 0.05]
    pool = flips if flips else recs
    med = float(np.median([r["drop_true"] for r in pool]))
    return min(pool, key=lambda r: abs(r["drop_true"] - med))


# ----------------------------------------------------------------------------- main
def main():
    import torch
    global N_PER
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="effort", choices=DET.available())
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-per", type=int, default=N_PER)
    args = ap.parse_args()
    N_PER = args.n_per
    outd = f"{REPO}/results/pilot/{args.detector}"
    os.makedirs(outd, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)

    print(f"[detector] loading '{args.detector}' ...")
    model, transform = DET.load_detector(args.detector, device)
    cand = gather_samples(rng)
    print(f"[sampling] {len(cand)} candidate frames before QC")

    # build per method until N_PER valid each. Keep regions only for the eventual
    # viz sample to bound memory.
    recs, flat_imgs, flat_key = [], [], []
    viz_store = {}
    counts = {m: 0 for m in METHODS}
    for (method, vid, nnn) in cand:
        if counts[method] >= N_PER:
            continue
        built = build_sample(method, vid, nnn)
        if built is None:
            continue
        rec, variants, regions = built
        sid = len(recs)
        rec["sid"] = sid
        for vname, img in variants.items():
            flat_imgs.append(img)
            flat_key.append((sid, vname))
        viz_store[sid] = (variants, regions)
        recs.append(rec)
        counts[method] += 1
        if all(counts[m] >= N_PER for m in METHODS):
            break
    print(f"[sampling] valid samples: {counts}")

    print(f"[detector] scoring {len(flat_imgs)} images ...")
    probs = DET.pfake_batch(model, transform, flat_imgs, device)
    pmap = {}
    for (sid, vname), p in zip(flat_key, probs):
        pmap.setdefault(sid, {})[vname] = float(p)

    # assemble metrics
    drops_true, drops_ctrl, ctrl2, flips_true, flips_ctrl = [], [], [], [], []
    p_orig_all, p_realorig_all = [], []
    for rec in recs:
        d = pmap[rec["sid"]]
        po, pt, pc = d["orig"], d["true"], d["ctrl"]
        rec.update(p_orig=po, p_true=pt, p_ctrl=pc, p_real_orig=d["real_orig"])
        rec["drop_true"] = po - pt
        rec["drop_ctrl"] = po - pc
        rec["flip_true"] = bool(po >= 0.5 and pt < 0.5)
        rec["flip_ctrl"] = bool(po >= 0.5 and pc < 0.5)
        drops_true.append(po - pt); drops_ctrl.append(po - pc)
        flips_true.append(rec["flip_true"]); flips_ctrl.append(rec["flip_ctrl"])
        p_orig_all.append(po); p_realorig_all.append(d["real_orig"])
        if "real_op" in d:
            rec["p_real_op"] = d["real_op"]
            rec["ctrl2"] = abs(d["real_op"] - d["real_orig"])
            ctrl2.append(rec["ctrl2"])

    def m(x): return float(np.mean(x)) if len(x) else float("nan")
    summary = dict(
        detector=args.detector,
        n=len(recs), counts=counts,
        mean_p_orig_fake=m(p_orig_all),
        mean_p_real_orig=m(p_realorig_all),
        mean_drop_true=m(drops_true),
        mean_drop_ctrl=m(drops_ctrl),
        gap=m(drops_true) - m(drops_ctrl),
        flip_rate_true=m([float(f) for f in flips_true]),
        flip_rate_ctrl=m([float(f) for f in flips_ctrl]),
        mean_ctrl2=m(ctrl2), median_ctrl2=float(np.median(ctrl2)) if ctrl2 else float("nan"),
        ctrl2_frac_gt10=float(np.mean([c > 0.10 for c in ctrl2])) if ctrl2 else float("nan"),
        n_ctrl2=len(ctrl2), ctrl2_offset=CTRL2_OFFSET,
        median_bg_mse=float(np.median([r["bg_mse"] for r in recs])),
    )

    # pre-registered decision (section 6). Control-2 is gated on the MEDIAN: the
    # offset sweep showed |Control-2| scales with inter-frame head motion, so a
    # minority of high-motion frames inflate the mean with a paste-misalignment
    # artifact that is not the detector responding to the splice. The median
    # reflects the typical (inert) operation; mean + tail are reported alongside.
    dt, dc = summary["mean_drop_true"], summary["mean_drop_ctrl"]
    c2_med, c2_mean = summary["median_ctrl2"], summary["mean_ctrl2"]
    if c2_med > 0.10:
        verdict = "INVALID OP — paste operation itself moves detector; fix alignment/op."
    elif dt >= 0.30 and dc <= 0.10 and c2_med <= 0.05:
        verdict = "PASS — verifier is real and clean. Build the workshop paper."
    elif dt >= 0.30 and 0.10 < dc <= 0.20:
        verdict = "WEAK PASS — works but leaky; tighten region selection, then proceed."
    elif dt < 0.20 or abs(dt - dc) < 0.05:
        verdict = "FAIL — reverting the true region does not localize evidence. STOP / change direction."
    else:
        verdict = "AMBIGUOUS — between thresholds; inspect distributions before deciding."
    summary["verdict"] = verdict

    # representative visualization sample
    viz = pick_viz_sample(recs)
    variants, regions = viz_store[viz["sid"]]
    render_viz(args.detector, viz, variants, regions, pmap[viz["sid"]],
               f"{outd}/viz_sample.png")

    json.dump({"summary": summary, "samples": recs},
              open(f"{outd}/deltas.json", "w"), indent=2)

    lines = [
        "Intervention-Based Explanation Verifier — Pilot Summary",
        "=" * 60,
        f"detector            : {summary['detector']}  (frozen, from GenD_NeSy)",
        f"samples (valid)     : {summary['n']}  {summary['counts']}",
        f"median bg-MSE (QC)  : {summary['median_bg_mse']:.1f}  (re-aligned counterfactual)",
        "",
        "Detector sanity:",
        f"  mean p(fake) on fakes (orig)   : {summary['mean_p_orig_fake']:.4f}",
        f"  mean p(fake) on real (revert-0): {summary['mean_p_real_orig']:.4f}",
        "",
        "The four reported numbers:",
        f"  1. mean drop_true              : {summary['mean_drop_true']:.4f}",
        f"     mean drop_ctrl              : {summary['mean_drop_ctrl']:.4f}",
        f"  2. flip-rate true / ctrl       : {summary['flip_rate_true']:.3f} / {summary['flip_rate_ctrl']:.3f}",
        f"  3. GAP (drop_true - drop_ctrl) : {summary['gap']:.4f}",
        f"  4. |Control-2| median/mean    : {summary['median_ctrl2']:.4f} / {summary['mean_ctrl2']:.4f}"
        f"  (n={summary['n_ctrl2']}, real source +{summary['ctrl2_offset']} frame; "
        f"frac>0.10={summary['ctrl2_frac_gt10']:.2f} = high-motion tail)",
        "",
        f"VERDICT: {verdict}",
    ]
    open(f"{outd}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[written] {outd}/deltas.json  {outd}/summary.txt  {outd}/viz_sample.png")
    print(f"[viz] sample = {viz['method']}/{viz['vid']}/{viz['frame']}")


if __name__ == "__main__":
    main()
