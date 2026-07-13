"""
Pilot 1 rev3, whole_face spectral interventions (PLANNING section 4, CLAUDE_CODE_PILOTS
Pilot 1 step 9).

Frequency and noise predicates bind to whole_face and cannot be verified by spatial
repair, a global fingerprint survives local inpainting. They get predicate-targeted
spectral interventions instead. For each fake we apply three interventions to the whole
crop, re-run the frozen detector, and record the p(fake) drop. The same operations are
applied to the paired real crop to measure the offset distribution.

  notch_hf            zero a high-frequency annulus in the 2D FFT (removes the band a
                      frequency_anomaly predicate would cite), then invert.
  checkerboard_supp   detect periodic upsampling peaks in the magnitude spectrum outside
                      the low-frequency disk and attenuate them to the local median.
  residual_renorm     shrink the SRM-style high-frequency residual energy toward the
                      paired real crop's residual energy, testing noise_inconsistency.

These are predicate-level (verified individually), so predicate-level necessity is
claimed here, unlike the region-level spatial gate.

Run:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=~/.cache/huggingface \
  /data/umar/miniconda3/envs/GenD/bin/python scripts/pilot1_spectral.py --detector effort --gpu 0
"""
import os, sys, json, random, argparse
import numpy as np
import cv2

REPO = "/data/umar/Repos/DFB_NeSyNS"
sys.path.insert(0, f"{REPO}/scripts")
import intervention_pilot as P
import pilot_detectors as DET

METHODS = P.METHODS
SEED = 0


# --------------------------------------------------------------------------- spectral ops
def _fft_channels(img):
    """Return per-channel shifted FFT of a float BGR image."""
    return [np.fft.fftshift(np.fft.fft2(img[:, :, c])) for c in range(3)]


def _ifft_channels(fchs, shape):
    out = np.zeros(shape, np.float32)
    for c, F in enumerate(fchs):
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    return np.clip(out, 0, 255).astype(np.uint8)


def _radius_grid(h, w):
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return r / r.max()          # normalized 0..1


def notch_hf(img_bgr, r_lo=0.55, r_hi=0.85):
    """Zero a high-frequency annulus r_lo..r_hi (normalized radius)."""
    f = img_bgr.astype(np.float32)
    h, w = f.shape[:2]
    r = _radius_grid(h, w)
    band = (r >= r_lo) & (r <= r_hi)
    fchs = _fft_channels(f)
    for F in fchs:
        F[band] = 0.0
    return _ifft_channels(fchs, f.shape)


def checkerboard_supp(img_bgr, r_min=0.35, pct=99.5):
    """Attenuate strong isolated peaks (upsampling harmonics) outside the low-freq disk."""
    f = img_bgr.astype(np.float32)
    h, w = f.shape[:2]
    r = _radius_grid(h, w)
    outer = r >= r_min
    fchs = _fft_channels(f)
    for F in fchs:
        mag = np.abs(F)
        thr = np.percentile(mag[outer], pct)
        peaks = outer & (mag > thr)
        # replace peak magnitude with local median magnitude, keep phase
        if peaks.sum():
            med = np.median(mag[outer])
            scale = np.where(peaks, med / (mag + 1e-6), 1.0)
            F *= scale
    return _ifft_channels(fchs, f.shape)


def residual_renorm(img_bgr, ref_bgr):
    """Shrink the high-freq residual std toward the reference crop's residual std."""
    def residual(x):
        blur = cv2.GaussianBlur(x, (5, 5), 0).astype(np.float32)
        return x.astype(np.float32) - blur, blur
    res, blur = residual(img_bgr)
    ref_res, _ = residual(ref_bgr)
    s_img = res.std() + 1e-6
    s_ref = ref_res.std() + 1e-6
    factor = min(1.0, s_ref / s_img)          # only shrink, never amplify
    out = blur + res * factor
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- sample build
def build(method, vid, nnn):
    target = vid.split("_")[0]
    base = f"{P.PP}/manipulated_sequences/{method}/c23"
    fake = cv2.imread(f"{base}/frames/{vid}/{nnn}.png")
    mraw = cv2.imread(f"{base}/masks/{vid}/{nnn}.png", 0)
    lp = f"{base}/landmarks/{vid}/{nnn}.npy"
    if fake is None or mraw is None or not os.path.exists(lp):
        return None
    fg = (mraw > 127).mean()
    if not (P.FG_MIN <= fg <= P.FG_MAX):
        return None
    lm = np.load(lp)
    raw = P.read_raw_frame(f"{P.RAW}/original_sequences/youtube/c23/videos/{target}.mp4", int(nnn))
    if raw is None:
        return None
    real, _ = P.align_face(raw, lm)
    if real is None:
        return None
    bg_mse = float(((fake.astype(float) - real.astype(float))[~(mraw > 127)] ** 2).mean())
    if bg_mse > P.QC_MSE:
        return None
    fake_variants = dict(
        orig=fake,
        notch=notch_hf(fake),
        checker=checkerboard_supp(fake),
        residual=residual_renorm(fake, real),
    )
    real_variants = dict(
        real_orig=real,
        real_notch=notch_hf(real),
        real_checker=checkerboard_supp(real),
        real_residual=residual_renorm(real, fake),
    )
    return dict(method=method, vid=vid, frame=nnn, bg_mse=bg_mse), fake_variants, real_variants


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="effort", choices=DET.available())
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-per", type=int, default=50)
    args = ap.parse_args()
    outd = f"{REPO}/results/pilot_rev3/{args.detector}"
    os.makedirs(outd, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)

    print(f"[detector] loading '{args.detector}' ...")
    model, transform = DET.load_detector(args.detector, device)
    cand = P.gather_samples(rng)

    recs, imgs, keys = [], [], []
    counts = {m: 0 for m in METHODS}
    for (method, vid, nnn) in cand:
        if counts[method] >= args.n_per:
            continue
        built = build(method, vid, nnn)
        if built is None:
            continue
        rec, fv, rv = built
        sid = len(recs); rec["sid"] = sid
        for k, im in {**fv, **rv}.items():
            imgs.append(im); keys.append((sid, k))
        recs.append(rec); counts[method] += 1
        if all(counts[m] >= args.n_per for m in METHODS):
            break
    print(f"[sampling] valid: {counts}; scoring {len(imgs)} images ...")
    probs = DET.pfake_batch(model, transform, imgs, device)
    pmap = {}
    for (sid, k), p in zip(keys, probs):
        pmap.setdefault(sid, {})[k] = float(p)

    def drops(fake_key):
        return [pmap[r["sid"]]["orig"] - pmap[r["sid"]][fake_key] for r in recs]

    def offsets(real_key):
        return [abs(pmap[r["sid"]][real_key] - pmap[r["sid"]]["real_orig"]) for r in recs]

    def m(x): return float(np.mean(x))
    def md(x): return float(np.median(x))
    S = dict(detector=args.detector, n=len(recs), counts=counts,
             mean_p_orig=m([pmap[r["sid"]]["orig"] for r in recs]))
    for name, fk, rk in [("notch", "notch", "real_notch"),
                         ("checkerboard", "checker", "real_checker"),
                         ("residual_renorm", "residual", "real_residual")]:
        d, o = drops(fk), offsets(rk)
        S[name] = dict(mean_drop=m(d), median_drop=md(d),
                       mean_real_offset=m(o), median_real_offset=md(o))

    json.dump({"summary": S, "samples": recs}, open(f"{outd}/spectral.json", "w"), indent=2)
    lines = [f"Pilot 1 rev3 -- whole_face spectral interventions [{args.detector}]",
             "=" * 66,
             f"samples: {S['n']} {S['counts']}   mean p(fake) orig: {S['mean_p_orig']:.4f}", ""]
    for name in ["notch", "checkerboard", "residual_renorm"]:
        s = S[name]
        lines.append(f"  {name:16s} drop mean/med {s['mean_drop']:+.4f}/{s['median_drop']:+.4f}   "
                     f"real offset mean/med {s['mean_real_offset']:.4f}/{s['median_real_offset']:.4f}")
    open(f"{outd}/spectral.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {outd}/spectral.json  {outd}/spectral.txt")


if __name__ == "__main__":
    main()
