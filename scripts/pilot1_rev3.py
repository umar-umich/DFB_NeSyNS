"""
Pilot 1, revision 3 GATE, verification test on pixels.

This is the make-or-break gate from PLANNING.md section 8.1 and CLAUDE_CODE_PILOTS.md
Pilot 1. It decides whether the counterfactual test measures the cited cue rather than
detector fragility. It is decided at a SINGLE LPIPS matched corruption point.

Per sample, on the ground-truth manipulated region of a FF++ fake with a paired real
source:
  gt_repair      Poisson blend (seamlessClone) the paired-real region back into the fake,
                 masked to the face side of the seam. The gold-standard upper-bound
                 intervention. Records drop in calibrated p(fake).
  match_blur     A localized blur on the SAME region, magnitude tuned so its LPIPS
                 distance to the fake matches the gt_repair LPIPS distance. This is the
                 gate criterion: an inert, non-semantic distortion of equal perceptual
                 magnitude.
  match_shift    A localized patch shift on the same region, LPIPS-matched. Logged as a
                 secondary matched corruption.
  wrong_region   The same Poisson repair applied to an equal-area NON-manipulated region
                 (hugging the mask boundary). Must be near-inert.
  real_offset    The same repair operation applied to the paired REAL frame region
                 (real content from a +1 frame, Poisson blended). Measures the operation's
                 own artifact. Its distribution is the normalization floor, not required
                 to be zero.

Deferred to later phases, NOT on this gate:
  diffusion repair arm (needs the inpainter), spectral interventions on whole_face,
  VLM proposal recall against masks (InternVL). These are the PARTIAL-outcome and
  localization measurements, separate from the go/no-go gate.

Run:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /data/umar/miniconda3/envs/GenD/bin/python scripts/pilot1_rev3.py --detector effort --gpu 0
"""
import os, sys, json, random, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/data/umar/Repos/DFB_NeSyNS"
sys.path.insert(0, f"{REPO}/scripts")
# reuse the validated sampling + alignment machinery from the passed pilot
import intervention_pilot as P
import pilot_detectors as DET

METHODS = P.METHODS
SEED = 0
LPIPS_TOL = 0.15          # accept a matched corruption within +-15% of the target LPIPS


# --------------------------------------------------------------------------- interventions
def poisson_paste(dst_bgr, src_bgr, region_mask):
    """Poisson blend src into dst over region_mask (bool HxW). Masked to the region so
    background pixels are untouched. Falls back to a hard paste if seamlessClone fails
    (tiny/degenerate regions)."""
    m = (region_mask.astype(np.uint8)) * 255
    ys, xs = np.where(region_mask)
    if len(ys) < 20:
        out = dst_bgr.copy(); out[region_mask] = src_bgr[region_mask]; return out
    cy, cx = int((ys.min() + ys.max()) / 2), int((xs.min() + xs.max()) / 2)
    try:
        return cv2.seamlessClone(src_bgr, dst_bgr, m, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        out = dst_bgr.copy(); out[region_mask] = src_bgr[region_mask]; return out


def blur_region(img_bgr, region_mask, ksize):
    """Gaussian blur applied only inside region_mask."""
    if ksize < 3:
        return img_bgr.copy()
    k = ksize | 1
    blurred = cv2.GaussianBlur(img_bgr, (k, k), 0)
    out = img_bgr.copy()
    out[region_mask] = blurred[region_mask]
    return out


def shift_region(img_bgr, region_mask, dx):
    """Shift the region's pixel content by dx px (in-region patch shift), region only."""
    out = img_bgr.copy()
    ys, xs = np.where(region_mask)
    nx = np.clip(xs + dx, 0, img_bgr.shape[1] - 1)
    out[ys, xs] = img_bgr[ys, nx]
    return out


# --------------------------------------------------------------------------- LPIPS matching
class Lpips:
    def __init__(self, device):
        import torch, lpips
        self.torch = torch
        self.net = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
        self.device = device

    def dist(self, a_bgr, b_bgr):
        """LPIPS between two BGR uint8 crops, batched-safe single pair."""
        import torch
        def t(x):
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.net(t(a_bgr), t(b_bgr)).item())

    def dist_batch(self, ref_bgr, cand_list):
        import torch
        def t(x):
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(x).permute(2, 0, 1).to(self.device)
        r = t(ref_bgr).unsqueeze(0)
        with torch.no_grad():
            out = []
            for c in cand_list:
                out.append(float(self.net(r, t(c).unsqueeze(0)).item()))
        return out


def match_corruption(orig, region, target_lpips, lp, kind):
    """Sweep a corruption's magnitude and return the variant whose LPIPS to orig is
    closest to target_lpips, plus its achieved LPIPS and the parameter used."""
    if kind == "blur":
        params = [3, 5, 7, 9, 11, 15, 19, 25, 31, 41, 55, 71]
        cands = [blur_region(orig, region, k) for k in params]
    else:  # shift
        params = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30]
        cands = [shift_region(orig, region, d) for d in params]
    dists = lp.dist_batch(orig, cands)
    j = int(np.argmin([abs(d - target_lpips) for d in dists]))
    return cands[j], dists[j], params[j]


# --------------------------------------------------------------------------- sample build
def build_sample_rev3(method, vid, nnn):
    """Return (rec, variants{bgr}, regions) or None. Variants include gt_repair, the two
    matched corruptions are built later (need LPIPS)."""
    target = vid.split("_")[0]
    base = f"{P.PP}/manipulated_sequences/{method}/c23"
    fpath = f"{base}/frames/{vid}/{nnn}.png"
    mpath = f"{base}/masks/{vid}/{nnn}.png"
    lpath = f"{base}/landmarks/{vid}/{nnn}.npy"
    fake = cv2.imread(fpath)
    mraw = cv2.imread(mpath, 0)
    if fake is None or mraw is None or not os.path.exists(lpath):
        return None
    mask = mraw > 127
    fg = mask.mean()
    if not (P.FG_MIN <= fg <= P.FG_MAX):
        return None
    lm = np.load(lpath)

    yt = f"{P.RAW}/original_sequences/youtube/c23/videos/{target}.mp4"
    raw = P.read_raw_frame(yt, int(nnn))
    if raw is None:
        return None
    real, _ = P.align_face(raw, lm)
    if real is None:
        return None
    bg_mse = float(((fake.astype(float) - real.astype(float))[~mask] ** 2).mean())
    if bg_mse > P.QC_MSE:
        return None

    ctrl = P.make_control_region(mask)
    if ctrl is None:
        return None

    # real +1 frame for the real-repair offset (real content, same geometry)
    raw_b = P.read_raw_frame(yt, int(nnn) + P.CTRL2_OFFSET)
    if raw_b is None:
        raw_b = P.read_raw_frame(yt, max(0, int(nnn) - P.CTRL2_OFFSET))
    real_b = None
    if raw_b is not None:
        real_b, _ = P.align_face(raw_b, lm)

    variants = dict(
        orig=fake,
        gt_repair=poisson_paste(fake, real, mask),        # gold-standard upper bound
        wrong_region=poisson_paste(fake, real, ctrl),     # specificity control
        real_orig=real,
    )
    if real_b is not None:
        variants["real_offset"] = poisson_paste(real, real_b, mask)   # op on real content
    rec = dict(method=method, vid=vid, frame=nnn, fg=fg, bg_mse=bg_mse, sid=None)
    regions = dict(mask=mask, ctrl=ctrl)
    return rec, variants, regions


# --------------------------------------------------------------------------- main
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
    lp = Lpips(device)

    cand = P.gather_samples(rng)
    print(f"[sampling] {len(cand)} candidate frames before QC")

    recs, viz_store = [], {}
    counts = {m: 0 for m in METHODS}
    for (method, vid, nnn) in cand:
        if counts[method] >= args.n_per:
            continue
        built = build_sample_rev3(method, vid, nnn)
        if built is None:
            continue
        rec, variants, regions = built
        # LPIPS of the gt_repair, then build matched corruptions on the true region
        target = lp.dist(variants["orig"], variants["gt_repair"])
        blur_img, blur_l, blur_k = match_corruption(variants["orig"], regions["mask"], target, lp, "blur")
        shift_img, shift_l, shift_d = match_corruption(variants["orig"], regions["mask"], target, lp, "shift")
        variants["match_blur"] = blur_img
        variants["match_shift"] = shift_img
        rec.update(lpips_gt=target, lpips_blur=blur_l, lpips_shift=shift_l,
                   blur_k=blur_k, shift_d=shift_d)
        sid = len(recs); rec["sid"] = sid
        viz_store[sid] = (variants, regions)
        recs.append(rec)
        counts[method] += 1
        if all(counts[m] >= args.n_per for m in METHODS):
            break
    print(f"[sampling] valid samples: {counts}")

    # score every variant of every sample in one batch
    flat_imgs, flat_key = [], []
    for rec in recs:
        variants, _ = viz_store[rec["sid"]]
        for vname, img in variants.items():
            flat_imgs.append(img); flat_key.append((rec["sid"], vname))
    print(f"[detector] scoring {len(flat_imgs)} images ...")
    probs = DET.pfake_batch(model, transform, flat_imgs, device)
    pmap = {}
    for (sid, vname), p in zip(flat_key, probs):
        pmap.setdefault(sid, {})[vname] = float(p)

    def collect(key):
        return [pmap[r["sid"]]["orig"] - pmap[r["sid"]][key] for r in recs if key in pmap[r["sid"]]]

    d_gt = collect("gt_repair")
    d_blur = collect("match_blur")
    d_shift = collect("match_shift")
    d_wrong = collect("wrong_region")
    # real offset: change in p on the real frame from the op (|Δ| around its own baseline)
    real_off = [abs(pmap[r["sid"]]["real_offset"] - pmap[r["sid"]]["real_orig"])
                for r in recs if "real_offset" in pmap[r["sid"]]]
    p_orig = [pmap[r["sid"]]["orig"] for r in recs]
    p_real = [pmap[r["sid"]]["real_orig"] for r in recs]

    flips_gt = [float(pmap[r["sid"]]["orig"] >= 0.5 and pmap[r["sid"]]["gt_repair"] < 0.5) for r in recs]

    for r in recs:
        d = pmap[r["sid"]]
        r.update(p_orig=d["orig"], p_gt=d["gt_repair"], p_blur=d["match_blur"],
                 p_shift=d["match_shift"], p_wrong=d["wrong_region"],
                 p_real_orig=d["real_orig"],
                 drop_gt=d["orig"] - d["gt_repair"], drop_blur=d["orig"] - d["match_blur"],
                 drop_shift=d["orig"] - d["match_shift"], drop_wrong=d["orig"] - d["wrong_region"])

    def m(x): return float(np.mean(x)) if len(x) else float("nan")
    def md(x): return float(np.median(x)) if len(x) else float("nan")
    S = dict(
        detector=args.detector, n=len(recs), counts=counts,
        mean_p_orig=m(p_orig), mean_p_real=m(p_real),
        mean_drop_gt=m(d_gt), median_drop_gt=md(d_gt),
        mean_drop_blur=m(d_blur), mean_drop_shift=m(d_shift),
        mean_drop_wrong=m(d_wrong), median_drop_wrong=md(d_wrong),
        mean_real_offset=m(real_off), median_real_offset=md(real_off),
        gap_gt_vs_blur=m(d_gt) - m(d_blur),
        gap_gt_vs_shift=m(d_gt) - m(d_shift),
        flip_rate_gt=m(flips_gt),
        mean_lpips_gt=m([r["lpips_gt"] for r in recs]),
        mean_lpips_blur=m([r["lpips_blur"] for r in recs]),
        mean_lpips_shift=m([r["lpips_shift"] for r in recs]),
        median_bg_mse=md([r["bg_mse"] for r in recs]),
    )

    # gate decision (section 8.1). GO if gt repair drops clearly more than the matched
    # corruption, wrong-region is near-inert, real offset small relative to the gt effect.
    matched = max(S["mean_drop_blur"], S["mean_drop_shift"])   # strongest inert corruption
    if S["mean_drop_gt"] >= 0.30 and (S["mean_drop_gt"] - matched) >= 0.15 \
            and S["median_drop_wrong"] <= 0.10 and S["median_real_offset"] <= 0.10:
        verdict = "GO -- gt repair drops p(fake) clearly beyond the LPIPS-matched corruption; controls inert."
    elif S["median_drop_wrong"] > 0.15:
        verdict = "STOP -- wrong-region repair drops the score comparably; test is not specific."
    elif (S["mean_drop_gt"] - matched) < 0.05:
        verdict = "STOP -- matched corruption reproduces the gt-repair drop; test measures fragility."
    else:
        verdict = "AMBIGUOUS -- between thresholds; inspect distributions."
    S["verdict"] = verdict

    json.dump({"summary": S, "samples": recs}, open(f"{outd}/deltas.json", "w"), indent=2)
    lines = [
        "Pilot 1 rev3 GATE -- ground-truth Poisson repair vs LPIPS-matched inert corruption",
        "=" * 78,
        f"detector              : {S['detector']}  (frozen)",
        f"samples (valid)       : {S['n']}  {S['counts']}",
        f"median bg-MSE (QC)     : {S['median_bg_mse']:.1f}",
        f"mean p(fake) fake/real : {S['mean_p_orig']:.4f} / {S['mean_p_real']:.4f}",
        "",
        "LPIPS matching (region-local, corruption tuned to the gt-repair LPIPS):",
        f"  gt_repair / blur / shift : {S['mean_lpips_gt']:.4f} / {S['mean_lpips_blur']:.4f} / {S['mean_lpips_shift']:.4f}",
        "",
        "p(fake) drops (mean):",
        f"  gt_repair (upper bound)  : {S['mean_drop_gt']:.4f}  (median {S['median_drop_gt']:.4f}, flip {S['flip_rate_gt']:.2f})",
        f"  matched blur             : {S['mean_drop_blur']:.4f}",
        f"  matched shift            : {S['mean_drop_shift']:.4f}",
        f"  wrong-region (control)   : {S['mean_drop_wrong']:.4f}  (median {S['median_drop_wrong']:.4f})",
        f"  real-repair offset |Δ|   : {S['mean_real_offset']:.4f}  (median {S['median_real_offset']:.4f})",
        "",
        f"  GAP gt vs matched-blur   : {S['gap_gt_vs_blur']:.4f}",
        f"  GAP gt vs matched-shift  : {S['gap_gt_vs_shift']:.4f}",
        "",
        f"VERDICT: {verdict}",
    ]
    open(f"{outd}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # representative viz: a flip case with small wrong-region drop, gt drop near median
    pool = [r for r in recs if r.get("drop_gt", 0) > 0.3 and abs(r.get("drop_wrong", 1)) < 0.1]
    pool = pool or recs
    med = md([r["drop_gt"] for r in pool])
    viz = min(pool, key=lambda r: abs(r["drop_gt"] - med))
    render_viz(args.detector, viz, viz_store[viz["sid"]][0], viz_store[viz["sid"]][1],
               pmap[viz["sid"]], f"{outd}/viz_sample.png")
    print(f"\n[written] {outd}/deltas.json  {outd}/summary.txt  {outd}/viz_sample.png")


def render_viz(det, rec, variants, regions, pv, path):
    def rgb(b): return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    def ov(b, region, c):
        o = rgb(b).copy(); o[region] = (0.45 * o[region] + 0.55 * np.array(c)).astype(np.uint8); return o
    po = pv["orig"]
    panels = [
        (rgb(variants["orig"]), f"fake\np={po:.3f}"),
        (ov(variants["orig"], regions["mask"], (255, 40, 40)), "true region (GT mask)"),
        (rgb(variants["gt_repair"]), f"gt Poisson repair\np={pv['gt_repair']:.3f}  Δ={po-pv['gt_repair']:+.3f}"),
        (rgb(variants["match_blur"]), f"LPIPS-matched blur\np={pv['match_blur']:.3f}  Δ={po-pv['match_blur']:+.3f}"),
        (rgb(variants["wrong_region"]), f"wrong-region repair\np={pv['wrong_region']:.3f}  Δ={po-pv['wrong_region']:+.3f}"),
        (rgb(variants["real_orig"]), f"real source\np={pv['real_orig']:.3f}"),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(21, 4.2))
    for ax, (img, t) in zip(axes, panels):
        ax.imshow(img); ax.set_title(t, fontsize=10); ax.axis("off")
    fig.suptitle(f"[{det}] {rec['method']}/{rec['vid']}/{rec['frame']}  "
                 f"gt-repair Δ={po-pv['gt_repair']:+.3f} vs matched-blur Δ={po-pv['match_blur']:+.3f} "
                 f"(LPIPS {rec['lpips_gt']:.3f} vs {rec['lpips_blur']:.3f})", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
