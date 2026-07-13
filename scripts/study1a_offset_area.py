"""
Study 1A, inpainting offset versus region area (PILOTS_2.md, runs alongside Pilot 1.5,
no gate).

Purpose. Pilot 1.5 STOPped, no whole-region pixel replacement verifies. This study finds
the operating regime where SMALL-region inpainting is still admissible, the largest region
area at which the inpainting offset on a REAL frame stays below a nuisance threshold
relative to the phase-1 GT effect. That threshold feeds Pilot 1L and 2V.

Procedure. On 50 real FF++ frames (the paired reals from phase 1), inpaint nested regions
of increasing area centered on the face, at six area levels from a single-landmark box up
to the full inner-face mask. Record p(fake) change per detector at each level. Repeat the
two smallest levels with LaMa to confirm operator independence.

Run:
  HF_HOME=~/.cache/huggingface \
  /data/umar/miniconda3/envs/GenD/bin/python scripts/study1a_offset_area.py \
      --detector effort --det-gpu 1 --sd-gpu 2
"""
import os, sys, json, random, argparse
import numpy as np
import cv2

REPO = "/data/umar/Repos/DFB_NeSyNS"
sys.path.insert(0, f"{REPO}/scripts")
import intervention_pilot as P
import pilot_detectors as DET
from pilot1_diffusion import SDInpaint, LaMaInpaint

METHODS = P.METHODS
SEED = 0
AREA_LEVELS = [0.01, 0.03, 0.08, 0.15, 0.30]     # fraction of the 256x256 crop; + full mask
CENTER = (128, 150)                               # face center on the aligned crop


def box_region(frac, shape):
    """Square region of the given area fraction, centered on the face."""
    H, W = shape[:2]
    half = int(round(0.5 * np.sqrt(frac * H * W)))
    cy, cx = CENTER
    m = np.zeros((H, W), bool)
    m[max(0, cy - half):min(H, cy + half), max(0, cx - half):min(W, cx + half)] = True
    return m


def build_real(method, vid, nnn):
    target = vid.split("_")[0]
    base = f"{P.PP}/manipulated_sequences/{method}/c23"
    fake = cv2.imread(f"{base}/frames/{vid}/{nnn}.png")
    mraw = cv2.imread(f"{base}/masks/{vid}/{nnn}.png", 0)
    lp = f"{base}/landmarks/{vid}/{nnn}.npy"
    if fake is None or mraw is None or not os.path.exists(lp):
        return None
    mask = mraw > 127
    if not (P.FG_MIN <= mask.mean() <= P.FG_MAX):
        return None
    lm = np.load(lp)
    raw = P.read_raw_frame(f"{P.RAW}/original_sequences/youtube/c23/videos/{target}.mp4", int(nnn))
    if raw is None:
        return None
    real, _ = P.align_face(raw, lm)
    if real is None:
        return None
    if float(((fake.astype(float) - real.astype(float))[~mask] ** 2).mean()) > P.QC_MSE:
        return None
    return dict(method=method, vid=vid, frame=nnn), real, mask


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="effort", choices=DET.available())
    ap.add_argument("--det-gpu", type=int, default=1)
    ap.add_argument("--sd-gpu", type=int, default=2)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    outd = f"{REPO}/results/pilot1_5"
    os.makedirs(outd, exist_ok=True)
    ddev = torch.device(f"cuda:{args.det_gpu}")
    rng = random.Random(SEED)

    print("[load] detector + SD + LaMa ...")
    model, transform = DET.load_detector(args.detector, ddev)
    sd = SDInpaint(args.sd_gpu)
    lama = LaMaInpaint(args.sd_gpu)

    cand = P.gather_samples(rng)
    reals, seen = [], 0
    counts = {m: 0 for m in METHODS}
    for (method, vid, nnn) in cand:
        if sum(counts.values()) >= args.n:
            break
        b = build_real(method, vid, nnn)
        if b is None:
            continue
        reals.append(b); counts[method] = counts.get(method, 0) + 1
    print(f"[sampling] {len(reals)} real frames {counts}")

    imgs, keys = [], []
    for i, (rec, real, mask) in enumerate(reals):
        imgs.append(real); keys.append((i, "orig"))
        for lvl in AREA_LEVELS:
            reg = box_region(lvl, real.shape)
            imgs.append(sd.repair(real, reg)); keys.append((i, f"sd_{lvl}"))
        imgs.append(sd.repair(real, mask)); keys.append((i, "sd_full"))
        # LaMa on the two smallest levels for operator independence
        for lvl in AREA_LEVELS[:2]:
            reg = box_region(lvl, real.shape)
            imgs.append(lama.repair(real, reg)); keys.append((i, f"lama_{lvl}"))
        if (i + 1) % 10 == 0:
            print(f"  built {i+1}/{len(reals)}")

    print(f"[detector] scoring {len(imgs)} images ...")
    probs = DET.pfake_batch(model, transform, imgs, ddev)
    pmap = {}
    for (i, k), p in zip(keys, probs):
        pmap.setdefault(i, {})[k] = float(p)

    def offsets(key):
        return [abs(pmap[i][key] - pmap[i]["orig"]) for i in range(len(reals)) if key in pmap[i]]

    def m(x): return float(np.mean(x))
    def md(x): return float(np.median(x))
    curve = {}
    for lvl in AREA_LEVELS:
        o = offsets(f"sd_{lvl}")
        curve[f"{lvl}"] = dict(mean=m(o), median=md(o))
    full = offsets("sd_full"); curve["full_mask"] = dict(mean=m(full), median=md(full))
    lama_curve = {f"{lvl}": dict(mean=m(offsets(f'lama_{lvl}')), median=md(offsets(f'lama_{lvl}')))
                  for lvl in AREA_LEVELS[:2]}

    # admissible area, largest level whose MEDIAN offset stays below a nuisance threshold.
    # nuisance threshold, 15 percent of the phase-1 GT effect (~0.45) = 0.067.
    NUIS = 0.067
    admissible = 0.0
    for lvl in AREA_LEVELS:
        if curve[f"{lvl}"]["median"] <= NUIS:
            admissible = lvl
    S = dict(detector=args.detector, n=len(reals), area_levels=AREA_LEVELS,
             sd_offset_curve=curve, lama_offset_curve=lama_curve,
             nuisance_threshold=NUIS, admissible_max_area=admissible)
    json.dump(S, open(f"{outd}/study1a_{args.detector}.json", "w"), indent=2)

    lines = [f"Study 1A -- inpainting offset vs region area [{args.detector}]  (n={len(reals)} real frames)",
             "=" * 72,
             f"nuisance threshold (15% of phase-1 GT effect): {NUIS:.3f}",
             "",
             f"{'area frac':>10} {'SD offset mean/med':>24} {'LaMa offset mean/med':>24}"]
    for lvl in AREA_LEVELS:
        c = curve[f"{lvl}"]
        lc = lama_curve.get(f"{lvl}")
        ls = f"{lc['mean']:.3f}/{lc['median']:.3f}" if lc else "-"
        lines.append(f"{lvl:>10} {c['mean']:.3f}/{c['median']:>10.3f}        {ls:>18}")
    lines.append(f"{'full_mask':>10} {curve['full_mask']['mean']:.3f}/{curve['full_mask']['median']:>10.3f}")
    lines += ["", f"=> largest admissible area (median offset <= {NUIS}): {admissible} "
              f"of the crop. Small-region inpainting is admissible at or below this; "
              f"inadmissible above it."]
    open(f"{outd}/study1a_{args.detector}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {outd}/study1a_{args.detector}.txt")


if __name__ == "__main__":
    main()
