"""
Study 1B, compression sensitivity of spectral evidence on DF40 (PILOTS_2.md, no gate).

Disambiguates the phase-1 FF++ spectral null, was it compression that destroyed the
fingerprint, or do blending-based manipulations carry no global spectral fingerprint at
all. DF40 is the better testbed since its GAN and diffusion families are where global
spectral fingerprints are expected.

Three fake groups, GAN synthesis, diffusion, blending face_swap, plus FFHQ reals as the
real class (DF40 ships no reals, FFHQ is out-of-domain, stated as a caveat). A
self-controlled JPEG compression ladder, native, q90 (c23-equivalent), q50
(c40-equivalent). At each level, spectral intervention drops on fakes (effort detector),
and per-feature DISCERN spectral/noise separability AUC, real vs fake.

The sharpest single result, the family contrast, if the blending swap group shows near
zero drops at native while GAN and diffusion show real drops at the same quality, the
manipulation-type explanation for the phase-1 FF++ null is confirmed without raw FF++.

Run:
  HF_HOME=~/.cache/huggingface \
  /data/umar/miniconda3/envs/GenD/bin/python scripts/study1b_compression.py \
      --detector effort --gpu 1
"""
import os, sys, json, random, argparse, glob
import numpy as np
import cv2

REPO = "/data/umar/Repos/DFB_NeSyNS"
DF = "/data/umar/Datasets/df40"
FFHQ = "/data/umar/Datasets/ffhq256_subset"
sys.path.insert(0, f"{REPO}/scripts")
sys.path.insert(0, f"{REPO}/preprocessing")
import pilot_detectors as DET
from pilot1_spectral import notch_hf, checkerboard_supp, residual_renorm
import forensic_helpers as FH

SEED = 0
GROUPS = {
    "GAN":       ["StyleGAN2", "StyleGAN3", "StyleGANXL"],
    "diffusion": ["ddim", "sd2.1", "DiT"],
    "swap":      ["simswap", "inswap", "blendface"],
}
N_PER_GROUP = 60
LADDER = {"native": None, "q90": 90, "q50": 50}


def list_df40(gen):
    fs = glob.glob(f"{DF}/test/{gen}/**/*.png", recursive=True) + \
         glob.glob(f"{DF}/test/{gen}/**/*.jpg", recursive=True)
    return sorted(fs)


def jpeg_recode(bgr, q):
    if q is None:
        return bgr
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else bgr


def forensic_vec(bgr):
    """Parsing-free DISCERN spectral+noise bank, per-group concatenation with labels."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    parts, names = [], []
    for fn, nm in [(lambda: FH.extract_spectral(gray), "spectral"),
                   (lambda: FH.extract_srm(gray), "srm"),
                   (lambda: FH.extract_ppnc(gray), "ppnc"),
                   (lambda: FH.extract_ccnc(bgr), "ccnc"),
                   (lambda: FH.extract_multiscale_noise(gray), "multiscale")]:
        v = np.asarray(fn(), np.float32).ravel()
        parts.append(v); names += [f"{nm}_{i}" for i in range(len(v))]
    return np.concatenate(parts), names


def main():
    import torch
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="effort", choices=DET.available())
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    outd = f"{REPO}/results/study1b"; os.makedirs(outd, exist_ok=True)
    ddev = torch.device(f"cuda:{args.gpu}")
    rng = random.Random(SEED)

    print("[load] detector ...")
    model, transform = DET.load_detector(args.detector, ddev)

    # sample fakes per group + reals
    fakes = {g: [] for g in GROUPS}
    for g, gens in GROUPS.items():
        pool = []
        for gen in gens:
            pool += [(gen, f) for f in list_df40(gen)]
        rng.shuffle(pool)
        for gen, f in pool[:N_PER_GROUP]:
            im = cv2.imread(f)
            if im is not None:
                fakes[g].append(cv2.resize(im, (256, 256)))
        print(f"[data] {g}: {len(fakes[g])} frames")
    reals = []
    for f in sorted(glob.glob(f"{FFHQ}/*.png"))[:N_PER_GROUP]:
        im = cv2.imread(f)
        if im is not None:
            reals.append(cv2.resize(im, (256, 256)))
    print(f"[data] reals(FFHQ): {len(reals)}")

    results = {"spectral_drops": {}, "feature_auc": {}, "detector": args.detector}

    for level, q in LADDER.items():
        # ---- spectral intervention drops per group (effort) ----
        results["spectral_drops"][level] = {}
        for g in GROUPS:
            imgs, keys = [], []
            for i, im in enumerate(fakes[g]):
                r = jpeg_recode(im, q)
                imgs += [r, notch_hf(r), checkerboard_supp(r), residual_renorm(r, r)]
                keys += [(i, "o"), (i, "n"), (i, "c"), (i, "res")]
            probs = DET.pfake_batch(model, transform, imgs, ddev)
            pm = {}
            for (i, k), p in zip(keys, probs):
                pm.setdefault(i, {})[k] = float(p)
            def dr(k): return float(np.median([pm[i]["o"] - pm[i][k] for i in pm]))
            def mn(k): return float(np.mean([pm[i]["o"] - pm[i][k] for i in pm]))
            results["spectral_drops"][level][g] = dict(
                p_fake=float(np.mean([pm[i]["o"] for i in pm])),
                notch_med=dr("n"), checker_med=dr("c"), residual_med=dr("res"),
                notch_mean=mn("n"), checker_mean=mn("c"))

        # ---- per-feature DISCERN AUC, real vs fake, per group ----
        real_feats = np.stack([forensic_vec(jpeg_recode(im, q))[0] for im in reals])
        names = forensic_vec(reals[0])[1]
        results["feature_auc"][level] = {}
        for g in GROUPS:
            fake_feats = np.stack([forensic_vec(jpeg_recode(im, q))[0] for im in fakes[g]])
            X = np.vstack([real_feats, fake_feats])
            y = np.array([0] * len(real_feats) + [1] * len(fake_feats))
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            # per feature-group mean |AUC-0.5|, summarizing survivability
            group_auc = {}
            for grp in ["spectral", "srm", "ppnc", "ccnc", "multiscale"]:
                idx = [j for j, nm in enumerate(names) if nm.startswith(grp)]
                aucs = []
                for j in idx:
                    col = X[:, j]
                    if np.std(col) < 1e-9:
                        continue
                    a = roc_auc_score(y, col)
                    aucs.append(max(a, 1 - a))       # separability regardless of sign
                group_auc[grp] = float(np.mean(aucs)) if aucs else float("nan")
            results["feature_auc"][level][g] = group_auc

    json.dump(results, open(f"{outd}/study1b_{args.detector}.json", "w"), indent=2)

    lines = [f"Study 1B -- DF40 compression ladder, spectral evidence [{args.detector}]",
             "=" * 74,
             f"groups: GAN{GROUPS['GAN']} diffusion{GROUPS['diffusion']} swap{GROUPS['swap']}",
             f"reals: FFHQ (out-of-domain, caveat). {N_PER_GROUP}/group. ladder: native, q90, q50",
             "",
             "SPECTRAL INTERVENTION DROPS (median p(fake) drop on fakes):",
             f"{'level':>7} {'group':>10} {'p(fake)':>8} {'notch':>8} {'checker':>8} {'residual':>9}"]
    for level in LADDER:
        for g in GROUPS:
            s = results["spectral_drops"][level][g]
            lines.append(f"{level:>7} {g:>10} {s['p_fake']:>8.3f} {s['notch_med']:>8.4f} "
                         f"{s['checker_med']:>8.4f} {s['residual_med']:>9.4f}")
    lines += ["", "DISCERN FEATURE SEPARABILITY (mean |AUC-.5|+.5, real vs fake, by group):",
              f"{'level':>7} {'group':>10} {'spectral':>9} {'srm':>7} {'ppnc':>7} {'ccnc':>7} {'multisc':>8}"]
    for level in LADDER:
        for g in GROUPS:
            a = results["feature_auc"][level][g]
            lines.append(f"{level:>7} {g:>10} {a['spectral']:>9.3f} {a['srm']:>7.3f} "
                         f"{a['ppnc']:>7.3f} {a['ccnc']:>7.3f} {a['multiscale']:>8.3f}")
    # family contrast at native
    nat = results["spectral_drops"]["native"]
    lines += ["", "FAMILY CONTRAST at native quality (the sharpest single result):",
              f"  GAN checker drop {nat['GAN']['checker_med']:+.4f} | "
              f"diffusion {nat['diffusion']['checker_med']:+.4f} | "
              f"swap {nat['swap']['checker_med']:+.4f}",
              "  If swap ~0 while GAN/diffusion show real drops, the phase-1 FF++ null is "
              "manipulation-type, not compression."]
    open(f"{outd}/study1b_{args.detector}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {outd}/study1b_{args.detector}.txt")


if __name__ == "__main__":
    main()
