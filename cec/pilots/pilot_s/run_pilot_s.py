"""PILOT S — the make-or-break gate for the dual-instrument design.

Two questions, per Implementation v2 Task 5:
  (i)  Does a frequency detector SEPARATE on our crops?  -> AUC(fake vs real).
  (ii) Does it RESPOND to spectral interventions where CLIP detectors don't?
       -> mean Δp(fake) under notch / checkerboard / residual, for the freq
          detectors (should be ≫ 0) vs a CLIP detector (should be ≈ 0).

Datasets (Umar, both): FF++ Deepfakes+FaceSwap (paired reals via alignment) and
DF40 generative families (StyleGAN/diffusion, their own reals). FF++ is the
contrast where CLIP separates but spectral may not; DF40 is where the frequency
fingerprint lives.

Detectors: effort (CLIP-family spatial, the "CLIP ≈ 0" reference) + freqnet + npr
(the two spectral candidates).

Gate (pre-committed): pass -> dual-instrument adopted; fail -> fallback (single
instrument + spectral characterization, or DISCERN spectral-bank probe).

NOTE: writes results, does not decide. The gate call is Umar's on the numbers.

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/umar/miniconda3/envs/GenD/bin/python cec/pilots/pilot_s/run_pilot_s.py --n 100
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from cec.data.df40 import (  # noqa: E402
    GENERATIVE_FAMILIES, family_native_resolution, load_samples,
)
from cec.data.pairing import build_pair, fake_dir  # noqa: E402
from cec.instruments.spatial import SpatialInstrument  # noqa: E402
from cec.instruments.spectral import SpectralInstrument  # noqa: E402

SEED = 0
INTERVENTIONS = ["spectral_notch", "checkerboard_suppress", "residual_renormalize"]
OUT = REPO / "results" / "pilot_s"


# ------------------------------------------------------------------ sample gathering
def gather_ffpp(n, rng):
    """n fakes + their paired reals from FF++ (Deepfakes+FaceSwap, test split)."""
    from cec.data.pairing import PP
    test_pairs = json.load(open(PP / "test.json"))
    test_vids = set()
    for a, b in test_pairs:
        test_vids.add(f"{a}_{b}")
        test_vids.add(f"{b}_{a}")

    fakes, reals = [], []
    per = {"Deepfakes": 0, "FaceSwap": 0}
    target = n // 2
    for method in ("Deepfakes", "FaceSwap"):
        fdir = fake_dir(method) / "frames"
        vids = [v.name for v in fdir.iterdir() if v.name in test_vids]
        rng.shuffle(vids)
        for vid in vids:
            if per[method] >= target:
                break
            frames = sorted(p.stem for p in (fdir / vid).glob("*.png"))
            rng.shuffle(frames)
            for frame in frames[:1]:
                pair = build_pair(method, vid, frame)
                if pair is not None:
                    fakes.append(pair.fake)
                    reals.append(pair.real)
                    per[method] += 1
                    break
    return fakes, reals


def gather_df40(n, rng):
    """n//2 fakes + n//2 reals per generative family.

    Returns {family: (fakes, reals, native_resolution)}. native_resolution flags
    whether fake and real share native resolution — an unmatched family has a
    resize confound and its AUC is not a clean spectral number.
    """
    per_family = max(1, (n // 2) // len(GENERATIVE_FAMILIES))
    out = {}
    for fam in GENERATIVE_FAMILIES:
        samples = load_samples(fam, per_family, rng)
        fakes = [s.image for s in samples if s.label == 1]
        reals = [s.image for s in samples if s.label == 0]
        out[fam] = (fakes, reals, family_native_resolution(samples))
    return out


# ------------------------------------------------------------------ scoring
def score(instrument, fakes, reals):
    """AUC(fake vs real) + mean p on each class."""
    pf = instrument.p_fake_batch(fakes)
    pr = instrument.p_fake_batch(reals)
    labels = np.r_[np.ones(len(pf)), np.zeros(len(pr))]
    scores = np.r_[pf, pr]
    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")
    return auc, float(pf.mean()), float(pr.mean())


def intervention_response(instrument, fakes, is_spectral):
    """Mean Δp(fake) per intervention. Spatial instruments have no spectral ops,
    so we route the SAME crops through the spectral op battery regardless — a
    CLIP detector SHOULD be ≈ 0, which is the point of arm (ii)."""
    from networks.cec.spectral_ops import INTERVENTIONS as OPS  # noqa: E402
    out = {}
    for it in INTERVENTIONS:
        intervened = []
        for f in fakes:
            # residual needs a reference; use the fake's own blurred self as a
            # neutral reference so the op is defined without a paired real here.
            if it == "residual_renormalize":
                intervened.append(OPS[it](f, f))
            else:
                intervened.append(OPS[it](f))
        p0 = instrument.p_fake_batch(fakes)
        p1 = instrument.p_fake_batch(intervened)
        out[it] = float(np.mean(p0 - p1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="samples per dataset (half fake, half real)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print("[gather] FF++ ...")
    ff_fakes, ff_reals = gather_ffpp(args.n, rng)
    print(f"  FF++: {len(ff_fakes)} fake / {len(ff_reals)} real")
    print("[gather] DF40 generative families ...")
    df40 = gather_df40(args.n, rng)
    results_res = {}
    for fam, (fk, rl, res) in df40.items():
        tag = "res-matched" if res["res_matched"] else "RES-CONFOUND"
        print(f"  {fam}: {len(fk)} fake / {len(rl)} real  "
              f"native fake={res['fake']} real={res['real']}  [{tag}]")
        results_res[fam] = res

    instruments = {
        "effort": ("clip", SpatialInstrument("effort", device=args.device)),
        "freqnet": ("spectral", SpectralInstrument("freqnet", device=args.device)),
        "npr": ("spectral", SpectralInstrument("npr", device=args.device)),
    }

    results = {"n": args.n, "seed": SEED, "auc": {}, "intervention_response": {},
               "df40_native_resolution": results_res}

    # Build the dataset -> (fakes, reals) map. Pool only the resolution-matched
    # DF40 families for the headline "df40:matched"; keep every family individually
    # so the confounded ones are visible but not silently averaged in.
    datasets = {"ffpp": (ff_fakes, ff_reals)}
    matched_fake, matched_real = [], []
    for fam, (fk, rl, res) in df40.items():
        datasets[f"df40:{fam}"] = (fk, rl)
        if res["res_matched"]:
            matched_fake += fk
            matched_real += rl
    if matched_fake:
        datasets["df40:matched"] = (matched_fake, matched_real)

    print("\n[score] AUC (fake vs real)")
    print(f"{'dataset':<18} {'effort':>10} {'freqnet':>10} {'npr':>10}")
    for ds, (fk, rl) in datasets.items():
        results["auc"][ds] = {}
        row = {}
        for name, (_, inst) in instruments.items():
            auc, mpf, mpr = score(inst, fk, rl)
            results["auc"][ds][name] = {"auc": auc, "mean_pf": mpf, "mean_pr": mpr}
            row[name] = auc
        print(f"{ds:<18} {row['effort']:>10.3f} {row['freqnet']:>10.3f} {row['npr']:>10.3f}")

    print("\n[score] intervention response — mean Δp(fake), CLIP should be ≈0")
    resp_datasets = [ds for ds in ("ffpp", "df40:matched") if ds in datasets]
    for ds in resp_datasets:
        fk = datasets[ds][0]
        results["intervention_response"][ds] = {}
        print(f"  {ds}:")
        for name, (kind, inst) in instruments.items():
            resp = intervention_response(inst, fk, kind == "spectral")
            results["intervention_response"][ds][name] = resp
            s = "  ".join(f"{k.split('_')[0]}={v:+.3f}" for k, v in resp.items())
            print(f"    {name:<9} {s}")

    (OUT / "pilot_s.json").write_text(json.dumps(results, indent=2))
    print(f"\n[written] {OUT / 'pilot_s.json'}")
    print("\nGate is Umar's call on these numbers (pass -> dual-instrument; "
          "fail -> pre-committed fallback).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
