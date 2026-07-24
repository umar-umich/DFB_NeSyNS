"""T21 — PILOT D: re-certification, base vs DPO-tuned proposer (the headline experiment).

Both variants run over the SAME held-out images, through the SAME UNTOUCHED gate.
The held-out split is pre-committed in `cec/registration/dpo.yaml`
(`splits.heldout_eval_split`) and the DPO data never saw it — so the evaluation
split cannot be chosen after seeing results.

Reported metrics (region and composite ALWAYS separate, never pooled):
  FP-claim rate on reals   raw, pre-suppression
  Coverage                 certifiable fakes with >=1 certified claim
  region cert rate · composite cert rate · region:composite ratio
  oracle hit-rate          cited a region the gate independently certifies
  abstention rate          reals / weak fakes / all
  claim diversity          unique artifact x region pairs (detector-mimic check)

PRE-COMMITTED success criteria (stated here so they are not decided afterwards):
  PRIMARY   FP-claim rate drops WHILE coverage holds (within a few points). Both,
            or it is not a win.
  SECONDARY region:composite ratio shifts toward region (the tier taught
            specificity, not just "whole face").
  FAILURE MODES to report honestly: coverage collapses (model went mute) ·
            FP drops only because it abstains on everything · diversity collapses
            to a single artifact x region (detector mimicry).
Any honest outcome publishes.

Run (🔴 UMAR-RUNS, after T20):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/umar/miniconda3/envs/GenD/bin/python cec/pilots/pilot_d/run_pilot_d.py \
        --model internvl3-8b --adapter cec/dpo/lora --videos 40
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from cec.gate.certify import CertificationGate  # noqa: E402
from cec.proposer import Proposer, validate  # noqa: E402
from cec.registration import load_params  # noqa: E402
from cec.repair import Lpips  # noqa: E402
from cec.scripts.run_audit import (  # noqa: E402  (reuse the audit's samplers)
    GROUPS, gather_fakes, gather_reals, oracle_certifiable,
)
from cec.data.pairing import split_video_ids  # noqa: E402

SEED = 0
OUT = REPO / "results" / "pilot_d"


def evaluate(proposer, gate, fakes, reals, with_oracle=True):
    """All Pilot D metrics for one proposer variant."""
    n_fake = n_real = 0
    covered = 0                      # fakes with >=1 certified claim
    region_cert = composite_cert = 0
    fp_images = 0                    # reals asserting >=1 manipulation claim
    abstain_real = abstain_weakfake = 0
    hits = hit_eligible = 0
    diversity = Counter()

    for pair in fakes:
        vr = validate(proposer.propose(pair.image_path))
        if not vr.schema_valid:
            continue
        n_fake += 1
        rec = gate.certify_image(pair, vr, {"proposer": proposer.name})
        certs = [c for c in rec["claims"] if c.get("label") == "CERTIFIED"]
        region_cert += sum(1 for c in certs if c.get("scope") == "region")
        composite_cert += sum(1 for c in certs if c.get("scope") == "composite")
        if certs:
            covered += 1
        else:
            if vr.abstained:
                abstain_weakfake += 1
        for c in vr.manipulation_claims:
            diversity[(c.artifact, c.location)] += 1
        if with_oracle:
            oc = oracle_certifiable(gate, pair)
            if oc:                                   # only images where a hit is possible
                hit_eligible += 1
                if set(oc) & {c.location for c in vr.manipulation_claims}:
                    hits += 1

    for pair in reals:
        vr = validate(proposer.propose(pair.image_path))
        if not vr.schema_valid:
            continue
        n_real += 1
        if len(vr.manipulation_claims) > 0:
            fp_images += 1           # RAW, pre-suppression
        else:
            abstain_real += 1
        for c in vr.manipulation_claims:
            diversity[(c.artifact, c.location)] += 1

    n_all = n_fake + n_real
    ratio = (region_cert / composite_cert) if composite_cert else None
    return {
        "n_fake": n_fake, "n_real": n_real,
        "fp_claim_rate": round(fp_images / n_real, 4) if n_real else None,
        "coverage": round(covered / n_fake, 4) if n_fake else None,
        "region_certified_claims": region_cert,
        "composite_certified_claims": composite_cert,
        "region_composite_ratio": round(ratio, 3) if ratio is not None else None,
        "oracle_hit_rate": round(hits / hit_eligible, 4) if hit_eligible else None,
        "abstention_real": round(abstain_real / n_real, 4) if n_real else None,
        "abstention_weakfake": round(abstain_weakfake / n_fake, 4) if n_fake else None,
        "abstention_all": round((abstain_real + abstain_weakfake) / n_all, 4) if n_all else None,
        "claim_diversity": len(diversity),
        "top_claims": [f"{a}/{l}:{n}" for (a, l), n in diversity.most_common(5)],
    }


def verdict(base, tuned, coverage_tol=0.05):
    """Apply the PRE-COMMITTED success criteria. No post-hoc goalposts."""
    notes = []
    fp_b, fp_t = base["fp_claim_rate"], tuned["fp_claim_rate"]
    cov_b, cov_t = base["coverage"], tuned["coverage"]
    if None in (fp_b, fp_t, cov_b, cov_t):
        return "INCONCLUSIVE (missing metrics)", notes
    fp_drop = fp_t < fp_b
    cov_held = cov_t >= cov_b - coverage_tol
    primary = fp_drop and cov_held
    if not fp_drop:
        notes.append("FP-claim rate did not drop")
    if not cov_held:
        notes.append(f"coverage fell more than {coverage_tol:.0%} (model may have gone mute)")
    if tuned["claim_diversity"] <= 1:
        notes.append("diversity collapsed to a single artifact x region (detector mimicry)")
    if tuned.get("abstention_all", 0) and tuned["abstention_all"] > 0.9:
        notes.append("abstains on almost everything — FP drop is not a real win")
    rb, rt = base["region_composite_ratio"], tuned["region_composite_ratio"]
    if rb is not None and rt is not None:
        notes.append(f"region:composite {rb} -> {rt} "
                     f"({'shifted toward region' if rt > rb else 'no shift toward region'})")
    return ("PRIMARY MET (FP down, coverage held)" if primary
            else "PRIMARY NOT MET"), notes


def main():
    cfg = yaml.safe_load((REPO / "cec" / "registration" / "dpo.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg["model"]["order"][0])
    ap.add_argument("--adapter", required=True, help="LoRA dir from T20")
    ap.add_argument("--split", default=cfg["splits"]["heldout_eval_split"])
    ap.add_argument("--group", default="all", choices=list(GROUPS))
    ap.add_argument("--videos", type=int, default=40)
    ap.add_argument("--instrument", default="fsfm")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    params = load_params()
    fpv = params.qc["frames_per_video"]
    split_vids = split_video_ids(args.split)

    fakes = gather_fakes(GROUPS[args.group], args.videos, fpv, split_vids, rng)
    reals = gather_reals(len(fakes), fpv, split_vids, rng)
    print(f"[pilot D] held-out split '{args.split}' (pre-committed): "
          f"{len(fakes)} fakes / {len(reals)} reals · gate={args.instrument} UNTOUCHED")

    gate = CertificationGate(args.instrument, device=args.device, lpips=Lpips(args.device))
    results = {}
    for variant, adapter in (("base", None), ("tuned", args.adapter)):
        print(f"[pilot D] evaluating {variant} ...")
        prop = Proposer(args.model, device=args.device, adapter=adapter)
        results[variant] = evaluate(prop, gate, fakes, reals, with_oracle=not args.no_oracle)
        del prop

    v, notes = verdict(results["base"], results["tuned"])
    results["verdict"] = v
    results["notes"] = notes
    results["config"] = {"model": args.model, "adapter": args.adapter, "split": args.split,
                         "instrument": args.instrument, "seed": SEED}
    (OUT / "pilot_d.json").write_text(json.dumps(results, indent=2))

    rows = ["fp_claim_rate", "coverage", "region_certified_claims",
            "composite_certified_claims", "region_composite_ratio", "oracle_hit_rate",
            "abstention_real", "abstention_weakfake", "abstention_all", "claim_diversity"]
    print(f"\n{'metric':<28}{'base':>12}{'tuned':>12}")
    for r in rows:
        print(f"{r:<28}{str(results['base'][r]):>12}{str(results['tuned'][r]):>12}")
    print(f"\nVERDICT (pre-committed): {v}")
    for n in notes:
        print(f"  - {n}")
    print(f"\n[written] {OUT / 'pilot_d.json'}")
    print("Any honest outcome publishes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
