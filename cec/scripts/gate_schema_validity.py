"""TASK 6 GATE — schema-validity probe (>=95%).

Runs each pinned proposer over 100 FF++ images through the frozen Type-B prompt,
validates every response, and reports the schema-validity rate. Gate: >=95%
(one recorded prompt revision allowed if it misses). Also reports the claim-route
distribution (spatial / spectral / untestable) — informative for the DPO pool
and for how many claims the single-instrument spatial gate can actually test.

Run:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /data/umar/miniconda3/envs/GenD/bin/python cec/scripts/gate_schema_validity.py --n 100
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cec.data.pairing import fake_dir  # noqa: E402
from cec.proposer import validate  # noqa: E402

PP = Path("/data/umar/Datasets/preprocessed/FaceForensics++")
OUT = REPO / "results" / "proposer_gate"
SEED = 0
GATE = 0.95


def gather(n, rng):
    imgs = []
    for method in ("Deepfakes", "FaceSwap"):
        fdir = fake_dir(method) / "frames"
        vids = sorted(p.name for p in fdir.iterdir())
        rng.shuffle(vids)
        for vid in vids[: n // 4]:
            fr = sorted((fdir / vid).glob("*.png"))
            if fr:
                imgs.append(str(fr[len(fr) // 2]))
    rdir = PP / "original_sequences" / "youtube" / "c23" / "frames"
    vids = sorted(p.name for p in rdir.iterdir())
    rng.shuffle(vids)
    for vid in vids[: n // 2]:
        fr = sorted((rdir / vid).glob("*.png"))
        if fr:
            imgs.append(str(fr[len(fr) // 2]))
    return imgs[:n]


def run_proposer(name, images, device):
    from cec.proposer import Proposer
    prop = Proposer(name, device=device)
    n_valid, routes = 0, Counter()
    n_claims = 0
    for path in images:
        v = validate(prop.propose(path))
        n_valid += v.schema_valid
        if v.schema_valid:
            for c in v.claims:
                routes[c.route] += 1
                n_claims += 1
    rate = n_valid / len(images)
    return {"schema_valid": n_valid, "n": len(images), "rate": rate,
            "n_claims": n_claims, "routes": dict(routes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--proposers", nargs="+", default=["internvl3-8b", "qwen2.5-vl-32b-instruct"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    images = gather(args.n, rng)
    print(f"[gather] {len(images)} FF++ images")

    results = {"n": len(images), "seed": SEED, "gate": GATE, "proposers": {}}
    for name in args.proposers:
        print(f"\n[proposer] {name} ...")
        r = run_proposer(name, images, args.device)
        results["proposers"][name] = r
        verdict = "PASS" if r["rate"] >= GATE else "FAIL (prompt revision allowed)"
        print(f"  schema-valid {r['schema_valid']}/{r['n']} = {r['rate']:.1%}  [{verdict}]")
        print(f"  claims: {r['n_claims']} total  routes={r['routes']}")

    (OUT / "schema_validity.json").write_text(json.dumps(results, indent=2))
    print(f"\n[written] {OUT / 'schema_validity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
