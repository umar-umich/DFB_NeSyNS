"""TASK 11 — build DPO preference pairs from Certified Evidence Records.

The training signal, and the paper's central novelty: the gate's own labels teach
the proposer to cite load-bearing evidence. For each image with BOTH a certified
and a rejected claim, emit a preference pair:
    {prompt: <image + frozen prompt>, chosen: <CERTIFIED claim>, rejected: <REJECTED claim>}
Same image, so the only difference the proposer learns is which cited evidence
survived the counterfactual gate — not image content.

Reports pair counts + the positive-source breakdown (which method/region), so a
scarce pool is visible before training (🟡 gate for Task 11).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cec.records import RecordStore  # noqa: E402
from cec.registration import load_prompt  # noqa: E402

OUT = REPO / "cec" / "dpo" / "data"


def _claim_text(claim: dict) -> str:
    """The proposer-style JSON fragment for one claim (chosen/rejected target)."""
    return json.dumps({"artifact": claim.get("artifact"), "location": claim.get("location")})


def build(record_name: str):
    store = RecordStore(record_name)
    prompt, prompt_hash = load_prompt("proposer_type_b")
    pairs = []
    src = Counter()
    for rec in store.read():
        certified = [c for c in rec["claims"] if c.get("label") == "CERTIFIED"]
        rejected = [c for c in rec["claims"] if c.get("label") == "REJECTED"]
        if not certified or not rejected:
            continue
        # strongest certified (max NM) vs strongest rejected — same image.
        chosen = max(certified, key=lambda c: c.get("NM", 0))
        rej = max(rejected, key=lambda c: c.get("NM", -1))
        pairs.append({
            "image": rec["image"],
            "prompt": prompt,
            "prompt_hash": prompt_hash,
            "chosen": _claim_text(chosen),
            "rejected": _claim_text(rej),
        })
        method = rec["image"].split("/")[0]
        src[f"{method}:{chosen.get('location')}"] += 1
    return pairs, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="record store name (cec/records/data/<name>.jsonl)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pairs, src = build(args.records)
    dest = OUT / f"{args.records}_prefs.jsonl"
    with dest.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")

    print(f"preference pairs: {len(pairs)}")
    print(f"positive-source breakdown (method:region -> count):")
    for k, v in src.most_common(15):
        print(f"  {k}: {v}")
    print(f"\n[written] {dest}")
    if len(pairs) < 50:
        print("🟡 SCARCE positive pool (<50 pairs) — Task 11 gate: report to Umar before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
