"""TASK 11 / T15c — three-case DPO preference builder from Certified Evidence Records.

The training signal (the paper's novelty): the gate's own labels teach the
proposer to cite load-bearing evidence and to abstain honestly. Three cases:

  | case                      | chosen                       | rejected                  |
  | fake, evidence certifies  | a CERTIFIED claim            | an uncertified claim, same image |
  | real image                | contentful abstention        | any manipulation claim it made   |
  | weak fake (nothing cert.) | contentful abstention        | any manipulation claim it made   |

Preference TIERS within the fake case: region-certified > composite-certified >
rejected — teaches specificity without punishing honest composite claims on
overdetermined swaps. NEVER use an empty list as `chosen` (trains toward silence);
the abstention target is the contentful `verdict_support` object (T15a).

Reports pair counts by case / scope / proposer, so a scarce pool is visible
before training (T18 gate: >=500 pairs).
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

# The contentful abstention target (T15a). Never an empty list.
ABSTENTION_TARGET = json.dumps({"verdict_support": "no_certified_evidence",
                                "examined": ["skin", "blending", "eyes", "mouth"]})

# scope preference tier for the fake case (higher = more preferred as `chosen`).
_TIER = {"region": 2, "composite": 1}


def _claim_text(claim: dict) -> str:
    return json.dumps({"artifact": claim.get("artifact"), "location": claim.get("location")})


def _manip_claims(rec):
    """Claims that assert a manipulation (exclude abstention/untestable-abstention)."""
    return [c for c in rec.get("claims", []) if c.get("reason") != "abstention"]


def build(record_names):
    """Build preference pairs from one or more record stores (pooled)."""
    if isinstance(record_names, str):
        record_names = [record_names]
    prompt, prompt_hash = load_prompt("proposer_type_b")
    pairs = []
    by_case, by_scope, by_proposer = Counter(), Counter(), Counter()

    records = (rec for name in record_names for rec in RecordStore(name).read())
    for rec in records:
        proposer = rec.get("provenance", {}).get("proposer", "?")
        base = {"image": rec["image"], "prompt": prompt, "prompt_hash": prompt_hash,
                "proposer": proposer}
        certified = [c for c in rec["claims"] if c.get("label") == "CERTIFIED"]
        rejected = [c for c in _manip_claims(rec) if c.get("label") == "REJECTED"]
        is_real = rec.get("split_label") == "real"

        if certified:  # fake, evidence certifies
            chosen = max(certified, key=lambda c: (_TIER.get(c.get("scope"), 0), c.get("NM", 0)))
            if not rejected:
                continue  # need a contrasting negative on the same image
            rej = max(rejected, key=lambda c: c.get("NM", -1))
            pairs.append({**base, "case": "fake_certified", "scope": chosen.get("scope"),
                          "chosen": _claim_text(chosen), "rejected": _claim_text(rej)})
            by_case["fake_certified"] += 1
            by_scope[chosen.get("scope")] += 1
            by_proposer[proposer] += 1
        else:
            # real image OR weak fake (nothing certified): prefer abstention over any
            # manipulation claim the proposer made. Skip if it made none (nothing to correct).
            manip = _manip_claims(rec)
            if not manip:
                continue
            rej = manip[0]
            case = "real_abstention" if is_real else "weakfake_abstention"
            pairs.append({**base, "case": case, "scope": "abstention",
                          "chosen": ABSTENTION_TARGET, "rejected": _claim_text(rej)})
            by_case[case] += 1
            by_scope["abstention"] += 1
            by_proposer[proposer] += 1

    return pairs, by_case, by_scope, by_proposer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True,
                    help="one or more record stores; pooled into a single pref set")
    ap.add_argument("--name", default=None, help="output name (default: joined record names)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pairs, by_case, by_scope, by_proposer = build(args.records)
    out_name = args.name or ("+".join(args.records) if len(args.records) > 1 else args.records[0])
    dest = OUT / f"{out_name}_prefs.jsonl"
    with dest.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")

    print(f"preference pairs: {len(pairs)}")
    print(f"  by case:     {dict(by_case)}")
    print(f"  by scope:    {dict(by_scope)}")
    print(f"  by proposer: {dict(by_proposer)}")
    print(f"\n[written] {dest}")
    if len(pairs) < 500:
        print(f"🟡 pool < 500 (Umar's trigger) — T18: STOP, present the escalation ladder.")
    else:
        print(f"✓ pool >= 500 (Umar's trigger cleared) — proceed to DPO on Umar's go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
