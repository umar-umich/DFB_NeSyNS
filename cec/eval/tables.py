"""TASK 12 — eval tables from the Certified Evidence Records.

Every number traceable to records/ + a commit hash. Lean set (Implementation v2
Task 12): the audit table (per proposer: certified/rejected/untestable rates,
raw-score NM per detector) and the certification-rate summary. AUC tables come
from the frozen pilot outputs (pilot_gen), not recomputed here.

NM reporting rules (params.yaml, obeyed): NM is a raw per-detector score drop;
never a probability; never averaged/compared across detectors.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cec.records import RecordStore  # noqa: E402


def audit_table(record_name: str) -> dict:
    """Per-record-store: claim-label rates + NM distribution (raw drops)."""
    store = RecordStore(record_name)
    labels = Counter()
    nms = []
    n_images = 0
    abstained = 0
    detector = None
    for rec in store.read():
        n_images += 1
        detector = rec.get("detector")
        certified_here = 0
        for c in rec["claims"]:
            labels[c["label"]] += 1
            if c["label"] == "CERTIFIED":
                certified_here += 1
                if "NM" in c:
                    nms.append(c["NM"])
        if certified_here == 0:
            abstained += 1
    total = sum(labels.values()) or 1
    return {
        "record_store": record_name,
        "detector": detector,
        "n_images": n_images,
        "abstention_rate": round(abstained / max(1, n_images), 3),
        "claim_counts": dict(labels),
        "certified_rate": round(labels["CERTIFIED"] / total, 3),
        "rejected_rate": round(labels["REJECTED"] / total, 3),
        "untestable_rate": round(labels["UNTESTABLE"] / total, 3),
        # raw NM drops for CERTIFIED claims — NOT a probability, this detector only
        "certified_NM_median": round(float(np.median(nms)), 4) if nms else None,
        "certified_NM_iqr": [round(float(np.percentile(nms, 25)), 4),
                             round(float(np.percentile(nms, 75)), 4)] if nms else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--out", default=str(REPO / "results" / "eval"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tables = {name: audit_table(name) for name in args.records}
    (out / "audit_table.json").write_text(json.dumps(tables, indent=2))

    print(f"{'record store':<34}{'imgs':>6}{'cert%':>7}{'rej%':>7}{'untest%':>9}{'abstain%':>9}{'NM med':>8}")
    for name, t in tables.items():
        print(f"{name:<34}{t['n_images']:>6}{t['certified_rate']:>7}{t['rejected_rate']:>7}"
              f"{t['untestable_rate']:>9}{t['abstention_rate']:>9}"
              f"{(t['certified_NM_median'] or 0):>8}")
    print(f"\n[written] {out / 'audit_table.json'}")
    print("NM = raw per-detector score drop (params.yaml reporting rules): not a probability, "
          "not comparable across detectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
