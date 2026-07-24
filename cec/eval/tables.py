"""T22 — eval tables, regenerated from records. Nothing hand-entered.

Tables produced:
  audit        per record-store: claim-label rates, abstention, raw NM distribution
  certification by method x SCOPE x proposer, with the mandated 0.05/0.10
               **sensitivity row** — region and composite are reported SEPARATELY
               and never pooled; the region:composite ratio is a headline diagnostic
  pool         DPO preference-pool composition (by case / scope / proposer)

Reporting rules obeyed (params.yaml): NM is a raw per-detector score drop; never a
probability; never averaged or compared across detectors.

The sensitivity row exists because the region margin floor was set to 0.05 (the
control-p99 rule). Disclosing certification at BOTH 0.05 and 0.10 is the defense
against a post-hoc-threshold critique.

Run:
    /data/umar/miniconda3/envs/GenD/bin/python cec/eval/tables.py --records <store> [...]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cec.records import RecordStore  # noqa: E402
from cec.registration import load_params  # noqa: E402


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


def _passes_at(claim: dict, margin: float) -> bool:
    """Re-evaluate a recorded claim at an alternative margin (sensitivity row).

    Only the margin varies; gap/wrong/offset keep their recorded outcomes, so this
    isolates the margin choice. Requires the record to carry `checks` + `NM`
    (added by the T13 instrumentation).
    """
    checks = claim.get("checks")
    if checks is None or claim.get("NM") is None:
        return claim.get("label") == "CERTIFIED"
    others = all(v for k, v in checks.items() if k != "margin")
    return bool(others and claim["NM"] >= margin)


def certification_table(record_names) -> dict:
    """method x scope x proposer, with the 0.05/0.10 sensitivity row."""
    margins = load_params().raw["gate_region"].get("sensitivity_margins", [0.05, 0.10])
    cells = defaultdict(lambda: {"claims": 0, **{f"cert@{m}": 0 for m in margins}})
    for name in record_names:
        for rec in RecordStore(name).read():
            if rec.get("split_label") == "real":
                continue  # certification rates are a fakes-only quantity
            proposer = rec.get("provenance", {}).get("proposer", "?")
            family = rec.get("family", "?")
            for c in rec["claims"]:
                scope = c.get("scope")
                if scope not in ("region", "composite"):
                    continue
                key = (family, scope, proposer)
                cells[key]["claims"] += 1
                for m in margins:
                    # composite is judged by the FROZEN gate: its bar does not vary
                    if scope == "composite":
                        cells[key][f"cert@{m}"] += c.get("label") == "CERTIFIED"
                    else:
                        cells[key][f"cert@{m}"] += _passes_at(c, m)
    out = {"margins": margins, "rows": []}
    ratio_num = Counter()
    for (family, scope, proposer), v in sorted(cells.items()):
        row = {"family": family, "scope": scope, "proposer": proposer, "claims": v["claims"]}
        for m in margins:
            row[f"cert_rate@{m}"] = round(v[f"cert@{m}"] / v["claims"], 3) if v["claims"] else None
        out["rows"].append(row)
        ratio_num[(family, proposer, scope)] = v[f"cert@{margins[0]}"]
    # region:composite ratio (headline diagnostic), at the primary margin
    ratios = {}
    for (family, proposer, scope), n in ratio_num.items():
        if scope == "region":
            comp = ratio_num.get((family, proposer, "composite"), 0)
            ratios[f"{family}|{proposer}"] = round(n / comp, 3) if comp else None
    out["region_composite_ratio"] = ratios
    return out


def pool_table(prefs_names) -> dict:
    """DPO preference-pool composition."""
    out = {}
    for name in prefs_names:
        path = REPO / "cec" / "dpo" / "data" / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        out[name] = {
            "total_pairs": len(rows),
            "by_case": dict(Counter(r.get("case") for r in rows)),
            "by_scope": dict(Counter(r.get("scope") for r in rows)),
            "by_proposer": dict(Counter(r.get("proposer") for r in rows)),
            "meets_500_trigger": len(rows) >= 500,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--prefs", nargs="*", default=[])
    ap.add_argument("--out", default=str(REPO / "results" / "eval"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    audit = {name: audit_table(name) for name in args.records}
    cert = certification_table(args.records)
    pool = pool_table(args.prefs) if args.prefs else {}
    (out / "audit_table.json").write_text(json.dumps(audit, indent=2))
    (out / "certification_table.json").write_text(json.dumps(cert, indent=2))
    if pool:
        (out / "pool_table.json").write_text(json.dumps(pool, indent=2))

    print(f"{'record store':<40}{'imgs':>6}{'cert%':>7}{'untest%':>9}{'abstain%':>9}{'NM med':>8}")
    for name, t in audit.items():
        print(f"{name:<40}{t['n_images']:>6}{t['certified_rate']:>7}"
              f"{t['untestable_rate']:>9}{t['abstention_rate']:>9}"
              f"{str(t['certified_NM_median']):>8}")

    m = cert["margins"]
    print(f"\ncertification by family x scope x proposer (SENSITIVITY: margin {m[0]} vs {m[1]})")
    print(f"{'family':<18}{'scope':<11}{'proposer':<24}{'claims':>7}"
          + "".join(f"{'@'+str(x):>9}" for x in m))
    for r in cert["rows"]:
        print(f"{r['family']:<18}{r['scope']:<11}{r['proposer']:<24}{r['claims']:>7}"
              + "".join(f"{str(r[f'cert_rate@{x}']):>9}" for x in m))
    if cert["region_composite_ratio"]:
        print(f"\nregion:composite ratio @margin {m[0]} — {cert['region_composite_ratio']}")
    if pool:
        print(f"\npool composition: {json.dumps(pool, indent=2)}")
    print(f"\n[written] {out}/audit_table.json · certification_table.json"
          + (" · pool_table.json" if pool else ""))
    print("NM = raw per-detector score drop: not a probability, not comparable across "
          "detectors. Region and composite are never pooled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
