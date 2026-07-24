"""T15d — headline metrics: FP-claim rate and coverage.

Reported as a PAIR. "Maximize abstention" is NOT the objective — it collapses to
silence. The framework is good iff it abstains on reals (low FP-claim rate) WHILE
still certifying evidence on genuine fakes (high coverage).

  FP-claim rate  fraction of REAL images on which the proposer asserts >=1
                 manipulation claim. Computed on RAW proposer output, BEFORE any
                 disagreement suppression (a real record stores the raw
                 manipulation-claim count).
  Coverage       fraction of certifiable FAKES where >=1 claim CERTIFIES.

Records carry `split_label` ('real'|'fake'). Real records carry
`n_manipulation_claims` (raw). Fake records carry gate labels on their claims.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cec.records import RecordStore  # noqa: E402


def fp_claim_rate(real_records) -> dict:
    """Over real images: fraction asserting >=1 manipulation claim (raw)."""
    n = 0
    asserted = 0
    for rec in real_records:
        n += 1
        # raw count stored at audit time; fall back to counting non-abstention claims
        k = rec.get("n_manipulation_claims")
        if k is None:
            k = sum(1 for c in rec.get("claims", []) if c.get("label") != "UNTESTABLE"
                    or c.get("reason") not in ("abstention",))
        asserted += 1 if k and k > 0 else 0
    return {"n_real": n, "fp_images": asserted,
            "fp_claim_rate": round(asserted / n, 4) if n else None}


def coverage(fake_records) -> dict:
    """Over fakes: fraction with >=1 CERTIFIED claim."""
    n = 0
    covered = 0
    for rec in fake_records:
        n += 1
        if any(c.get("label") == "CERTIFIED" for c in rec.get("claims", [])):
            covered += 1
    return {"n_fake": n, "covered": covered,
            "coverage": round(covered / n, 4) if n else None}


def compute(record_name: str) -> dict:
    recs = list(RecordStore(record_name).read())
    reals = [r for r in recs if r.get("split_label") == "real"]
    fakes = [r for r in recs if r.get("split_label") == "fake"]
    return {"record_store": record_name, "n_records": len(recs),
            **fp_claim_rate(reals), **coverage(fakes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--out", default=str(REPO / "results" / "eval"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = {name: compute(name) for name in args.records}
    (out / "metrics.json").write_text(json.dumps(table, indent=2))
    print(f"{'record store':<34}{'n_real':>7}{'FP-claim%':>11}{'n_fake':>7}{'coverage%':>11}")
    for name, m in table.items():
        fp = f"{100 * m['fp_claim_rate']:.1f}" if m["fp_claim_rate"] is not None else "-"
        cov = f"{100 * m['coverage']:.1f}" if m["coverage"] is not None else "-"
        print(f"{name:<34}{m['n_real']:>7}{fp:>11}{m['n_fake']:>7}{cov:>11}")
    print(f"\n[written] {out / 'metrics.json'}")
    print("Report FP-claim rate and coverage as a PAIR (abstention alone is not the goal).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
