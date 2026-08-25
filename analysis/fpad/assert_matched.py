#!/usr/bin/env python3
"""Prove two Stage-A students are MATCHED before the Stage-2 gate compares them.

    python analysis/fpad/assert_matched.py \
        --runs logs/fpad/studentA_ordinary_seed42 logs/fpad/studentA_preserve_seed42

The Stage-2 gate's whole claim is that the ordinary-LoRA and preservation deltas differ because of
`L_preserve` and nothing else. If the two students saw different data, a different order, different
augmentation, a different LoRA target set or a different number of steps, the gate is measuring
training maturity and would still produce a clean-looking per-layer table.

This is checkable rather than assumed, because each run records a `batch_order_fingerprint` per
epoch — a hash of the sample order actually consumed. Two runs whose fingerprints agree epoch for
epoch, at equal step counts and identical config, differed only in the loss.

It found a real failure the first time it ran. Two independent causes:

* `MatchedAugment` derived its per-pair seed from Python's `hash()`, which is randomized per
  process unless `PYTHONHASHSEED` is set — so the two runs augmented the same pair differently;
* `abstract_dataset.py:346` shuffles the collected frame list with `random.shuffle` on the GLOBAL
  module RNG, and the trainers seeded `torch` and `numpy` but not `random`. The same 115,198 FF++
  train frames hashed to two different orders across two processes.

Both are fixed. This script exists so a regression cannot pass unnoticed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Fields that must agree for the comparison to isolate L_preserve. `lambda_preserve` is
# deliberately absent: it is the one thing that is SUPPOSED to differ.
MUST_MATCH = ("seed", "sampling", "layers", "lora", "readout", "batch_size", "epochs",
              "frames_per_video", "amp")


def load(run: Path) -> dict:
    meta_path = run / "run_meta.json"
    log_path = run / "metrics.jsonl"
    if not meta_path.is_file():
        raise SystemExit(f"{meta_path} missing — is {run} a Stage-A run directory?")
    if not log_path.is_file():
        raise SystemExit(f"{log_path} missing")
    rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    return {"run": run, "meta": json.loads(meta_path.read_text()), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, nargs=2, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    a, b = (load(r) for r in args.runs)
    problems: list[str] = []

    # 1. config
    for field in MUST_MATCH:
        va, vb = a["meta"].get(field), b["meta"].get(field)
        if va != vb:
            problems.append(f"config `{field}` differs: {va!r} vs {vb!r}")

    # 2. the thing that is supposed to differ actually does
    la, lb = a["meta"].get("lambda_preserve"), b["meta"].get("lambda_preserve")
    if la == lb:
        problems.append(
            f"both runs used lambda_preserve = {la}. There is no preservation contrast to gate "
            f"on — one of them was meant to be the ordinary-LoRA student.")
    if not ((la == 0) ^ (lb == 0)):
        problems.append(
            f"exactly one run must have lambda_preserve = 0 (the ordinary-LoRA student); got "
            f"{la} and {lb}")

    # 3. step count
    if len(a["rows"]) != len(b["rows"]):
        problems.append(f"different epoch counts: {len(a['rows'])} vs {len(b['rows'])}")
    n_batches = [(r["train"].get("n_batches"), s["train"].get("n_batches"))
                 for r, s in zip(a["rows"], b["rows"])]
    for i, (na, nb) in enumerate(n_batches):
        if na != nb:
            problems.append(f"epoch {i}: {na} vs {nb} optimizer steps")

    # 4. batch order, epoch by epoch — the one that catches silent non-determinism
    mismatched = []
    for i, (r, s) in enumerate(zip(a["rows"], b["rows"])):
        fa = r["train"].get("batch_order_fingerprint")
        fb = s["train"].get("batch_order_fingerprint")
        if fa != fb:
            mismatched.append((i, fa, fb))
    if mismatched:
        problems.append(
            f"batch order differs on {len(mismatched)} of {len(a['rows'])} epochs "
            f"(first: epoch {mismatched[0][0]}, {mismatched[0][1]} vs {mismatched[0][2]}). The "
            f"two students did not see the same data in the same order, so any difference "
            f"between their deltas includes a data difference.")

    # 5. Epoch-0 classification loss: INFORMATIONAL, not a criterion.
    #
    # An earlier version asserted these must be equal, reasoning that L_preserve is ~0 at
    # initialization because the teacher and student start bit-identical. That reasoning is right
    # for the FIRST optimizer step and wrong for the logged quantity, which is the MEAN over an
    # epoch: from step 2 onward the preservation student's weights have already been steered by
    # L_preserve, so its classification loss legitimately diverges over the remaining steps.
    # Divergence here is the mechanism working. The check was rejecting correctly-matched runs.
    #
    # It is still worth reporting: a LARGE gap would suggest something other than L_preserve is
    # differing. The guarantees that actually matter are the config, the step counts and the
    # batch-order fingerprints above, all of which are exact.
    notes = []
    if a["rows"] and b["rows"]:
        ca = a["rows"][0]["train"].get("loss_cls_direct")
        cb = b["rows"][0]["train"].get("loss_cls_direct")
        if ca is not None and cb is not None:
            rel = abs(ca - cb) / max(abs(ca), abs(cb), 1e-12)
            notes.append(
                f"epoch-0 mean L_cls_direct {ca:.6f} vs {cb:.6f} ({rel:.1%} apart) — expected to "
                f"differ, since L_preserve steers the preservation student from step 2 onward")
            if rel > 0.25:
                problems.append(
                    f"epoch-0 mean L_cls_direct differs by {rel:.0%} ({ca:.6f} vs {cb:.6f}). "
                    f"L_preserve should perturb the classification loss, not dominate it; a gap "
                    f"this large suggests the two runs differ in more than the preservation term.")

    report = {
        "runs": [str(a["run"]), str(b["run"])],
        "lambda_preserve": {str(a["run"]): la, str(b["run"]): lb},
        "epochs": len(a["rows"]),
        "matched": not problems,
        "problems": problems,
        "notes": notes,
    }
    print(f"run A: {a['run']}  lambda_preserve = {la}")
    print(f"run B: {b['run']}  lambda_preserve = {lb}")
    print(f"epochs: {len(a['rows'])} / {len(b['rows'])}")
    for n in notes:
        print(f"  note: {n}")
    if problems:
        print(f"\nNOT MATCHED — {len(problems)} problem(s):")
        for p in problems:
            print(f"  * {p}")
        print("\nThe Stage-2 gate must not run on these. It would attribute a data or schedule "
              "difference to L_preserve.")
    else:
        print("\nMATCHED. The two students share seed, sampling, LoRA targets, readout, batch "
              "order and step count, and differ only in lambda_preserve. The Stage-2 gate's "
              "comparison isolates preservation.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
