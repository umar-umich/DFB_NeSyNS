#!/usr/bin/env python3
"""§11 — select the expert checkpoint on VAL_select only.

    🔴 UMAR-RUNS (seconds, no GPU):

    python analysis/discern_v2/select_checkpoint.py --run logs/v1/stage_b_seed42

§11's rule, implemented literally: primary metric is **video-level AUROC on VAL_select**,
tie-broken by lower ECE and then lower validation loss. No OOD source may participate, so the
script refuses to run if the metrics file contains anything that looks like one — epoch-wise OOD
scores are saved for later analysis and must not reach this decision.

Why this script says more than "epoch N wins"
---------------------------------------------
§22 sets a noise floor: a difference below 0.01 AUROC is not a difference. On FF++ in-domain
validation every late epoch scores above 0.99, so the whole run typically lands inside that band
and the primary metric produces an ordering without producing a ranking. Reporting the winner
alone invites the run to be described as "we selected the best checkpoint" when the honest
statement is "the candidates were indistinguishable on the primary metric".

Two readings then differ, and the script reports both rather than hiding the choice:

* **literal §11** — AUROC decides even when the margin is 0.001; ECE breaks exact ties only.
* **noise-aware §11+§22** — inside the band the epochs are indistinguishable, so ECE decides.

The literal reading is the default, because it is what §11 says. When the two disagree the report
says so and marks it 🟡 ASK-UMAR, since the selected checkpoint is what Stages D and E are built
on and reselecting later invalidates both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Anything matching these in a metrics key means an OOD source reached the selection file.
OOD_TOKENS = ("celeb", "cdf", "dfdc", "df40", "dfd", "uadfv", "deepfake-eval", "eval24",
              "faceshifter", "ood")
NOISE_FLOOR = 0.01          # §22


def load_metrics(run: Path) -> list[dict]:
    path = run / "metrics.jsonl"
    if not path.is_file():
        raise SystemExit(f"{path} not found — is {run} a Stage-B run directory?")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    epochs = [r["epoch"] for r in rows]
    if len(set(epochs)) != len(epochs):
        raise SystemExit(
            f"{path} contains duplicate epochs {sorted(e for e in set(epochs) if epochs.count(e) > 1)}"
            f" — two runs wrote into one directory and their metrics are interleaved, so the "
            f"selection would mix checkpoints from different configurations.")
    return rows


def assert_no_ood(rows: list[dict]) -> None:
    """§11: never select using CDF/DFDC/DF40 or any OOD source."""
    offenders = set()
    for row in rows:
        for key in row:
            if any(token in key.lower() for token in OOD_TOKENS):
                offenders.add(key)
        for section in ("val_select", "train"):
            for key in (row.get(section) or {}):
                if any(token in key.lower() for token in OOD_TOKENS):
                    offenders.add(f"{section}.{key}")
    if offenders:
        raise SystemExit(
            f"the metrics file carries OOD quantities {sorted(offenders)}. §11 forbids selecting "
            f"on them; keep epoch-wise OOD scores in a separate file for later analysis.")


def select(rows: list[dict], noise_aware: bool = False) -> dict:
    """§11's ordering: video AUROC desc, then ECE asc, then val loss asc, then earliest epoch.

    `noise_aware` changes ONE thing, and it is a deliberate interpretation rather than the spec's
    literal text: §11 orders by AUROC and tie-breaks on ECE, while §22 says an AUROC difference
    below 0.01 is not a difference. Read together, the epochs inside that band are indistinguishable
    on the primary metric and ECE should decide among them. Read literally, AUROC still decides
    even when the gap is 0.001. Both are reported; the literal reading is the default so the spec
    is followed unless someone chooses otherwise. 🟡 ASK-UMAR.
    """
    candidates = []
    for row in rows:
        val = row.get("val_select") or {}
        if val.get("video_auroc") is None:
            continue
        candidates.append({
            "epoch": row["epoch"],
            "video_auroc": float(val["video_auroc"]),
            "frame_auroc": float(val.get("frame_auroc", float("nan"))),
            "ece": float(val.get("ece", float("nan"))),
            "val_loss": float(val.get("loss", float("nan"))),
            "train_loss": float((row.get("train") or {}).get("loss", float("nan"))),
        })
    if not candidates:
        raise SystemExit("no epoch has a VAL_select video AUROC to select on")

    literal = sorted(candidates,
                     key=lambda c: (-c["video_auroc"], c["ece"], c["val_loss"], c["epoch"]))
    top = max(c["video_auroc"] for c in candidates)
    within = [c for c in candidates if top - c["video_auroc"] < NOISE_FLOOR]
    noise_aware_pick = sorted(within, key=lambda c: (c["ece"], c["val_loss"], c["epoch"]))[0]

    chosen = noise_aware_pick if noise_aware else literal[0]
    indistinguishable = len(within) > 1
    if noise_aware and indistinguishable:
        decided_by = ("ECE among the epochs inside §22's noise floor (noise-aware reading of "
                      "§11 + §22)")
    elif indistinguishable:
        decided_by = ("VAL_select video AUROC, as §11 orders it — but note the margin over the "
                      "other candidates is INSIDE §22's noise floor, so this ordering is not a "
                      "meaningful ranking")
    else:
        decided_by = "VAL_select video AUROC, with a margin above §22's noise floor"

    return {
        "selected_epoch": chosen["epoch"],
        "selected": chosen,
        "ranking": literal,
        "primary_metric": "VAL_select video AUROC",
        "rule": "noise_aware" if noise_aware else "literal_§11",
        "noise_floor": NOISE_FLOOR,
        "n_within_noise_floor": len(within),
        "epochs_within_noise_floor": sorted(c["epoch"] for c in within),
        "auroc_spread_within": (max(c["video_auroc"] for c in within)
                                - min(c["video_auroc"] for c in within)),
        "primary_metric_is_decisive": not indistinguishable,
        "decided_by": decided_by,
        "literal_pick": literal[0],
        "noise_aware_pick": noise_aware_pick,
        "rules_agree": literal[0]["epoch"] == noise_aware_pick["epoch"],
    }


def render(result: dict, run: Path) -> str:
    s = result["selected"]
    lines = [
        "# Checkpoint selection (§11)",
        "",
        f"**Selected: epoch {result['selected_epoch']}** — `{run}/epoch_{s['epoch']:03d}.pth`",
        "",
        f"VAL_select video AUROC {s['video_auroc']:.4f} · frame {s['frame_auroc']:.4f} · "
        f"ECE {s['ece']:.4f}",
        "",
        f"Selected on **VAL_select only**. Decided by: {result['decided_by']}.",
        "",
    ]
    if result["n_within_noise_floor"] > 1:
        lines += [
            f"> **{result['n_within_noise_floor']} of the run's epochs sit within §22's "
            f"{result['noise_floor']:.2f} AUROC noise floor of the leader** "
            f"(spread {result['auroc_spread_within']:.4f}, epochs "
            f"{result['epochs_within_noise_floor']}). The primary metric therefore does not "
            f"produce a meaningful ranking here — whichever epoch is chosen, the honest statement "
            f"is \"the candidates were indistinguishable on VAL_select AUROC\", not \"this was "
            f"the best epoch\".",
            "",
            f"> Two defensible rules, and they {'agree' if result['rules_agree'] else 'DISAGREE'}:",
            "",
            f"> - **literal §11** (AUROC first, ECE only on an exact tie): epoch "
            f"{result['literal_pick']['epoch']} "
            f"(AUROC {result['literal_pick']['video_auroc']:.4f}, "
            f"ECE {result['literal_pick']['ece']:.4f})",
            f"> - **noise-aware §11+§22** (ECE decides inside the band): epoch "
            f"{result['noise_aware_pick']['epoch']} "
            f"(AUROC {result['noise_aware_pick']['video_auroc']:.4f}, "
            f"ECE {result['noise_aware_pick']['ece']:.4f})",
            "",
            f"> This report used **{result['rule']}**. 🟡 ASK-UMAR if the other reading is "
            f"preferred — it changes which checkpoint Stages D and E are built on."
            if not result["rules_agree"] else
            f"> Both rules select epoch {result['literal_pick']['epoch']}, so the choice is "
            f"robust to the interpretation.",
            "",
        ]
    lines += ["## Ranking (§11 order: AUROC ↓, ECE ↑, val loss ↑)", "",
              "| rank | epoch | video AUROC | frame AUROC | ECE | train loss |",
              "|---:|---:|---:|---:|---:|---:|"]
    for i, c in enumerate(result["ranking"], start=1):
        mark = " **←**" if c["epoch"] == result["selected_epoch"] else ""
        lines.append(f"| {i} | {c['epoch']}{mark} | {c['video_auroc']:.4f} | "
                     f"{c['frame_auroc']:.4f} | {c['ece']:.4f} | {c['train_loss']:.4f} |")
    lines += [
        "",
        "## What happens next",
        "",
        "The selected checkpoint is frozen for Stages D and E: the applicability gates are trained "
        "against *this* checkpoint's opinions (§13's target is defined from the frozen selected "
        "experts), and the risk model is fit on the out-of-fold `q` those gates produce. Reselecting "
        "later would invalidate both.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="Stage-B run directory")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to <run>/SELECTED.json + SELECTION.md")
    ap.add_argument("--noise-aware", action="store_true",
                    help="inside §22's noise floor let ECE decide instead of AUROC (an "
                         "interpretation of §11+§22, not §11's literal text)")
    args = ap.parse_args()

    rows = load_metrics(args.run)
    assert_no_ood(rows)
    result = select(rows, noise_aware=args.noise_aware)

    checkpoint = args.run / f"epoch_{result['selected_epoch']:03d}.pth"
    if not checkpoint.is_file():
        raise SystemExit(f"selected epoch {result['selected_epoch']} but {checkpoint} is missing")
    result["checkpoint"] = str(checkpoint)

    out = args.out or args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "SELECTED.json").write_text(json.dumps(result, indent=2))
    (out / "SELECTION.md").write_text(render(result, args.run))

    print(f"epochs evaluated: {len(rows)}")
    print(f"within §22's {NOISE_FLOOR} noise floor of the leader: "
          f"{result['n_within_noise_floor']} "
          f"(spread {result['auroc_spread_within']:.4f})")
    print(f"rule: {result['rule']}")
    print(f"decided by: {result['decided_by']}")
    if not result["rules_agree"]:
        print(f"  NOTE the two readings disagree: literal §11 -> epoch "
              f"{result['literal_pick']['epoch']} (ECE {result['literal_pick']['ece']:.4f}), "
              f"noise-aware -> epoch {result['noise_aware_pick']['epoch']} "
              f"(ECE {result['noise_aware_pick']['ece']:.4f}). 🟡 ASK-UMAR.")
    print(f"\nSELECTED epoch {result['selected_epoch']}  ->  {checkpoint}")
    s = result["selected"]
    print(f"  VAL_select video AUROC {s['video_auroc']:.4f}  ECE {s['ece']:.4f}")
    print(f"\nwrote {out}/SELECTED.json and {out}/SELECTION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
