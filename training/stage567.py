#!/usr/bin/env python3
"""Stages 5, 6 and 7 — applicability gates, the fusion comparison, and reliability/defer.

    🔴 UMAR-RUNS (GPU, ~15 min):

    python training/stage567.py \
        --run logs/v1/stage_b_paired_seed42 \
        --output logs/phase2/stage567/epoch_NNN --device cuda:N

One pass over VAL_meta, because all three stages read the same frozen expert opinions and scoring
twice would let them drift apart. Everything happens with the selected checkpoint frozen and on
FF++ VAL_meta only (`--val-protocol ffpp`, the brief's ground rule 2).

What is NEW here relative to V1's `stage_de.py`
-----------------------------------------------
**Stage 5 — a different applicability target.** V1 used a pairwise target: does adding specialist
`b` to the anchor alone improve the fused cross-entropy. The brief replaces it with the
specialist's exact **Shapley value** in the cooperative game over all specialists, computed under
the fusion operator that will be used downstream:

    phi_b = sum_{S subset of specialists\\{b}} w(|S|) [U(anchor+S+b) - U(anchor+S)]
    U(S)  = -CE(p_fusion(S), y)                      w(s) = s!(n-1-s)!/n!

For two specialists this is exactly the brief's
`0.5[U(A+ref) - U(A)] + 0.5[U(A+ref+rate) - U(A+rate)]`. The difference from V1 matters when
specialists overlap: a pairwise target credits both members of a redundant pair with the full gain,
so both gates learn to admit, and the redundancy is only discovered at fusion time.

**Stage 6 — three fusion arms**, plus one the brief does not ask for and that its own comparison
needs (see below).

**Stage 7 — `A` is renamed `U_sup`.** Only the name changes. V1's `reliability()` already computes
`w_b = q_b(1-u_b)`, `C = sum w_i w_j JS / (sum w_i w_j + eps)` and `1 - mean_b w_b`, and over two
specialists `1 - mean` IS the brief's `1 - 0.5*sum`. The numbers are directly comparable to V1's;
the rename exists to stop `A` colliding with aleatoric uncertainty.

The fourth arm, and why it is not scope creep
----------------------------------------------
The brief names three arms — Applicability-CCF, Applicability-DS, Equal-CCF — and says to run them
"on identical discounted opinions", while Stage 5 says to learn applicability "under the fusion
operator that will be used downstream". Those two instructions cannot both hold across all three
arms: if each applicability arm learns its own `q`, then Applicability-CCF vs Applicability-DS
differs in BOTH the operator and the gate, and the CCF-vs-DS comparison is confounded.

Equal-CCF vs Applicability-CCF stays clean either way (same operator, `q` vs `q=1`), and the brief
correctly identifies that as "the number that tells you whether applicability earns its place".
But the operator comparison needs one more arm, so `applicability_ds_shared_q` fuses with DS over
the CCF arm's `q`. One extra fusion, no extra training, and the operator effect is isolated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

from networks.discern_v2 import applicability_gate as G  # noqa: E402
from networks.discern_v2 import risk_model as R  # noqa: E402
from networks.discern_v2.ccf_fusion import ccf_combine, ccf_is_non_associative  # noqa: E402
from networks.discern_v2.ds_fusion import (  # noqa: E402
    Opinion, discount, ds_combine, reliability)
from eval_v1 import load_model  # noqa: E402
from stage_de import (  # noqa: E402
    VAL_PROTOCOLS, load_calibration, resolve_checkpoint, score, video_auroc)
from train_v1 import ViewMaker, prepare_dataset_config  # noqa: E402

ANCHOR = "sem"
SPECIALISTS = ("ref", "proc", "rate")
NOISE_FLOOR = 0.01

# Every arm: (label, fusion operator, which q to use).
#   `own`    -> the q learned under this arm's own operator (the brief's three arms)
#   `equal`  -> q = 1 for every specialist (the critical control)
#   `ccf`    -> the CCF arm's q, so the operator is the only thing that changes
ARMS = (
    ("applicability_ccf", "ccf", "own"),
    ("applicability_ds", "ds", "own"),
    ("equal_ccf", "ccf", "equal"),
    ("equal_ds", "ds", "equal"),
    ("applicability_ds_shared_q", "ds", "ccf"),
)


# --------------------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------------------

def combine(opinions: list[Opinion], operator: str) -> Opinion:
    if operator == "ds":
        fused, _ = ds_combine(opinions)
        return fused
    if operator == "ccf":
        return ccf_combine(opinions)
    raise ValueError(f"unknown fusion operator {operator!r}")


def fuse(opinions: dict[str, Opinion], names: list[str], q: dict[str, torch.Tensor],
         operator: str) -> dict:
    """Discount the specialists, fuse, and recompute reliability under this operator.

    Applicability discounting (brief Stage 6): `b' = q*b`, `u' = (1-q) + q*u`, so an inapplicable
    specialist contributes IGNORANCE rather than evidence for Real. The anchor is undiscounted.
    """
    weights, parts = {}, []
    for name in names:
        if name == ANCHOR:
            weights[name] = torch.ones(opinions[name].batch_size)
            parts.append(opinions[name])
        else:
            w = q[name].clamp(0.0, 1.0)
            weights[name] = w
            parts.append(discount(opinions[name], w))
    fused = combine(parts, operator)
    present = tuple(n for n in SPECIALISTS if n in names)
    rel = reliability({n: opinions[n] for n in names}, weights, fused, specialists=present)
    prob = fused.fake_prob()
    return {
        "fused": fused, "prob": prob, "weights": weights,
        # V is u_f, so it is operator-dependent and MUST be recomputed per arm (brief Stage 6)
        "V": rel["V"], "C": rel["C"],
        # renamed from A; identical quantity (brief Stage 7)
        "U_sup": rel["A"],
        "margin": (prob - 0.5).abs(),
    }


# --------------------------------------------------------------------------------------
# Stage 5 — the applicability target
# --------------------------------------------------------------------------------------

def _cross_entropy(p_fake: torch.Tensor, y: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = p_fake.clamp(eps, 1.0 - eps)
    yf = y.float()
    return -(yf * torch.log(p) + (1.0 - yf) * torch.log(1.0 - p))


def marginal_utility(opinions: dict[str, Opinion], labels: torch.Tensor,
                     specialists: list[str], operator: str) -> dict[str, torch.Tensor]:
    """Exact per-sample Shapley value of each specialist, under `operator` (brief Stage 5).

    Utilities are computed with q = 1 throughout: the target answers "would admitting this
    specialist improve the collective decision on this sample", which is the question the gate
    then learns to predict. Folding a current q into the utility would make the target depend on
    the gate being trained.

    n <= 3, so all 2^n subsets are enumerated and no approximation is needed.
    """
    import itertools
    from math import factorial

    n = len(specialists)
    if n == 0:
        return {}
    ones = {name: torch.ones(opinions[name].batch_size) for name in opinions}
    utility: dict[frozenset, torch.Tensor] = {}
    for size in range(n + 1):
        for subset in itertools.combinations(specialists, size):
            p = fuse(opinions, [ANCHOR, *subset], ones, operator)["prob"]
            utility[frozenset(subset)] = -_cross_entropy(p, labels)

    phi = {}
    for b in specialists:
        others = [s for s in specialists if s != b]
        total = torch.zeros_like(labels, dtype=torch.float32)
        for size in range(n):
            weight = factorial(size) * factorial(n - 1 - size) / factorial(n)
            for subset in itertools.combinations(others, size):
                key = frozenset(subset)
                total = total + weight * (utility[key | {b}] - utility[key])
        phi[b] = total
    return phi


def train_gates(opinions: dict[str, Opinion], labels: torch.Tensor, fold: torch.Tensor,
                specialists: list[str], operator: str, delta: float) -> tuple[dict, dict, dict]:
    """Cross-fitted gates and out-of-fold q, with the Shapley target under `operator`."""
    phi = marginal_utility(opinions, labels, specialists, operator)
    gates, q_oof, report = {}, {}, {}
    for name in specialists:
        target = (phi[name] > delta).long()
        features = G.gate_features(opinions[ANCHOR], opinions[name])
        result = G.cross_fit(features, target, fold)
        gates[name] = result["gate"]
        q_oof[name] = result["q_out_of_fold"]
        m = result["metrics"]
        report[name] = {
            "target_positive_rate": float(target.float().mean()),
            "mean_phi": float(phi[name].mean()),
            "median_phi": float(phi[name].median()),
            "metrics_out_of_fold": m,
            "per_fold": result["per_fold"],
            "beats_always_admit": bool(m["auroc"] > 0.5 + NOISE_FLOOR),
        }
        print(f"    q_{name}: target positive rate {report[name]['target_positive_rate']:.3f} · "
              f"mean phi {report[name]['mean_phi']:+.4f} · OOF AUROC {m['auroc']:.4f} "
              f"(always-admit baseline accuracy {m['majority_baseline_accuracy']:.4f}) · "
              f"mean q {m['mean_q']:.3f}")
        if not report[name]["beats_always_admit"]:
            print(f"      NOTE q_{name} does not beat chance out of fold. Its applicability is "
                  f"not inferable from these features, so any gain this arm shows is not the "
                  f"gate's.")
    return gates, q_oof, report


# --------------------------------------------------------------------------------------
# Stage 7 — risk and defer, per arm
# --------------------------------------------------------------------------------------

def fit_risk(arm: dict, labels: torch.Tensor, budget: float, provenance: str) -> dict:
    """Logistic risk on [V, C, U_sup, margin], then the frozen Real/Fake/Defer policy.

    Refit per arm on purpose: V and the fused margin are operator-dependent, so a risk model fit
    under one operator and applied under another would be reading two of its four inputs off the
    wrong scale.
    """
    threshold = R.eer_threshold(labels, arm["prob"])
    wrong = ((arm["prob"] >= threshold).long() != labels).float()
    features = R.risk_features(arm["V"], arm["C"], arm["U_sup"], arm["prob"])
    model, info = R.fit_risk_model(features, wrong)
    with torch.no_grad():
        risk = model(features)
    policy = R.freeze_thresholds(labels, arm["prob"], risk, abstention_budget=budget,
                                 source=provenance)
    report = R.report(risk, wrong, labels, arm["prob"], budget)
    applied = R.evaluate_policy(policy, labels, arm["prob"], risk)

    coefficients = dict(zip(("V", "C", "U_sup", "fused_margin"), info["coefficients"]))
    # The brief's pass condition, checked rather than asserted: more vacuity, more informative
    # conflict and more support deficit each RAISE risk; a more decided margin LOWERS it.
    expected = {"V": +1, "C": +1, "U_sup": +1, "fused_margin": -1}
    signs = {k: (int(np.sign(v)) == expected[k]) for k, v in coefficients.items()}
    return {
        "model": model, "risk": risk, "wrong": wrong, "policy": policy,
        "decision_threshold": threshold,
        "coefficients": coefficients, "coefficient_signs_as_expected": signs,
        "all_signs_expected": all(signs.values()),
        "error_rate_full_coverage": info["error_rate"],
        "error_detection_auroc": report["error_detection_auroc"],
        "selective_risk_at_budget": 1.0 - applied["selective_accuracy"],
        "risk_at_full_coverage": 1.0 - applied["full_coverage_accuracy"],
        "coverage_at_budget": applied["coverage"],
        "risk_coverage": report.get("risk_coverage"),
        "report": report, "applied": applied,
    }


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------

def render(payload: dict) -> str:
    arms = payload["arms"]
    lines = [
        "# Stages 5-7 — applicability, fusion comparison, reliability",
        "",
        f"Checkpoint `{payload['checkpoint']}` (epoch {payload['epoch']}) · calibration "
        f"{' + '.join(payload['calibration_sources'])} · {payload['n_frames']} frames · "
        f"specialists {payload['specialists']}.",
        "",
    ]
    if payload["sources_no_longer_zero_shot"]:
        lines += [f"> ⚠️ **Not zero-shot:** "
                  f"{', '.join(payload['sources_no_longer_zero_shot'])} — these sources' test "
                  f"videos calibrated the gates and the defer policy. Label them calibrated, "
                  f"never OOD.", ""]

    lines += ["## Stage 5 — gates", "",
              "Target: exact per-sample Shapley marginal utility under each arm's own fusion "
              "operator, `t_b = 1[phi_b > 0]`. Inputs are label-free; the target is not, and is "
              "never an input.", "",
              "| operator | gate | target positive rate | mean phi | OOF AUROC | always-admit "
              "accuracy | mean q | beats chance |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for operator, gates in payload["gate_reports"].items():
        for name, r in gates.items():
            m = r["metrics_out_of_fold"]
            lines.append(
                f"| `{operator}` | `q_{name}` | {r['target_positive_rate']:.3f} | "
                f"{r['mean_phi']:+.4f} | {m['auroc']:.4f} | "
                f"{m['majority_baseline_accuracy']:.4f} | {m['mean_q']:.3f} | "
                f"{'yes' if r['beats_always_admit'] else '**no**'} |")

    lines += ["", "## Stage 6 — fusion comparison", "",
              "| arm | operator | q | video AUROC | mean V | mean C | mean U_sup |",
              "|---|---|---|---:|---:|---:|---:|"]
    for name, a in arms.items():
        lines.append(f"| `{name}` | {a['operator']} | {a['q_source']} | "
                     f"{a['video_auroc']:.4f} | {a['mean_V']:.4f} | {a['mean_C']:.4f} | "
                     f"{a['mean_U_sup']:.4f} |")

    lines += ["", "### The comparisons that decide things", "",
              "| question | comparison | delta | verdict |", "|---|---|---:|---|"]
    for key, c in payload["comparisons"].items():
        lines.append(f"| {c['question']} | `{c['better']}` − `{c['worse']}` | "
                     f"{c['delta']:+.4f} | {c['verdict']} |")
    lines += ["", f"Measured non-associativity of CCF on these opinions: "
                  f"`{payload['ccf_non_associative']}` — this is why the operator is implemented "
                  f"as a genuine multi-source fusion rather than a pairwise chain.", ""]

    lines += ["## Stage 7 — reliability and defer", "",
              "Pass condition: more vacuity, more informative conflict and more support deficit "
              "each RAISE risk; a more decided margin LOWERS it. Checked per arm, not asserted.",
              "", "| arm | V | C | U_sup | margin | signs OK | error-detection AUROC | "
              "risk @ full coverage | selective risk @ budget |",
              "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for name, a in arms.items():
        r = a["risk"]
        c = r["coefficients"]
        lines.append(
            f"| `{name}` | {c['V']:+.3f} | {c['C']:+.3f} | {c['U_sup']:+.3f} | "
            f"{c['fused_margin']:+.3f} | {'yes' if r['all_signs_expected'] else '**NO**'} | "
            f"{r['error_detection_auroc']:.4f} | {r['risk_at_full_coverage']:.4f} | "
            f"{r['selective_risk_at_budget']:.4f} |")
    lines += ["", "Directly comparable to V1's reliability report: `A` was renamed `U_sup` and "
                  "nothing else changed — V1 already computed `w_b = q_b(1-u_b)` and, over two "
                  "specialists, `1 - mean_b w_b` IS the brief's `1 - 0.5*sum_b w_b`.", ""]

    failures = [n for n, a in arms.items() if not a["risk"]["all_signs_expected"]]
    if failures:
        lines += [f"> ⚠️ Stage 7's pass condition FAILS on {', '.join(failures)}. A coefficient "
                  f"with the wrong sign means the risk model is reading that signal backwards, "
                  f"and the defer policy built on it will abstain on the wrong samples.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--detector-config", type=Path,
                    default=REPO / "training/config/detector/nesy_defake_d1_v.yaml")
    ap.add_argument("--split-file", type=Path,
                    default=REPO / "configs/discern_v2/meta_split.json")
    ap.add_argument("--val-protocol", choices=sorted(VAL_PROTOCOLS), default="ffpp",
                    help="the brief's ground rule 2 requires FF++ only; `diverse` is available "
                         "and every artifact records which was used")
    ap.add_argument("--val-sources", nargs="+", default=None)
    ap.add_argument("--force-forbidden-sources", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--abstention-budget", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.0,
                    help="brief Stage 5: t_b = 1[phi_b > delta]; config, never tuned on OOD")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    checkpoint, selection = resolve_checkpoint(args.run, args.checkpoint)
    model, cfg, epoch = load_model(checkpoint, args.detector_config, args.device)
    model.eval()
    print(f"checkpoint {checkpoint} (epoch {epoch})")

    data_cfg = prepare_dataset_config(args.detector_config, args.batch_size, args.workers)
    clip_norm = data_cfg["foundation_models"]["spatial"]["normalization"]
    views = ViewMaker(clip_norm["mean"], clip_norm["std"], args.device, augment=False)
    targets = args.val_sources or VAL_PROTOCOLS[args.val_protocol]
    print(f"calibration sources: {targets}")
    datasets, folds, overlaps = load_calibration(data_cfg, args.split_file, targets,
                                                 args.force_forbidden_sources)
    compromised = sorted(t for t, ov in overlaps.items() if ov.get("n_shared_with_test", 0) > 0)
    dataset = (datasets[0] if len(datasets) == 1
               else torch.utils.data.ConcatDataset(datasets))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=int(data_cfg["workers"]), collate_fn=datasets[0].collate_fn)

    scored = score(model, loader, views, args.device, folds, args.max_batches)
    opinions, labels, fold = scored["opinions"], scored["labels"], scored["fold"]
    videos = scored["video_id"]
    specialists = [n for n in SPECIALISTS if n in opinions]
    if not specialists:
        raise SystemExit("no specialist branch is present; there is nothing to gate or compare")
    names = [ANCHOR, *specialists]
    print(f"  scored {len(labels)} frames · specialists {specialists}")

    # ---------------- Stage 5 ----------------
    print("\nStage 5 — cross-fitted applicability gates (Shapley marginal utility)")
    gates, q_by_operator, gate_reports = {}, {}, {}
    for operator in ("ccf", "ds"):
        print(f"  under `{operator}` fusion:")
        g, q, report = train_gates(opinions, labels, fold, specialists, operator, args.delta)
        gates[operator] = g
        q_by_operator[operator] = q
        gate_reports[operator] = report

    ones = {n: torch.ones(opinions[n].batch_size) for n in names}

    # ---------------- Stage 6 ----------------
    print("\nStage 6 — fusion comparison")
    arms: dict[str, dict] = {}
    for label, operator, q_kind in ARMS:
        q = (ones if q_kind == "equal" else
             q_by_operator[operator] if q_kind == "own" else q_by_operator[q_kind])
        arm = fuse(opinions, names, q, operator)
        arm.update({
            "operator": operator,
            "q_source": ("q = 1 (control)" if q_kind == "equal" else
                         f"learned under {operator}" if q_kind == "own" else
                         f"learned under {q_kind}"),
            "video_auroc": video_auroc(arm["prob"].numpy(), labels.numpy(), videos),
            "mean_V": float(arm["V"].mean()), "mean_C": float(arm["C"].mean()),
            "mean_U_sup": float(arm["U_sup"].mean()),
            "mean_margin": float(arm["margin"].mean()),
        })
        arms[label] = arm
        print(f"  {label:28s} video AUROC {arm['video_auroc']:.4f}  "
              f"V {arm['mean_V']:.4f}  C {arm['mean_C']:.4f}  U_sup {arm['mean_U_sup']:.4f}")

    def compare(better: str, worse: str, question: str) -> dict:
        d = arms[better]["video_auroc"] - arms[worse]["video_auroc"]
        return {"better": better, "worse": worse, "question": question, "delta": d,
                "verdict": (f"inside the {NOISE_FLOOR} noise band — a confirming seed is required"
                            if abs(d) < NOISE_FLOOR else
                            f"{better} wins" if d > 0 else f"{worse} wins")}

    comparisons = {
        "applicability_earns_its_place": compare(
            "applicability_ccf", "equal_ccf",
            "does LEARNED applicability beat equal standing, same operator?"),
        "applicability_earns_its_place_ds": compare(
            "applicability_ds", "equal_ds",
            "the same question under DS, as a robustness check"),
        "ccf_vs_ds_confounded": compare(
            "applicability_ccf", "applicability_ds",
            "CCF vs DS — CONFOUNDED: the gate differs too (brief's three arms)"),
        "ccf_vs_ds_isolated": compare(
            "applicability_ccf", "applicability_ds_shared_q",
            "CCF vs DS on the SAME q — the operator effect, isolated"),
    }
    for c in comparisons.values():
        print(f"  {c['question']}\n      {c['better']} − {c['worse']} = {c['delta']:+.4f} "
              f"→ {c['verdict']}")

    # is the operator's non-associativity real on THIS data, or only in principle?
    non_assoc = False
    if len(names) >= 3:
        parts = [opinions[n] for n in names[:3]]
        non_assoc = ccf_is_non_associative(*parts)

    # ---------------- Stage 7 ----------------
    print("\nStage 7 — reliability, risk and Real/Fake/Defer")
    provenance = " + ".join(targets)
    if compromised:
        provenance += f" [NOT zero-shot for: {', '.join(compromised)}]"
    for label, arm in arms.items():
        arm["risk"] = fit_risk(arm, labels, args.abstention_budget,
                               f"{provenance} (out-of-fold q, epoch {epoch}, arm {label})")
        r = arm["risk"]
        print(f"  {label:28s} coeffs "
              f"V {r['coefficients']['V']:+.2f} C {r['coefficients']['C']:+.2f} "
              f"U_sup {r['coefficients']['U_sup']:+.2f} "
              f"margin {r['coefficients']['fused_margin']:+.2f}  "
              f"signs {'OK' if r['all_signs_expected'] else 'WRONG'}  "
              f"err-AUROC {r['error_detection_auroc']:.4f}  "
              f"risk {r['risk_at_full_coverage']:.4f} -> "
              f"{r['selective_risk_at_budget']:.4f}")

    # ---------------- artifacts ----------------
    args.output.mkdir(parents=True, exist_ok=True)
    primary = "applicability_ccf"
    torch.save({
        "epoch": epoch, "checkpoint": str(checkpoint), "primary_arm": primary,
        "specialists": specialists,
        "gates": {op: {n: g.state_dict() for n, g in gs.items()}
                  for op, gs in gates.items()},
        "gate_feature_names": list(G.FEATURE_NAMES),
        "risk_models": {label: a["risk"]["model"].state_dict() for label, a in arms.items()},
        "risk_feature_names": ("V", "C", "U_sup", "fused_margin"),
        "policies": {label: a["risk"]["policy"].as_dict() for label, a in arms.items()},
        "calibration_sources": targets,
        "sources_no_longer_zero_shot": compromised,
        "delta": args.delta, "val_protocol": args.val_protocol,
    }, args.output / "stage567.pt")

    # Per-sample export WITH the per-branch evidence columns. V1's VAL_meta parquet carried only
    # q/V/C/A, which meant any later analysis wanting to refit a gate on protocol-valid data had
    # to fall back to FF++ TEST rows. Exporting the evidence removes that fallback.
    export = {"key": scored["key"], "video_id": videos, "fold": fold.numpy(),
              "label": labels.numpy()}
    for name in names:
        op = opinions[name]
        # EVIDENCE, not belief. The two differ by the Dirichlet strength (e_k = b_k * K/u), and
        # exporting belief under an `e_*` column name would make every downstream reconstruction
        # of the opinion wrong while still looking like a valid opinion.
        state = op.to_dirichlet()
        export[f"p_{name}"] = op.fake_prob().numpy()
        export[f"u_{name}"] = op.vacuity.squeeze(1).numpy()
        export[f"e_{name}_real"] = state.evidence[:, 0].numpy()
        export[f"e_{name}_fake"] = state.evidence[:, 1].numpy()
        export[f"valid_{name}"] = (op.vacuity.squeeze(1) < 1.0 - 1e-6).numpy()
    for operator, q in q_by_operator.items():
        for name, v in q.items():
            export[f"q_{name}_{operator}"] = v.numpy()
    for label, arm in arms.items():
        export[f"prob_{label}"] = arm["prob"].numpy()
        export[f"V_{label}"] = arm["V"].numpy()
        export[f"C_{label}"] = arm["C"].numpy()
        export[f"U_sup_{label}"] = arm["U_sup"].numpy()
        export[f"risk_{label}"] = arm["risk"]["risk"].numpy()
    pd.DataFrame(export).to_parquet(args.output / "val_meta_per_sample.parquet", index=False)

    payload = {
        "checkpoint": str(checkpoint), "epoch": epoch, "specialists": specialists,
        "calibration_sources": targets, "val_protocol": args.val_protocol,
        "sources_no_longer_zero_shot": compromised, "calibration_overlap": overlaps,
        "n_frames": int(len(labels)), "delta": args.delta,
        "selection": selection, "primary_arm": primary,
        "ccf_non_associative": non_assoc,
        "gate_reports": gate_reports,
        "arms": {label: {k: v for k, v in a.items()
                         if k in ("operator", "q_source", "video_auroc", "mean_V", "mean_C",
                                  "mean_U_sup", "mean_margin")}
                 | {"risk": {k: v for k, v in a["risk"].items()
                             if k not in ("model", "risk", "wrong", "policy", "report",
                                          "applied")}
                    | {"policy": a["risk"]["policy"].as_dict()}}
                 for label, a in arms.items()},
        "comparisons": comparisons,
    }
    (args.output / "stage567.json").write_text(json.dumps(payload, indent=2, default=str))
    (args.output / "STAGE567.md").write_text(render(payload))
    print(f"\nwrote {args.output}/STAGE567.md, stage567.json, stage567.pt and "
          f"val_meta_per_sample.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
