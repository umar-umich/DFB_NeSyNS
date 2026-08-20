#!/usr/bin/env python3
"""Stages D and E — applicability gates, then the risk/defer policy (§9 D-E, §12, §13, §18).

    🔴 UMAR-RUNS:

    python training/stage_de.py \
        --run logs/v1/stage_b_seed42 \
        --output logs/v1/stage_de/epoch_007 --device cuda:N

Everything here happens on **VAL_meta only**, with the selected checkpoint frozen:

    D  score VAL_meta with the frozen experts
       -> fusion-utility target t_b from undiscounted pairwise DS      (§13)
       -> 5-fold cross-fit the gates, keep OUT-OF-FOLD q              (§12)
       -> refit each gate on all of VAL_meta for deployment
    E  gated fusion with the out-of-fold q -> V/C/A
       -> logistic risk model on [V, C, A, fused_margin]              (§18)
       -> freeze the Real/Fake/Defer policy on FF++ validation only

Why the out-of-fold q matters here specifically
-----------------------------------------------
The risk model is fit on the SAME samples the gates were trained on. Using each gate's in-sample
q would let the risk model calibrate against a gate that had already seen those samples, and the
defer policy would look better offline than it can behave. §12's cross-fitting exists for exactly
this, and §18 says to use its output; the deployment gates (refit on everything) are saved
separately and are what inference uses.

What this script deliberately does not do
-----------------------------------------
It never touches an OOD source. The gates, the risk model and both thresholds come from FF++
validation, and §11's checkpoint selection has already happened on VAL_select — a disjoint
partition, by identity, verified by `meta_split.py`.
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
from networks.discern_v2.ds_fusion import Opinion, apply_validity  # noqa: E402
from eval_v1 import load_model  # noqa: E402
from train_v1 import ViewMaker, prepare_dataset_config, video_of  # noqa: E402

SPECIALISTS = ("ref", "proc")


def resolve_checkpoint(run: Path, explicit: Path | None) -> tuple[Path, dict | None]:
    """The §11-selected checkpoint, or an explicit one.

    Reading SELECTED.json rather than taking "the last epoch" or "the best val number" keeps
    Stage D bound to the checkpoint selection that was actually recorded — §13's target is
    defined from the frozen SELECTED experts, so silently gating a different epoch would make the
    gates describe a model nobody evaluated.
    """
    if explicit:
        return explicit, None
    path = run / "SELECTED.json"
    if not path.is_file():
        raise SystemExit(
            f"{path} not found. Run analysis/discern_v2/select_checkpoint.py first (§11), or pass "
            f"--checkpoint explicitly. Stage D must be bound to a recorded selection.")
    selection = json.loads(path.read_text())
    return Path(selection["checkpoint"]), selection


def load_val_meta(data_cfg: dict, split_file: Path):
    """FF++ val restricted to VAL_meta, with each frame's fold attached."""
    import cache_encoder_features as CE

    mapping = json.loads(split_file.read_text())
    partition, folds = mapping["video_to_partition"], mapping["video_to_fold"]

    dataset = CE.load_split(data_cfg, "FaceForensics++", "val")
    keep = [i for i, p in enumerate(dataset.image_list)
            if partition.get(video_of(p if isinstance(p, str) else p[0])) == "VAL_meta"]
    if not keep:
        raise SystemExit("no VAL_meta frames matched the split file")
    dataset.image_list = [dataset.image_list[i] for i in keep]
    dataset.label_list = [dataset.label_list[i] for i in keep]
    dataset.data_dict = {"image": dataset.image_list, "label": dataset.label_list}
    dataset._build_source_video_maps()
    print(f"  VAL_meta: {len(keep)} frames")
    return dataset, folds


@torch.no_grad()
def score(model, loader, views, device: str, folds: dict, max_batches: int = 0) -> dict:
    """Per-frame opinions from the frozen experts, plus labels, video ids and fold ids."""
    from tqdm import tqdm

    collected: dict[str, list] = {"label": [], "video_id": [], "fold": [], "key": []}
    evidence: dict[str, list] = {}
    valid: dict[str, list] = {}
    for i, batch in enumerate(tqdm(loader, desc="  VAL_meta", leave=False)):
        if max_batches and i >= max_batches:
            break
        out = model(views(batch, device))
        collected["label"].append(torch.where(batch["label"] != 0, 1, 0).numpy())
        videos = [video_of(p) for p in batch["name"]]
        collected["video_id"].extend(videos)
        collected["key"].extend(batch["name"])
        collected["fold"].extend(folds.get(v, -1) for v in videos)
        for name, b in out["branches"].items():
            if name == "direct":
                continue
            evidence.setdefault(name, []).append(b["evidence"].cpu())
            valid.setdefault(name, []).append(b["valid"].cpu())

    labels = torch.from_numpy(np.concatenate(collected["label"])).long()
    fold = torch.tensor(collected["fold"])
    if (fold < 0).any():
        missing = int((fold < 0).sum())
        raise SystemExit(f"{missing} frames had no fold assignment — the split file and the "
                         f"dataset disagree about video names")
    opinions = {}
    for name in evidence:
        ev = torch.cat(evidence[name])
        va = torch.cat(valid[name]).float()
        opinions[name] = apply_validity(Opinion.from_evidence(ev), va)
    return {"opinions": opinions, "labels": labels, "fold": fold,
            "video_id": collected["video_id"], "key": collected["key"]}


def video_auroc(prob: np.ndarray, labels: np.ndarray, videos: list[str]) -> float:
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame({"p": prob, "y": labels, "v": videos})
    agg = df.groupby("v", as_index=False).agg(p=("p", "mean"), y=("y", "max"))
    if agg["y"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(agg["y"], agg["p"]))


def fuse(opinions: dict, q: dict | None) -> dict:
    """The deployment fusion path, reused so Stage E calibrates what inference computes."""
    from networks.discern_v2.ds_fusion import discount, ds_combine, reliability

    order = [n for n in ("sem", "ref", "proc") if n in opinions]
    weights = {}
    for name in order:
        if name == "sem" or q is None:
            weights[name] = torch.ones(opinions[name].batch_size)
        else:
            weights[name] = q[name].clamp(0, 1)
    discounted = [opinions[n] if n == "sem" else discount(opinions[n], weights[n])
                  for n in order]
    fused, diag = ds_combine(discounted)
    rel = reliability(opinions, weights, fused, specialists=SPECIALISTS)
    return {"fused": fused, "prob": fused.fake_prob(), **rel, "diag": diag}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="Stage-B run directory")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="override the §11 selection (normally leave unset)")
    ap.add_argument("--detector-config", type=Path,
                    default=REPO / "training/config/detector/nesy_defake_d1_v.yaml")
    ap.add_argument("--split-file", type=Path,
                    default=REPO / "configs/discern_v2/meta_split.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--abstention-budget", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=G.DEFAULT_DELTA,
                    help="§13's admission margin; config, never tuned on OOD")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint, selection = resolve_checkpoint(args.run, args.checkpoint)
    model, cfg, epoch = load_model(checkpoint, args.device)
    model.set_stage("D")                     # §19: experts frozen from here on
    model.assert_frozen_protocol()
    live = {k: v for k, v in model.trainable_parameters().items() if v}
    if live:
        raise SystemExit(f"experts still trainable in stage D: {live}")
    print(f"frozen selected checkpoint: epoch {epoch} ({checkpoint})")

    data_cfg = prepare_dataset_config(args.detector_config, args.batch_size, args.workers)
    clip_norm = data_cfg["foundation_models"]["spatial"]["normalization"]
    views = ViewMaker(clip_norm["mean"], clip_norm["std"], args.device, augment=False)
    dataset, folds = load_val_meta(data_cfg, args.split_file)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=int(data_cfg["workers"]), collate_fn=dataset.collate_fn)

    scored = score(model, loader, views, args.device, folds, args.max_batches)
    opinions, labels, fold = scored["opinions"], scored["labels"], scored["fold"]
    videos = scored["video_id"]
    print(f"  scored {len(labels)} frames over folds {sorted(set(fold.tolist()))}")

    # ---------------- Stage D: the gates ----------------
    print("\nStage D — fusion-utility-supervised gates (§13)")
    gates, q_oof, gate_report = {}, {}, {}
    for name in SPECIALISTS:
        if name not in opinions:
            continue
        target = G.fusion_utility_target(opinions["sem"], opinions[name], labels,
                                         delta=args.delta)
        features = G.gate_features(opinions["sem"], opinions[name])
        result = G.cross_fit(features, target["target"], fold)
        gates[name] = result["gate"]
        q_oof[name] = result["q_out_of_fold"]
        gate_report[name] = {
            "target_positive_rate": float(target["target"].mean()),
            "mean_delta_fuse": float(target["delta_fuse"].mean()),
            "metrics_out_of_fold": result["metrics"],
            "per_fold": result["per_fold"],
        }
        m = result["metrics"]
        print(f"  q_{name}: target positive rate {float(target['target'].mean()):.3f} · "
              f"OOF AUROC {m['auroc']:.4f} · accuracy {m['accuracy']:.4f} "
              f"(always-admit baseline {m['majority_baseline_accuracy']:.4f}) · "
              f"mean q {m['mean_q']:.3f}")

    # ---------------- plain DS vs DS + applicability ----------------
    plain = fuse(opinions, None)
    gated = fuse(opinions, q_oof)
    comparison = {
        "plain_ds_video_auroc": video_auroc(plain["prob"].numpy(), labels.numpy(), videos),
        "gated_video_auroc": video_auroc(gated["prob"].numpy(), labels.numpy(), videos),
    }
    comparison["delta"] = comparison["gated_video_auroc"] - comparison["plain_ds_video_auroc"]
    print(f"\nVAL_meta video AUROC — plain DS {comparison['plain_ds_video_auroc']:.4f} · "
          f"with applicability {comparison['gated_video_auroc']:.4f} "
          f"({comparison['delta']:+.4f})")
    if abs(comparison["delta"]) < 0.01:
        print("  NOTE this delta is inside §22's noise floor; it is in-sample for the gates and "
              "must not be reported as the applicability layer's benefit — the OOD suite decides.")

    # ---------------- Stage E: the risk / defer policy ----------------
    print("\nStage E — risk model and frozen defer policy (§18)")
    decision_threshold = R.eer_threshold(labels, gated["prob"])
    predicted = (gated["prob"] >= decision_threshold).long()
    wrong = (predicted != labels).float()
    features = R.risk_features(gated["V"], gated["C"], gated["A"], gated["prob"])
    risk_model, risk_info = R.fit_risk_model(features, wrong)
    with torch.no_grad():
        risk = risk_model(features)
    policy = R.freeze_thresholds(labels, gated["prob"], risk,
                                 abstention_budget=args.abstention_budget,
                                 source=f"FF++ VAL_meta (out-of-fold q, epoch {epoch})")
    report = R.report(risk, wrong, labels, gated["prob"], args.abstention_budget)
    applied = R.evaluate_policy(policy, labels, gated["prob"], risk)

    print(f"  error rate at full coverage {risk_info['error_rate']:.4f}")
    print(f"  coefficients {risk_info['coefficients']}")
    print(f"  error-detection AUROC {report['error_detection_auroc']:.4f}")
    print(f"  at a {args.abstention_budget:.0%} budget: coverage {applied['coverage']:.3f}, "
          f"selective accuracy {applied['selective_accuracy']:.4f} "
          f"(full coverage {applied['full_coverage_accuracy']:.4f})")
    print(f"  policy: {policy.provenance}")

    # ---------------- artifacts ----------------
    torch.save({
        "epoch": epoch, "checkpoint": str(checkpoint),
        "gates": {n: g.state_dict() for n, g in gates.items()},
        "gate_feature_names": list(G.FEATURE_NAMES),
        "risk_model": risk_model.state_dict(),
        "risk_feature_names": list(R.FEATURE_NAMES),
        "policy": policy.as_dict(),
        "delta": args.delta,
    }, args.output / "stage_de.pt")

    payload = {
        "checkpoint": str(checkpoint), "epoch": epoch,
        "selection": selection,
        "n_frames": int(len(labels)), "n_videos": len(set(videos)),
        "stage_d": gate_report,
        "plain_vs_applicability_on_VAL_meta": {
            **comparison,
            "caveat": ("in-sample for the gates and computed on FF++ validation; the honest "
                       "comparison is the OOD suite with the frozen gates applied"),
        },
        "stage_e": {"risk_fit": risk_info, "selective": report, "policy": policy.as_dict(),
                    "applied_to_VAL_meta": applied,
                    "decision_threshold_source": "EER on VAL_meta fused probability"},
    }
    (args.output / "stage_de.json").write_text(json.dumps(payload, indent=2, default=str))

    pd.DataFrame({
        "key": scored["key"], "video_id": videos, "fold": fold.numpy(),
        "label": labels.numpy(), "prob_plain": plain["prob"].numpy(),
        "prob_gated": gated["prob"].numpy(), "risk": risk.numpy(),
        "V": gated["V"].numpy(), "C": gated["C"].numpy(), "A": gated["A"].numpy(),
        **{f"q_{n}": q_oof[n].numpy() for n in q_oof},
    }).to_parquet(args.output / "val_meta_per_sample.parquet", index=False)

    print(f"\nwrote {args.output}/stage_de.pt, stage_de.json and val_meta_per_sample.parquet")
    print("Next: §21 evaluation with these gates and this policy applied — and they are FROZEN, "
          "so no OOD source may retune them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
