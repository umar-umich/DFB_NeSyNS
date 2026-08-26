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

Which data calibrates the gates
-------------------------------
`--val-protocol ffpp` (the default, and what §1/§11/§18 require) uses FF++ VAL_meta only, so no
OOD sample touches the gates, the risk model or either threshold.

`--val-protocol ffpp_cdf2_TESTCONTAMINATED` (formerly `diverse`, and it needs
--force-forbidden-sources) reproduces the protocol in
`analysis/discern_v2/SELECTION_PROTOCOL_RESULT.md` — FF++ **and** Celeb-DF-v2, with the remaining
sources untouched. That is a deliberate deviation from the V1 spec, and it has a price that is
computed rather than described: for every corpus here except FF++ the shipped `val` split IS the
test split (Celeb-DF-v1/v2/v3, DFD and UADFV are exact copies; DFDCP differs by two videos). Any
source used for calibration therefore stops being zero-shot, and the artifact records it under
`sources_no_longer_zero_shot` so a results table can label it instead of implying otherwise.

The motivation is real and documented in this repo: `A2_gate/A2b_PROTOCOL.md` records that
"in-domain FF++ validation cannot rank cross-domain detectors", and on this run FF++ VAL_meta is
saturated — plain and gated fusion both reach 1.0000 video AUROC, and the gates learn to admit
almost everything. A2b also forbids DF40, CDFv3 and Deepfake-Eval-2024 at any stage of gate
fitting; those are refused unless `--force-forbidden-sources` is passed.
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


# Formerly `diverse`. RENAMED because that name described an intent and hid two costs, and this
# is calibration data — what the gates, the risk model and both thresholds are fitted on.
#
#   1. Celeb-DF-v2's shipped `val` split IS its `test` split, 518/518 verified. Calibrating here
#      is calibrating on test, and Celeb-DF-v2 stops being a zero-shot number afterwards.
#   2. Even with a clean split it breaks the brief's firewall, which puts gates, risk and
#      thresholds on FF++ VAL_meta ONLY. Diverse-validation labels may select checkpoints and
#      nothing else.
#
# It is kept rather than deleted because it reproduces a recorded historical result
# (analysis/discern_v2/SELECTION_PROTOCOL_RESULT.md), but it now names its own cost and cannot be
# selected by accident. Do NOT reach for it to get "more diverse calibration": the diverse split
# that exists for selection is VALmix
# (preprocessing/build_valmix_manifest.py, tbiom/VALMIX.md), and it does not belong here either.
VAL_PROTOCOLS = {
    "ffpp": ["FaceForensics++:val"],
    "ffpp_cdf2_TESTCONTAMINATED": ["FaceForensics++:val", "Celeb-DF-v2:val"],
}
RENAMED_PROTOCOLS = {"diverse": "ffpp_cdf2_TESTCONTAMINATED"}
# A2b_PROTOCOL.md: "No DF40, CDFv3, or Deepfake-Eval-2024 data at any stage of gate fitting,
# including scaler and calibrator fitting." Refused unless explicitly forced.
A2B_FORBIDDEN = ("DF40", "Celeb-DF-v3", "Deepfake-Eval-2024")


def calibration_overlap(source: str, split: str, data_cfg: dict) -> dict:
    """How many calibration videos of `source` also appear in that source's TEST split.

    Reported because for every corpus here except FF++ the shipped `val` split IS the test split
    (Celeb-DF-v1/v2/v3, DFD and UADFV are exact copies; DFDCP differs by 2 videos). Calibrating
    on such a slice means the source's own test numbers are no longer zero-shot, so the number is
    computed and carried into the artifact rather than left as a caveat someone has to remember.
    """
    folder = Path(data_cfg["dataset_json_folder"])
    candidates = [folder / f"{source}.json"]
    blob = next((json.loads(c.read_text()) for c in candidates if c.is_file()), None)
    if blob is None:
        return {"status": "no json", "source": source}
    root = blob[next(iter(blob))]
    vids = {"val": set(), "test": set()}
    for _label, splits in root.items():
        for name in vids:
            section = splits.get(name) or {}
            if section and all(k in ("c23", "c40", "raw") for k in list(section)[:2]):
                section = section.get(data_cfg.get("compression", "c23"), {})
            vids[name] |= set(section)
    shared = vids[split] & vids["test"]
    return {"source": source, "split": split, "n_calibration_videos": len(vids[split]),
            "n_test_videos": len(vids["test"]), "n_shared_with_test": len(shared),
            "fraction_of_test_seen": (len(shared) / len(vids["test"])) if vids["test"] else 0.0,
            "zero_shot_after_calibration": len(shared) == 0}


def load_calibration(data_cfg: dict, split_file: Path, targets: list[str],
                     force: bool = False) -> tuple[list, dict, dict]:
    """Build the calibration set from one or more SOURCE:SPLIT slices.

    FF++ keeps its identity-disjoint VAL_meta restriction and the folds `meta_split.py` verified.
    Any other source has no such split file, so its folds are assigned by video group here —
    grouped, so no video's frames straddle two folds, but NOT identity-disjoint across the corpus
    the way FF++'s are.
    """
    import cache_encoder_features as CE

    mapping = json.loads(split_file.read_text())
    partition, ffpp_folds = mapping["video_to_partition"], mapping["video_to_fold"]

    datasets, folds, overlaps = [], {}, {}
    for target in targets:
        source, split = target.rsplit(":", 1)
        if any(bad.lower() in source.lower() for bad in A2B_FORBIDDEN) and not force:
            raise SystemExit(
                f"{source} is forbidden as calibration data by A2b_PROTOCOL.md (\"No DF40, CDFv3, "
                f"or Deepfake-Eval-2024 data at any stage of gate fitting\"). Pass "
                f"--force-forbidden-sources to override, and expect to drop it from the zero-shot "
                f"claims.")
        overlaps[target] = calibration_overlap(source, split, data_cfg)

        dataset = CE.load_split(data_cfg, source, split)
        if source == "FaceForensics++":
            keep = [i for i, p in enumerate(dataset.image_list)
                    if partition.get(video_of(p if isinstance(p, str) else p[0])) == "VAL_meta"]
            if not keep:
                raise SystemExit("no VAL_meta frames matched the split file")
            dataset.image_list = [dataset.image_list[i] for i in keep]
            dataset.label_list = [dataset.label_list[i] for i in keep]
            dataset.data_dict = {"image": dataset.image_list, "label": dataset.label_list}
            dataset._build_source_video_maps()
            folds.update(ffpp_folds)
            print(f"  {target}: {len(keep)} frames (VAL_meta, identity-disjoint folds)")
        else:
            videos = sorted({video_of(p if isinstance(p, str) else p[0])
                             for p in dataset.image_list})
            clash = sorted(set(videos) & set(folds))
            if clash:
                raise SystemExit(
                    f"{target}: {len(clash)} video names collide with an earlier calibration "
                    f"source (e.g. {clash[:3]}). Folds are keyed by video name, so one source's "
                    f"assignment would silently overwrite the other's and frames of the same "
                    f"video could land in different folds.")
            for i, video in enumerate(videos):
                folds[video] = i % 5              # grouped by video, deterministic
            ov = overlaps[target]
            print(f"  {target}: {len(dataset.image_list)} frames, {len(videos)} videos "
                  f"(folds by video group) — shares {ov.get('n_shared_with_test', '?')} of "
                  f"{ov.get('n_test_videos', '?')} test videos with its own test split")
        datasets.append(dataset)
    return datasets, folds, overlaps


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
    ap.add_argument("--val-protocol", choices=sorted(VAL_PROTOCOLS) + sorted(RENAMED_PROTOCOLS),
                    default="ffpp",
                    help="ffpp = FF++ VAL_meta only, which the V1 spec and the brief's firewall "
                         "both require. ffpp_cdf2_TESTCONTAMINATED reproduces the historical "
                         "SELECTION_PROTOCOL_RESULT.md run and calibrates on Celeb-DF-v2's val "
                         "split, which IS its test split; it needs --force-forbidden-sources.")
    ap.add_argument("--val-sources", nargs="+", default=None,
                    help="explicit SOURCE:SPLIT calibration slices, overriding --val-protocol")
    ap.add_argument("--force-forbidden-sources", action="store_true",
                    help="allow DF40 / CDFv3 / Deepfake-Eval-2024 as calibration data, which "
                         "A2b_PROTOCOL.md forbids")
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
    if args.val_protocol in RENAMED_PROTOCOLS:
        raise SystemExit(
            f"`--val-protocol {args.val_protocol}` was renamed to "
            f"`{RENAMED_PROTOCOLS[args.val_protocol]}`. The old name described the intent and hid "
            f"the cost: it calibrates the gates, the risk model and both thresholds on "
            f"Celeb-DF-v2's val split, which IS its test split (518/518), and the brief's "
            f"firewall puts calibration on FF++ VAL_meta only. Pass the new name with "
            f"--force-forbidden-sources if you are deliberately reproducing the historical run, "
            f"or use the default `ffpp`.")
    if args.val_protocol == "ffpp_cdf2_TESTCONTAMINATED" and not args.force_forbidden_sources:
        raise SystemExit(
            "`--val-protocol ffpp_cdf2_TESTCONTAMINATED` calibrates on Celeb-DF-v2's val split, "
            "which IS its test split (518/518 verified), and breaks the brief's firewall besides "
            "— gates, risk and thresholds are FF++ VAL_meta only. Re-run with "
            "--force-forbidden-sources to do it anyway; every Celeb-DF-v2 number afterwards must "
            "then be labelled as not zero-shot.")
    targets = args.val_sources or VAL_PROTOCOLS[args.val_protocol]
    print(f"calibration sources: {targets}")
    datasets, folds, overlaps = load_calibration(data_cfg, args.split_file, targets,
                                                 args.force_forbidden_sources)
    compromised = {t: ov for t, ov in overlaps.items()
                   if ov.get("n_shared_with_test", 0) > 0}
    if compromised:
        print("\n  !! these sources are NO LONGER ZERO-SHOT — their test videos were used to "
              "calibrate:")
        for target, ov in compromised.items():
            print(f"     {target}: {ov['n_shared_with_test']}/{ov['n_test_videos']} test videos "
                  f"seen ({ov['fraction_of_test_seen']:.0%}). Report it as calibrated, not OOD.")
    dataset = (datasets[0] if len(datasets) == 1
               else torch.utils.data.ConcatDataset(datasets))
    collate = datasets[0].collate_fn
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=int(data_cfg["workers"]), collate_fn=collate)

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
    # The provenance string must name the sources that ACTUALLY calibrated the policy. It used to
    # say "FF++ VAL_meta" unconditionally, which under --val-protocol diverse would have shipped a
    # frozen policy claiming FF++-only provenance while Celeb-DF-v2 test videos were in the fit —
    # the one thing a provenance field exists to prevent.
    calibrated_on = " + ".join(targets)
    if compromised:
        calibrated_on += f" [NOT zero-shot for: {', '.join(sorted(compromised))}]"
    policy = R.freeze_thresholds(labels, gated["prob"], risk,
                                 abstention_budget=args.abstention_budget,
                                 source=f"{calibrated_on} (out-of-fold q, epoch {epoch})")
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
        "calibration_sources": targets,
        "sources_no_longer_zero_shot": sorted(compromised),
        "gate_feature_names": list(G.FEATURE_NAMES),
        "risk_model": risk_model.state_dict(),
        "risk_feature_names": list(R.FEATURE_NAMES),
        "policy": policy.as_dict(),
        "delta": args.delta,
    }, args.output / "stage_de.pt")

    payload = {
        "checkpoint": str(checkpoint), "epoch": epoch,
        "selection": selection,
        "calibration": {
            "protocol": args.val_protocol if not args.val_sources else "explicit",
            "sources": targets,
            "overlap_with_test": overlaps,
            "sources_no_longer_zero_shot": sorted(compromised),
            "note": ("every source listed under sources_no_longer_zero_shot had its own TEST "
                     "videos used for calibration; its numbers must be reported as calibrated "
                     "rather than zero-shot (§1, §11, §18 require FF++-only calibration, so this "
                     "is a recorded deviation)")
            if compromised else "calibration touched no test video",
        },
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
