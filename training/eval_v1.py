#!/usr/bin/env python3
"""Evaluate a Stage-B checkpoint on FF++ test and the OOD suite (§20, §21).

    🔴 UMAR-RUNS.

    python training/eval_v1.py \
        --checkpoint logs/v1/stage_b_seed42/epoch_001.pth \
        --datasets FaceForensics++ Celeb-DF-v2 Celeb-DF-v3 DFDC DFDCP \
        --output logs/v1/eval/epoch_001 --device cuda:N

**This is not checkpoint selection.** §11 selects on VAL_select only and says epoch-wise OOD
scores are saved "for later analysis only". Nothing this script writes may pick an epoch, a
hyperparameter, or a threshold; the output filename carries the epoch so the distinction survives
into the results directory.

What "fused" means here, precisely
---------------------------------
At Stage B there is no applicability gate (Stage D) and no defer policy (Stage E), so fusion runs
**ungated, q = 1** — the same configuration Stage B trains under. Every table this writes is
therefore *plain DS over three ungated experts*, which is the baseline the applicability layer is
later supposed to beat, not the V1 system. Reported explicitly rather than left for the reader to
infer, because "DISCERN-v2 V1" in a paper table means the gated, calibrated system.

Per-sample instrumentation (§20)
--------------------------------
Writes one row per frame with everything §20 lists: per-branch evidence / p / u, `e_direct` and
its prediction, `branch_valid_*`, reference and process diagnostics, the DS conflict and its
degenerate flag, fused p / u, and V / C / A. That file is the input to the §20 domain-detector
distribution audit and to the post-hoc contribution diagnostics, so those never need a second
forward pass.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "analysis" / "discern_v2"))

from networks.discern_v2.discern_v1_model import build_v1_model  # noqa: E402
from networks.discern_v2 import risk_model as R  # noqa: E402
from networks.discern_v2.dirichlet import to_dirichlet  # noqa: E402
from train_v1 import ViewMaker, prepare_dataset_config, video_of  # noqa: E402

BRANCHES = ("sem", "ref", "proc", "rate")


def load_model(checkpoint: Path, device: str):
    """Rebuild the model from the checkpoint's own config snapshot, then load its weights.

    The config travels inside the checkpoint, so an evaluation cannot silently use a different
    reference artifact or process statistics than the run was trained with.
    """
    blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = build_v1_model({
        "backbone": cfg["semantic"]["backbone"],
        "semantic_feature_dim": cfg["semantic"]["feature_dim"],
        "enable_lora": cfg["semantic"]["enable_lora"],
        "reference": {**cfg["reference"],
                      "artifact_path": str(REPO / cfg["reference"]["artifact_path"]),
                      "fsvfm_checkpoint": str(REPO / cfg["reference"]["fsvfm_checkpoint"])},
        "process": {**cfg["process"],
                    "stats_artifact": str(REPO / cfg["process"]["stats_artifact"])},
    }, stage="B")
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing:
        raise SystemExit(
            f"{len(missing)} parameters missing from {checkpoint} (first: {list(missing)[:5]}). "
            f"Refusing to evaluate a partially-restored model.")
    if unexpected:
        print(f"  note: {len(unexpected)} unexpected keys ignored (first: {list(unexpected)[:3]})")
    model.to(device).eval()
    model.assert_frozen_protocol()
    return model, cfg, blob.get("epoch")


def load_stage_de(path: Path, device: str, arm: str | None = None) -> dict:
    """The frozen Stage-D gates, Stage-E risk model and defer policy.

    Loaded read-only and applied unchanged: §18's policy carries the thresholds it was frozen
    with, and re-deriving either of them on an evaluation source is the leak §11 and §18 forbid.
    """
    from networks.discern_v2.applicability_gate import ApplicabilityGate
    from networks.discern_v2.risk_model import DeferPolicy, RiskModel

    blob = torch.load(str(path), map_location="cpu", weights_only=False)

    # Two artifact formats. V1's `stage_de.pt` holds one gate set, one risk model and one policy;
    # phase-2's `stage567.pt` holds a gate set PER fusion operator and a risk model and policy per
    # ARM, because V and the fused margin are operator-dependent. Detected from the blob rather
    # than from a flag, so an artifact from either stage can be applied without the caller having
    # to know which produced it.
    phase2 = "risk_models" in blob
    if phase2:
        arm = arm or blob.get("primary_arm")
        if arm not in blob["risk_models"]:
            raise SystemExit(
                f"arm {arm!r} is not in this artifact; available: "
                f"{sorted(blob['risk_models'])}. The arm decides which fusion operator and which "
                f"frozen policy are applied, so it cannot be guessed.")
        operator = "ccf" if arm.endswith("ccf") else "ds"
        # Gates were cross-fit under a specific operator; the arm names which. `equal_*` arms have
        # no gate at all — every specialist enters at q = 1 by definition.
        gate_states = {} if arm.startswith("equal") else blob["gates"][operator]
        risk_state = blob["risk_models"][arm]
        policy_dict = blob["policies"][arm]
    else:
        arm, operator = "v1_ds", "ds"
        gate_states = blob["gates"]
        risk_state = blob["risk_model"]
        policy_dict = blob["policy"]

    gates = {}
    for name, state in gate_states.items():
        gate = ApplicabilityGate(n_features=len(blob["gate_feature_names"]))
        gate.load_state_dict(state)
        gate.eval()
        gates[name] = gate.to(device)
    risk = RiskModel(len(blob["risk_feature_names"]))
    risk.load_state_dict(risk_state)
    risk.eval()
    policy = DeferPolicy(**policy_dict)
    print(f"  frozen policy [{arm} / {operator}]: gates {sorted(gates) or 'none (q = 1)'} "
          f"from epoch {blob.get('epoch')}; policy {policy.as_dict()}")
    if blob.get("sources_no_longer_zero_shot"):
        print(f"  !! calibrated on {blob['sources_no_longer_zero_shot']} — those sources are "
              f"NOT zero-shot in any table built from this run")
    return {"gates": gates, "risk": risk.to(device), "policy": policy, "arm": arm,
            "operator": operator, "epoch": blob.get("epoch"), "source": str(path),
            "calibration_sources": blob.get("calibration_sources"),
            "sources_no_longer_zero_shot": blob.get("sources_no_longer_zero_shot") or []}


@torch.no_grad()
def apply_gates(model, branches: dict, stage_de: dict) -> dict:
    """q_b from the frozen gates, then the model's own fusion path with those weights.

    `model.fuse` is reused rather than reimplemented so the evaluated fusion is byte-for-byte the
    deployment fusion; a second implementation here could drift from the one Stage E calibrated
    against, and the defer thresholds would then be applied to a slightly different quantity.
    """
    from networks.discern_v2.applicability_gate import gate_features
    from networks.discern_v2.ds_fusion import Opinion, apply_validity

    def opinion_of(name: str) -> Opinion:
        b = branches[name]
        return apply_validity(Opinion.from_evidence(b["evidence"]), b["valid"].float())

    sem = opinion_of("sem")
    q = {}
    for name, gate in stage_de["gates"].items():
        if name in branches:
            q[name] = gate(gate_features(sem, opinion_of(name)))
    fusion_set = {k: v for k, v in branches.items() if k != "direct"}
    # The operator travels with the artifact: a policy frozen against CCF's vacuity must be
    # applied to CCF's vacuity, or two of the risk model's four inputs are on the wrong scale.
    return {**model.fuse(fusion_set, q, operator=stage_de.get("operator", "ds")), "q": q}


@torch.no_grad()
def evaluate_dataset(model, loader, views, device: str, dataset: str,
                     max_batches: int = 0, stage_de: dict | None = None) -> pd.DataFrame:
    from tqdm import tqdm

    rows: list[dict] = []
    for i, batch in enumerate(tqdm(loader, desc=f"  {dataset}", leave=False)):
        if max_batches and i >= max_batches:
            break
        labels = torch.where(batch["label"] != 0, 1, 0)
        out = model(views(batch, device))
        branches = out["branches"]

        record = {
            "dataset": dataset,
            "key": batch["name"],
            "video_id": [video_of(p) for p in batch["name"]],
            "label": labels.numpy(),
            "prob_fused": out["prob"].cpu().numpy(),
            "u_fused": out["fused"].vacuity.squeeze(1).cpu().numpy(),
            "V": out["V"].cpu().numpy(),
            "C": out["C"].cpu().numpy(),
            "A": out["A"].cpu().numpy(),
            "ds_conflict": out["ds_conflict"].cpu().numpy(),
            "ds_degenerate": out["ds_degenerate"].cpu().numpy(),
        }
        for name in BRANCHES:
            if name not in branches:
                continue
            state = to_dirichlet(branches[name]["evidence"])
            record[f"p_{name}"] = state.fake_prob().cpu().numpy()
            record[f"u_{name}"] = state.vacuity.cpu().numpy()
            record[f"e_{name}_real"] = state.evidence[:, 0].cpu().numpy()
            record[f"e_{name}_fake"] = state.evidence[:, 1].cpu().numpy()
            record[f"valid_{name}"] = branches[name]["valid"].cpu().numpy()
        if "direct" in branches:
            direct = to_dirichlet(branches["direct"]["evidence"])
            record["p_direct"] = direct.fake_prob().cpu().numpy()
            record["u_direct"] = direct.vacuity.cpu().numpy()
        if "proc" in branches and "raw_stats" in branches["proc"]["diagnostics"]:
            stats = branches["proc"]["diagnostics"]["raw_stats"].cpu().numpy()
            for j, sname in enumerate(model.process.stat_names):
                record[f"proc_{sname}"] = stats[:, j]
        if "ref" in branches:
            diag = branches["ref"]["diagnostics"]
            for key, col in (("reference_residual_norm", "ref_residual_norm"),
                             ("reference_angle", "ref_angle")):
                if key in diag:
                    record[col] = diag[key].cpu().numpy()
        if "rate" in branches and "raw_stats" in branches["rate"]["diagnostics"]:
            # The K rate-distortion components, one column each. Exported RAW rather than
            # standardized: Stage 2.3 asks whether R(x) carries family-specific SHAPE, and
            # exporting only the head's summary would answer a different question — the head is
            # trained to separate real from fake, so its output cannot show whether the curve
            # itself was structured.
            for sname, values in branches["rate"]["diagnostics"]["raw_stats"].items():
                record[sname] = values.cpu().numpy()

        if stage_de is not None:
            gated = apply_gates(model, branches, stage_de)
            # `A` and `U_sup` are the same tensor (renamed in phase 2); read the canonical name
            risk_features = R.risk_features(gated["V"], gated["C"],
                                            gated.get("U_sup", gated["A"]), gated["prob"])
            risk = stage_de["risk"](risk_features)
            decision = stage_de["policy"].decide(gated["prob"], risk)
            record.update({
                "prob_gated": gated["prob"].cpu().numpy(),
                "u_gated": gated["fused"].vacuity.squeeze(1).cpu().numpy(),
                "V_gated": gated["V"].cpu().numpy(),
                "C_gated": gated["C"].cpu().numpy(),
                "A_gated": gated["A"].cpu().numpy(),
                "risk": risk.cpu().numpy(),
                "decision": decision.cpu().numpy(),
            })
            for name, values in gated["q"].items():
                record[f"q_{name}"] = values.cpu().numpy()
        rows.append(pd.DataFrame(record))
    if not rows:
        raise RuntimeError(f"{dataset}: no batches produced predictions")
    return pd.concat(rows, ignore_index=True)


def fix_degenerate_video_ids(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, str | None]:
    """Fall back to per-file ids where the parent directory is not a video.

    `video_of()` takes the parent directory, which is the video for every frame-extracted corpus.
    DF40's whole-image generators are stored flat as `<method>/<half>/<half>/<n>.jpg`, so the
    parent directory IS the class: every real image becomes one "video" called `real` and every
    fake one called `fake`. Video AUROC over two points whose scores are the class means is then
    exactly 1.0 or 0.0 — a spectacular-looking number that measures nothing at all, which is
    worse than a missing one because it looks like a result.

    Detected from the data rather than from a method whitelist, so any flat corpus is caught:
    a handful of distinct ids covering many frames is not a video structure.
    """
    n_ids = df["video_id"].nunique()
    if n_ids > 4 or len(df) < 4 * max(n_ids, 1):
        return df, None
    # The FULL path minus its suffix, not the bare file stem. DF40's flat methods store the two
    # classes as .../<half>/<half>/<n>.jpg, so `real/real/1332.jpg` and `fake/fake/1332.jpg` share
    # the stem `1332`: grouping on it pairs each real with its fake, label=max() marks every group
    # fake, and the video-level array becomes single-class -> AUROC nan for every branch while the
    # frame AUROC still looks fine. The path is unique per file by construction.
    df = df.assign(video_id=[str(Path(k).with_suffix("")) for k in df["key"]])
    note = (f"{dataset}: only {n_ids} distinct parent directories over {len(df)} frames — the "
            f"layout is flat, so each file is its own sample and video aggregation now uses the "
            f"full file path. Video AUROC therefore EQUALS frame AUROC here: there is no video "
            f"level in this corpus. Grouping on the parent would have given a 2-point AUROC of "
            f"1.0 or 0.0, and grouping on the bare stem gives nan (real and fake share stems).")
    return df, note


def metrics_for(df: pd.DataFrame, score_col: str) -> dict:
    """Frame and video AUROC. Video aggregation is mean frame p(fake), per §21."""
    from sklearn.metrics import roc_auc_score

    def auc(y, s):
        return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")

    video = df.groupby(["dataset", "video_id"], as_index=False).agg(
        score=(score_col, "mean"), label=("label", "max"))
    # DF40's whole-image methods list the same authentic image under several entries, so the
    # frame-level number double-counts those reals while the grouped number counts each file
    # once. Reported so the gap between the two is explained rather than puzzling.
    duplicates = int(len(df) - df["key"].nunique())
    conflicting = int((df.groupby("key")["label"].nunique() > 1).sum())
    return {
        "duplicate_keys": duplicates,
        "keys_with_conflicting_labels": conflicting,
        "frame_auroc": auc(df["label"].to_numpy(), df[score_col].to_numpy()),
        "video_auroc": auc(video["label"].to_numpy(), video["score"].to_numpy()),
        "n_frames": int(len(df)),
        "n_videos": int(len(video)),
        "n_real_videos": int((video["label"] == 0).sum()),
        "n_fake_videos": int((video["label"] == 1).sum()),
    }


def selective_summary(df: pd.DataFrame) -> dict:
    """Coverage and selective accuracy under the FROZEN policy, at video level (§18, §21).

    Coverage is expected to DIFFER from the calibration budget on every OOD source: the threshold
    was frozen on FF++ validation, so a harder source defers more. Coverage pinned at the budget
    everywhere would mean the threshold had been retuned per source.
    """
    from networks.discern_v2.risk_model import DEFER, DECISION_NAMES

    # Frame-level decisions are authoritative — the policy is applied per frame — so a video is
    # answered when fewer than half its frames were deferred.
    video = df.groupby(["dataset", "video_id"], as_index=False).agg(
        prob=("prob_gated", "mean"), label=("label", "max"),
        defer_frac=("decision", lambda d: float((d == DEFER).mean())))
    answered = video["defer_frac"] < 0.5
    predicted = (video["prob"] >= 0.5).astype(int)
    correct = (predicted == video["label"]) & answered
    return {
        "frame_level": {DECISION_NAMES[k]: int((df["decision"] == k).sum())
                        for k in DECISION_NAMES},
        "frame_coverage": float((df["decision"] != DEFER).mean()),
        "video_coverage": float(answered.mean()),
        "video_selective_accuracy": (float(correct.sum() / answered.sum())
                                     if int(answered.sum()) else float("nan")),
        "n_videos": int(len(video)),
        "note": ("coverage is expected to differ from the calibration budget — the threshold is "
                 "frozen on FF++ validation, so a harder source defers more"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--detector-config", type=Path,
                    default=REPO / "training/config/detector/nesy_defake_d1_v.yaml")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="pick the GPU explicitly; this script never chooses one for you")
    ap.add_argument("--seed", type=int, default=42,
                    help="seeds the dataset shuffle. `abstract_dataset.py:346` shuffles the "
                         "collected frame list with the GLOBAL `random` module, so an unseeded "
                         "run covers a DIFFERENT subset of frames than the next one. With "
                         "--max-batches that is not cosmetic: this export and a score_fpad.py "
                         "export of the same sources shared only 11,810 of ~14,000 videos, and "
                         "any complementarity computed across them would have mixed a sampling "
                         "difference into the comparison.")
    ap.add_argument("--df40", action="store_true",
                    help="treat --datasets as DF40 per-method names: use DF40's json folder, "
                         "inject its per-method label_dict, and remap its relative frame paths")
    ap.add_argument("--dataset-json-folder", type=Path, default=None,
                    help="override the json folder (DF40 keeps its own)")
    ap.add_argument("--arm", default=None,
                    help="phase-2 stage567.pt only: which fusion arm's frozen gates, risk model "
                         "and policy to apply (default: the artifact's primary_arm)")
    ap.add_argument("--stage-de", type=Path, default=None,
                    help="stage_de.pt from training/stage_de.py: apply the FROZEN gates and defer "
                         "policy, giving true V1 numbers alongside the ungated baseline")
    ap.add_argument("--overwrite", action="store_true",
                    help="discard existing results for this epoch in --output instead of refusing")
    args = ap.parse_args()

    # Seed BEFORE any dataset is constructed: the shuffle happens at collection time, so seeding
    # later would not change which frames this export covers.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    model, cfg, epoch = load_model(args.checkpoint, args.device)
    print(f"loaded epoch {epoch} from {args.checkpoint}")

    # Outputs are named by epoch, so a second invocation for the SAME epoch into the SAME
    # directory silently overwrites the first run's results and per-sample export — which is
    # exactly what a follow-up run (say, adding DF40) looks like. Refused rather than clobbered;
    # domain_audit.py accepts several parquets, so separate directories are the natural pattern.
    existing = [p for p in (args.output / f"results_epoch_{epoch}.json",
                            args.output / f"per_sample_epoch_{epoch}.parquet") if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"{args.output} already holds results for epoch {epoch} "
            f"({', '.join(p.name for p in existing)}). Use a different --output (e.g. "
            f"{args.output}_df40) and pass both parquets to domain_audit.py, or --overwrite to "
            f"discard what is there.")

    data_cfg = prepare_dataset_config(args.detector_config, args.batch_size, args.workers)
    clip_norm = data_cfg["foundation_models"]["spatial"]["normalization"]
    views = ViewMaker(clip_norm["mean"], clip_norm["std"], args.device, augment=False)

    stage_de = (load_stage_de(args.stage_de, args.device, args.arm)
                if args.stage_de else None)
    if stage_de and stage_de.get("epoch") not in (None, epoch):
        raise SystemExit(
            f"the Stage-D/E artifact was built on epoch {stage_de['epoch']} but this checkpoint is "
            f"epoch {epoch}. §13's gate target is defined from the frozen SELECTED experts, so "
            f"gates from one epoch applied to another describe a model that was never calibrated.")

    df40 = None
    if args.df40:
        from dataset import df40_paths as df40

        json_dir = args.dataset_json_folder or df40.DF40_JSON_DIR
        data_cfg["dataset_json_folder"] = str(json_dir)
        # DF40 labels are per method and absent from the shared config, which RAISES on unknown
        # keys; injected for this run only so test_config.yaml stays untouched.
        data_cfg["label_dict"] = {**data_cfg.get("label_dict", {}),
                                  **df40.label_dict_for(args.datasets, json_dir)}
        print(f"DF40 mode: json folder {json_dir}, "
              f"{len(data_cfg['label_dict'])} label keys after injection")
    elif args.dataset_json_folder:
        data_cfg["dataset_json_folder"] = str(args.dataset_json_folder)

    from dataset.nesy_defake_dataset import NeSyDeFakeDataset

    results: dict[str, dict] = {}
    resolution: dict[str, dict] = {}
    aggregation_notes: dict[str, str] = {}
    frames: list[pd.DataFrame] = []
    for name in args.datasets:
        print(f"\n=== {name} ===")
        cfg_ds = {**data_cfg, "test_dataset": name}
        dataset = NeSyDeFakeDataset(cfg_ds, mode="test")
        if df40 is not None:
            report = df40.remap_dataset(dataset)
            resolution[name] = {**report, "family": df40.family_of(name)}
            print(f"  path remap: kept {report['kept']}, dropped {report['dropped']} "
                  f"(rate {report['resolution_rate']:.3f}, family {resolution[name]['family']})")
            if report["resolution_rate"] < 0.99:
                print(f"  WARNING: {name} is missing "
                      f"{(1 - report['resolution_rate']) * 100:.1f}% of its frames — the AUROC "
                      f"below is computed over a partial method")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=int(data_cfg["workers"]), collate_fn=dataset.collate_fn)
        df = evaluate_dataset(model, loader, views, args.device, name, args.max_batches,
                              stage_de=stage_de)
        df, note = fix_degenerate_video_ids(df, name)
        if note:
            print(f"  NOTE {note}")
            aggregation_notes[name] = note
        frames.append(df)

        per_source = {"fused_ungated": metrics_for(df, "prob_fused")}
        for branch in BRANCHES:
            if f"p_{branch}" in df:
                per_source[branch] = metrics_for(df, f"p_{branch}")
        if "p_direct" in df:
            per_source["direct_probe_control"] = metrics_for(df, "p_direct")
        per_source["reliability_means"] = {k: float(df[k].mean()) for k in ("V", "C", "A")}
        if stage_de is not None:
            per_source["fused_gated"] = metrics_for(df, "prob_gated")
            per_source["gate_means"] = {c: float(df[c].mean()) for c in df.columns
                                        if c.startswith("q_")}
            per_source["reliability_means_gated"] = {
                k: float(df[f"{k}_gated"].mean()) for k in ("V", "C", "A")}
            per_source["selective"] = selective_summary(df)
            # the §20 diagnostic §24 hinges on: does applicability beat plain DS on this source?
            per_source["applicability_delta_video_auroc"] = (
                per_source["fused_gated"]["video_auroc"]
                - per_source["fused_ungated"]["video_auroc"])
        per_source["ds_degenerate_rate"] = float(df["ds_degenerate"].mean())
        per_source["branch_validity"] = {
            f"valid_{b}": float(df[f"valid_{b}"].mean()) for b in BRANCHES
            if f"valid_{b}" in df}
        results[name] = per_source

        f = per_source["fused_ungated"]
        print(f"  fused (ungated)  video AUROC {f['video_auroc']:.4f}  "
              f"frame {f['frame_auroc']:.4f}  ({f['n_videos']} videos)")
        if stage_de is not None:
            g = per_source["fused_gated"]
            sel = per_source["selective"]
            print(f"  fused (GATED)    video AUROC {g['video_auroc']:.4f}  frame "
                  f"{g['frame_auroc']:.4f}   applicability delta "
                  f"{per_source['applicability_delta_video_auroc']:+.4f}")
            print(f"  defer            video coverage {sel['video_coverage']:.3f}, selective "
                  f"accuracy {sel['video_selective_accuracy']:.4f}")
        for key in [*BRANCHES, "direct_probe_control"]:
            if key in per_source:
                print(f"  {key:22s} video AUROC {per_source[key]['video_auroc']:.4f}")

    all_frames = pd.concat(frames, ignore_index=True)
    all_frames.to_parquet(args.output / f"per_sample_epoch_{epoch}.parquet", index=False)

    payload = {
        "checkpoint": str(args.checkpoint),
        "epoch": epoch,
        "IMPORTANT": (
            "`fused_ungated` is plain DS over three ungated experts (q=1) — the baseline the "
            "applicability layer must beat, NOT the V1 system. `fused_gated` (present only with "
            "--stage-de) is the V1 system: frozen Stage-D gates and the frozen Stage-E defer "
            "policy, neither retuned on any evaluation source."
            if stage_de else
            "Stage-B checkpoint scored with UNGATED fusion (q=1): no applicability gate (Stage D) "
            "and no defer policy (Stage E) were supplied. These are plain-DS numbers over three "
            "ungated experts — the baseline the applicability layer must beat, NOT the V1 system. "
            "Pass --stage-de to score the V1 system."),
        "stage_de": ({"source": stage_de["source"], "epoch": stage_de["epoch"],
                      "arm": stage_de.get("arm"), "operator": stage_de.get("operator"),
                      "calibration_sources": stage_de.get("calibration_sources"),
                      "sources_no_longer_zero_shot":
                          stage_de.get("sources_no_longer_zero_shot", []),
                      "policy": stage_de["policy"].as_dict()} if stage_de else None),
        "selection_note": ("§11: OOD scores are for later analysis only and must not select an "
                           "epoch, a hyperparameter, or a threshold."),
        "video_aggregation": "mean frame p(fake) (§21)",
        "df40_path_resolution": resolution or None,
        "video_aggregation_notes": aggregation_notes or None,
        "results": results,
    }
    (args.output / f"results_epoch_{epoch}.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.output}/results_epoch_{epoch}.json and the per-sample parquet")
    print("Reminder: this cannot select the checkpoint — §11 keeps that on VAL_select.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
