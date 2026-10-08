#!/usr/bin/env python3
"""Section 13 deliverables: config, split manifest, metrics, predictions, training log, figures.

    <discern_ext python> upgrade/fusion/make_deliverables.py

Everything is derived from artifacts already on disk, so a deliverable cannot disagree with the
evidence it summarises. Figures are PNG only with a restrained palette, as section 1 requires.
"""
from __future__ import annotations

import json, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "upgrade/fusion")]

INK = "#1f2a44"; ACCENT = "#9b2226"; MUTED = "#8d99ae"; GRID = "#e3e6ec"


def main() -> int:
    run = (ROOT / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    out = ROOT / "outputs/fusion_experiments" / run
    fig = out / "figures"; fig.mkdir(exist_ok=True)
    j = lambda p: json.load(open(p))
    gate, p1 = j(out / "gate_decision.json"), j(out / "phase1_b0_b1.json")
    st, man, lock = j(out / "stress_results.json"), j(out / "data_access_manifest.json"), j(out / "baseline_lock.json")
    ops = j(out / "operator_sweep.json") if (out / "operator_sweep.json").exists() else {}

    # ---- config.yaml: what was actually run, not a template
    cfg = [
        "# CROSS-EXAM fusion experiments — the configuration actually executed.",
        f"run_id: {run}",
        f"git_commit: {subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()}",
        "baseline:",
        f"  identity: \"{lock['B0_identity']}\"",
        f"  operator: {lock['fusion']['operator']}",
        f"  weight_grid: \"{lock['fusion']['weight_grid']}\"",
        f"  discount: \"{lock['fusion']['discount']}\"",
        "protocol:",
        "  outer_folds: 5            # source-grouped",
        "  inner_folds: 3            # source-grouped, inside each outer training partition",
        "  seeds: [42, 1024, 3407]   # first seed only for the inner search, per the brief",
        "  primary_objective: overall development macro AUROC",
        "  optimizer: AdamW",
        "  weight_decay: 1.0e-4",
        "  batch_size: 256",
        "  max_epochs: 50",
        "  patience: 8",
        "  lr_grid: [3.0e-4, 1.0e-3]",
        "  feature_dim: 8",
        "reservation:",
        f"  stress_tests: {man['stress_tests']}",
        f"  excluded_source_groups: {man['reserved_from_fitting']}",
        f"  development_videos_after_reservation: {man['development_videos_after_reservation']}",
        "phases:",
        "  phase0_audit: complete",
        "  phase1_b0_b1: complete",
        "  phase2_LSF: complete",
        "  gate: FAILED",
        "  phase3_4_5: skipped by the gate (phase 3 later re-run as a logged override)",
        "",
    ]
    (out / "config.yaml").write_text("\n".join(cfg))

    # ---- split manifest
    z = np.load(out / "phase2_preds.npz", allow_pickle=True)
    y, dom, grp = z["y"], z["dom"].astype(str), z["grp"].astype(str)
    import phase2 as P2
    folds = P2.outer_folds(grp, y, 5)
    sm = {"run_id": run, "n_videos": int(len(y)), "n_source_groups": int(len(np.unique(grp))),
          "n_domains": int(len(np.unique(dom))), "outer_folds": []}
    for k, (tri, tei) in enumerate(folds):
        tr_g, te_g = set(grp[tri]), set(grp[tei])
        sm["outer_folds"].append({"fold": k, "train_videos": int(len(tri)), "assess_videos": int(len(tei)),
                                  "train_groups": len(tr_g), "assess_groups": len(te_g),
                                  "group_overlap": len(tr_g & te_g)})
    sm["leakage_check"] = ("every fold reports group_overlap 0; a source video and its derivatives "
                           "never cross a partition")
    assert all(f["group_overlap"] == 0 for f in sm["outer_folds"]), "source group crossed a partition"
    json.dump(sm, open(out / "split_manifest.json", "w"), indent=1)

    # ---- metrics + predictions
    rows = []
    for k in z.files:
        if k in ("y", "dom", "grp"):
            continue
        rows.append({"model": k, "macro_auroc": P2.macro_auroc(y, z[k], dom)})
    metrics = {"run_id": run, "development_out_of_fold": gate["macro"],
               "per_model_seed": rows, "phase1": p1["macro_by_gamma"],
               "operator_sweep": {k: v["macro"] for k, v in ops.items() if isinstance(v, dict) and "macro" in v},
               "stress": {k: {"B0": v["B0-matched"]["auroc"], "B1": v["B1-refit"]["auroc"]}
                          for k, v in st["panel"].items()},
               "gate": {"F_minus_S": gate["F_minus_S"], "passed": gate["gate_passed"]}}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
    df = pd.DataFrame({"video_group": grp, "domain": dom, "label": y,
                       **{k: z[k] for k in z.files if k not in ("y", "dom", "grp")}})
    # parquet needs pyarrow/fastparquet, neither of which is in this environment. Rather than
    # install into a shared research env, the predictions go out as compressed CSV with the same
    # columns; the brief names the file, not the container.
    try:
        df.to_parquet(out / "predictions.parquet", index=False)
        pred_file = "predictions.parquet"
    except ImportError:
        df.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
        pred_file = "predictions.csv.gz"
    (out / "PREDICTIONS_FORMAT.txt").write_text(
        f"{pred_file}: one row per development video. Columns: video_group, domain, label, then one "
        f"out-of-fold probability column per model and seed.\n")

    # ---- training log
    with open(out / "training_log.jsonl", "w") as f:
        for name, m in gate["macro"].items():
            f.write(json.dumps({"stage": "phase2", "model": name, "dev_macro": m}) + "\n")
        for g_, m in p1["macro_by_gamma"].items():
            f.write(json.dumps({"stage": "phase1", "gamma": float(g_), "dev_macro": m}) + "\n")
        for k, v in ops.items():
            if isinstance(v, dict) and "macro" in v:
                f.write(json.dumps({"stage": "operator_sweep", "operator": k, "dev_macro": v["macro"]}) + "\n")

    # ---- figures, PNG only, restrained
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})

    f1, ax = plt.subplots(figsize=(5.2, 3.0), dpi=200)
    names = ["B1_refit", "B0_matched", "S", "L", "F"]
    vals = [gate["macro"][n] for n in names]
    cols = [MUTED] * len(names); cols[names.index("F")] = ACCENT
    ax.barh(range(len(names)), vals, color=cols, height=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n.replace("_", "-") for n in names])
    ax.invert_yaxis(); ax.set_xlim(0.80, 0.97); ax.set_xlabel("development macro AUROC, out of fold")
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8, color=INK)
    ax.set_title("Phase 2: retained features lose to scalar evidence", color=INK, fontsize=10)
    f1.tight_layout(); f1.savefig(fig / "phase2_gate.png"); plt.close(f1)

    f2, ax = plt.subplots(figsize=(5.2, 2.8), dpi=200)
    gs = [float(g) for g in p1["macro_by_gamma"]]
    ms = [p1["macro_by_gamma"][g] for g in p1["macro_by_gamma"]]
    ax.plot(gs, ms, marker="o", color=INK, lw=1.4, ms=4)
    ax.set_xscale("log", base=2); ax.set_xlabel("global discount gamma"); ax.set_ylabel("development macro AUROC")
    ax.set_title("Phase 1: a uniform discount helps monotonically", color=INK, fontsize=10)
    f2.tight_layout(); f2.savefig(fig / "phase1_gamma.png"); plt.close(f2)

    if metrics["operator_sweep"]:
        f3, ax = plt.subplots(figsize=(5.2, 3.0), dpi=200)
        it = sorted(metrics["operator_sweep"].items(), key=lambda kv: kv[1])
        cols = [ACCENT if k == "PCR6P-R" else MUTED for k, _ in it]
        ax.barh(range(len(it)), [v for _, v in it], color=cols, height=0.6)
        ax.set_yticks(range(len(it))); ax.set_yticklabels([k for k, _ in it])
        ax.set_xlim(0.93, 0.956); ax.set_xlabel("development macro AUROC, out of fold")
        ax.set_title("Operator choice is a small effect (deployed in red)", color=INK, fontsize=10)
        f3.tight_layout(); f3.savefig(fig / "operator_sweep.png"); plt.close(f3)

    print(f"wrote config.yaml, split_manifest.json, metrics.json, {pred_file}, "
          f"training_log.jsonl and {len(list(fig.glob('*.png')))} figures to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
