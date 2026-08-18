#!/usr/bin/env python3
"""Task 0 — the encoder decision. Does LN tuning earn a second encoder?

Phase 2 asks one question before Stage II: measure how much LN tuning contributes to the
visual branch on the conventional axis, then

    within noise  -> ONE frozen CLIP for both e_sem and the reference input f_0 (simplest)
    material drop -> DUAL encoder: E_vis (LN-tuned) for e_sem, E_ref (frozen) for f_0

Either way the reference is fit on a frozen space; what is at stake is whether `e_sem` gives
up anything by sharing that space. 🟡 The number is reported here; **Umar locks the encoder**.

Why a linear probe rather than two training runs
-----------------------------------------------
The instructions call Task 0 cheap and put it before Stage II. Two full arms would answer the
question at the cost of the sweep they are supposed to gate. A linear probe on cached features
is the standard way to compare two frozen feature spaces, and it isolates exactly what is being
decided: the *feature space*, not the head, the loss, or the schedule — all of which are
identical downstream of `spatial_raw`. What it cannot see is an interaction between LN tuning
and the rest of the v2 stack, so a borderline verdict should escalate to real arms rather than
be resolved here.

Read the two axes separately
----------------------------
The tuned encoder's LayerNorms were moved by FF++ labels, so its **in-domain** probe number is
optimistically biased and is reported as context, not evidence. The decision rests on the
**cross-dataset** rows, which is also the axis DISCERN v2 exists to improve.

Noise band, stated before the verdict
-------------------------------------
"Within noise" needs a definition or the verdict is a judgement call dressed as a measurement.
The band is the wider of (a) the paired-bootstrap 95% CI half-width of the per-source delta and
(b) the spread across probe seeds. A delta inside the band is reported as *not resolved*, never
as "no difference" — an underpowered comparison is not evidence of equality.

Threshold discipline
--------------------
Every headline number is threshold-free (AUROC), so no operating point is chosen on any OOD
source. Probe regularisation is selected by cross-validation on the FF++ **train** split only;
standardisation uses FF++ train statistics only.

Usage (🔴 UMAR-RUNS):

    python analysis/discern_v2/t0_encoder_decision.py \
        --frozen-cache cache/discern_v2/clip_frozen \
        --tuned-cache  cache/discern_v2/clip_tuned \
        --eval FaceForensics++:test Celeb-DF-v3:test DFDC:test DFDCP:test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_io as C  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent
SEED = 42
# Fixed grid, searched on FF++ train only. Not tuned per encoder arm by hand: a grid searched
# identically for both arms is the only way the delta describes the features rather than the
# effort spent on each probe.
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]


def parse_targets(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for it in items:
        if ":" not in it:
            raise SystemExit(f"--eval takes SOURCE:SPLIT, got {it!r}")
        source, split = it.rsplit(":", 1)
        out.append((source, split))
    return out


def fit_probe(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int
              ) -> tuple[LogisticRegression, float]:
    """Linear probe with regularisation chosen by grouped CV on the training source only.

    Folds are grouped by video: frames from one video are near-duplicates, so a random split
    puts the same face in train and validation and the search picks a C that is too weak.
    """
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=min(4, max(2, n_groups)))
    search = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=seed),
        {"C": C_GRID}, scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(x, y, groups=groups)
    return search.best_estimator_, float(search.best_params_["C"])


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC, or nan for a single-class slice rather than a raised exception.

    Some OOD splits are reals-only by construction; a nan row is honest and keeps the rest of
    the table computable.
    """
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def paired_bootstrap(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, n: int = 2000,
                     seed: int = SEED) -> tuple[float, float]:
    """95% CI for AUROC(b) - AUROC(a), resampling videos jointly.

    Paired: both arms are scored on the same videos, so resampling them independently would
    add variance that does not exist in the comparison and widen the band until nothing is
    ever material.
    """
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    deltas = []
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[take])) < 2:
            continue
        deltas.append(auc(y[take], s_b[take]) - auc(y[take], s_a[take]))
    if not deltas:
        return float("nan"), float("nan")
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def drift(frozen: C.Cache, tuned: C.Cache) -> dict:
    """How far the tuned encoder moved the space for the same frames.

    This is the quantity the stationarity argument rests on, measured directly instead of
    inferred from a hash mismatch: if `1 - cos` is ~0 the two spaces are effectively the same
    and the probe delta must be ~0 too (a useful internal consistency check); if it is large,
    a reference fit in one space and read in the other is measuring drift.
    """
    fa, fb, _meta = C.join_on_key(frozen, tuned)
    num = (fa * fb).sum(axis=1)
    den = np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1) + 1e-12
    cos = num / den
    rel = np.linalg.norm(fb - fa, axis=1) / (np.linalg.norm(fa, axis=1) + 1e-12)
    return {"n": int(len(cos)), "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
            "one_minus_cos_mean": float((1 - cos).mean()),
            "relative_shift_mean": float(rel.mean())}


def evaluate(cache_dir: Path, train_source: str, train_split: str,
             targets: list[tuple[str, str]], seeds: list[int]
             ) -> tuple[dict, dict, dict]:
    """Fit probes on the training source and score every target. One encoder arm."""
    train = C.load_cache(cache_dir, train_source, train_split)
    mu, sigma = C.train_standardiser(train.features)
    x_train = C.apply_standardiser(train.features, mu, sigma)
    groups = (train.meta["method"].astype(str) + "/" + train.meta["video_id"].astype(str)).to_numpy()

    caches = {(s, sp): C.load_cache(cache_dir, s, sp) for s, sp in targets}
    C.assert_same_encoder(train, *caches.values())

    per_seed: dict[int, dict] = {}
    video_scores: dict[tuple[str, str], np.ndarray] = {}
    video_labels: dict[tuple[str, str], np.ndarray] = {}
    chosen_C: dict[int, float] = {}
    for seed in seeds:
        probe, best_c = fit_probe(x_train, train.labels, groups, seed)
        chosen_C[seed] = best_c
        row = {}
        for key, cache in caches.items():
            s = probe.decision_function(C.apply_standardiser(cache.features, mu, sigma))
            v = C.video_level(cache.meta, {"score": s})
            row[f"{key[0]}:{key[1]}"] = {
                "frame_auc": auc(cache.labels, s),
                "video_auc": auc(v["label"].to_numpy(), v["score"].to_numpy()),
                "n_frames": int(len(s)), "n_videos": int(len(v)),
            }
            if seed == seeds[0]:
                video_scores[key] = v["score"].to_numpy()
                video_labels[key] = v["label"].to_numpy()
        per_seed[seed] = row

    provenance = {
        "encoder_mode": train.encoder_mode,
        "fingerprint": train.fingerprint,
        "checkpoint": train.manifest.get("checkpoint"),
        "train": f"{train_source}/{train_split}",
        "n_train_frames": int(len(train.labels)),
        "probe_C_by_seed": chosen_C,
        "partial_cache": bool(train.manifest.get("partial")),
    }
    return per_seed, {"scores": video_scores, "labels": video_labels}, provenance


def summarise(per_seed: dict, level: str = "video_auc") -> dict[str, dict]:
    """mean / spread across seeds, per target."""
    targets = list(next(iter(per_seed.values())).keys())
    out = {}
    for t in targets:
        vals = np.array([per_seed[s][t][level] for s in per_seed], dtype=float)
        out[t] = {"mean": float(np.nanmean(vals)),
                  "spread": float(np.nanmax(vals) - np.nanmin(vals)) if len(vals) > 1 else 0.0}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frozen-cache", type=Path, required=True)
    ap.add_argument("--tuned-cache", type=Path, required=True)
    ap.add_argument("--train-source", default="FaceForensics++")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval", nargs="+", required=True,
                    help="SOURCE:SPLIT targets, e.g. FaceForensics++:test Celeb-DF-v3:test")
    ap.add_argument("--in-domain", default="FaceForensics++:test",
                    help="the target treated as in-domain context, not decision evidence")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=Path, default=OUT_ROOT / "T0_encoder")
    args = ap.parse_args()

    targets = parse_targets(args.eval)
    args.out.mkdir(parents=True, exist_ok=True)

    # checked before any probe is fit: if the two arms are the same encoder the whole
    # comparison is void, and finding that out after two full sweeps wastes the run.
    C.assert_different_encoder(
        C.load_cache(args.frozen_cache, args.train_source, args.train_split),
        C.load_cache(args.tuned_cache, args.train_source, args.train_split))

    print("=== frozen encoder arm ===")
    fro_seed, fro_v, fro_prov = evaluate(args.frozen_cache, args.train_source,
                                         args.train_split, targets, args.seeds)
    print("=== LN-tuned encoder arm ===")
    tun_seed, tun_v, tun_prov = evaluate(args.tuned_cache, args.train_source,
                                         args.train_split, targets, args.seeds)

    fro_sum, tun_sum = summarise(fro_seed), summarise(tun_seed)

    rows = []
    for source, split in targets:
        key = f"{source}:{split}"
        lo, hi = paired_bootstrap(fro_v["labels"][(source, split)],
                                  fro_v["scores"][(source, split)],
                                  tun_v["scores"][(source, split)])
        delta = tun_sum[key]["mean"] - fro_sum[key]["mean"]
        seed_band = max(fro_sum[key]["spread"], tun_sum[key]["spread"])
        ci_band = abs(hi - lo) / 2 if np.isfinite(hi) else float("nan")
        band = float(np.nanmax([seed_band, ci_band]))
        rows.append({
            "target": key,
            "frozen_video_auc": fro_sum[key]["mean"],
            "tuned_video_auc": tun_sum[key]["mean"],
            "delta": delta,
            "ci_lo": lo, "ci_hi": hi,
            "seed_spread": seed_band, "band": band,
            "material": bool(np.isfinite(band) and abs(delta) > band),
            "in_domain": key == args.in_domain,
        })
    table = pd.DataFrame(rows)

    # drift on every source present in both caches
    drifts = {}
    for source, split in targets:
        try:
            drifts[f"{source}:{split}"] = drift(
                C.load_cache(args.frozen_cache, source, split),
                C.load_cache(args.tuned_cache, source, split))
        except Exception as exc:                      # noqa: BLE001 — reported, not fatal
            drifts[f"{source}:{split}"] = {"error": str(exc)}

    ood = table[~table["in_domain"]]
    mean_ood_delta = float(ood["delta"].mean()) if len(ood) else float("nan")
    material_rows = ood[ood["material"]]
    n_material_down = int((material_rows["delta"] < 0).sum())
    n_material_up = int((material_rows["delta"] > 0).sum())

    if not len(ood):
        verdict, recommendation = "NOT RESOLVED", "no cross-dataset target was supplied"
    elif len(material_rows) == 0:
        verdict = "WITHIN NOISE"
        recommendation = (
            "SINGLE frozen CLIP for both e_sem and the reference input f_0. No cross-dataset "
            "row moved by more than its own noise band, so a second encoder would add a "
            "feature space, a set of weights and a drift risk to buy nothing measured here.")
    elif n_material_up and not n_material_down:
        verdict = "MATERIAL GAIN FROM LN TUNING"
        recommendation = (
            "DUAL encoder: E_vis (LN-tuned) for e_sem, E_ref (frozen snapshot) for f_0 and the "
            "reference. LN tuning earns its keep on the conventional axis, and the reference "
            "still must not read a drifting space.")
    elif n_material_down and not n_material_up:
        verdict = "MATERIAL LOSS FROM LN TUNING"
        recommendation = (
            "SINGLE frozen CLIP — and note the stronger finding: LN tuning *hurts* "
            "cross-dataset transfer here, so freezing is not merely the simpler choice.")
    else:
        verdict = "MIXED"
        recommendation = (
            "NOT RESOLVED by the probe: some sources gain and others lose materially. Escalate "
            "to two real D1-V arms (train_layernorms true/false) before locking the encoder — "
            "a probe cannot see interactions with the rest of the v2 stack.")

    result = {
        "verdict": verdict,
        "recommendation": recommendation,
        "mean_ood_delta_video_auc": mean_ood_delta,
        "n_material_sources": {"gain": n_material_up, "loss": n_material_down,
                               "total_ood": int(len(ood))},
        "table": rows,
        "frame_level": {"frozen": summarise(fro_seed, "frame_auc"),
                        "tuned": summarise(tun_seed, "frame_auc")},
        "drift": drifts,
        "provenance": {"frozen": fro_prov, "tuned": tun_prov, "seeds": args.seeds,
                       "probe": "logistic regression, C by GroupKFold CV on the train source"},
    }
    (args.out / "t0_result.json").write_text(json.dumps(result, indent=2, default=str))
    table.to_csv(args.out / "t0_table.csv", index=False)
    (args.out / "T0_ENCODER_DECISION.md").write_text(render(result, args))
    print(table.to_string(index=False))
    print(f"\nVERDICT: {verdict}\n{recommendation}")
    print(f"\nwrote {args.out}/T0_ENCODER_DECISION.md")
    print("🟡 ASK-UMAR: this recommends; the encoder lock is yours.")
    return 0


def render(result: dict, args) -> str:
    t = pd.DataFrame(result["table"])
    lines = [
        "# Task 0 — encoder decision",
        "",
        f"**Verdict: {result['verdict']}.** {result['recommendation']}",
        "",
        "🟡 ASK-UMAR: this is a recommendation from a linear probe. The encoder lock is Umar's.",
        "",
        "Generated by `analysis/discern_v2/t0_encoder_decision.py`. Every number below comes "
        "from the run recorded in `t0_result.json`; nothing is asserted from memory.",
        "",
        "## Method",
        "",
        f"Linear probe (logistic regression) on cached `spatial_raw` features, fit on "
        f"`{result['provenance']['frozen']['train']}` "
        f"({result['provenance']['frozen']['n_train_frames']} frames) for each encoder arm, "
        f"seeds {result['provenance']['seeds']}. Regularisation chosen by video-grouped CV on "
        f"the training source only; standardisation uses training statistics only; every "
        f"headline number is threshold-free AUROC at video level.",
        "",
        f"A delta counts as **material** only if it exceeds that source's noise band — the "
        f"wider of the paired-bootstrap 95% CI half-width and the across-seed spread. Deltas "
        f"inside the band are *not resolved*, which is not the same as equal.",
        "",
        "Encoder provenance:",
        "",
        f"- frozen: fingerprint `{result['provenance']['frozen']['fingerprint'][:16]}…`",
        f"- tuned:  fingerprint `{result['provenance']['tuned']['fingerprint'][:16]}…`, "
        f"checkpoint `{result['provenance']['tuned']['checkpoint']}`",
        "",
        "## Result (video-level AUROC)",
        "",
        "| target | frozen | LN-tuned | delta | 95% CI | band | material |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result["table"]:
        tag = " *(in-domain, context only)*" if r["in_domain"] else ""
        lines.append(
            f"| {r['target']}{tag} | {r['frozen_video_auc']:.4f} | {r['tuned_video_auc']:.4f} | "
            f"{r['delta']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['band']:.4f} | "
            f"{'**yes**' if r['material'] else 'no'} |")
    lines += [
        "",
        f"Mean cross-dataset delta (tuned − frozen): **{result['mean_ood_delta_video_auc']:+.4f}** "
        f"video AUROC over {result['n_material_sources']['total_ood']} OOD "
        f"target{'s' if result['n_material_sources']['total_ood'] != 1 else ''}; "
        f"{result['n_material_sources']['gain']} materially better, "
        f"{result['n_material_sources']['loss']} materially worse.",
        "",
        "The in-domain row is context, not evidence: the tuned encoder's LayerNorms were moved "
        "by FF++ labels, so its FF++ number is optimistically biased.",
        "",
        "## Feature-space drift",
        "",
        "How far LN tuning moved the space for the *same* frames. This is the stationarity "
        "concern measured directly rather than inferred from a hash mismatch.",
        "",
        "| target | n | mean 1−cos | mean ‖Δf‖/‖f‖ |",
        "|---|---|---|---|",
    ]
    for key, d in result["drift"].items():
        if "error" in d:
            lines.append(f"| {key} | — | — | (not computed: {d['error'][:60]}) |")
        else:
            lines.append(f"| {key} | {d['n']} | {d['one_minus_cos_mean']:.4f} | "
                         f"{d['relative_shift_mean']:.4f} |")
    lines += [
        "",
        "## What this cannot decide",
        "",
        "A probe compares feature spaces with everything downstream held identical. It cannot "
        "see an interaction between LN tuning and the rest of the v2 stack (fusion, the "
        "applicability gate, the process branch). A MIXED or borderline verdict should escalate "
        "to two real D1-V arms rather than be resolved here.",
        "",
        "Whatever is locked, `assert_reference_config` still requires the reference's input "
        "space to be frozen: under the dual-encoder option `e_sem` may read the tuned encoder, "
        "but `f_0` reads the frozen snapshot.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
