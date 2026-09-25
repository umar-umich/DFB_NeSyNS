#!/usr/bin/env python3
"""Gate G0b: would the IBDC hinge ever be active during training?

    <discern_ext python> training/joint_ibdc/gate_g0b.py

On the FF++ c23 training cache with the J0 heads, report the distribution of kappa and the fraction
of training samples where kappa > u_f -- the samples on which max(0, kappa - u_f) is non-zero and the
loss therefore has something to act on. PASS if that is at least 1% of training samples.

Reported per augmentation level, which is well defined here because the draws are matched across
views: one arm-B setting per (video, draw), so a sample has ONE level rather than one per view.

THE TRUST WEIGHTS ARE AMBIGUOUS IN THE BRIEF, SO BOTH READINGS ARE REPORTED. It says the weights are
"fixed at the deployed staged values during training", but the deployed staged values are fitted on
DEVELOPMENT and differ per fold, so there is no single deployed value to fix them at without
importing development information into training. Experiment 6 hit the same fork and resolved it
explicitly -- it trained against the EQUAL-TRUST fusion "since DS-R's weights are fitted on
development, so they are kept out of the training objective" (`tbiom/reports/EXP_JOINT_ABLATION.md`).
Equal trust is therefore the default here and the per-fold deployed triples are reported alongside,
because the choice materially changes kappa: discounting a view towards zero makes it vacuous, and a
vacuous view contributes no conflict at all.
"""
from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext")
sys.path[:0] = [str(ROOT / "analysis/tbiom/pcr6p"), str(ROOT / "analysis/tbiom/grouped"),
                str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/temporal"),
                str(Path(__file__).resolve().parent)]

import fusion_torch as FT        # noqa: E402
import draws as DR               # noqa: E402

CACHE = ROOT / "logs/joint_ibdc/cache"
OUT = ROOT / "analysis/joint_ibdc"
CKPT = EXT / "runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt"
BR = ROOT / "logs/tbiom/temporal/branches/vjepa2_lora_B_seed42"
VIEWS = ("T", "APP", "OM")
MIN_ACTIVE = 0.01


def spatial_evidence(sd, feat, norm_p, head_p):
    x = torch.as_tensor(feat, dtype=torch.float32)
    x = Fn.layer_norm(x, (x.shape[-1],), sd[f"{norm_p}.weight"].float(), sd[f"{norm_p}.bias"].float())
    h = Fn.relu(Fn.linear(x, sd[f"{head_p}.fc1.weight"].float(), sd[f"{head_p}.fc1.bias"].float()))
    return Fn.softplus(Fn.linear(h, sd[f"{head_p}.fc2.weight"].float(), sd[f"{head_p}.fc2.bias"].float()))


def temporal_evidence(feat):
    ck = torch.load(BR / "best.pt", map_location="cpu")
    head = torch.nn.Sequential(torch.nn.LayerNorm(1024), torch.nn.Linear(1024, 128),
                               torch.nn.GELU(), torch.nn.Linear(128, 2))
    head.load_state_dict(ck["head"]); head.eval()
    with torch.no_grad():
        return Fn.softplus(head(torch.as_tensor(feat, dtype=torch.float32)))


def weight_sets():
    """Equal trust, plus every per-fold deployed triple, so the choice is visible."""
    out = {"equal": (1.0, 1.0, 1.0)}
    f = ROOT / "tbiom/reports/EXP_PCR6P_DEVELOPMENT.json"
    if f.exists():
        for s in json.load(open(f))["42"]:
            out[f"deployed_fold{s['split']}"] = tuple(s["w"]["PCR6P-R"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=3)
    a = ap.parse_args()

    missing = [d for d in range(a.draws)
               if not (CACHE / f"FFpp_train_B_d{d}.npz").exists()
               or not (CACHE / f"FFpp_train_T_B_d{d}.npz").exists()]
    if missing:
        print(f"G0b cannot run yet; draws not cached: {missing}")
        return 2

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    WS = weight_sets()
    per_draw, agg = [], {k: {"active": 0, "n": 0} for k in WS}
    by_level = {k: {} for k in WS}

    for d in range(a.draws):
        zs = np.load(CACHE / f"FFpp_train_B_d{d}.npz", allow_pickle=True)
        zt = np.load(CACHE / f"FFpp_train_T_B_d{d}.npz", allow_pickle=True)
        e_app = spatial_evidence(sd, zs["semantic"], "model.semantic_norm", "model.semantic_head")
        e_om = spatial_evidence(sd, zs["artifact"], "model.artifact_norm", "model.artifact_head")
        e_t_v = temporal_evidence(zt["feature"])
        tv = {str(v): i for i, v in enumerate(zt["video"])}
        frame_vid = [DR.video_of(str(k)) for k in zs["key"]]
        keep = np.array([i for i, v in enumerate(frame_vid) if v in tv])
        idx = np.array([tv[frame_vid[i]] for i in keep])
        lvl = np.array([str(s) for s in zs["setting"]])[keep]
        e = torch.stack([e_t_v[idx], e_app[keep], e_om[keep]], dim=1)      # (n, 3, 2) in VIEWS order

        row = {"draw": d, "frames": int(len(keep)), "dropped_no_clip": int(len(frame_vid) - len(keep))}
        for name, w in WS.items():
            m = FT.opinion_of_evidence(e * torch.tensor(w).view(1, -1, 1))
            k = FT.kappa(m).numpy()
            uf = FT.vacuity_of(FT.fuse(m, plus=True)).numpy()
            act = k > uf
            row[name] = {"frac_active": float(act.mean()),
                         "kappa_mean": float(k.mean()), "kappa_p50": float(np.percentile(k, 50)),
                         "kappa_p90": float(np.percentile(k, 90)), "kappa_max": float(k.max()),
                         "uf_mean": float(uf.mean())}
            agg[name]["active"] += int(act.sum()); agg[name]["n"] += len(act)
            for L in np.unique(lvl):
                s = lvl == L
                b = by_level[name].setdefault(L, {"active": 0, "n": 0})
                b["active"] += int(act[s].sum()); b["n"] += int(s.sum())
        per_draw.append(row)
        print(f"  draw {d}: {len(keep)} frames, equal-trust active {row['equal']['frac_active']:.4f}, "
              f"kappa mean {row['equal']['kappa_mean']:.4f}", flush=True)

    overall = {k: (v["active"] / v["n"] if v["n"] else float("nan")) for k, v in agg.items()}
    ok = overall["equal"] >= MIN_ACTIVE
    res = {"gate": "G0b", "verdict": "PASS" if ok else "FAIL",
           "rule": f"hinge active on at least {MIN_ACTIVE:.0%} of training samples",
           "primary_weights": "equal (see the module docstring for why)",
           "frac_active_overall": overall, "per_draw": per_draw,
           "per_augmentation_level": {k: {L: {"frac_active": b["active"] / b["n"], "n": b["n"]}
                                          for L, b in v.items()} for k, v in by_level.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "gate_g0b.json", "w"), indent=1)

    print(f"\n  {'weights':18s} {'frac active':>12}")
    for k, v in overall.items():
        print(f"  {k:18s} {v:12.4f}")
    print(f"\n  per augmentation level (equal trust):")
    for L, b in sorted(by_level["equal"].items()):
        print(f"    {L:14s} {b['active'] / b['n']:.4f}  (n={b['n']})")
    print(f"\nG0b {res['verdict']} -> {OUT / 'gate_g0b.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
