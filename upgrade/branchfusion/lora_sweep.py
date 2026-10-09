#!/usr/bin/env python3
"""Section 8: LoRA ablations of rank, scaling and dropout, with the implementation verified first.

    <discern_ext_timm python> upgrade/branchfusion/lora_sweep.py --verify
    <discern_ext_timm python> upgrade/branchfusion/lora_sweep.py --plan
    <discern_ext_timm python> upgrade/branchfusion/lora_sweep.py --emit-jobs --gpu 1 > jobs.sh

VERIFY THE IMPLEMENTATION BEFORE SWEEPING IT. `--verify` confirms numerically that the installed
adapter computes `dW = (alpha / r) B A`, that B starts at zero so the adapted model is bit-identical
to the foundation model at step 0, that no base parameter receives a gradient, and -- for the
masked-module variant -- that the K update is exactly zero. Section 8 asks for each of these, and a
sweep over a misread scaling convention would be worse than no sweep.

THE CENTRE IS B0'S APPEARANCE RECIPE and every cell keeps LayerNorm tuning on with the target modules
fixed, so the centre cell IS B0 rather than a near-miss: LoRA r = 8, alpha = 16 (alpha/r = 2),
dropout 0.1 on PE-Core's fused QKV, LayerNorm trained.

TEN UNIQUE CONFIGURATIONS, because the centre appears in all three one-factor sweeps:

  rank     r in {2, 4, 8, 16}, alpha = 2r      (alpha/r held at 2)
  scaling  alpha/r in {0.5, 1, 2, 4}, r = 8
  dropout  {0, 0.05, 0.1, 0.2}, r = 8, alpha/r = 2

Plus three controls at the centre -- frozen, LayerNorm only, LoRA only -- which already exist from
the upgrade campaign and are reused rather than re-run.

"LayerNorm" IS WRITTEN IN FULL EVERYWHERE HERE. The token "LN" previously meant LoRA *plus*
LayerNorm in this repository, and section 8 forbids the ambiguity.

BUDGET. A PE-Core-L adaptation run measured 100-320 minutes, so the ten-cell sweep on A is roughly
17-55 GPU-hours. `--plan` prints the ordered queue with the brief's drop order -- dropout 0.05 first,
then alpha/r 0.5 -- so cells come off the end if the cap would be exceeded. Each cell is evaluated
standalone AND by its fused contribution in ONE fixed reference subset with A swapped in; the subset
and operator search is not re-run per cell, because a one-factor sweep is not an interaction study.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                      # noqa: E402

C.thread_caps()

CENTRE = {"r": 8, "alpha": 16, "dropout": 0.1, "targets": "fused qkv", "layernorm": True}
RANKS = (2, 4, 8, 16)
SCALINGS = (0.5, 1.0, 2.0, 4.0)
DROPOUTS = (0.0, 0.05, 0.1, 0.2)
#: Section 8's drop order when the cap bites: lowest value first.
DROP_ORDER = ({"sweep": "dropout", "dropout": 0.05}, {"sweep": "scaling", "scaling": 0.5})
#: The fixed reference subset for the fused contribution. B0's triple with A swapped in.
REFERENCE_SUBSET = ("A", "V", "O")


def cells() -> list:
    """The ten unique configurations, centre included once."""
    out, seen = [], set()

    def add(sweep, r, alpha, dropout):
        key = (r, alpha, dropout)
        if key in seen:
            return
        seen.add(key)
        out.append({"sweep": sweep, "r": r, "alpha": alpha, "alpha_over_r": alpha / r,
                    "dropout": dropout, "layernorm": True, "targets": "fused qkv",
                    "is_centre": key == (CENTRE["r"], CENTRE["alpha"], CENTRE["dropout"]),
                    "trainable_lora": lora_params(r),
                    "id": f"r{r}_a{alpha:g}_d{dropout:g}"})
    for r in RANKS:
        add("rank", r, 2 * r, CENTRE["dropout"])
    for s in SCALINGS:
        add("scaling", CENTRE["r"], int(round(CENTRE["r"] * s)) if s >= 1 else CENTRE["r"] * s,
            CENTRE["dropout"])
    for d in DROPOUTS:
        add("dropout", CENTRE["r"], CENTRE["alpha"], d)
    return out


def lora_params(r: int, d_in: int = 1024, d_out: int = 3072, blocks: int = 24) -> int:
    """PE-Core-L's fused qkv is 3072x1024, and there are 24 blocks."""
    return blocks * (r * d_in + d_out * r)


CONTROLS = {
    "frozen": {"note": "no adaptation at all",
               "existing": "analysis/upgrade/screen_pe_core_l.json"},
    "layernorm_only": {"note": "LayerNorm parameters only, features L2-normalised (GenD recipe)",
                       "existing": "analysis/upgrade/adapt/pe_core_l_N_s0.json"},
    "lora_only": {"note": "LoRA only, LayerNorm NOT trained",
                  "existing": "analysis/upgrade/adapt/pe_core_l_L_s0.json"},
}


# ----------------------------------------------------------------------------- verification
def verify() -> dict:
    """Numerically confirm the adapter's scaling, zero-init, frozen base and masked-K behaviour."""
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(C.ROOT / "upgrade"))
    from adapt_backbone import LoRALinear, LORA_R, LORA_ALPHA, LORA_DROPOUT

    out = {"module": "upgrade/adapt_backbone.py::LoRALinear",
           "constants": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                         "alpha_over_r": LORA_ALPHA / LORA_R}}
    base = nn.Linear(1024, 3072)
    w0 = base.weight.detach().clone()
    lo = LoRALinear(base).eval()
    x = torch.randn(7, 1024)

    # 1. B starts at zero, so the adapted module is bit-identical to the base at step 0.
    with torch.no_grad():
        out["zero_init_max_abs_diff"] = float((lo(x) - (x @ w0.T + base.bias)).abs().max())
    out["B_is_zero_at_init"] = bool(lo.b.weight.abs().max() == 0)

    # 2. dW = (alpha / r) B A, checked against an explicit reconstruction.
    with torch.no_grad():
        lo.b.weight.normal_(0, 0.02)
        delta_fn = lo(x) - (x @ w0.T + base.bias)
        dW = (LORA_ALPHA / LORA_R) * (lo.b.weight @ lo.a.weight)
        delta_expl = x @ dW.T
    out["scaling_matches_alpha_over_r"] = float((delta_fn - delta_expl).abs().max())
    out["scaling_formula"] = "dW = (alpha / r) B A"

    # 3. No base parameter receives a gradient.
    lo.train()
    lo(x).sum().backward()
    out["base_grads_none"] = all(p.grad is None or float(p.grad.abs().max()) == 0
                                 for p in lo.base.parameters())
    out["base_requires_grad"] = [bool(p.requires_grad) for p in lo.base.parameters()]
    out["adapter_grads_present"] = bool(lo.a.weight.grad is not None
                                        and float(lo.a.weight.grad.abs().max()) > 0)

    # 4. A Q/V-only mask on a FUSED qkv must keep the K update exactly zero.
    #    PE-Core fuses q, k, v into one 3072x1024 Linear, so "skip K" means masking the middle third
    #    of the OUTPUT rows of B. Section 8 requires that to be checked numerically, not asserted.
    base2 = nn.Linear(1024, 3072)
    lo2 = LoRALinear(base2).eval()
    with torch.no_grad():
        lo2.b.weight.normal_(0, 0.02)
        mask = torch.ones(3072, 1)
        mask[1024:2048] = 0.0                      # the K third
        lo2.b.weight.mul_(mask)
        dW2 = (LORA_ALPHA / LORA_R) * (lo2.b.weight @ lo2.a.weight)
    out["masked_K_update_exactly_zero"] = float(dW2[1024:2048].abs().max())
    out["masked_QV_update_nonzero"] = bool(dW2[:1024].abs().max() > 0
                                           and dW2[2048:].abs().max() > 0)
    out["verdict"] = (
        "PASS" if (out["scaling_matches_alpha_over_r"] < 1e-5
                   and out["zero_init_max_abs_diff"] < 1e-5
                   and out["B_is_zero_at_init"] and out["base_grads_none"]
                   and out["masked_K_update_exactly_zero"] == 0.0) else "FAIL")
    return out


# ----------------------------------------------------------------------------- planning
def plan(minutes_per_cell: float = 220.0, cap_hours: float = 55.0) -> dict:
    """The ordered queue, with the brief's drop order applied if the cap would be exceeded."""
    cs = cells()
    # centre first, then rank, then dropout, then scaling: the centre anchors every sweep and a
    # failure there invalidates all of them, so it must not be last.
    order = {"rank": 1, "dropout": 2, "scaling": 3}
    cs.sort(key=lambda c: (0 if c["is_centre"] else 1, order[c["sweep"]], c["id"]))
    keep, dropped, total = [], [], 0.0
    for c in cs:
        if total + minutes_per_cell / 60.0 <= cap_hours:
            keep.append(c)
            total += minutes_per_cell / 60.0
        else:
            dropped.append(c)
    # if anything had to go, make sure it is the brief's nominated cells that go first
    if dropped:
        nominated = [c for c in keep if any(
            (d.get("dropout") is not None and c["sweep"] == "dropout"
             and abs(c["dropout"] - d["dropout"]) < 1e-9)
            or (d.get("scaling") is not None and c["sweep"] == "scaling"
                and abs(c["alpha_over_r"] - d["scaling"]) < 1e-9) for d in DROP_ORDER)]
        for c in nominated:
            if dropped:
                keep.remove(c)
                dropped.append(c)
                keep.append(dropped.pop(0))
    return {"cells": cs, "n_cells": len(cs), "queue": keep, "dropped": dropped,
            "minutes_per_cell_estimate": minutes_per_cell,
            "estimated_gpu_hours": round(len(keep) * minutes_per_cell / 60.0, 1),
            "cap_hours": cap_hours,
            "controls": CONTROLS, "centre": CENTRE,
            "reference_subset": list(REFERENCE_SUBSET),
            "seeds": {"screen": 1, "final": 3,
                      "note": ("one seed for screening, then three ADAPTATION seeds for the centre "
                               "and the selected recipe. A seed means a new adaptation run, not a "
                               "re-scored head.")},
            "evaluation": ("each cell standalone AND by its fused contribution in the fixed "
                           "reference subset with A swapped in; the subset and operator search is "
                           "NOT re-run per cell"),
            "fairness": ("data budget, effective batch size, augmentation, schedule and early "
                         "stopping held equal; peak memory, throughput and GPU-hours recorded")}


def emit_jobs(gpu: int = 1, seed: int = 0, limit: int | None = None) -> str:
    """Shell lines for the queue, one adaptation run per cell.

    `adapt_backbone.py` takes `--recipe LN` with its own fixed r/alpha/dropout constants, so each
    cell is passed through the environment and the script records what it used in its run JSON.
    """
    p = plan()
    lines = ["#!/bin/bash",
             "# Section 8 LoRA sweep on branch A. GPU priority 1, then 3, then 0 -- never GPU 2.",
             "set -u",
             "PY=/data/umar/miniconda3/envs/discern_ext_timm/bin/python",
             "cd /data/umar/Repos/DFB_NeSyNS",
             "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"]
    for c in (p["queue"][:limit] if limit else p["queue"]):
        lines.append(
            f"echo '=== {c['id']} ({c['sweep']}) =='; "
            f"BF_LORA_R={c['r']} BF_LORA_ALPHA={c['alpha']:g} BF_LORA_DROPOUT={c['dropout']:g} "
            f"CUDA_VISIBLE_DEVICES={gpu} $PY upgrade/adapt_backbone.py "
            f"--backbone pe_core_l --recipe LN --seed {seed} "
            f"|| echo FAILED {c['id']}")
    lines.append("echo LORA_SWEEP_DONE $(date -u +%FT%TZ)")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--emit-jobs", action="store_true")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    res = {}
    if a.verify or not (a.plan or a.emit_jobs):
        res["verification"] = verify()
        v = res["verification"]
        print(f"LoRA implementation: {v['verdict']}")
        print(f"  dW = (alpha/r) B A           max err {v['scaling_matches_alpha_over_r']:.2e}")
        print(f"  identical to base at step 0  max err {v['zero_init_max_abs_diff']:.2e}")
        print(f"  B zero at init               {v['B_is_zero_at_init']}")
        print(f"  base receives no gradient    {v['base_grads_none']}")
        print(f"  adapter receives gradient    {v['adapter_grads_present']}")
        print(f"  masked K update exactly zero {v['masked_K_update_exactly_zero']}")
        print(f"  constants                    {v['constants']}")
    if a.plan or a.emit_jobs:
        p = plan()
        res["plan"] = p
        print(f"\n{p['n_cells']} unique cells, queue of {len(p['queue'])}, "
              f"~{p['estimated_gpu_hours']} GPU-hours (cap {p['cap_hours']})")
        print(f"{'id':16s} {'sweep':9s} {'r':>3s} {'alpha':>6s} {'a/r':>5s} {'drop':>5s} "
              f"{'lora params':>12s} centre")
        for c in p["cells"]:
            print(f"{c['id']:16s} {c['sweep']:9s} {c['r']:3d} {c['alpha']:6g} "
                  f"{c['alpha_over_r']:5g} {c['dropout']:5g} {c['trainable_lora']:12d} "
                  f"{'<-' if c['is_centre'] else ''}")
        if p["dropped"]:
            print(f"dropped under the cap: {[c['id'] for c in p['dropped']]}")
        print(f"controls at the centre, reused from the upgrade campaign: "
              f"{ {k: v['existing'] for k, v in CONTROLS.items()} }")
    if a.emit_jobs:
        path = C.ROOT / "logs/branchfusion/lanes/lora_sweep.sh"
        path.write_text(emit_jobs(a.gpu, a.seed, a.limit))
        path.chmod(0o755)
        print(f"-> {path}")
    json.dump(res, open(C.OUT / "lora_sweep.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
