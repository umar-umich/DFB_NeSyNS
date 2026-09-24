#!/usr/bin/env python3
"""Two things the manuscript needs that no existing report holds: per-view AUROC on the ten test sets, and the
trainable-parameter counts of the adapted views.

    OMP_NUM_THREADS=8 python analysis/tbiom/paper/extra_facts.py

Per-view scores are read straight off the cached opinions -- a single view needs no fusion, so the numbers are
operator-independent. Parameter counts are read from the checkpoint state dicts, not estimated from the architecture.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "pcr6p"), str(HERE.parent / "grouped"),
                str(HERE.parent), str(HERE.parent / "temporal")]
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import confirm as C  # noqa: E402

R = Path("tbiom/reports")
TESTS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM",
         "CDFv3 reenactment", "KoDF first-order motion"]
VIEWS = ("T", "APP", "OM")


def params():
    out = {}
    try:
        import torch
        ck = Path("logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt")
        if ck.exists():
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            # the branch checkpoint stores only what was trained: the adapted blocks and the evidential head
            # `trainable_blocks` stores the adapted attention projections: the LoRA factors, which are trained,
            # alongside their frozen base weights, which are not. Only the former count as trainable.
            tb = sd.get("trainable_blocks", {})
            n = lambda d: int(sum(v.numel() for v in d.values() if hasattr(v, "numel")))
            out["temporal_adapters"] = n({k: v for k, v in tb.items() if "lora" in k.lower()})
            out["temporal_frozen_base_stored"] = n({k: v for k, v in tb.items() if "lora" not in k.lower()})
            out["temporal_head"] = n(sd.get("head", {}))
            out["temporal_trainable"] = out["temporal_adapters"] + out["temporal_head"]
            out["temporal_adapted_modules"] = len([k for k in tb if k.endswith("lora_A.weight")])
            out["selected_epoch"] = int(sd.get("epoch", -1))
            out["tensors"] = len(sd.get("trainable_blocks", {})) + len(sd.get("head", {}))
    except Exception as e:                                     # reported, not silently dropped
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    dev = C.dev_table("warmgate_e19", "vjepa2_lora_B_seed42")
    tests = D.load("vjepa2_lora_B_seed42")[2]
    per = {}
    for tn in TESTS:
        if tn not in tests:
            continue
        t = tests[tn]; y = t.y.to_numpy()
        per[tn] = {v: A.fast_auc(t[f"p_{v}"].to_numpy()[y == 0], t[f"p_{v}"].to_numpy()[y == 1]) for v in VIEWS}
        per[tn]["n_real"] = int((y == 0).sum()); per[tn]["n_fake"] = int((y == 1).sum())
    dv = {v: float(np.mean(dev[f"u_{v}"])) for v in VIEWS}
    out = {"per_view_test": per, "dev_mean_vacuity": dv, "params": params()}
    json.dump(out, open(R / "PAPER_EXTRA_FACTS.json", "w"), indent=1, default=float)
    for tn, d in per.items():
        print(f"{tn:26s} T {d['T']:.4f}  APP {d['APP']:.4f}  OM {d['OM']:.4f}   ({d['n_real']}r/{d['n_fake']}f)")
    print("\nmean vacuity on development:", {k: round(v, 4) for k, v in dv.items()})
    print("params:", out["params"])


if __name__ == "__main__":
    main()
