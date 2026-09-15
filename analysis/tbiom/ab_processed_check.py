#!/usr/bin/env python3
"""Go/no-go for paired seeds: v3.1-selected A (epoch 5) vs B (epoch 8) under processing.
Video-level DS replay, tau = each arm's own FF++ val EER from its v3 selection record (deployable);
TPR at matched FPR is retrospective. Prints what exists; missing conditions are marked."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corrected_table_v2 import videos, rates, tpr_at_fpr, CONDS, PROC_SETS  # noqa: E402
from health import auroc  # noqa: E402

S23, S2 = Path("logs/tbiom/stage23"), Path("logs/tbiom/stage2")
ARMS = {"A e05": ("sbiarmA", "v3_sbiarmA_"), "B e08": ("sbiarmB", "v2_sbiarmB_")}
tau = {k: json.load(open(S23 / f"{a}_selection_v3.json"))["selected"]["tau_ffpp_val"] for k, (a, _) in ARMS.items()}
print("tau:", {k: round(v, 4) for k, v in tau.items()})
print(f"{'set':6} {'cond':11} {'arm':6} {'AUROC':>7} {'FPR@tau':>8} {'TPR@tau':>8} {'TPR@5%':>7} {'TPR@10%':>8}")
for ds in PROC_SETS:
    for c, label in CONDS:
        for k, (_, pre) in ARMS.items():
            g = videos(S2 / f"p_{pre}{ds}_{c}.csv")
            if g is None:
                print(f"{ds:6} {label:11} {k:6}  (missing)"); continue
            f, t = rates(g, tau[k])
            print(f"{ds:6} {label:11} {k:6} {auroc(g.y.to_numpy(), g.s.to_numpy()):7.4f} {f:8.3f} {t:8.3f} "
                  f"{tpr_at_fpr(g, .05):7.3f} {tpr_at_fpr(g, .10):8.3f}")
