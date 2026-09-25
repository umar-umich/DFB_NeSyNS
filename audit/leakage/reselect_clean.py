#!/usr/bin/env python3
"""Did the identity overlap change which checkpoint was selected? Read-only, cached exports only.

    OMP_NUM_THREADS=8 python audit/leakage/reselect_clean.py

Branch TRAINING is clean: FF++ c23 train has zero overlap with all ten test sets at video, source and identity level
(`overlap_*.csv`). The step that did see related identities is checkpoint SELECTION, which used VALmix-v2. This
re-runs the v3.1 selection rule twice for the spatial branch -- once on the full development pool, once on the clean
subset with every video whose Celeb-DF source or identity appears in any evaluation set removed -- and reports whether
the winning epoch changes.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path[:0] = [str(Path("analysis/tbiom/pcr6p").resolve()), str(Path("analysis/tbiom/grouped").resolve()),
                str(Path("analysis/tbiom").resolve()), str(Path("analysis/tbiom/temporal").resolve()),
                str(Path(__file__).resolve().parent)]
import numpy as np  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
import keys as K  # noqa: E402
import refit_control as RC  # noqa: E402

OUT = Path("audit/leakage")
S23 = Path("logs/tbiom/stage23")
DEV = ("VALmix", "VALmixV2", "FFpp_val")
EPOCHS = ("e13", "e17", "e19")          # the spatial checkpoints retained for this run


def dev_table(tag):
    return A.dev_tables({n: S23 / f"warmgate_{tag}_{n}.csv" for n in DEV}, [1.0])


def main():
    esrc, eids = RC.eval_keys()
    dvk = RC.dev_video_keys()

    def clean_mask(t):
        out = []
        for v in t.v:
            s, i = dvk.get(str(v).split("/")[-1], (set(), set()))
            out.append(not ((s & esrc) or (i & eids)))
        return np.array(out)

    rows = []
    for tag in EPOCHS:
        try:
            t = dev_table(tag)
        except Exception as e:
            print(f"skip {tag}: {type(e).__name__}: {e}"); continue
        m = clean_mask(t)
        full = A.v31(t.p_fused.to_numpy() if "p_fused" in t else t.p_semantic.to_numpy(), A.cells(t))
        sub = t[m].reset_index(drop=True)
        clean = A.v31(sub.p_fused.to_numpy() if "p_fused" in sub else sub.p_semantic.to_numpy(), A.cells(sub))
        rows.append({"epoch": tag, "n_full": len(t), "n_clean": int(m.sum()),
                     "v31_full": full[0], "v31_clean": clean[0],
                     "types_full": full[1], "types_clean": clean[1]})
        print(f"{tag}: full {full[0]:.4f} (n={len(t)})   clean {clean[0]:.4f} (n={int(m.sum())})", flush=True)

    if not rows:
        print("no per-epoch development exports usable"); return
    win_full = max(rows, key=lambda r: r["v31_full"])["epoch"]
    win_clean = max(rows, key=lambda r: r["v31_clean"])["epoch"]
    tr = json.load(open("logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/train_record.json"))
    hist = tr["history"]

    L = ["# Did the identity overlap change checkpoint selection?", "",
         "## 1. Branch training is clean, and that is the part that would force a retrain", "",
         "| level | FF++ c23 train x the ten test sets | FF++ c23 train x DF40 evaluation |", "|---|---|---|",
         "| video | **0** | 0 |", "| source video | **0** | 720 |", "| identity | **0** | 0 |", "",
         "All three branches are trained on FF++ c23 train only. It shares no video, no source video and no identity "
         "with any of the ten test sets, so **no branch was trained on test material and no branch needs retraining "
         "on that account.** The single training-side overlap is 720 FF++ source videos reappearing inside DF40's "
         "FF++-sourced generators, which is a property of DF40 and affects DF40 numbers only.", "",
         "## 2. Checkpoint selection is the step that saw related identities", "",
         "All three branches were selected on the v3.1 score over VALmix + VALmix-v2 + FF++ val, and that pool shares "
         "Celeb-DF identities with CDFv2, CDFv3, DFDM and DF40. The test below re-runs the selection on the clean "
         "subset only.", "",
         "### Spatial branch (Appearance and Off-manifold)", "",
         "| checkpoint | v3.1 on the full pool | v3.1 on the clean subset | videos full / clean |",
         "|---|---:|---:|---|"]
    for r in rows:
        L.append(f"| {r['epoch']} | {r['v31_full']:.4f} | {r['v31_clean']:.4f} | {r['n_full']} / {r['n_clean']} |")
    same = win_full == win_clean
    L += ["", f"Winner on the full pool: **{win_full}**. Winner on the clean subset: **{win_clean}**. "
          + ("**Unchanged** — the deployed checkpoint is the one a leakage-free selection would also have chosen, so "
             "there is nothing to re-select." if same else
             "**CHANGED** — the deployed checkpoint is not the one a clean selection would choose, which is a real "
             "finding and would justify re-selecting."), "",
          "### Temporal branch (V-JEPA 2)", "",
          "**FLAG — cannot be re-selected from retained artefacts.** Only the selected epoch's weights (`best.pt`) "
          "and its exported opinions were kept; the per-epoch checkpoints were not. Re-selecting on a clean subset "
          "would require re-exporting per-epoch opinions, which needs the discarded checkpoints.", "",
          "What the retained record does show is the shape of the selection curve, from "
          "`train_record.json`:", "",
          "| epoch | dev v3.1 | swap | reenactment | in-the-wild |", "|---|---:|---:|---:|---:|"]
    for e in hist:
        star = " **<- selected**" if e["epoch"] == tr["best_epoch"] else ""
        L.append(f"| {e['epoch']}{star} | {e['dev_v31']:.4f} | {e['types']['swap']:.4f} | "
                 f"{e['types']['reenactment']:.4f} | {e['types']['in_the_wild']:.4f} |")
    best = max(hist, key=lambda e: e["dev_v31"])
    margin = sorted((e["dev_v31"] for e in hist), reverse=True)
    L += ["", f"The selected epoch {tr['best_epoch']} leads the runner-up by {margin[0] - margin[1]:.4f} on the full "
          f"pool. Task 8 showed that removing the contaminated half of that pool moves a *fused* score by at most "
          f"0.0020; a margin of {margin[0] - margin[1]:.4f} is "
          + ("larger than that, so the selected epoch would very likely survive a clean re-selection, but this is an "
             "argument from the size of the margin rather than a direct measurement."
             if margin[0] - margin[1] > 0.002 else
             "smaller than that, so the selected epoch cannot be assumed to survive a clean re-selection."), "",
          "## 3. Verdict on retraining", "",
          "| question | answer |", "|---|---|",
          "| Was any branch trained on test material? | **No** — zero overlap at all three levels |",
          "| Do the branches need retraining? | **No**, on this evidence |",
          f"| Was the spatial checkpoint choice affected? | **{'No' if same else 'Yes'}** — measured directly above |",
          "| Was the temporal checkpoint choice affected? | **Unmeasurable** from retained artefacts; the margin "
          "argument above is the best available |",
          "| Is joint training indicated by the leakage? | **No.** Joint training would change the architecture, not "
          "the exposure. The exposure is in selection and weight fitting, and Task 8 already shows refitting those "
          "on clean data moves results by at most 0.0020 |", "",
          "The one methodological change the audit does support is unrelated to leakage: selection and trust-weight "
          "fitting share the same pool, which makes development numbers optimistic as held-out estimates. Separating "
          "a selection pool from a weight-fitting pool would fix that, and it costs no retraining.", ""]
    (OUT / "reselect_clean.md").write_text("\n".join(L) + "\n")
    json.dump({"rows": rows, "winner_full": win_full, "winner_clean": win_clean,
               "temporal_selected_epoch": tr["best_epoch"],
               "temporal_margin": margin[0] - margin[1]}, open(OUT / "reselect_clean.json", "w"), indent=1,
              default=float)
    print(f"\nspatial winner: full={win_full} clean={win_clean} -> {'UNCHANGED' if same else 'CHANGED'}")
    print(f"temporal selected epoch {tr['best_epoch']}, margin over runner-up {margin[0] - margin[1]:.4f}")


if __name__ == "__main__":
    main()
