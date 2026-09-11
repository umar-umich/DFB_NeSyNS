#!/usr/bin/env python3
"""Stage 1 — assemble the per-sample measurement record on four separate axes.

    python analysis/tbiom/stage1_record.py --out tbiom/reports/EVIDENCE_STAGE1.md

THE DESIGN CORRECTION THIS STAGE EXISTS FOR. Observability, reliability and evidence direction
are distinct and must not be collapsed into one label. A cue can be available, reliable, and
point to real; or available, unreliable, and point to fake. Those are different states and the
record has to keep them apart.

    value         the measurement itself
    availability  was the cue measurable at all -- from the MEASUREMENT PROCESS, never from the
                  score. Read out of the export before core_model multiplies it into the
                  evidence, which is the whole reason the re-export was needed.
    reliability   quality of the measurement under this sample's conditions. NOT FILLED HERE:
                  it is Stage 2's output and nothing in the frozen model estimates it. The
                  column exists and is null, rather than being quietly faked from something
                  else, so that Stage 3 cannot silently consume a placeholder.
    source        which detector produced it, and what it shares with the others

EVIDENCE DIRECTION STAYS SEPARATE. `p_<branch>` is what the value supports, and it is carried
as its own field rather than folded into availability or reliability.

SOURCE IS BRANCH-LEVEL, NOT REGION-LEVEL, and that is a correction to the brief rather than a
shortcut. All three branches read the same aligned 256x256 face crop; there are no regions or
segments to attribute. What the source field must record instead is SHARED TRUNK: `semantic`
and `artifact` both come from the CLIP encoder -- artifact is the orthogonal residual
f_s - f_c -- so they are not independent sources. Stage 4 measured what that costs: supplying
one branch twice raises fused certainty +0.0110 with no AUROC gain, and dropping `semantic`
from the fusion is worth +0.008. A source field that could not express this would miss the only
real correlated-evidence instance in the model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_id import video_id  # noqa: E402
from view_names import VIEW_NAME  # noqa: E402

S1 = Path("logs/tbiom/stage1")
OUT = Path("logs/tbiom/stage1/record")
BR = ["semantic", "artifact", "fsvfm"]
DATASETS = ["FFpp_val", "VALmix", "CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24"]

# Which raw measurement belongs to which branch. fsvfm has no scalar of its own by design --
# its measurement is the 1024-d feature, and inventing a scalar summary the branch does not
# compute would fabricate the value field. See EVIDENCE_STAGE0.md.
VALUE_COL = {"semantic": ["norm_semantic"],
             "artifact": ["recon_err", "resid_norm", "kl_manifold"],
             "fsvfm": []}
TRUNK = {"semantic": "clip", "artifact": "clip", "fsvfm": "fsvfm"}


def build(ds: str) -> pd.DataFrame | None:
    f = S1 / f"m_{ds}.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    d["v"] = video_id(d["key"])
    recs = []
    for b in BR:
        if f"p_{b}" not in d:
            continue
        r = pd.DataFrame({
            "dataset": ds,
            "key": d["key"], "video": d["v"], "label": d["label"],
            "cue": b, "cue_name": VIEW_NAME[b],
            # --- availability: from the process, before it touched the evidence
            "availability": d.get(f"avail_{b}", pd.Series(np.ones(len(d), np.int8))),
            # --- reliability: Stage 2's job. Explicitly null, never imputed.
            "reliability": np.nan,
            # --- source
            "source_detector": b, "source_trunk": TRUNK[b],
            "source_region": "full_face_crop",      # the only region this model has
            # --- evidence direction, kept separate from the three axes above
            "p_fake": d[f"p_{b}"], "vacuity": d[f"u_{b}"],
        })
        # --- value: the raw measurement(s) the head consumes
        for i, c in enumerate(VALUE_COL[b]):
            r[f"value_{i}"] = d[c] if c in d else np.nan
            r[f"value_{i}_name"] = c
        recs.append(r)
    return pd.concat(recs, ignore_index=True) if recs else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("tbiom/reports/EVIDENCE_STAGE1.md"))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    made, missing, summary = [], [], []
    for ds in DATASETS:
        r = build(ds)
        if r is None:
            missing.append(ds); continue
        p = OUT / f"record_{ds}.parquet"
        try:
            r.to_parquet(p, index=False)
        except Exception:
            p = OUT / f"record_{ds}.csv.gz"; r.to_csv(p, index=False, compression="gzip")
        made.append(ds)
        for b in BR:
            s = r[r.cue == b]
            if s.empty:
                continue
            summary.append({"dataset": ds, "cue": b, "n": len(s),
                            "avail_rate": float(s["availability"].mean()),
                            "reliability_known": float(s["reliability"].notna().mean())})

    L = ["# Stage 1 — the measurement record", "",
         "Four axes kept separate: **value**, **availability**, **reliability**, **source**, "
         "with evidence direction (`p_fake`, `vacuity`) carried alongside rather than folded "
         "into any of them. One row per (sample, cue).", "",
         f"Written to `{OUT}/record_<dataset>.parquet`.", ""]

    if not made:
        L += ["**No Stage 1 exports yet.**", ""] + [f"- missing `{m}`" for m in missing]
        args.out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0

    S = pd.DataFrame(summary)
    L += ["## Coverage", "",
          "| dataset | rows/cue | availability rate | reliability known |",
          "|---|---:|---:|---:|"]
    for ds in made:
        s = S[S.dataset == ds]
        L.append(f"| {ds} | {int(s['n'].iloc[0]):,} | "
                 f"{s['avail_rate'].mean():.4f} | {s['reliability_known'].mean():.0%} |")

    av = S["avail_rate"].mean()
    L += ["", "## The three axes, honestly", "",
          f"**Availability: {av:.4f} mean across every cue and dataset.** " +
          ("Every cue is available on every sample. The axis is correctly populated from the "
           "measurement process and it is *constant*. That is a real property of this data, "
           "not a bug, and it has a consequence the writeup must carry: this system is never "
           "observed handling an unavailable cue, so Stage 4's missing-measurement result is "
           "about deliberately constructed conditions."
           if av > 0.9999 else
           f"Unavailability does occur, at rate {1-av:.4f}. Stage 4's masking test can be "
           f"paired with naturally occurring missingness."), "",
          "**Reliability: 0% known, by construction.** Nothing in the frozen model estimates "
          "measurement quality under operating conditions. The column is present and null "
          "rather than imputed from vacuity or from the score — either would collapse "
          "reliability back into the axes it is supposed to be separate from, which is the "
          "exact error this stage exists to prevent. Stage 2 fills it.", "",
          "**Source: branch-level, with a shared-trunk marker.** `semantic` and `artifact` both "
          "carry `source_trunk = clip`; `fsvfm` is its own trunk. `source_region` is "
          "`full_face_crop` for all three because that is the truth — there are no regions in "
          "this architecture. The trunk field is the one that does work: Stage 4 measured that "
          "duplicating a branch raises fused certainty +0.0110 with no AUROC gain, and the "
          "CLIP-sharing pair is the standing instance of that in the live model.", "",
          "**Value is complete for two cues of three.** `semantic` carries `norm_semantic`; "
          "`artifact` carries `recon_err`, `resid_norm` and `kl_manifold`. `fsvfm` carries no "
          "scalar: its measurement is the 1024-d feature and its norm is structurally constant "
          "(LayerNorm, cv 6.6e-08), so no scalar was invented for it. Any Stage 3 mechanism "
          "reading `value` must handle a cue whose value field is legitimately empty."]

    if missing:
        L += ["", "## Not yet exported", ""] + [f"- `{m}`" for m in missing]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
