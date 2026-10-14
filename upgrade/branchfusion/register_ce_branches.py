#!/usr/bin/env python3
"""Register App's and LFS's cross-entropy counterparts, so the CE arm is a complete framework.

    <discern_ext python> upgrade/branchfusion/register_ce_branches.py

The CE row of Table B is held until every branch has a CE twin, because a row mixing evidential
and cross-entropy branches answers no question and is easy to mistake for one that does. TD and
STT already have theirs from `train_ce.py`. These are the remaining two, and neither needs
training:

  AppCE   the CE head from `train_head_ce.py`, replayed over the SAME frozen feature caches the
          evidential head reads, with a softmax readout and App's own frame-mean reduction.
  LFSCE   the artifact view of DISCERN_Ext's `run1d_simplece_seed42`, already exported per frame
          by `export_views.py`, reduced to one opinion per video the same way.

`u` IS NOT A VACUITY in either. A cross-entropy model has no Dirichlet; the column holds
`1 - max p`, the only uncertainty such a model emits, and is written so a CE branch can pass
through the same code paths. The provenance JSON records `uncertainty_is_margin: true`, and the
CE arm is fused by mean-of-logits, never by DS or PCR6+.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                               # noqa: E402
import branches as B                                             # noqa: E402

CE_HEAD = C.ROOT / "analysis/upgrade/heads_ce/view_pe_core_lLN_s0.pt"
O_CE = C.ROOT / "logs/branchfusion/o_ce"
SETS = list(C.DEV_SETS) + list(C.TEST_SETS)


def app_ce() -> int:
    sys.path[:0] = [str(C.ROOT / "upgrade")]
    from train_head_ce import CEHead
    st = torch.load(CE_HEAD, map_location="cpu")
    head = CEHead(st["dim"] if "dim" in st else 1024)
    head.load_state_dict(st["state"]); head.eval()
    out = C.BRANCH_OUT / "AppCE"; out.mkdir(parents=True, exist_ok=True)
    done, missing = {}, []
    for ds in SETS:
        try:
            D = C.load_frames("pe_core_lLN", ds)
        except FileNotFoundError:
            missing.append(ds); continue
        ps = []
        with torch.no_grad():
            for i in range(0, len(D["x"]), 8192):
                lg = head(torch.from_numpy(D["x"][i:i + 8192]))
                ps.append(torch.softmax(lg.float(), 1)[:, 1].numpy())
        p = C.video_mean(D, np.concatenate(ps))
        pd.DataFrame({"v": [str(v) for v in D["order"]], "y": D["yv"], "p": p,
                      "u": 1.0 - np.maximum(p, 1.0 - p)}).to_csv(out / f"{ds}.csv", index=False)
        done[ds] = int(len(D["order"]))
        print(f"  AppCE {ds}: {len(D['order'])} videos", flush=True)
    return _finish("AppCE", out, done, missing, "App CE head, softmax, frame mean")


def lfs_ce() -> int:
    out = C.BRANCH_OUT / "LFSCE"; out.mkdir(parents=True, exist_ok=True)
    done, missing = {}, []
    for ds in SETS:
        f = O_CE / f"{ds}.csv"
        if not f.exists():
            missing.append(ds); continue
        d = pd.read_csv(f)
        # Same scope filter the evidential LFS export applies, so the two arms cover the same
        # videos and a difference between them is the loss and not the population. This was a
        # FIFTH hardcoded copy of that predicate, and it is why LFSCE stayed at the v3.1 census
        # (CDFv3 5118, KoDF 1395) while the evidential branches widened -- putting a CE row and
        # an evidential row in one table over different videos.
        sub = d.key.astype(str).str.split("::").str[-1].str.split("/").str[-3]
        d = d[B.in_scope(sub).to_numpy()]
        # NOT the file's own `video` column. It holds `id0` for every frame of id0_0001, id0_0002
        # and so on -- the id collision DiCoME's own adaptation notes document -- which collapsed
        # CDFv2 from 518 videos to 124 and FFIW from 3470 to 1. `video_id` is the project's single
        # rule, shared with every other consumer, and `to_video_level` is its aggregation.
        sys.path[:0] = [str(C.ROOT / "analysis/tbiom")]
        from video_id import to_video_level
        g = to_video_level(d.rename(columns={"p_artifact": "_p"}), "_p", key_col="key",
                           label_col="label").set_index("video")
        p = g.p.to_numpy()
        pd.DataFrame({"v": [str(v) for v in g.index], "y": g.y.to_numpy(), "p": p,
                      "u": 1.0 - np.maximum(p, 1.0 - p)}).to_csv(out / f"{ds}.csv", index=False)
        done[ds] = int(len(g))
        print(f"  LFSCE {ds}: {len(g)} videos", flush=True)
    return _finish("LFSCE", out, done, missing, "DISCERN_Ext simple-CE artifact view, per video")


def _finish(label, out, done, missing, note) -> int:
    rec = {"label": label, "datasets": done, "missing": missing,
           "uncertainty_is_margin": True, "note": note, "dir": str(out)}
    (C.ROOT / f"analysis/branchfusion/{label}_export.json").write_text(json.dumps(rec, indent=1))
    test_missing = [d for d in C.TEST_SETS if d not in done]
    if test_missing:
        print(f"  {label} NOT registered: missing {test_missing}", flush=True)
        return 1
    B.register(label, out, note=note + "; u is a margin, not a vacuity")
    print(f"  registered {label}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(app_ce() + lfs_ce())
