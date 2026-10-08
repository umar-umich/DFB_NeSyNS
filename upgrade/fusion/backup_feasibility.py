#!/usr/bin/env python3
"""Appendix backup, step 3: are there genuine/manipulated pairs from the same source content?

    <discern_ext python> upgrade/fusion/backup_feasibility.py

The appendix requires this BEFORE anything is fitted, and requires the backup to stop if valid pairs
are unavailable -- explicitly forbidding substitution with unrelated genuine videos or an extra
training dataset. Pairs are needed only to fit the subspace; the fusion head may still train on all
eligible development examples, and inference must require only the queried video.

A pair is two videos from the SAME original source: one genuine, one manipulated. Dataset membership
is not a pair. The pairing key is derived from the video identifier and is reported per domain so a
family or domain with no usable pairs cannot hide inside an aggregate count.
"""
from __future__ import annotations

import json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "upgrade"), str(ROOT / "analysis/tbiom/pcr6p"),
                str(ROOT / "analysis/tbiom/grouped"), str(ROOT / "analysis/tbiom"),
                str(ROOT / "analysis/tbiom/temporal")]
import dsr_system as DS   # noqa: E402
import final_eval as FE   # noqa: E402


def source_key(v: str, subset: str):
    """The original content a video derives from.

    FF++ names a fake `<target>_<source>` against a real `<target>`, so the target id is the key.
    Celeb-DF names `idA_idB_nnnn` against real `idA_nnnn`. Anything whose key cannot be derived is
    returned as None and counted as unmatched rather than guessed at.
    """
    stem = v.rstrip("/").rsplit("/", 1)[-1]
    stem = re.sub(r"^[A-Za-z0-9.\-]+__", "", stem)          # strip a leading "<subset>__"
    m = re.match(r"^(\d{3})_(\d{3})$", stem)                # FF++ fake 000_003
    if m:
        return m.group(1)
    if re.match(r"^\d{3}$", stem):                          # FF++ real 000
        return stem
    m = re.match(r"^id(\d+)_id(\d+)_(\d+)$", stem)          # Celeb-DF fake id0_id16_0003
    if m:
        return f"id{m.group(1)}_{m.group(3)}"
    m = re.match(r"^id(\d+)_(\d+)$", stem)                  # Celeb-DF real id0_0003
    if m:
        return f"id{m.group(1)}_{m.group(2)}"
    return None


def main() -> int:
    brun = (ROOT / "outputs/fusion_experiments/CURRENT_BACKUP_RUN").read_text().strip()
    out = ROOT / "outputs/fusion_experiments" / brun
    main_run = (ROOT / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    man = json.load(open(ROOT / "outputs/fusion_experiments" / main_run / "data_access_manifest.json"))

    dev_t, _dp, _tests = DS.load("vjepa2_lora_B_seed42")
    op = FE.opinions("pe_core_lN", 0, "cuda", ("VALmix", "VALmixV2", "FFpp_val"))
    t = FE.swap_app(dev_t, op, "development")
    keep = ~t["source"].astype(str).isin(man["reserved_from_fitting"]).to_numpy()
    t = t[keep].reset_index(drop=True)

    y = t.y.to_numpy()
    dom = t["source"].astype(str).to_numpy()
    sub = t["subset"].astype(str).to_numpy()
    vid = t["v"].astype(str).to_numpy()

    per_dom = defaultdict(lambda: {"real": {}, "fake": defaultdict(list), "unmatched": 0})
    for i, v in enumerate(vid):
        k = source_key(v, sub[i])
        d = per_dom[dom[i]]
        if k is None:
            d["unmatched"] += 1
            continue
        if y[i] == 0:
            d["real"][k] = v
        else:
            d["fake"][k].append((sub[i], v))

    report, total_pairs, fams = {}, 0, set()
    for d, s in sorted(per_dom.items()):
        matched = {k: fs for k, fs in s["fake"].items() if k in s["real"]}
        npair = sum(len(f) for f in matched.values())
        total_pairs += npair
        f_here = {a for fs in matched.values() for a, _ in fs}
        fams |= f_here
        report[d] = {"real_videos": len(s["real"]), "fake_videos": sum(len(f) for f in s["fake"].values()),
                     "source_groups_with_a_genuine_partner": len(matched),
                     "aligned_pairs": npair, "unmatched_videos": s["unmatched"],
                     "families_with_pairs": sorted(f_here)}
        print(f"{d:14s} real {len(s['real']):5d}  fake {sum(len(f) for f in s['fake'].values()):5d}  "
              f"paired groups {len(matched):5d}  pairs {npair:5d}  unmatched {s['unmatched']:5d}")

    ok = total_pairs >= 200 and len([d for d in report.values() if d["aligned_pairs"] > 0]) >= 1
    res = {"backup_run_id": brun, "main_run_id": main_run,
           "pair_rule": "same original source content; FF++ target id, Celeb-DF identity+clip",
           "per_domain": report, "total_aligned_pairs": total_pairs,
           "families_with_pairs": sorted(fams),
           "feasible": bool(ok),
           "decision": ("proceed: the subspace can be fitted on these pairs" if ok else
                        "STOP: insufficient aligned pairs; the appendix forbids substituting "
                        "unrelated genuine videos or importing another dataset")}
    json.dump(res, open(out / "backup_feasibility.json", "w"), indent=1)
    print(f"\ntotal aligned pairs {total_pairs} across {len(fams)} families")
    print(f"FEASIBLE: {ok} -- {res['decision']}")
    print(f"wrote {out}/backup_feasibility.json")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
