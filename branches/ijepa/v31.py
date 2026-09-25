#!/usr/bin/env python3
"""The v3.1 development score, used for checkpoint selection on VALmix + VALmix-v2 + FF++ val.

TYPES and EXCLUDED are not copied. They are parsed out of `analysis/tbiom/select_checkpoint_v3.py`
with `ast.literal_eval` at import time, so this module cannot drift from the deployed definition and
cannot silently score a different set of cells. Parsing also avoids importing that module, which pulls
in the whole checkpoint-selection stack.

A cell is (source, forgery type): the AUROC of that source's fakes of that type against that source's
genuine videos, except DF40train, which is scored against FF++ validation genuine because it ships no
genuine videos of its own. A type's score is the mean of its cells; the v3.1 score is the mean of the
three type scores, so swap, reenactment and in-the-wild weigh equally however many cells each has.

KEY PARSING, FROM THE END. `subset` is parts[-3] and `source` is parts[-4], with CDFv2val, DFDCPval
and DFEval24val recognised by substring because VALmix flattens them into the subset component. The
depths genuinely differ -- VALmix keys are 4 components, VALmix-v2 keys are 5 -- which is the
mismatch that produced a false PASS in the leakage audit when indexing ran from the front.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

SRC = Path("/data/umar/Repos/DFB_NeSyNS/analysis/tbiom/select_checkpoint_v3.py")
SPECIAL = ("CDFv2val", "DFDCPval", "DFEval24val")


def _parse(names=("TYPES", "EXCLUDED", "RULE_VERSION")):
    tree = ast.parse(SRC.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in names:
                out[t.id] = ast.literal_eval(node.value)
    missing = set(names) - set(out)
    if missing:
        raise RuntimeError(f"{SRC.name} no longer defines {sorted(missing)}")
    return out


_P = _parse()
TYPES, EXCLUDED, RULE_VERSION = _P["TYPES"], _P["EXCLUDED"], _P["RULE_VERSION"]


def subset_of(key: str) -> str:
    return str(key).split("/")[-3]


def source_of(key: str) -> str:
    k = str(key)
    for s in SPECIAL:
        if s in k:
            return s
    p = k.split("/")
    return p[-4] if len(p) >= 4 else p[0]


def excluded_subsets() -> set:
    """Every (source, subset) the protocol never scores: lip sync only, entire face synthesis."""
    out = set()
    for group in EXCLUDED.values():
        for src, subs in group.items():
            for sub in subs:
                out.add((src, sub))
    return out


def fast_auc(sr, sf) -> float:
    if len(sr) == 0 or len(sf) == 0:
        return float("nan")
    r = rankdata(np.concatenate([sr, sf]))
    return float((r[len(sr):].sum() - len(sf) * (len(sf) + 1) / 2) / (len(sr) * len(sf)))


def score(dev_keys, dev_p, dev_y, ff_keys, ff_p, ff_y):
    """v3.1 score from VIDEO-level inputs.

    dev_* are the concatenated VALmix and VALmix-v2 videos, ff_* the FF++ validation videos.
    Returns (score, per-type means, per-cell AUROCs).
    """
    dev_p, dev_y = np.asarray(dev_p, float), np.asarray(dev_y, int)
    src = np.array([source_of(k) for k in dev_keys])
    sub = np.array([subset_of(k) for k in dev_keys])
    ffreal = np.asarray(ff_p, float)[np.asarray(ff_y, int) == 0]

    cells, types = {}, {}
    for typ, srcs in TYPES.items():
        vals = []
        for s, subs in srcs.items():
            m = src == s
            fake = m & (dev_y == 1) if subs == "*" else m & (dev_y == 1) & np.isin(sub, subs)
            real = ffreal if s == "DF40train" else dev_p[m & (dev_y == 0)]
            a = fast_auc(real, dev_p[fake])
            cells[(typ, s)] = a
            vals.append(a)
        types[typ] = float(np.nanmean(vals))
    return float(np.mean(list(types.values()))), types, cells


def _selftest():
    assert set(TYPES) == {"swap", "reenactment", "in_the_wild"}, TYPES.keys()
    print(f"v31.py parsed rule {RULE_VERSION} from {SRC.name}")
    for t, srcs in TYPES.items():
        print(f"  {t:12s} {len(srcs)} sources: {', '.join(srcs)}")
    ex = excluded_subsets()
    assert ("CDFv3valG", "IP_LAP") in ex and ("DF40train", "wav2lip") in ex
    print(f"  excluded: {len(ex)} (source, subset) pairs never scored")

    # key parsing on the real shapes
    assert source_of("VALmix/CDFv2val-Celeb-real/cdf2-id0_0005/000.png") == "CDFv2val"
    assert subset_of("VALmix/CDFv2val-Celeb-real/cdf2-id0_0005/000.png") == "CDFv2val-Celeb-real"
    k2 = "VALmixV2/CDFv3valG/AniTalker/id0_0007_test/frame_0000.png"
    assert source_of(k2) == "CDFv3valG" and subset_of(k2) == "AniTalker"
    assert source_of("FFpp_val/Deepfakes/Deepfakes__004_982/000.png") == "FFpp_val"

    # every source the protocol scores exists in the manifests
    splits = Path("/data/umar/Repos/DISCERN_Ext/eval_adaptation/data/splits")
    present = set()
    for ds in ("VALmix", "VALmixV2"):
        for m in (splits / ds).glob("*.txt"):
            for line in m.read_text().splitlines():
                if line.strip():
                    present.add((source_of(line), subset_of(line)))
    want_src = {s for srcs in TYPES.values() for s in srcs}
    have_src = {s for s, _ in present}
    assert want_src <= have_src, f"sources missing from the dev pools: {sorted(want_src - have_src)}"
    for typ, srcs in TYPES.items():
        for s, subs in srcs.items():
            if subs != "*":
                miss = [x for x in subs if (s, x) not in present]
                assert not miss, f"{typ}/{s}: subsets absent from the manifests: {miss}"
    print(f"  all {len(want_src)} scored sources and every named subset are present in the dev pools")

    # a perfect and an inverted ranking bracket the score
    rng = np.random.default_rng(0)
    keys, y = [], []
    for s, sub in sorted(present):
        for i in range(4):
            keys.append(f"VALmixV2/{s}/{sub}/vid{i}/f.png" if s not in SPECIAL
                        else f"VALmix/{sub}/vid{i}/f.png")
            y.append(0 if "real" in sub.lower() else 1)
    y = np.array(y)
    ffk = [f"FFpp_val/FFpp-real/v{i}/f.png" for i in range(8)]
    ffy = np.zeros(8, int)
    good, _, _ = score(keys, y.astype(float), y, ffk, np.zeros(8), ffy)
    bad, _, _ = score(keys, 1.0 - y, y, ffk, np.ones(8), ffy)
    assert good > 0.99 and bad < 0.01, (good, bad)
    print(f"  perfect ranking scores {good:.3f}, inverted {bad:.3f}")
    print("v31.py self-test OK")


if __name__ == "__main__":
    _selftest()
