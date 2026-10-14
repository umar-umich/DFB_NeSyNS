#!/usr/bin/env python3
"""Assemble one video-level opinion table per branch per dataset, from each branch's REAL export.

Section 2 of the brief. This is the data layer every later module reads, and it exists so that a row
in the combined table traces to an exact branch export (section 12) instead of to whichever cache was
convenient.

    <discern_ext_timm python> upgrade/branchfusion/branches.py --export A,O --sets dev,test
    <discern_ext_timm python> upgrade/branchfusion/branches.py --table --sets dev

WHERE EACH BRANCH COMES FROM, and why it is not interchangeable with any other source:

  A  `logs/upgrade/feats/pe_core_lLN_<set>.npz` frame features through the trained evidential head
     `analysis/upgrade/heads/view_pe_core_lLN_s0.pt`, frame-mean to video. This reproduces the
     measured B0 appearance view rather than approximating it.
  S  `logs/branchfusion/seq/` -- written by `train_seq.py`, which must run first.
  V  `logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/opinions/<set>.csv`, the verified V-JEPA 2
     branch. Its opinions live in their OWN directory, not in the stage-23 export; reading the
     export's third view instead silently scores a system we do not deploy, which has happened.
  O  `logs/tbiom/stage23/` -- `p_artifact`/`u_artifact`, the CLIP/beta-VAE orthogonal residual. Kept
     under the explicit identifier `legacy_om` per section 1, because "OM", "LFS" and
     "facial structure" have all named this same computation at different times.
  M  `logs/tbiom/temporal/branches/videomae_lora_seed42/opinions/<set>.csv`, already trained with
     full dev+test coverage, so section 2's 8-GPU-hour fallback for M does not arise.
  P  `logs/tbiom/temporal/branches/pecore_l14_lora_B_seed42/opinions/<set>.csv`, copied from Dr
     Kutub's run. Trained by him, not by this campaign -- recorded as such in every table, because
     its adaptation budget (7.6M trainable) is eight times V's and a fair row has to say so.
  R  `logs/branchfusion/patch/` -- written by `extract_patch.py` plus a head, both of which must run
     first, and R enters the subset search only if section 2.3's gate passes.

The test-set video filter is the one B0 uses (`dsr_system.load`): IP_LAP, audio-driven and adversarial
subsets are excluded. Reusing it is what keeps O's video set identical to the other branches' rather
than a superset that would quietly change every macro mean.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                      # noqa: E402

import exp1_fusion_audit as A                           # noqa: E402
import torch                                            # noqa: E402

C.thread_caps()

#: Section 1 asks for the legacy name to stay explicit. This is the identifier, O is the table label.
LEGACY_OM = "legacy_om"

#: Branch -> how to build it. `kind` picks the reader below.
SOURCES = {
    "A": {"kind": "head", "backbone": "pe_core_lLN", "head": "view_pe_core_lLN_s0.pt",
          "note": "PE-Core-L/336 LoRA+LayerNorm frame features, frame-mean, B0 appearance recipe"},
    "S": {"kind": "export", "dir": C.SEQ / "opinions", "note": "PE sequence transformer (train_seq.py)"},
    "V": {"kind": "temporal", "branch": "vjepa2_lora_B_seed42", "note": "V-JEPA 2 LoRA, verified"},
    "O": {"kind": "stage23", "view": "artifact", "spatial": "warmgate_e19", "sel": "warmgate_sel",
          "note": f"{LEGACY_OM}: CLIP/beta-VAE orthogonal residual"},
    "M": {"kind": "temporal", "branch": "videomae_lora_seed42", "note": "VideoMAE LoRA, encoder panel"},
    # R is derived the same way A is -- cached features through a trained evidential head -- so it
    # needs no separate export step. `Rcov` is the covariance cache registered into logs/upgrade/feats
    # by `extract_patch.py --register`, and `Rmean` is its mean-pooled control.
    "R": {"kind": "head", "backbone": "Rcov", "head": "view_Rcov_s0.pt",
          "note": "PE block-16 patch covariance (528) + log norm, frozen trunk"},
    "Rmean": {"kind": "head", "backbone": "Rmean", "head": "view_Rmean_s0.pt",
              "note": "R's control: mean-pooled block-16 features, same depth"},
    # Added 2026-10-08 from Dr Kutub's trained run; see the branch's PROVENANCE.md. Read through the
    # same `temporal` reader as V and M, so no new code path can treat it differently.
    "P": {"kind": "temporal", "branch": "pecore_l14_lora_B_seed42",
          "note": "PE-Core L/14 temporal, LoRA + 2 unfrozen blocks (7.6M trainable), 16 frames"},
}

# ----------------------------------------------------------------------------- dynamic registry
#: Branches produced by a later experiment register themselves here rather than by editing SOURCES,
#: so an arm can be added without a code change and, more importantly, so the ledger records WHEN
#: and WHY each one appeared. The file is a dict of {label: {dir, note}}; the reader is always
#: `export`, because anything registered this way ships v,y,p,u CSVs like S does.
#:
#: A registered branch is NOT automatically in any screen: `subsets.py` takes its branch list
#: explicitly, and `dev_table(..., require_all=True)` will reject a branch whose coverage is partial.
#: That is deliberate -- a half-covered branch silently shrinking the development pool is exactly how
#: a comparison stops being between the same videos.
REGISTRY = C.BRANCH_OUT / "registry.json"

#: The built-in labels, frozen BEFORE any registry merge. The collision test below has to ask
#: "is this one of the hand-written branches?", and it cannot ask that of `SOURCES` itself: the
#: module merges the registry into `SOURCES` at import, so a second call -- which `register`
#: makes, to refresh after writing -- would find every registered label already there and reject
#: it as a collision with itself. That is why registering any branch failed as soon as the
#: registry was non-empty, and why only the very first one ever succeeded.
BUILTIN = frozenset(SOURCES)


def _load_registry() -> dict:
    if not REGISTRY.exists():
        return {}
    try:
        reg = json.loads(REGISTRY.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{REGISTRY} is not readable JSON: {e}") from e
    out = {}
    for label, spec in reg.items():
        if label in BUILTIN:
            raise ValueError(f"registry label {label!r} collides with a built-in branch; "
                             f"pick another label rather than shadowing {SOURCES[label]['note']!r}")
        out[label] = {"kind": "export", "dir": Path(spec["dir"]),
                      "note": spec.get("note", ""), "registered": spec.get("registered", "")}
    return out


def register(label: str, directory, note: str = "") -> dict:
    """Add an export branch to the registry on disk. Idempotent for the same label and directory."""
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    if label in BUILTIN:
        raise ValueError(f"{label!r} is a built-in branch; choose another label")
    reg[label] = {"dir": str(directory), "note": note,
                  "registered": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    REGISTRY.write_text(json.dumps(reg, indent=1))
    SOURCES.update(_load_registry())
    return reg[label]


SOURCES.update(_load_registry())


# ----------------------------------------------------------------------------- readers
def _head_opinions(backbone: str, head_file: str, dataset: str) -> pd.DataFrame:
    """Frame features -> evidential head -> frame-mean video opinion.

    Mirrors `upgrade/screen_eval.candidate_opinions`, including the frame-mean reduction, so A is the
    same quantity the screens reported and not a re-derivation of it.
    """
    from train_head import Head
    ck = ROOT_HEADS / head_file
    st = torch.load(ck, map_location="cpu")
    D = C.load_frames(backbone, dataset)
    head = Head(st["dim"])
    head.load_state_dict(st["state"])
    head.eval()
    ps, us = [], []
    with torch.no_grad():
        for i in range(0, len(D["x"]), 8192):
            ev = head(torch.from_numpy(D["x"][i:i + 8192]))
            a = ev + 1.0
            s = a.sum(1, keepdim=True)
            ps.append((a / s)[:, 1].numpy())            # EDL order: index 1 is fake
            us.append((C.K / s.squeeze(1)).numpy())
    p = C.video_mean(D, np.concatenate(ps))
    u = C.video_mean(D, np.concatenate(us))
    return pd.DataFrame({"v": [str(x) for x in D["order"]], "y": D["yv"], "p": p, "u": u})


ROOT_HEADS = C.ROOT / "analysis/upgrade/heads"


def _temporal_opinions(branch: str, dataset: str) -> pd.DataFrame:
    return C.read_opinions(C.TEMPORAL / branch / "opinions" / f"{dataset}.csv")


def _export_opinions(d: Path, dataset: str) -> pd.DataFrame:
    return C.read_opinions(d / f"{dataset}.csv")


def sel_csv(dataset: str) -> Path:
    """The stage-23 selection export for one dataset, under EITHER name the campaigns wrote.

    LFS reads `warmgate_sel_<set>.csv`. The DF40 cross-manipulation campaign wrote the SAME
    checkpoint's export -- `warm_dicome_gate_seed42` epoch 19, through the same `warmgate_<set>.yaml`
    config family -- under the tag `ours_<set>.csv`, because that run emitted two checkpoints side
    by side and named them `ours` and `dicome`. So the file exists for all 41 DF40 generator sets;
    only the prefix differs. Returning the first that exists is what lets LFS and the metadata cover
    DF40 without copying 41 files under a second name, and the identity of the checkpoint is what
    makes the two prefixes interchangeable -- a DIFFERENT model's export must never be reached here.
    """
    for tag in ("warmgate_sel", "ours"):
        f = C.S23 / f"{tag}_{dataset}.csv"
        if f.exists():
            return f
    return C.S23 / f"warmgate_sel_{dataset}.csv"          # the name to report as missing


def in_scope(d):
    """Which rows survive the evaluation-scope rule, as ONE function used by every caller.

    THE FILTER IS APPLIED AT READ TIME for the stage-23 views, from exports that already hold
    every video -- so widening the scope costs nothing but this predicate, where the other
    branches needed their features re-encoded.

    `BF_SCOPE` selects which groups stay excluded, and `v31` is the default so every published
    number is reproducible unchanged:

        v31       lip-sync/audio, adversarial and whole-face synthesis all excluded
        nolip     lip-sync/audio ADMITTED
        nolipadv  lip-sync/audio and adversarial admitted
        all       nothing excluded

    This lived twice in this file as a hardcoded boolean expression. Two copies of a filter is
    how one gets widened and the other does not, and the two readers would then disagree about
    which videos a dataset contains while both reporting a plausible AUROC.
    """
    import os
    scope = os.environ.get("BF_SCOPE", "v31")
    # Either a frame carrying `subset` or the subset Series itself: `sota_table1` has only the
    # Series, derived from the baseline's own file paths, and giving it a reason to rebuild the
    # predicate is how the fourth copy of this filter came to exist.
    sub = d if isinstance(d, pd.Series) else d.subset
    keep = pd.Series(True, index=sub.index)
    if scope in ("v31",):
        keep &= ~sub.str.contains("IP_LAP") & (sub != "fake-audio-driven")
    if scope in ("v31", "nolip"):
        keep &= ~sub.str.startswith("adv-")
    return keep


def _stage23_opinions(spec: dict, dataset: str) -> pd.DataFrame:
    """`p_artifact`/`u_artifact` from the DISCERN_Ext export, reduced to video level.

    Development sets use `A.dev_tables`, which also attaches the `source` label the v3.1 cells and the
    FF++-val EER threshold need. Test sets use `warmgate_sel_<set>.csv` with B0's exclusion filter.
    """
    view = spec["view"]
    if dataset in C.DEV_SETS:
        t = A.dev_tables({dataset: C.S23 / f"{spec['spatial']}_{dataset}.csv"}, [1.0])
    else:
        d = A.load_frames(sel_csv(dataset))
        keep = in_scope(d)
        t = A.video_table(d[keep].reset_index(drop=True), [1.0])
    return pd.DataFrame({"v": [str(x) for x in t.v], "y": t.y.to_numpy(),
                         "p": t[f"p_{view}"].to_numpy(), "u": t[f"u_{view}"].to_numpy()})


def branch_opinions(branch: str, dataset: str) -> pd.DataFrame:
    """One branch's video-level `(v, y, p, u)` for one dataset, from its real export."""
    spec = SOURCES[branch]
    k = spec["kind"]
    if k == "head":
        return _head_opinions(spec["backbone"], spec["head"], dataset)
    if k == "temporal":
        return _temporal_opinions(spec["branch"], dataset)
    if k == "stage23":
        return _stage23_opinions(spec, dataset)
    if k == "export":
        return _export_opinions(spec["dir"], dataset)
    raise ValueError(k)


def available(branch: str, dataset: str) -> bool:
    spec = SOURCES[branch]
    try:
        if spec["kind"] == "head":
            return (C.feat_cache(spec["backbone"], dataset).exists()
                    and (ROOT_HEADS / spec["head"]).exists())
        if spec["kind"] == "temporal":
            return (C.TEMPORAL / spec["branch"] / "opinions" / f"{dataset}.csv").exists()
        if spec["kind"] == "stage23":
            f = (C.S23 / f"{spec['spatial']}_{dataset}.csv" if dataset in C.DEV_SETS
                 else sel_csv(dataset))
            return f.exists()
        return (spec["dir"] / f"{dataset}.csv").exists()
    except Exception:
        return False


def coverage() -> pd.DataFrame:
    """Which branch has which dataset. Printed before anything launches, per section 3."""
    sets = list(C.DEV_SETS) + list(C.TEST_SETS)
    return pd.DataFrame({b: [available(b, s) for s in sets] for b in SOURCES}, index=sets)


# ----------------------------------------------------------------------------- caching the exports
def export(branch: str, datasets, force: bool = False) -> dict:
    """Materialise a branch's opinions into `logs/branchfusion/branches/<branch>/<set>.csv`.

    Caching here rather than re-deriving on every read means a later module cannot accidentally read
    a different version of A than the subset search used, and it makes the per-row provenance in
    section 12 a file path rather than a claim.
    """
    out = C.BRANCH_OUT / branch
    out.mkdir(parents=True, exist_ok=True)
    done = {}
    for ds in datasets:
        dst = out / f"{ds}.csv"
        if dst.exists() and not force:
            done[ds] = "cached"
            continue
        if not available(branch, ds):
            done[ds] = "missing"
            continue
        t = branch_opinions(branch, ds)
        C.assert_valid_opinions(t.p.to_numpy(), t.u.to_numpy(), tag=f"{branch}/{ds}")
        C.write_opinions(dst, t.v, t.y, t.p, t.u)
        done[ds] = f"{len(t)} videos"
        print(f"  {branch}/{ds}: {done[ds]} -> {dst}", flush=True)
    return done


def read(branch: str, dataset: str) -> pd.DataFrame:
    """Read a materialised export, falling back to deriving it."""
    f = C.BRANCH_OUT / branch / f"{dataset}.csv"
    return C.read_opinions(f) if f.exists() else branch_opinions(branch, dataset)


# ----------------------------------------------------------------------------- the joined tables
def meta_for(dataset: str) -> pd.DataFrame:
    """`source`, `subset` and `vname` per video, taken from the stage-23 export.

    The v3.1 development cells, the group union-find and the FF++-val threshold all key off these, so
    they come from the ONE place B0 takes them from rather than being reconstructed per branch.
    """
    spec = SOURCES["O"]
    if dataset in C.DEV_SETS:
        t = A.dev_tables({dataset: C.S23 / f"{spec['spatial']}_{dataset}.csv"}, [1.0])
    elif not sel_csv(dataset).exists():
        # A corpus packed by THIS project has no stage-23 selection file: that export belongs to
        # the earlier pipeline and was never run for it. Rebuild the metadata from whichever
        # branch has already been materialised. `source` is the dataset, which is exactly what the
        # test path sets below anyway; `subset` is the dataset too, which is sound here ONLY
        # because the pack step admitted in-scope generators alone, so no subset filter has
        # anything left to remove. A corpus packed with mixed generators would need its real
        # subsets and this fallback would be wrong for it.
        import glob as _glob
        hit = next(iter(sorted(_glob.glob(str(C.BRANCH_OUT / "*" / f"{dataset}.csv")))), None)
        if hit is None:
            raise FileNotFoundError(f"{dataset}: no stage-23 selection and no materialised branch")
        d = pd.read_csv(hit)
        v = [str(x) for x in d.v]
        return pd.DataFrame({"v": v, "source": dataset, "subset": dataset, "vname": v})
    else:
        d = A.load_frames(sel_csv(dataset))
        keep = in_scope(d)
        t = A.video_table(d[keep].reset_index(drop=True), [1.0])
        t["source"] = dataset
    cols = {"v": [str(x) for x in t.v]}
    for c in ("source", "subset", "vname"):
        if c in t.columns:
            cols[c] = t[c].to_numpy()
    if "vname" not in cols:
        cols["vname"] = cols["v"]
    return pd.DataFrame(cols)


def table(datasets, branches, require_all: bool = True) -> pd.DataFrame:
    """One wide table over the given datasets, carrying every requested branch.

    Development sets are concatenated into a single pool (the five source-grouped folds are drawn
    across it, as B0 does). Test sets are kept separate by the caller.
    """
    parts = []
    for ds in datasets:
        have = [b for b in branches if available(b, ds) or (C.BRANCH_OUT / b / f"{ds}.csv").exists()]
        if require_all and len(have) != len(branches):
            miss = [b for b in branches if b not in have]
            raise FileNotFoundError(f"{ds}: branches {miss} unavailable; run `--export` first "
                                    f"or drop them from the subset")
        tabs = {b: read(b, ds) for b in have}
        t = C.align(tabs, meta_for(ds), tag=ds)
        t["dataset"] = ds
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out.attrs["branches"] = list(branches)
    return out


def dev_table(branches, require_all: bool = True) -> pd.DataFrame:
    return table(C.DEV_SETS, branches, require_all)


def test_tables(branches, sets=C.TEST_SETS, require_all: bool = True) -> dict:
    return {s: table([s], branches, require_all) for s in sets}


# ----------------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="", help="comma-separated branches to materialise")
    ap.add_argument("--sets", default="dev,test", help="dev, test, or a comma list of names")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--table", action="store_true", help="build and summarise the joined table")
    a = ap.parse_args()

    names = []
    for s in a.sets.split(","):
        s = s.strip()
        if s == "dev":
            names += list(C.DEV_SETS)
        elif s == "test":
            names += list(C.TEST_SETS)
        elif s:
            names.append(s)

    if a.coverage or not (a.export or a.table):
        cov = coverage()
        print(cov.to_string())
        print("\nlegend:")
        for b, s in SOURCES.items():
            print(f"  {b}: {s['note']}")
        (C.OUT / "coverage.csv").parent.mkdir(parents=True, exist_ok=True)
        cov.to_csv(C.OUT / "coverage.csv")
        print(f"\n-> {C.OUT / 'coverage.csv'}")

    if a.export:
        rep = {}
        for b in [x.strip() for x in a.export.split(",") if x.strip()]:
            print(f"exporting {b}")
            rep[b] = export(b, names, a.force)
        json.dump(rep, open(C.OUT / "export_report.json", "w"), indent=1)
        print(f"-> {C.OUT / 'export_report.json'}")

    if a.table:
        br = [b for b in C.CORE if all(available(b, n) or (C.BRANCH_OUT / b / f"{n}.csv").exists()
                                       for n in names)]
        print(f"branches with full coverage of {len(names)} sets: {br}")
        if br:
            t = table(names, br, require_all=True)
            print(f"{len(t)} videos, {int((t.y == 1).sum())} fake, {int((t.y == 0).sum())} real")
            print(t.groupby("dataset").size().to_string())
            y = t.y.to_numpy()
            print("\nstandalone AUROC on this pool:")
            for b in br:
                print(f"  {b}: {C.auroc(y, t[f'p_{b}'].to_numpy()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
