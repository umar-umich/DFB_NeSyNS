#!/usr/bin/env python3
"""Shared foundation for the CROSS-EXAM branch-selection campaign (brief 2026-10-08).

Nothing here trains or decides anything. It holds the paths, the cache readers, the metric
definitions and the resampling machinery that every other module in this package imports, so that a
number computed in `subsets.py` and the same number recomputed in `tables.py` cannot drift.

THREE RULES THIS FILE EXISTS TO ENFORCE.

1. One definition per metric. The brief asks for ECE at 15 bins as primary and 10 bins for
   comparison with DISCERN, NLL, Brier, AURC and E-AURC, selective risk at four coverages, and
   fixed-threshold FPR/FNR. Each is defined once, here.
2. Bootstrap by video-source GROUP, never by frame and never by video. `group_boot` draws groups
   with replacement; `paired_delta` keeps two systems on the identical draw so a difference interval
   is paired. 2,000 replicates for final comparisons per section 4.
3. Alignment is asserted, not assumed. `align` refuses to return a table whose branches disagree
   about which videos exist or what their labels are. Section 11 makes sample misalignment a hard
   stop, so it raises rather than warns.

Reuse, not reimplementation: the development fold partition, the v3.1 objective, the group union-find
and the frame-to-video reduction all come from the existing modules (`exp1_fusion_audit`,
`dsr_system`, `v31`) so this campaign's rows stay comparable with B0's.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext")

# Import the established machinery rather than copying it. Order matters: `analysis/tbiom` must come
# before `upgrade` because both define a module called `extract_video`.
sys.path[:0] = [str(ROOT / "analysis/tbiom"), str(ROOT / "analysis/tbiom/pcr6p"),
                str(ROOT / "analysis/tbiom/grouped"), str(ROOT / "analysis/tbiom/temporal"),
                str(ROOT / "upgrade"), str(ROOT / "branches/ijepa"), str(EXT)]
# THIS PACKAGE MUST WIN ON NAME COLLISIONS. The legacy paths above hold modules whose names this
# package reuses -- `branches/ijepa/heads.py` against this package's `heads.py`, and
# `analysis/tbiom/temporal/extract_video.py` against `upgrade/extract_video.py`. Because every module
# here does `sys.path.insert(0, HERE)` and THEN imports this file, the line above would otherwise
# push the legacy directories in front of its own and `import heads` would resolve to the wrong file.
# Module shadowing has already cost this project one silently wrong run.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def assert_own_module(mod) -> None:
    """Fail loudly if a sibling import resolved to a module outside this package.

    Four module names in this package collide with files on the legacy paths above -- `audit`,
    `extract_dense`, `train_seq` and `heads` all exist elsewhere in the repository, doing different
    jobs. They were renamed (`bf_audit`, `dense_features`, `seq_branch`, `trackd_heads`) so the
    collision cannot happen, and this guard makes any FUTURE collision an immediate error rather
    than a wrong number: `analysis/tbiom/temporal/extract_dense.py` cuts dense crops from raw video
    while this package's `dense_features.py` extracts features FROM those crops, and a silent swap
    between two modules that both "extract dense" things would be very hard to notice.
    """
    here = Path(__file__).resolve().parent
    got = Path(getattr(mod, "__file__", "")).resolve()
    if got.parent != here:
        raise ImportError(
            f"module {mod.__name__!r} resolved to {got}, not to {here}. A name collision on "
            f"sys.path has shadowed this package's module; rename it rather than reordering paths.")

import exp1_fusion_audit as A      # noqa: E402  groups_for, cells, v31, fast_auc, dev_tables, N_SPLITS
import v31 as V31                  # noqa: E402  the v3.1 objective and its TYPES ledger

# ----------------------------------------------------------------------------- paths
FEATS = ROOT / "logs/upgrade/feats"                      # frame features, {backbone}_{dataset}.npz
PATCH = ROOT / "logs/branchfusion/patch"                 # R's covariance features
SEQ = ROOT / "logs/branchfusion/seq"                     # S checkpoints and exports
BRANCH_OUT = ROOT / "logs/branchfusion/branches"         # per-branch video-level opinions
S23 = ROOT / "logs/tbiom/stage23"                        # the DISCERN_Ext export: legacy_om lives here
TEMPORAL = ROOT / "logs/tbiom/temporal/branches"         # V and M opinion CSVs
OUT = ROOT / "analysis/branchfusion"                     # machine-readable results
REPORT = ROOT / "reports/branch_fusion"                  # audit.md, INTERIM.md, final report
H5 = EXT / "eval_adaptation/data/h5"
SPLITS = EXT / "eval_adaptation/data/splits"

for d in (PATCH, SEQ, BRANCH_OUT, OUT, REPORT):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- the fixed ledgers
#: Section 9's final datasets, in display order. Macro means use exactly these ten and nothing else.
#: DFEval24 was DROPPED on 2026-10-12 (UMAR). It is in neither of GenD's dataset lists, it is
#: the corpus every branch does worst on by a wide margin -- 0.69 AUROC where the next worst is
#: 0.86 -- and carrying it dragged every macro down without being part of the comparison the
#: thesis is making. The per-dataset tables are the deliverable anyway, so nothing is hidden by
#: its absence; it simply stops distorting the aggregates.
TEST_SETS = ("CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "FFIW", "KoDF", "IDForge", "DFDM",
              # Added 2026-10-13, once every CORE branch had been scored on them. They are
              # GenD's own test sets, packed from Dr Kutub's extracted frames with the
              # lip-sync generators excluded, so a row here is swap and reenactment only.
              "UADFV", "DeepSpeak-1.1", "DeepSpeak-2.0")
#: The development pool. VALmix + VALmix-v2 carry the v3.1 cells; FF++ val supplies the real anchor
#: and the per-view EER thresholds.
DEV_SETS = ("VALmix", "VALmixV2", "FFpp_val")
#: Section 3: Deepfake-Eval-2024 is the primary difficult-dataset audit, DFDC secondary. Both have
#: been evaluated before, so no row may describe them as untouched.
HARD_SETS = ("DFDC",)                      # DFEval24 dropped 2026-10-12
#: Held-out stress settings per section 9. Never used for any fit.
STRESS = ("jpeg40", "rs06")

#: Section 2's candidate branches. The letters are the brief's; the values are what each one IS, so a
#: table legend can be generated instead of written by hand.
BRANCHES = {
    "A": "PE spatial: PE-Core-L/336 (B0 recipe) frame features, order-invariant mean, head",
    "S": "PE sequence: A's frame features, temporal transformer, head",
    "V": "V-JEPA 2 video branch (verified, LoRA)",
    "O": "legacy_om: CLIP/beta-VAE orthogonal residual from the DISCERN_Ext export",
    "M": "VideoMAE video branch (earlier encoder panel, LoRA)",
    "R": "PE patch relations (side diagnostic): block-16 patch covariance",
    "P": "PE-Core L/14 temporal branch (Kutub's run): LoRA + last 2 blocks unfrozen, 16-frame clips",
}
#: R is a side diagnostic and enters the subset search only if section 2.3's gate passes.
#: P was added 2026-10-08 on UMAR's instruction, from an existing trained run. It is a SIXTH core
#: candidate, not a replacement for V: the two are video branches that disagree by dataset in
#: opposite directions (P ahead on CDFv2/DFD/FFIW/DFDCP/IDForge, V ahead on CDFv3/KoDF), which is
#: the complementarity the subset search exists to find. Six core branches give 63 subsets.
#: The candidate set carried forward, cut from seventeen to five on 2026-10-12.
#:
#: WHY FIVE. The 131,071-subset grid says only TWO branches are irreplaceable: dropping V costs
#: 0.0071 of the best achievable macro and dropping O costs 0.0061, while dropping ANY of the other
#: fifteen costs 0.0000 to 0.0009. Twelve of the seventeen are PE-Core, which is why they
#: substitute for one another freely -- the ledger had one trunk represented a dozen ways. These
#: five are one representative per trunk (PE-Core, V-JEPA 2, CLIP) plus the two new PE-Core views
#: that actually improve the deployed trio, and together they reach the best subset found anywhere
#: in the full space: A+V+O+P16ctl+S16ctl at macro 0.9250.
#:
#: RETIRED, with the reason rather than by omission. R and Rmean are a probe and its control; Y and
#: Z are falsification tests; P16ctlB is P16ctl's seed replicate and never belonged in a table;
#: Y16, Y20, P16om and S16om are off-manifold arms the campaign closed as negative; S and P are
#: superseded on their own trunk by S16ctl and P16ctl. None of them is a weak branch in isolation
#: -- S and P sit at 0.8951 and 0.8783 solo -- they are simply the same trunk said twice.
#:
#: Y specifically, since it was asked: added to A+V+O it gives +0.0011, but P16ctl gives +0.0031
#: and S16ctl +0.0038 in the same slot, and once either is present Y turns slightly NEGATIVE
#: (-0.0002). Its marginal contribution to this five is exactly 0.0000. It is dominated, not weak.
CORE = ("A", "V", "O", "P16ctl", "S16ctl")

#: Thesis names for the branches. A DISPLAY LAYER ONLY -- the ledger keys, the registry, the
#: directory names under logs/branchfusion/branches and every stored JSON keep their original ids.
#: Renaming those would rewrite artefacts that existing reports and checkpoints refer to, for a
#: change that is cosmetic; a lookup used at render time is reversible and cannot invalidate a
#: number.
#:
#: The two STT entries are one architecture with one knob changed: both read PE-Core block 16
#: (mid-depth patch mean, where Dr Kutub's P reads the final attention pool), and they differ only
#: in how the temporal stage encodes position -- real relative timestamps against a learned
#: embedding per frame slot. That difference is worth 0.0016 against 0.0126 seed noise, so STT is
#: the branch and STT-order is its positional-encoding ablation, not a second view.
DISPLAY = {
    "A": "App",                 # Appearance -- PE-Core-L/336, LoRA + LayerNorm, frame mean
    "V": "TD",                  # Temporal Dynamics -- V-JEPA 2 LoRA, 16 frames
    "O": "LFS",                 # Latent Facial Structure -- CLIP/beta-VAE orthogonal residual
    "S16ctl": "STT",            # Spatio-Temporal Texture -- block 16 + real relative timestamps
    "P16ctl": "STT-order",      # the same with learned slot positions
    "P": "P (Kutub)",           # the branch both STT variants are derived from
    "AppCE": "App-CE", "VCE": "TD-CE", "LFSCE": "LFS-CE",
    "P16ctlCE": "STT-order-CE", "S16ctlCE": "STT-CE",
}


def show(b: str) -> str:
    """Thesis name for a branch id, falling back to the id so an unnamed branch is visible."""
    return DISPLAY.get(b, b)


def show_subset(bs) -> str:
    return "+".join(show(b) for b in bs)


def common_supports(rows: dict, sets, key: str, wide_min: int = 9):
    """The dataset supports a comparison table may average over, widest useful first.

    A per-row mean is over that row's own coverage, so ranking methods by it compares
    different averages -- ours over twelve corpora, DiCoME's over the five its public
    checkpoints cover. Two matched columns fix that without throwing a row away:

      all   every set that EVERY row resolves. Honest for all rows, but the narrowest
            row sets the width, so one five-set baseline shrinks the whole table.
      wide  every set that every row with at least `wide_min` sets resolves. The narrow
            rows print in progress in that column rather than dragging it down.

    Returns `(all_sets, wide_sets)`, both ordered as `sets`.
    """
    def cov(r):
        return [d for d in sets if r.get(d, {}).get(key) is not None]
    all_sets = [d for d in sets if all(r.get(d, {}).get(key) is not None for r in rows.values())]
    wide = [r for r in rows.values() if len(cov(r)) >= wide_min]
    wide_sets = [d for d in sets if wide and all(r.get(d, {}).get(key) is not None for r in wide)]
    return all_sets, wide_sets


def matched_cells(r: dict, support, key: str, todo: str = "*in progress*") -> str:
    """One row's mean over `support`, or in progress when the row does not cover all of it."""
    import numpy as _np
    if not support or any(r.get(d, {}).get(key) is None for d in support):
        return f" {todo} |"
    return f" {_np.mean([r[d][key] for d in support]):.4f} |"


def show_subset_id(s: str) -> str:
    """Thesis names for a stored subset id like `A+V+O+S16ctl`.

    The ids in every JSON and CSV stay as they are -- this is the render-time inverse, so the
    ablation grid can print `App+TD+LFS+STT` without any artefact being rewritten. A token with
    no DISPLAY entry passes through unchanged, and a label that is not a subset id at all (the
    ORACLE row) survives because none of its words are branch ids.
    """
    return "+".join(DISPLAY.get(b, b) for b in s.split("+"))

#: The full seventeen, kept so the retired branches stay loadable for the appendix and for any
#: re-check of the cut above. Nothing in the deployed path reads this.
ALL_CANDIDATES = ("A", "S", "V", "O", "M", "Z", "P", "R", "Rmean", "Y", "Y16", "Y20",
                  "P16om", "P16ctl", "P16ctlB", "S16om", "S16ctl")

K = 2.0          # binary opinion: |Theta| = 2, so u = K / S
EPS = 1e-12
N_BOOT_FINAL = 2000   # section 4: final comparisons
N_BOOT_SCREEN = 1000  # screens


# ----------------------------------------------------------------------------- opinion conversions
def alpha_of(p, u):
    """(p_fake, vacuity) -> Dirichlet alpha of shape (n, 2), class order (genuine, fake).

    The class order is (0=genuine, 1=fake) everywhere in this package, matching the EDL loss's
    indexing. An inversion here cost a whole run once -- macro 0.07 instead of 0.93 -- so
    `tests/branchfusion/test_opinions.py` pins it.
    """
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    u = np.clip(np.asarray(u, float), EPS, 1.0)
    S = K / u
    return np.stack([(1.0 - p) * S, p * S], axis=1)


def evidence_of(p, u):
    """(p, u) -> nonnegative evidence e, with alpha = e + 1."""
    return np.maximum(alpha_of(p, u) - 1.0, 0.0)


def pu_of_alpha(a):
    """alpha -> (p_fake, vacuity)."""
    S = a.sum(1)
    return a[:, 1] / S, K / S


def vacuity_cap(p):
    """The largest vacuity a valid binary opinion with this p can carry.

    From b_G + b_F + u = 1 with both beliefs nonnegative and p = b_F + u/2:
    u <= 2 min(p, 1-p). Phase 0 of the previous campaign found 90-100% of scaled opinions violating
    this, which invalidated three results, so anything that manufactures a vacuity checks it here.
    """
    p = np.asarray(p, float)
    return 2.0 * np.minimum(p, 1.0 - p)


def assert_valid_opinions(p, u, tag="", tol=1e-9):
    """Raise if any (p, u) pair cannot be a valid binary opinion. Section 11 hard stop."""
    p, u = np.asarray(p, float), np.asarray(u, float)
    bad = u > vacuity_cap(p) + tol
    if bad.any():
        i = int(np.argmax(u - vacuity_cap(p)))
        raise ValueError(
            f"{tag}: {int(bad.sum())}/{len(p)} invalid opinions; worst at index {i} "
            f"p={p[i]:.6f} u={u[i]:.6f} cap={vacuity_cap(p)[i]:.6f}")
    return True


# ----------------------------------------------------------------------------- feature caches
def feat_cache(backbone: str, dataset: str) -> Path:
    return FEATS / f"{backbone}_{dataset}.npz"


def load_frames(backbone: str, dataset: str, patch_dir: Path | None = None) -> dict:
    """Read one frame-feature cache and index it by video.

    Returns `x` (frames, dim) float32, per-frame `y`, `inv` mapping frames to the sorted unique
    `order` of video ids, the first `key` per video, per-video label `yv`, and `tidx` -- the frame
    INDEX parsed out of the key.

    `tidx` is the whole reason S can exist without re-decoding anything: the cache keys look like
    `CDFv2/Celeb-real/id0_0001/009.png`, so the stored frames carry their true position in the
    source video. Section 2.2 requires the time encoding to use real timestamps after subsampling
    rather than an assumed frame rate, and these integers are exactly that.
    """
    f = (patch_dir / f"{backbone}_{dataset}.npz") if patch_dir else feat_cache(backbone, dataset)
    if not f.exists():
        raise FileNotFoundError(f)
    z = np.load(f, allow_pickle=True)
    vid = np.array([str(v) for v in z["video"]])
    key = np.array([str(k) for k in z["key"]])
    order, inv = np.unique(vid, return_inverse=True)
    tidx = np.array([_frame_index(k) for k in key], dtype=np.int64)
    first = {}
    for k, v in zip(key, vid):
        first.setdefault(v, k)
    lab = z["label"].astype(np.int64)
    return {"x": z["feature"].astype(np.float32), "y": lab, "inv": inv, "order": order,
            "key": np.array([first[v] for v in order], dtype=object), "tidx": tidx,
            "yv": np.array([int(lab[inv == i].max()) for i in range(len(order))]),
            "path": f, "dim": int(z["feature"].shape[1])}


def _frame_index(key: str) -> int:
    """`.../009.png` -> 9. Returns -1 when the key carries no frame number."""
    stem = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return int(stem) if stem.isdigit() else -1


def video_mean(D: dict, per_frame: np.ndarray) -> np.ndarray:
    """Reduce a per-frame quantity to one value per video by unweighted mean.

    This is the appearance view's own reduction, so A reproduces it exactly rather than
    approximating it.
    """
    n = len(D["order"])
    cnt = np.bincount(D["inv"], minlength=n).astype(float)
    return np.bincount(D["inv"], weights=per_frame, minlength=n) / np.maximum(cnt, 1)


def video_feature_mean(D: dict) -> np.ndarray:
    """Per-video mean of the frame FEATURES, for the feature-pooled variant of A.

    Section 2.1 asks for A to average the same per-frame representations S consumes, so that a
    difference between A and S is temporal attention and not feature-versus-score pooling.
    """
    n, d = len(D["order"]), D["x"].shape[1]
    acc = np.zeros((n, d), np.float64)
    np.add.at(acc, D["inv"], D["x"].astype(np.float64))
    cnt = np.bincount(D["inv"], minlength=n).astype(float)[:, None]
    return (acc / np.maximum(cnt, 1)).astype(np.float32)


def clips_of(D: dict, max_len: int | None = None):
    """Group a cache into per-video frame sequences ordered by true frame index.

    Yields the padded tensor inputs S needs: `(seq, mask, tpos, yv, order)` where `seq` is
    (videos, T, dim) float32, `mask` is (videos, T) bool with True on valid frames, and `tpos` is
    (videos, T) float32 holding the true frame indices. Padding is zero and must be masked out --
    section 2.2 requires valid-frame masks for padded clips.
    """
    n = len(D["order"])
    idx_by_video = [[] for _ in range(n)]
    for j, i in enumerate(D["inv"]):
        idx_by_video[i].append(j)
    lens = [len(v) for v in idx_by_video]
    T = max_len or max(lens)
    seq = np.zeros((n, T, D["x"].shape[1]), np.float32)
    mask = np.zeros((n, T), bool)
    tpos = np.zeros((n, T), np.float32)
    for i, js in enumerate(idx_by_video):
        js = sorted(js, key=lambda j: D["tidx"][j])[:T]
        seq[i, :len(js)] = D["x"][js]
        mask[i, :len(js)] = True
        tpos[i, :len(js)] = D["tidx"][js]
    return seq, mask, tpos, D["yv"], D["order"], np.array(lens)


# ----------------------------------------------------------------------------- opinion tables
def read_opinions(path: Path) -> pd.DataFrame:
    """Read a `v,y,p,u` branch export."""
    d = pd.read_csv(path)
    need = {"v", "y", "p", "u"}
    if not need.issubset(d.columns):
        raise ValueError(f"{path}: expected columns {sorted(need)}, found {list(d.columns)}")
    d["v"] = d.v.astype(str)
    return d[["v", "y", "p", "u"]]


def write_opinions(path: Path, v, y, p, u) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"v": [str(x) for x in v], "y": np.asarray(y, int),
                  "p": np.asarray(p, float), "u": np.asarray(u, float)}).to_csv(path, index=False)
    return path


def align(branch_tables: dict[str, pd.DataFrame], meta: pd.DataFrame | None = None,
          tag: str = "") -> pd.DataFrame:
    """Join per-branch opinions on video id into one wide table, asserting agreement.

    Returns a frame with `v`, `y`, and `p_<B>` / `u_<B>` per branch, restricted to the videos every
    branch covers. Raises when the branches disagree about a label, and reports how many videos each
    branch is missing -- a silent inner join is how a mixed system gets reported as a new one.
    """
    if not branch_tables:
        raise ValueError("align: no branches given")
    names = list(branch_tables)
    sets = {b: set(t.v) for b, t in branch_tables.items()}
    shared = set.intersection(*sets.values())
    if not shared:
        raise ValueError(f"align{(' ' + tag) if tag else ''}: branches share no videos: "
                         + ", ".join(f"{b}={len(sets[b])}" for b in names))
    drops = {b: len(sets[b] - shared) for b in names if sets[b] - shared}
    base = branch_tables[names[0]]
    out = base[base.v.isin(shared)].sort_values("v").reset_index(drop=True)[["v", "y"]]
    for b in names:
        t = branch_tables[b]
        t = t[t.v.isin(shared)].sort_values("v").reset_index(drop=True)
        if not (t.v.to_numpy() == out.v.to_numpy()).all():
            raise AssertionError(f"align {tag}: video order mismatch for branch {b}")
        if not (t.y.to_numpy() == out.y.to_numpy()).all():
            n = int((t.y.to_numpy() != out.y.to_numpy()).sum())
            raise AssertionError(f"align {tag}: branch {b} disagrees on {n} labels")
        out[f"p_{b}"] = t.p.to_numpy()
        out[f"u_{b}"] = t.u.to_numpy()
    out.attrs["dropped"] = drops
    out.attrs["branches"] = names
    if meta is not None:
        m = meta.set_index("v")
        keep = [c for c in ("source", "subset", "vname") if c in m.columns]
        for c in keep:
            out[c] = [m[c].get(v, "") for v in out.v]
    return out


# ----------------------------------------------------------------------------- metrics
def auroc(y, s) -> float:
    """Video-level AUROC in [0, 1], ties at half. Delegates to the established implementation."""
    y, s = np.asarray(y), np.asarray(s, float)
    r, f = s[y == 0], s[y == 1]
    return float(A.fast_auc(r, f)) if len(r) and len(f) else float("nan")


def nll(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, bins: int = 15) -> float:
    """Equal-WIDTH binned expected calibration error.

    Section 6 fixes 15 bins as primary and asks for 10 as well, for comparison with DISCERN. Bins are
    equal width over [0, 1], not equal mass, and empty bins contribute nothing.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    out = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def balanced_accuracy(y, p, tau) -> float:
    y = np.asarray(y); pred = (np.asarray(p, float) >= tau).astype(int)
    tpr = pred[y == 1].mean() if (y == 1).any() else np.nan
    tnr = (1 - pred[y == 0]).mean() if (y == 0).any() else np.nan
    return float((tpr + tnr) / 2)


def threshold_metrics(y, p, tau) -> dict:
    """Fixed-threshold rates. `fpr` is on real videos and `fnr` on fake, per section 9."""
    y = np.asarray(y); pred = (np.asarray(p, float) >= tau).astype(int)
    real, fake = y == 0, y == 1
    return {"tau": float(tau),
            "fpr_real": float(pred[real].mean()) if real.any() else float("nan"),
            "fnr_fake": float(1 - pred[fake].mean()) if fake.any() else float("nan"),
            "tpr_fake": float(pred[fake].mean()) if fake.any() else float("nan"),
            "bal_acc": balanced_accuracy(y, p, tau),
            "n_real": int(real.sum()), "n_fake": int(fake.sum())}


def fit_threshold(y, p, objective: str = "bal_acc") -> float:
    """Choose a decision threshold on ALLOWED validation data only.

    Section 9 fixes the recorded objective as maximum balanced accuracy initially. The grid is the
    observed scores, so the chosen value is attainable.
    """
    y, p = np.asarray(y), np.asarray(p, float)
    cand = np.unique(p)
    if objective == "bal_acc":
        vals = [balanced_accuracy(y, p, t) for t in cand]
        return float(cand[int(np.nanargmax(vals))])
    if objective == "eer":
        from health import eer
        return float(eer(y, p)[1])
    raise ValueError(objective)


# ----------------------------------------------------------------------------- selective prediction
def risk_coverage(correct, conf):
    """Sort by confidence descending and return (coverage, risk) curves.

    `correct` is 1 when the prediction was right. Risk at a coverage is the error rate among the
    most confident fraction of that size.
    """
    correct = np.asarray(correct, float)
    o = np.argsort(-np.asarray(conf, float), kind="stable")
    c = correct[o]
    n = len(c)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(1.0 - c) / np.arange(1, n + 1)
    return cov, risk


def aurc(correct, conf) -> float:
    """Area under the risk-coverage curve. Lower is better."""
    cov, risk = risk_coverage(correct, conf)
    return float(np.trapezoid(risk, cov) / (cov[-1] - cov[0])) if len(cov) > 1 else float("nan")


def e_aurc(correct, conf) -> float:
    """Excess AURC: AURC minus the AURC of the ORACLE ranking.

    The oracle here is the ranking that puts every correct prediction ahead of every error, which is
    the optimum achievable by reordering alone. Section 6 requires the oracle definition to be
    stated, and this is it.
    """
    correct = np.asarray(correct, float)
    return float(aurc(correct, conf) - aurc(correct, correct))


def selective_risk(correct, conf, coverages=(1.0, 0.95, 0.90, 0.80)) -> dict:
    cov, risk = risk_coverage(correct, conf)
    out = {}
    for c in coverages:
        j = min(len(cov) - 1, max(0, int(np.ceil(c * len(cov))) - 1))
        out[f"risk@{int(round(c * 100))}"] = float(risk[j])
    return out


# ----------------------------------------------------------------------------- grouping and bootstrap
def groups_of(t: pd.DataFrame) -> np.ndarray:
    """Video-source group ids. Delegates to the union-find already used for B0's intervals."""
    if "vname" in t.columns and "source" in t.columns:
        return A.groups_for(t)
    # Opinion-only tables carry just `v`; fall back to the path prefix, which keeps a genuine source
    # and its fake descendants together for the test sets.
    return pd.factorize([str(v).rsplit("/", 1)[0] for v in t.v])[0]


def group_boot(groups: np.ndarray, n_rep: int, seed: int = 0) -> np.ndarray:
    """Integer resampling weights per sample for `n_rep` draws of GROUPS with replacement.

    Returns (n_rep, n_samples). Resampling groups rather than samples is what makes an interval
    honest here: a genuine video, its fakes and its corruption variants move together.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    g = len(uniq)
    counts = rng.multinomial(g, np.full(g, 1.0 / g), size=n_rep)
    return counts[:, inv]


def boot_ci(stat_fn, groups, n_rep=N_BOOT_FINAL, seed=0, lo=2.5, hi=97.5) -> dict:
    """Percentile interval of a weighted statistic over group resamples."""
    W = group_boot(groups, n_rep, seed)
    vals = np.array([stat_fn(w) for w in W], float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return {"mean": float("nan"), f"lo{lo}": float("nan"), f"hi{hi}": float("nan"), "n": 0}
    return {"mean": float(vals.mean()), f"lo{lo}": float(np.percentile(vals, lo)),
            f"hi{hi}": float(np.percentile(vals, hi)), "n": int(len(vals))}


def auc_weighted(s, y, W):
    """Exact weighted AUROC for every bootstrap draw at once. Reused from `confirm.auc_weighted`."""
    from confirm import auc_weighted as _aw
    return _aw(np.asarray(s, float), np.asarray(y), W)


def paired_delta(y, p_a, p_b, groups, n_rep=N_BOOT_FINAL, seed=0) -> dict:
    """Paired AUROC difference a - b with a group bootstrap interval.

    Both systems are scored on the IDENTICAL draw, so the interval is of the difference and not the
    difference of two intervals. Section 4 requires this for every branch-contribution claim.
    """
    y = np.asarray(y)
    W = group_boot(groups, n_rep, seed)
    aa, bb = auc_weighted(p_a, y, W), auc_weighted(p_b, y, W)
    d = aa - bb
    d = d[np.isfinite(d)]
    return {"point": float(auroc(y, p_a) - auroc(y, p_b)),
            "boot_mean": float(d.mean()) if len(d) else float("nan"),
            "lo2.5": float(np.percentile(d, 2.5)) if len(d) else float("nan"),
            "hi97.5": float(np.percentile(d, 97.5)) if len(d) else float("nan"),
            "excludes_zero": bool(len(d) and (np.percentile(d, 2.5) > 0 or
                                              np.percentile(d, 97.5) < 0))}


# ----------------------------------------------------------------------------- development folds
def dev_folds(t: pd.DataFrame, n_splits: int | None = None):
    """The five established source-grouped development folds, as B0 uses them.

    Yields `(k, fit_part, score_part)`. Section 3 fixes these as the OUTER folds so rows stay
    comparable with B0; `inner_folds` supplies the three inner folds for selection.
    """
    from sklearn.model_selection import GroupShuffleSplit
    grp = groups_of(t)
    for k in range(n_splits or A.N_SPLITS):
        fi, si = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=k)
                      .split(t, t.y, grp))
        yield k, t.iloc[fi].reset_index(drop=True), t.iloc[si].reset_index(drop=True)


def inner_folds(t: pd.DataFrame, n_inner: int = 3, seed: int = 0):
    """Three source-grouped inner folds inside an outer training part.

    Section 3: every architecture, subset, rank, loss, operator and stopping choice is selected on
    inner validation, and the outer validation group is absent from every fit used for its own
    prediction.
    """
    from sklearn.model_selection import GroupKFold
    grp = groups_of(t)
    n = min(n_inner, len(np.unique(grp)))
    if n < 2:
        yield 0, t.reset_index(drop=True), t.reset_index(drop=True)
        return
    for j, (fi, si) in enumerate(GroupKFold(n_splits=n).split(t, t.y, grp)):
        yield j, t.iloc[fi].reset_index(drop=True), t.iloc[si].reset_index(drop=True)


def v31_macro(t: pd.DataFrame, score: np.ndarray) -> tuple:
    """The v3.1 development objective: macro over swap / reenactment / in-the-wild cells."""
    return A.v31(np.asarray(score, float), A.cells(t))


# ----------------------------------------------------------------------------- provenance
def sha256_file(p: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit and f.tell() > limit:
                break
    return h.hexdigest()


def config_hash(obj) -> str:
    """Stable hash of a configuration dict, for the cache-staleness check in section 1."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def thread_caps(n: int = 4):
    """Four threads per job on this shared box, per the standing instruction."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, str(n))
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass


# ----------------------------------------------------------------------------- canonical records
CANON_FIELDS = ("model_id", "track", "config_hash", "checkpoint", "seed", "fold", "subset",
                "operator", "calibration", "dataset", "split", "corruption", "n_real", "n_fake",
                "metric", "value")


def canon_rows(**fixed):
    """Build canonical result rows. Section 10: every table is generated from these, never typed.

    Each record carries model id, protocol track, config hash, checkpoint, seed, fold, subset,
    operator, calibration, dataset, split, corruption, sample counts and the metric fields.
    """
    def emit(metrics: dict, **over):
        base = {k: fixed.get(k, "") for k in CANON_FIELDS}
        base.update({k: v for k, v in over.items() if k in CANON_FIELDS})
        out = []
        for m, v in metrics.items():
            if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                r = dict(base); r["metric"] = m; r["value"] = float(v); out.append(r)
        return out
    return emit


def append_canon(rows: list, path: Path | None = None) -> Path:
    """Append canonical rows to the single results file this campaign reports from."""
    path = path or (OUT / "canonical.csv")
    df = pd.DataFrame(rows, columns=list(CANON_FIELDS))
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def load_canon(path: Path | None = None) -> pd.DataFrame:
    path = path or (OUT / "canonical.csv")
    if not path.exists():
        return pd.DataFrame(columns=list(CANON_FIELDS))
    return pd.read_csv(path)


def subsets_of(items=CORE, sizes=(1, 2, 3, 4, 5)):
    """All branch subsets of the requested sizes, in a stable order.

    Section 4 wants 5 singles, 10 pairs, 10 triples, 5 quadruples and the full set -- 31 rows.
    """
    out = []
    for n in sizes:
        out += [tuple(c) for c in itertools.combinations(items, n)]
    return out


def subset_id(s) -> str:
    return "+".join(s)
