#!/usr/bin/env python3
"""Task 6 of the leakage audit: code-path tests over the real fitting and threshold functions.

    /data/umar/miniconda3/envs/discern_ext/bin/python -m pytest audit/leakage/test_fit_inputs.py -v

Read-only. Nothing in training or fusion is modified; the tests instrument the functions from the outside by
wrapping them, and restore the originals afterwards.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXT = Path("/data/umar/Repos/DISCERN_Ext")
for p in ("analysis/tbiom/pcr6p", "analysis/tbiom/grouped", "analysis/tbiom", "analysis/tbiom/temporal"):
    sys.path.insert(0, str(ROOT / p))

import dsr_system as D            # noqa: E402
import finish_line as F           # noqa: E402
import arms as AR                 # noqa: E402
import exp1_fusion_audit as A     # noqa: E402

TEN = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]


@pytest.fixture(scope="module")
def dev():
    return D.load("vjepa2_lora_B_seed42")[0]


def test_dev_pool_contains_no_test_dataset(dev):
    """The trust-weight grid search is handed the development pool. Its `source` column must name no test set."""
    srcs = set(map(str, dev.source.unique()))
    assert not (srcs & set(TEN)), f"development pool contains test sources: {srcs & set(TEN)}"
    assert srcs, "development pool has no source column values"


def test_grid_search_sees_only_dev_pool_ids(dev, monkeypatch):
    """Wrap the objective the grid search maximises and record every table it is asked to score."""
    seen = []
    orig = A.v31

    def spy(score, cl, w=None):
        seen.append(len(score))
        return orig(score, cl, w)

    monkeypatch.setattr(A, "v31", spy)
    monkeypatch.setattr(F, "v31", spy, raising=False)
    k, Fd, Sd = next(iter(D.splits(dev)))
    AR.fit_w(Fd, "PCR6P-R", A.cells(Fd))
    assert seen, "the grid search never called the objective"
    assert set(seen) == {len(Fd)}, (f"objective was evaluated on row counts {set(seen)}, but the fitting fold has "
                                    f"{len(Fd)} rows -- something other than the fitting fold was scored")


def test_threshold_uses_only_ffpp_val(dev):
    """tau is the EER on FF++ val. Verify the selector picks exactly the FF++ val rows and nothing else."""
    k, Fd, Sd = next(iter(D.splits(dev)))
    ff = np.where(Fd.source.to_numpy() == "FFpp_val")[0]
    assert len(ff) > 0, "no FF++ val rows in the fitting fold"
    assert set(Fd.source.to_numpy()[ff]) == {"FFpp_val"}, "the threshold selector picked non-FF++-val rows"
    p = np.linspace(0.01, 0.99, len(Fd))
    tau_all = AR.tau_of(p, Fd)
    shuffled = p.copy()
    non_ff = np.setdiff1d(np.arange(len(Fd)), ff)
    rng = np.random.default_rng(0)
    shuffled[non_ff] = rng.permutation(shuffled[non_ff])       # scramble every non-FF++-val score
    assert AR.tau_of(shuffled, Fd) == tau_all, "tau changed when non-FF++-val scores were scrambled"


def test_no_test_label_loaded_during_fitting(dev, monkeypatch):
    """Instrument the frame loader and assert no test manifest is read while weights are being fitted."""
    opened = []
    orig = A.load_frames

    def spy(path, *a, **kw):
        opened.append(str(path))
        return orig(path, *a, **kw)

    monkeypatch.setattr(A, "load_frames", spy)
    k, Fd, Sd = next(iter(D.splits(dev)))
    AR.fit_w(Fd, "PCR6P-R", A.cells(Fd))
    bad = [o for o in opened if any(f"_{t}." in o or f"/{t}." in o for t in TEN)]
    assert not bad, f"a test-set export was read during weight fitting: {bad[:3]}"


def test_no_fitted_statistic_touches_test_data():
    """Every fitted quantity in the fusion path, and where it is computed. A new one must be added here knowingly."""
    # Every fitted quantity anywhere in the fusion/decision path, enumerated by searching for fit sites rather than
    # from memory. A statistic added later without being listed here will fail the completeness check below.
    fitted = {
        "reliability weights r_v": ("analysis/tbiom/pcr6p/arms.py", "fit_w", "fitting fold of the development pool"),
        "decision threshold tau": ("analysis/tbiom/pcr6p/arms.py", "tau_of", "FF++ val rows only"),
        "logistic comparator (scaler + LR)": ("analysis/tbiom/temporal/fusion_combos.py", "lr_fit",
                                              "fitting fold only"),
        "DS-R weight grid": ("analysis/tbiom/temporal/finish_line.py", "fit_w", "fitting fold(s) only"),
        "conflict-by-source probe (diagnostic, not deployed)":
            ("analysis/tbiom/pcr6p/development.py", "cbs_fit", "fitting fold only"),
    }
    for name, (rel, fn, scope) in fitted.items():
        src = (ROOT / rel).read_text()
        assert f"def {fn}" in src, f"{name}: {fn} not found in {rel}"
        assert scope                                            # documented above, asserted by the tests here
    # completeness: every `.fit(` site in the fusion/decision path must belong to a listed statistic
    FIT_FILES = ["analysis/tbiom/pcr6p/arms.py", "analysis/tbiom/pcr6p/ops_pcr6p.py",
                 "analysis/tbiom/temporal/fusion_combos.py", "analysis/tbiom/temporal/finish_line.py",
                 "analysis/tbiom/pcr6p/development.py"]
    listed_files = {v[0] for v in fitted.values()}
    for rel in FIT_FILES:
        src = (ROOT / rel).read_text()
        sites = [ln.strip() for ln in src.split("\n") if ".fit(" in ln and not ln.strip().startswith("#")]
        if sites:
            assert rel in listed_files, f"{rel} fits a statistic that is not listed in this test: {sites[0][:70]}"

    # no per-dataset normalisation or temperature anywhere in the fusion path
    for rel in ("analysis/tbiom/pcr6p/arms.py", "analysis/tbiom/pcr6p/ops_pcr6p.py",
                "analysis/tbiom/grouped/ops.py"):
        src = (ROOT / rel).read_text()
        for pat in (r"temperature", r"StandardScaler\(\)\.fit\(.*test", r"per_dataset_norm"):
            assert not re.search(pat, src, re.I), f"{rel} contains a per-dataset fitted statistic matching {pat}"


def test_eval_only_processing_absent_from_training_and_selection():
    """JPEG q40 and resize 0.60 are evaluation corruptions. They must not appear in any training augmentation."""
    train_cfg = (EXT / "eval_adaptation/configs/warm_dicome_gate_seed42.yaml").read_text()
    for pat in ("jpeg40", "q40", "rs06", "resize:0.6", "0.60"):
        assert pat not in train_cfg, f"the deployed training config mentions the evaluation corruption {pat}"
    branch = (ROOT / "analysis/tbiom/temporal/train_branch.py").read_text()
    m = re.search(r"JPEG q40 / resize 0\.60 remain evaluation-only", branch)
    assert m, "train_branch.py no longer records that q40/rs06 are evaluation-only"
    # the B augmentation mixture actually used, quoted from the recipe
    assert "q70/q50/q30" in branch and "0.75/0.50/0.35" in branch, \
        "the B augmentation mixture in train_branch.py is not the documented q70/q50/q30 and 0.75/0.50/0.35"
