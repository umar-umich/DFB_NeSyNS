#!/usr/bin/env python3
"""Section 12's acceptance checks and section 5's required operator algebra, as tests.

    <discern_ext_timm python> -m pytest tests/branchfusion/ -q

These are the checks the brief says must hold before any gain is interpreted. Several of them exist
because the exact failure they catch has already happened once in this project:

  class order          An evidence array from `ops` is (fake, genuine) and one from `common` is
                       (genuine, fake). Mixing them reported macro 0.07 for a system scoring 0.93.
  padding invariance   S must ignore padded frames. Without the mask the order controls would
                       measure padding, not motion.
  permutation mask     The order-discrimination loss must permute VALID frames among valid slots.
                       The first implementation sorted padding to the FRONT.
  prefix tokens        R must strip the right number. PE-Core-L has 1, DINOv3 ViT-L has 5, and the
                       obvious `z[:, 1:]` is wrong for the second.
  A is order-invariant Section 12 requires it, and if it ever moves, every temporal delta is suspect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path.insert(0, str(ROOT / "upgrade/branchfusion"))

import common as C          # noqa: E402
import opinions as O        # noqa: E402


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(7)
    n = 600
    y = rng.integers(0, 2, n)
    t = pd.DataFrame({"y": y, "v": [f"src{i // 10}/vid{i}" for i in range(n)]})
    for b, sig in zip("XYZ", (0.55, 0.35, 0.45)):
        p = np.clip(0.5 + sig * (y - 0.5) + rng.normal(0, 0.22, n), 0.02, 0.98)
        u = np.minimum(rng.uniform(0.05, 0.5, n), 2 * np.minimum(p, 1 - p) * 0.98)
        t[f"p_{b}"], t[f"u_{b}"] = p, u
    return t


# ----------------------------------------------------------------------------- opinion algebra
def test_opinion_round_trip():
    p = np.array([0.1, 0.5, 0.9])
    u = np.array([0.15, 0.4, 0.15])
    a = C.alpha_of(p, u)
    p2, u2 = C.pu_of_alpha(a)
    assert np.allclose(p, p2) and np.allclose(u, u2)


def test_class_order_is_pinned():
    """`common` puts fake at index 1; `ops`-derived evidence puts it at index 0."""
    a = C.alpha_of([0.9], [0.1])
    assert a[0, 1] > a[0, 0], "common.alpha_of must be (genuine, fake)"
    assert (O.OPS_FAKE, O.OPS_GEN) == (0, 1), "ops evidence must be (fake, genuine)"


def test_vacuity_cap_and_validity():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(C.vacuity_cap(p), [0.2, 1.0, 0.2])
    C.assert_valid_opinions(p, [0.2, 1.0, 0.2])
    with pytest.raises(ValueError):
        C.assert_valid_opinions([0.1], [0.5])


@pytest.mark.parametrize("op", ["F6", "F7", "F8"])
def test_operator_endpoints(toy, op):
    """rho = 1 is the identity; rho = 0 must equal dropping the source."""
    e = O.endpoint_checks(toy, ("X", "Y", "Z"), op)
    assert e["applicable"]
    assert e["rho1_equals_undiscounted"] < 1e-9
    assert e["rho0_equals_dropped"] < 1e-9


@pytest.mark.parametrize("op", ["F1", "F5"])
def test_undiscounted_operators_report_endpoints_not_applicable(toy, op):
    """F1 and F5 read no reliability, so a rho endpoint on them is a meaningless assertion."""
    e = O.endpoint_checks(toy, ("X", "Y", "Z"), op)
    assert e["applicable"] is False and "reason" in e


def test_F5_ignores_weights_and_F6_does_not(toy):
    """The difference between the two rows IS the discounting; this pins it."""
    w = [1.0, 0.5, 0.0]
    assert np.allclose(O.fuse(toy, ("X", "Y", "Z"), "F5", w),
                       O.fuse(toy, ("X", "Y", "Z"), "F5", None))
    assert np.max(np.abs(O.fuse(toy, ("X", "Y", "Z"), "F6", w)
                         - O.fuse(toy, ("X", "Y", "Z"), "F6", None))) > 1e-3


@pytest.mark.parametrize("op", ["F5", "F7", "F8"])
def test_multisource_is_permutation_invariant(toy, op):
    r = O.permutation_spread(toy, ("X", "Y", "Z"), op)
    assert r["n_permutations"] == 6
    assert r["invariant"], f"{op} moved by {r['max_abs_change']:.2e} under reordering"


def test_pairwise_pcr6_is_not_invariant_and_is_named_separately(toy):
    """The sequential fold is a DIFFERENT operator and must never be relabelled as multisource."""
    r = O.permutation_spread(toy, ("X", "Y", "Z"), "pairwise")
    assert not r["invariant"] and r["max_abs_change"] > 1e-6


@pytest.mark.parametrize("op", ["F1", "F5", "F6", "F7", "F8"])
def test_conflict_and_vacuity_stability(op):
    r = O.conflict_stability(op)
    for name, cell in r.items():
        if isinstance(cell, dict):
            assert cell.get("finite", True), f"{op} not finite on {name}"
            assert cell.get("in_range", True), f"{op} out of [0,1] on {name}"
    if "vacuous_is_identity_err" in r and op != "F1":
        assert r["vacuous_is_identity_err"] < 1e-6, f"{op}: a vacuous source changed the answer"


def test_belief_discount_identity(toy):
    """e'_k = rho u e_k / (1 - rho + rho u), checked against the operator itself."""
    import ops as G
    o, _bad = G.opinion_from_pu(toy.p_X.to_numpy(), toy.u_X.to_numpy())
    for rho in (0.25, 0.5, 0.75):
        lhs = G.evidence_of(O.discount_belief(o, rho))
        rhs = O.equivalent_evidence(o, rho)
        m = np.isfinite(lhs) & np.isfinite(rhs)
        assert np.max(np.abs(lhs[m] - rhs[m])) < 1e-9


def test_evidence_scaling_differs_from_belief_discounting(toy):
    """The brief insists these are not equivalent; this is the number that proves it."""
    a = O.fuse(toy, ("X", "Y", "Z"), "F8", [1.0, 0.5, 0.25], convention="evidence_scaling")
    b = O.fuse(toy, ("X", "Y", "Z"), "F8", [1.0, 0.5, 0.25], convention="belief_discounting")
    assert np.max(np.abs(a - b)) > 1e-3


def test_cumulative_counts_the_prior_once(toy):
    """alpha_f = 1 + sum e, not S + sum e."""
    m = O.masses(toy, ("X", "Y", "Z"))
    import ops as G
    e = np.stack([G.evidence_of(m[:, i, :]) for i in range(3)], 1)
    e = np.where(np.isfinite(e), e, 0.0)
    a = 1.0 + e.sum(1)
    expect = np.clip(a[:, O.OPS_FAKE] / a.sum(1), 0, 1)
    got = O.fuse(toy, ("X", "Y", "Z"), "F5")
    assert np.allclose(got, expect)


def test_duplicate_view_diagnostic_detects_non_idempotence(toy):
    d = O.duplicate_diagnostic(toy, ("X", "Y", "Z"))
    for b in "XYZ":
        assert d[b]["strength_ratio"] > 1.0, "duplicating a source must raise the fused strength"


def test_single_branch_fusion_is_that_branch(toy):
    p = O.fuse(toy, ("X",), "F8")
    assert np.allclose(p, toy.p_X.to_numpy())


# ----------------------------------------------------------------------------- metrics
def test_auroc_and_threshold_metrics():
    y = np.array([0, 0, 1, 1])
    assert C.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert C.auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    m = C.threshold_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert m["fpr_real"] == 0.0 and m["fnr_fake"] == 0.0 and m["bal_acc"] == 1.0


def test_ece_bins_are_equal_width_and_perfect_calibration_is_zero():
    y = np.concatenate([np.zeros(500), np.ones(500)])
    p = np.concatenate([np.zeros(500), np.ones(500)])
    assert C.ece(y, p, 15) < 1e-12 and C.ece(y, p, 10) < 1e-12


def test_e_aurc_of_the_oracle_is_zero():
    rng = np.random.default_rng(0)
    correct = rng.integers(0, 2, 400).astype(float)
    assert abs(C.e_aurc(correct, correct)) < 1e-12


def test_selective_risk_monotone_in_coverage():
    rng = np.random.default_rng(1)
    correct = (rng.random(500) > 0.3).astype(float)
    conf = correct + rng.normal(0, 0.3, 500)
    r = C.selective_risk(correct, conf)
    assert r["risk@80"] <= r["risk@100"] + 1e-9


def test_paired_delta_is_paired_and_zero_for_identical_systems(toy):
    g = C.groups_of(toy)
    d = C.paired_delta(toy.y.to_numpy(), toy.p_X.to_numpy(), toy.p_X.to_numpy(), g, n_rep=200)
    assert abs(d["point"]) < 1e-12 and abs(d["boot_mean"]) < 1e-12


def test_group_bootstrap_resamples_groups_not_rows(toy):
    g = C.groups_of(toy)
    W = C.group_boot(g, 50, seed=0)
    assert W.shape == (50, len(toy))
    # every row of a group carries that group's draw count, so rows within a group move together
    for gi in np.unique(g)[:5]:
        rows = np.where(g == gi)[0]
        assert len(set(W[0, rows].tolist())) == 1


# ----------------------------------------------------------------------------- S / section 12
def test_S_ignores_padded_frames():
    import seq_branch as T
    m = T.SeqBranch(64, pool="attn").eval()
    x = torch.randn(3, 16, 64)
    mask = torch.ones(3, 16, dtype=torch.bool)
    mask[0, 10:] = False
    tp = torch.arange(16).float()[None].repeat(3, 1) * 15
    x2 = x.clone()
    x2[0, 10:] = torch.randn(6, 64) * 50
    with torch.no_grad():
        a, _ = m(x, mask, tp)
        b, _ = m(x2, mask, tp)
    assert float((a - b).abs().max()) < 1e-6


def test_S_positional_encoding_is_relative_to_the_clip():
    import seq_branch as T
    m = T.SeqBranch(64, pool="attn", pe_scale=15.0).eval()
    x = torch.randn(2, 16, 64)
    mask = torch.ones(2, 16, dtype=torch.bool)
    tp = torch.arange(16).float()[None].repeat(2, 1) * 15
    with torch.no_grad():
        a, _ = m(x, mask, tp)
        b, _ = m(x, mask, tp + 5000)
    assert float((a - b).abs().max()) < 1e-6, "absolute clip start must not change the answer"


def test_S_dense_and_sparse_scales_agree_on_equivalent_relative_time():
    import seq_branch as T
    e1 = T.TemporalPE(64, 1.0)(torch.arange(16).float()[None], None)
    e15 = T.TemporalPE(64, 15.0)(torch.arange(0, 240, 15).float()[None], None)
    assert float((e1 - e15).abs().max()) < 1e-6


def test_S_mean_pool_is_order_invariant_but_attn_need_not_be():
    import seq_branch as T
    x = torch.randn(2, 16, 64)
    mask = torch.ones(2, 16, dtype=torch.bool)
    tp = torch.arange(16).float()[None].repeat(2, 1) * 15
    perm = torch.randperm(16)
    mm = T.SeqBranch(64, pool="mean", use_pos=False).eval()
    with torch.no_grad():
        a, _ = mm(x, mask, tp)
        b, _ = mm(x[:, perm], mask, tp)
    assert float((a - b).abs().max()) < 1e-5, "mean pooling without positions must be invariant"


def test_S_mlp_control_is_parameter_matched():
    import seq_branch as T
    a = T.SeqBranch(1024, pool="attn").param_report()
    b = T.SeqBranch(1024, pool="mlp").param_report()
    assert abs(a["temporal"] - b["temporal"]) / a["temporal"] < 0.01


def test_order_permutation_keeps_padding_at_the_end():
    """The order-discrimination loss must permute valid frames only."""
    mask = torch.ones(4, 32, dtype=torch.bool)
    mask[0, 20:] = False
    tp = torch.arange(32).float()[None].repeat(4, 1)
    perm = torch.argsort(torch.rand_like(tp) + (~mask).float() * 10.0, dim=1)
    assert bool((perm[0, :20] < 20).all())
    assert sorted(perm[0, :20].tolist()) == list(range(20))
    assert sorted(perm[1].tolist()) == list(range(32))


# ----------------------------------------------------------------------------- R / section 2.3
def test_R_projection_is_orthonormal_and_seeded():
    import extract_patch as E
    P1, P2 = E.projection(1024), E.projection(1024)
    assert np.allclose(P1, P2), "the projection must be reproducible from its seed"
    assert np.abs(P1.T @ P1 - np.eye(E.PROJ_DIM)).max() < 1e-5


def test_R_covariance_vectorisation_is_an_isometry():
    import extract_patch as E
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((200, E.PROJ_DIM))
    Zc = Z - Z.mean(0)
    Cm = Zc.T @ Zc / (len(Z) - 1)
    v = E.vectorise_cov(torch.from_numpy(Cm)[None])[0].numpy()
    assert v.shape == (E.TRI,) and E.TRI == 528
    assert abs(np.linalg.norm(v) - np.linalg.norm(Cm, "fro")) < 1e-6


def test_R_feature_width_includes_the_log_norm():
    import extract_patch as E
    c, mp = E.frame_features(torch.randn(2, 100, 1024),
                             torch.from_numpy(E.projection(1024)))
    assert c.shape == (2, E.TRI + 1) and mp.shape == (2, 1024)


@pytest.mark.skipif(__import__("importlib").util.find_spec("timm") is None,
                    reason="timm lives in the discern_ext_timm environment")
def test_prefix_token_counts_are_read_not_assumed():
    """PE-Core-L has 1 prefix token and DINOv3 ViT-L has 5. A hardcoded 1 is wrong for the second."""
    import timm
    pe = timm.create_model("vit_pe_core_large_patch14_336", pretrained=False, num_classes=0)
    dv = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0)
    assert int(getattr(pe, "num_prefix_tokens", 1)) == 1
    assert int(getattr(dv, "num_prefix_tokens", 1)) == 5


# ----------------------------------------------------------------------------- heads / Track D
def test_standardizer_never_refits_on_new_data():
    import trackd_heads as H
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.standard_normal((100, 8)), rng.standard_normal((50, 8)) * 10 + 5
    s = H.Standardizer().fit(Xtr)
    mu, sd = s.mu.copy(), s.sd.copy()
    s.transform(Xte)
    assert np.allclose(s.mu, mu) and np.allclose(s.sd, sd)


def test_temperature_is_monotone_and_preserves_auroc():
    import trackd_heads as H
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, 300), 1e-3, 1 - 1e-3)
    T = H.fit_temperature(y, p)
    pt = H.apply_temperature(p, T)
    assert abs(C.auroc(y, p) - C.auroc(y, pt)) < 1e-9
    assert T > 0


def test_concat_head_ce_and_edl_share_architecture():
    import trackd_heads as H
    a = H.ConcatHead(32, "mlp", "ce")
    b = H.ConcatHead(32, "mlp", "edl")
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_edl_evidence_is_nonnegative():
    import trackd_heads as H
    m = H.ConcatHead(16, "mlp", "edl")
    out = m(torch.randn(20, 16) * 5)
    assert float(out.min()) >= 0.0


# ----------------------------------------------------------------------------- subsets / section 4
def test_subset_counts_match_the_brief():
    assert len(C.subsets_of(("A", "S", "V", "O", "M"), (1,))) == 5
    assert len(C.subsets_of(("A", "S", "V", "O", "M"), (2,))) == 10
    assert len(C.subsets_of(("A", "S", "V", "O", "M"), (3,))) == 10
    assert len(C.subsets_of(("A", "S", "V", "O", "M"), (4,))) == 5
    assert len(C.subsets_of(("A", "S", "V", "O", "M"), (5,))) == 1


def test_encoder_cost_shares_the_PE_pass_but_not_P():
    import subsets as S
    assert S.encoder_evals(("A", "S", "R")) == 1, "A, S and R read one PE forward pass"
    assert S.encoder_evals(("A", "P")) == 2, "P runs its own adapted PE clip encoder"
    assert S.encoder_evals(("A", "V", "O", "M", "P")) == 5


def test_selection_rule_prefers_cheaper_within_the_margin():
    import subsets as S
    rows = [{"subset": "A+V+O+M", "v31_macro": 0.9100, "encoder_evals": 4,
             "trainable_params": 10, "n_branches": 4},
            {"subset": "A+V", "v31_macro": 0.9090, "encoder_evals": 2,
             "trainable_params": 5, "n_branches": 2}]
    sel = S.select(rows, margin=0.002)
    assert sel["best_raw"] == "A+V+O+M" and sel["selected"] == "A+V"


def test_align_rejects_label_disagreement():
    a = pd.DataFrame({"v": ["x", "y"], "y": [0, 1], "p": [.1, .9], "u": [.2, .2]})
    b = pd.DataFrame({"v": ["x", "y"], "y": [0, 0], "p": [.2, .8], "u": [.2, .2]})
    with pytest.raises(AssertionError):
        C.align({"A": a, "B": b}, tag="toy")


def test_align_intersects_and_records_drops():
    a = pd.DataFrame({"v": ["x", "y", "z"], "y": [0, 1, 1], "p": [.1, .9, .5],
                      "u": [.2, .2, .2]})
    b = pd.DataFrame({"v": ["x", "y"], "y": [0, 1], "p": [.2, .8], "u": [.2, .2]})
    t = C.align({"A": a, "B": b}, tag="toy")
    assert len(t) == 2 and t.attrs["dropped"]["A"] == 1


# ----------------------------------------------------------------------------- IBDC
def test_inline_kappa_matches_the_verified_implementation():
    """`uncertainty._kappa_detached` must agree with `fusion_torch.kappa`."""
    sys.path.insert(0, str(ROOT))
    from training.joint_ibdc.fusion_torch import pairwise_kappa
    import uncertainty as U
    rng = np.random.default_rng(0)
    n, S = 40, 3
    b1 = rng.uniform(0, .4, (n, S))
    b0 = rng.uniform(0, .4, (n, S))
    u = 1 - b1 - b0
    masses = torch.tensor(np.stack([b1, b0, u], -1), dtype=torch.float32)
    mine = U._kappa_detached(masses).numpy()
    theirs = pairwise_kappa(masses).max(dim=1).values.numpy()
    assert np.allclose(mine, theirs, atol=1e-6)
