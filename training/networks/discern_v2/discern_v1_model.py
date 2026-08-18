"""DISCERN v2 — the V1 three-branch model (spec §2, §4.2, §9, §14, §16, §17).

    aligned face crop
        |-- CLIP normalization ----> CLIP-L/14 + DiCoME LoRA ------------> e_sem   (anchor)
        |-- FS-VFM normalization --> frozen FS-VFM -+- P_R -> r_ref ------> e_ref   (Specialist 1)
        |                                           `- same-capacity head -> e_direct (control)
        `-- VAE-native ------------> frozen SDXL VAE cycle --------------> e_proc  (Specialist 2)
                                          |
                       q_ref, q_proc (utility-supervised; q_sem = 1)
                                          |
                       Shafer discounting -> DS combination -> V/C/A -> Real/Fake/Defer

Exactly three opinions reach fusion. `e_direct` is computed, logged and compared, and never
fused — §4.2 makes it the control that answers "does the P_R reference transformation add
anything over the raw pretrained representation, with head capacity held constant?".

Staged, because the stages are not interchangeable (§9)
------------------------------------------------------
The model carries a `stage` so that what is *supposed* to be frozen at each point is enforced
rather than remembered:

    B  per-branch heads train (and CLIP's LoRA). Gates are not trained and do not gate:
       every specialist enters at q = 1, so each branch's auxiliary EDL loss sees the same
       opinion it would contribute later.
    D  the applicability gates train on VAL_meta; branch heads are frozen.
    E+ everything is frozen; the risk model is fit outside this module on out-of-fold q.

The gate cannot be trained in Stage B because its target is defined from the *frozen selected
checkpoint's* pairwise DS utility (§13) — a gate trained against a moving expert is learning to
predict a quantity that no longer exists by the time it is used.

Availability is ignorance (§14.1). A specialist without a valid input contributes a vacuous
opinion, and its auxiliary loss is masked on that sample; it is never dropped, zero-filled, or
left for the head to interpret.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .branches import BranchOutput
from .dirichlet import to_dirichlet
from .ds_fusion import Opinion, apply_validity, discount, ds_combine, reliability
from .process_branch import ProcessEvidenceBranch
from .reference_branch import ReferenceEvidenceBranch
from .semantic_branch import SemanticEvidenceBranch

logger = logging.getLogger(__name__)

STAGES = ("B", "D", "E")
SPECIALISTS = ("ref", "proc")
# The order DS folds the opinions in. Fixed and recorded rather than incidental: §15 requires
# the order to be documented and its effect measured (`assert_order_invariant`).
FUSION_ORDER = ("sem", "ref", "proc")


class DiscernV1Model(nn.Module):
    """The V1 model. Branch construction is injected so each piece stays independently testable."""

    def __init__(self, semantic: SemanticEvidenceBranch,
                 reference: ReferenceEvidenceBranch | None = None,
                 process: ProcessEvidenceBranch | None = None,
                 fsvfm=None, direct_probe: nn.Module | None = None,
                 num_classes: int = 2, stage: str = "B"):
        super().__init__()
        self.num_classes = num_classes
        self.semantic = semantic
        self.reference = reference
        self.process = process
        self.fsvfm = fsvfm                       # frozen FS-VFM encoder, shared by ref + direct
        self.direct_probe = direct_probe         # §4.2 control head on z_ref; never fused
        self.set_stage(stage)

        if reference is not None and fsvfm is None:
            raise ValueError(
                "the reference branch needs the frozen FS-VFM encoder to produce z_ref; passing "
                "it separately is what lets the §4.2 direct probe share the same features "
                "instead of running the encoder twice")

    # ------------------------------------------------------------------ staging

    def set_stage(self, stage: str) -> "DiscernV1Model":
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        self.stage = stage
        if stage != "B":
            # §19: after checkpoint selection the experts are frozen. Enforced here so a later
            # stage cannot keep training them through a forgotten optimizer group.
            for module in (self.semantic, self.reference, self.process, self.direct_probe):
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad_(False)
        return self

    def trainable_parameters(self) -> dict:
        return {name: sum(p.numel() for p in m.parameters() if p.requires_grad)
                for name, m in (("semantic", self.semantic), ("reference", self.reference),
                                ("process", self.process), ("direct_probe", self.direct_probe))
                if m is not None}

    def assert_frozen_protocol(self) -> None:
        """§19's mechanism guards, run from the real build path rather than a dead check."""
        if self.fsvfm is not None:
            self.fsvfm.assert_frozen()
        if self.reference is not None:
            self.reference.assert_frozen()
        if self.process is not None:
            self.process.assert_frozen()
        if self.stage == "B":
            self.semantic.assert_lora_only()

    # ------------------------------------------------------------------ branches

    def _z_ref(self, batch: dict) -> torch.Tensor:
        """Frozen FS-VFM features, computed ONCE and shared by e_ref and e_direct.

        Sharing is what makes §4.2 a controlled comparison: both paths see the identical
        representation, so any difference is the P_R transformation and not encoder noise.
        """
        images = batch["fsvfm_frames"]
        with torch.no_grad():
            return self.fsvfm(images)

    def branch_outputs(self, batch: dict) -> dict:
        """Per-branch evidence, features, validity — before any gating or fusion."""
        out: dict[str, dict] = {}

        sem = self.semantic(batch["spatial_frames"])
        out["sem"] = {"evidence": sem["evidence"], "feature": sem["feature"],
                      "valid": torch.ones(sem["evidence"].shape[0], dtype=torch.bool,
                                          device=sem["evidence"].device),
                      "diagnostics": {}}

        if self.reference is not None:
            z = self._z_ref(batch)
            ref: BranchOutput = self.reference(z)
            valid = batch.get("branch_valid_ref")
            out["ref"] = {
                "evidence": ref.evidence, "feature": ref.features,
                "valid": (valid.bool() if valid is not None
                          else torch.ones(z.shape[0], dtype=torch.bool, device=z.device)),
                "diagnostics": ref.diagnostics,
            }
            if self.direct_probe is not None:
                # §4.2: logged, compared, NEVER fused. Kept out of `branch_outputs`' fusion set
                # by living in its own key rather than by a flag someone could flip.
                out["direct"] = {"evidence": F.softplus(self.direct_probe(z)),
                                 "feature": z, "valid": out["ref"]["valid"],
                                 "diagnostics": {"control": True}}

        if self.process is not None:
            proc = self.process(batch["process_frames"])
            valid = proc["valid"]
            if batch.get("branch_valid_proc") is not None:
                valid = valid & batch["branch_valid_proc"].bool()
            out["proc"] = {"evidence": proc["evidence"], "feature": proc["feature"],
                           "valid": valid, "diagnostics": {"raw_stats": proc["raw_stats"]}}
        return out

    # ------------------------------------------------------------------ fusion

    def fuse(self, branches: dict, q: dict[str, torch.Tensor] | None = None) -> dict:
        """Opinions -> validity -> discounting -> DS -> back to EDL -> V/C/A.

        `q` is optional: in Stage B there is no gate yet and every specialist enters at q = 1,
        which is deliberate. It keeps the auxiliary per-branch losses consistent with the
        opinion each branch will actually contribute, and it means "the gate helped" is later
        measured against a fusion that differs ONLY by the gate.
        """
        names = [n for n in FUSION_ORDER if n in branches]
        opinions, weights = {}, {}
        for name in names:
            b = branches[name]
            op = Opinion.from_evidence(b["evidence"])
            op = apply_validity(op, b["valid"].float())     # §14.1 before anything else
            opinions[name] = op

            if name == "sem":
                weights[name] = torch.ones(op.batch_size, device=op.belief.device)
            else:
                qb = (q or {}).get(name)
                qb = (torch.ones(op.batch_size, device=op.belief.device) if qb is None
                      else qb.squeeze(-1) if qb.dim() > 1 else qb)
                # an invalid branch is ignorance regardless of what the gate predicted
                weights[name] = qb * b["valid"].float()

        discounted = {n: (opinions[n] if n == "sem" else discount(opinions[n], weights[n]))
                      for n in names}
        fused, diagnostics = ds_combine([discounted[n] for n in names])
        state = fused.to_dirichlet()                        # §16
        rel = reliability(opinions, weights, fused, specialists=SPECIALISTS)

        return {
            "fused": fused,
            "evidence": state.evidence,
            "alpha": state.alpha,
            "p": state.p,
            "prob": fused.fake_prob(),
            "V": rel["V"], "C": rel["C"], "A": rel["A"],
            "weights": rel["weights"],
            "q": {n: weights[n] for n in names},
            "opinions": opinions,
            "discounted": discounted,
            "ds_conflict": diagnostics["ds_conflict_max"],
            "ds_degenerate": diagnostics["degenerate"],
            "order": names,
        }

    def forward(self, batch: dict, q: dict[str, torch.Tensor] | None = None) -> dict:
        branches = self.branch_outputs(batch)
        fusion = self.fuse({k: v for k, v in branches.items() if k != "direct"}, q)
        return {"branches": branches, **fusion}

    # ------------------------------------------------------------------ Stage B losses

    def auxiliary_losses(self, branches: dict, labels: torch.Tensor,
                         edl_loss_fn) -> dict[str, torch.Tensor]:
        """Per-branch EDL losses so each branch is individually usable (§9 Stage B).

        Masked by `branch_valid_b`: supervising a branch on a sample it could not process would
        teach it to produce a confident opinion from a placeholder input, which is precisely the
        behaviour §14.1 removes at inference. The control head is supervised too — it has to be
        trained to be a fair comparison — but its loss is reported separately so it can never be
        mistaken for part of the fused objective.
        """
        losses = {}
        for name, b in branches.items():
            valid = b["valid"]
            if not valid.any():
                losses[name] = torch.zeros((), device=labels.device)
                continue
            per_sample = edl_loss_fn(b["evidence"][valid], labels[valid])
            losses[name] = per_sample.mean() if per_sample.dim() > 0 else per_sample
        return losses


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def build_direct_probe(feature_dim: int = 1024, hidden_dim: int = 64,
                       num_classes: int = 2) -> nn.Module:
    """§4.2's same-capacity control head on raw frozen FS-VFM features.

    Same architecture and hidden width as the reference branch's evidence head, so the
    comparison isolates the P_R transformation rather than head capacity. The input widths
    differ by two (the reference head also reads the residual's magnitude and angle), which is
    the smallest honest way to give each path exactly the descriptors it has.
    """
    return nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, num_classes))


def build_v1_model(cfg: dict, stage: str = "B") -> DiscernV1Model:
    """Build from a config block. Every path is explicit — nothing is silently skipped.

    A missing reference artifact or VAE path is an ERROR, not a quietly-dropped branch: a V1 run
    with two branches that reports itself as three is the kind of failure that survives all the
    way into a results table.
    """
    from .fsvfm_encoder import FrozenFSVFM

    semantic = SemanticEvidenceBranch(
        backbone=cfg.get("backbone", "openai/clip-vit-large-patch14"),
        feature_dim=int(cfg.get("semantic_feature_dim", 64)),
        enable_lora=bool(cfg.get("enable_lora", True)))

    reference = fsvfm = direct = None
    ref_cfg = cfg.get("reference", {})
    if ref_cfg.get("enabled", True):
        artifact = ref_cfg.get("artifact_path")
        if not artifact:
            raise ValueError(
                "reference.artifact_path is required: Stage A fits P_R offline and Stage B loads "
                "it read-only. There is deliberately no path that constructs an unfitted "
                "reference here (spec §9 A, §19).")
        fsvfm = FrozenFSVFM(
            checkpoint=ref_cfg.get("fsvfm_checkpoint") or FrozenFSVFM.__init__.__defaults__[0],
            pooling=ref_cfg.get("pooling", "global_pool"))
        reference = ReferenceEvidenceBranch(
            artifact_path=artifact,
            hidden_dim=int(ref_cfg.get("hidden_dim", 64)),
            input_dim=int(ref_cfg.get("input_dim", 1024)))
        if ref_cfg.get("direct_probe", True):
            direct = build_direct_probe(
                feature_dim=int(ref_cfg.get("input_dim", 1024)),
                hidden_dim=int(ref_cfg.get("hidden_dim", 64)))

    process = None
    proc_cfg = cfg.get("process", {})
    if proc_cfg.get("enabled", True):
        vae_path = proc_cfg.get("vae_path")
        if not vae_path:
            raise ValueError("process.vae_path is required (a local stabilityai/sdxl-vae)")
        process = ProcessEvidenceBranch(
            vae_path=vae_path,
            resolution=int(proc_cfg.get("resolution", 256)),
            stats_artifact=proc_cfg.get("stats_artifact"))
        if proc_cfg.get("stats_artifact") is None:
            logger.warning(
                "process.stats_artifact is unset: the branch's standardisation is UNFITTED and "
                "will refuse at forward time. Run analysis/discern_v2/fit_process_stats.py "
                "before Stage B.")

    model = DiscernV1Model(semantic=semantic, reference=reference, process=process,
                           fsvfm=fsvfm, direct_probe=direct, stage=stage)
    logger.info(f"  DISCERN V1 [{stage}] trainable: {model.trainable_parameters()}")
    return model


def default_artifact_paths() -> dict:
    """Where the offline stages write what Stage B reads. One place, so configs agree."""
    root = Path(__file__).resolve().parents[3]
    return {
        "fsvfm_checkpoint": root / "weights" / "FS-VFM" / "checkpoint-599.pth",
        "reference_artifact": root / "configs" / "discern_v2" / "reference"
        / "reference_C3_ae_cosine.pt",
        "process_stats": root / "configs" / "discern_v2" / "process" / "process_stats.pt",
    }
