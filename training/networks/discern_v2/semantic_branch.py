"""Branch A — semantic evidence `e_sem`: CLIP ViT-L/14 with the ported DiCoME LoRA.

V1 spec §3. The anchor is the always-on generalist: `q_sem = 1`, never gated, never discounted.
The point of porting rather than re-deriving is that `e_sem` should behave like the DiCoME
semantic module we already reproduced, so any DISCERN-v2 shortfall is attributable to our
architecture rather than to a recipe difference we introduced by accident.

Traced from the local reproduced clone, not from the spec's prose
----------------------------------------------------------------
`/data/umar/Repos/DiCoME`, confirmed by reading the code (§3 says confirm, do not assume):

    encoder      src/encoders/clip_encoder.py:12  LoRA_CLIPEncoder
    LoRA         src/config.py:38-45   target_modules ["q_proj", "v_proj"], r 8, alpha 16,
                                       dropout 0.1, bias "none", task FEATURE_EXTRACTION
    backbone     src/config.py:61      openai/clip-vit-large-patch14
    projection   src/encoders/clip_encoder.py:52  Linear(1024 -> feature_dim)
    feature_dim  src/config.py:62      64
    head         src/modules/evidential_head.py   Linear(64) -> ReLU -> Linear(K) -> softplus
    pre-head LN  src/model/core_model.py:47       LayerNorm(feature_dim)

Every one matches the form §3 said to expect, so nothing here rests on memory.

The 64-D bottleneck is kept
---------------------------
§3 allows reusing it if code inspection shows it is recipe rather than a consequence of
Geometric View Purification. It is: the projection lives in the *encoder*, is applied to CLIP's
pooler output before anything view-specific exists, and feeds the semantic head directly. GVP
consumes `f_s` but did not create it. Kept, and listed in §24's iterate order as "LoRA with and
without the 64-D bottleneck" — a question for after V1 has numbers, not a silent redesign now.

What is deliberately NOT ported
-------------------------------
The beta-VAE semantic manifold and the orthogonal-projection artifact view (§25). V1's second
view is the frozen FS-VFM reference, whose whole argument is that it is fit on reals only and
frozen; DiCoME's co-trained in-manifold view would make the residual circular and would put both
views in one representation, which is the shared-ignorance failure our V/C/A exists to expose.

Output contract: non-negative `e_sem` of shape (B, K). The branch does no Dirichlet arithmetic of
its own — `dirichlet.to_dirichlet` and `ds_fusion.Opinion` own that (§7).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dirichlet import DirichletState, to_dirichlet

logger = logging.getLogger(__name__)

# Ported verbatim from DiCoME's config defaults (src/config.py:38-45). Held as a literal so a
# drift in that clone cannot silently change what "the ported recipe" means here; the build
# report records the trace.
DICOME_LORA = {
    "target_modules": ["q_proj", "v_proj"],
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "FEATURE_EXTRACTION",
    "inference_mode": False,
}
DICOME_BACKBONE = "openai/clip-vit-large-patch14"
DICOME_FEATURE_DIM = 64
DICOME_HEAD_HIDDEN = 64


class EvidentialHead(nn.Module):
    """Ported from DiCoME `src/modules/evidential_head.py`.

    softplus, so evidence is non-negative by construction rather than by clamping downstream.
    """

    def __init__(self, feature_dim: int = DICOME_FEATURE_DIM, num_classes: int = 2,
                 hidden: int = DICOME_HEAD_HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.fc2(F.relu(self.fc1(x))))


class SemanticEvidenceBranch(nn.Module):
    """CLIP-LoRA -> 64-D projection -> LayerNorm -> evidential head -> `e_sem` (B, K)."""

    name = "sem"

    def __init__(self, backbone: str = DICOME_BACKBONE,
                 feature_dim: int = DICOME_FEATURE_DIM, num_classes: int = 2,
                 lora: dict | None = None, enable_lora: bool = True,
                 local_files_only: bool = True):
        super().__init__()
        from transformers import CLIPModel

        self.backbone_name = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.lora_config_used = dict(lora or DICOME_LORA) if enable_lora else None

        clip_vision = CLIPModel.from_pretrained(
            backbone, local_files_only=local_files_only).vision_model
        self.vfm_output_dim = clip_vision.config.hidden_size

        if enable_lora:
            from peft import LoraConfig, get_peft_model
            self.vision_model = get_peft_model(clip_vision, LoraConfig(**self.lora_config_used))
        else:
            # the §24 control (LoRA vs LN vs frozen); NOT a V1 configuration
            self.vision_model = clip_vision

        # Ported: Identity when the widths already match, else a learned projection.
        self.projection = (nn.Identity() if self.vfm_output_dim == feature_dim
                           else nn.Linear(self.vfm_output_dim, feature_dim))
        self.norm = nn.LayerNorm(feature_dim)
        self.head = EvidentialHead(feature_dim, num_classes)

        self.freeze_base()

    # ------------------------------------------------------------------ freezing

    def freeze_base(self) -> "SemanticEvidenceBranch":
        """§19: every CLIP parameter frozen EXCEPT the LoRA adapters.

        `get_peft_model` already does this, but it is asserted here rather than trusted: a
        base-CLIP parameter left trainable would be a full fine-tune wearing the label "LoRA",
        which changes what the anchor is and would not show up as an error anywhere.
        """
        if self.lora_config_used is None:
            for p in self.vision_model.parameters():
                p.requires_grad_(False)
            return self
        for name, p in self.vision_model.named_parameters():
            if "lora_" not in name:
                p.requires_grad_(False)
        return self

    def assert_lora_only(self) -> dict:
        """The trainable set is exactly LoRA + projection + norm + head. Returns the counts."""
        trainable_backbone = [n for n, p in self.vision_model.named_parameters()
                              if p.requires_grad]
        if self.lora_config_used is not None:
            leaked = [n for n in trainable_backbone if "lora_" not in n]
            if leaked:
                raise RuntimeError(
                    f"{len(leaked)} non-LoRA CLIP parameters are trainable (e.g. {leaked[:3]}). "
                    f"That is a full fine-tune labelled as LoRA — the anchor would no longer be "
                    f"the ported DiCoME semantic module.")
            if not trainable_backbone:
                raise RuntimeError(
                    "no LoRA parameter is trainable; the adapters were frozen along with the "
                    "base model, so the anchor cannot learn at all")
        return {
            "lora": sum(p.numel() for n, p in self.vision_model.named_parameters()
                        if p.requires_grad),
            "frozen_backbone": sum(p.numel() for n, p in self.vision_model.named_parameters()
                                   if not p.requires_grad),
            "projection": sum(p.numel() for p in self.projection.parameters()),
            "norm": sum(p.numel() for p in self.norm.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
        }

    # ------------------------------------------------------------------ forward

    def features(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(projected 64-D feature, raw CLIP pooled feature).

        Ported call convention: with a PEFT wrapper the underlying vision model is reached via
        `.base_model`; the adapters are injected into the modules themselves, so they still
        apply.
        """
        if hasattr(self.vision_model, "base_model"):
            outputs = self.vision_model.base_model(pixel_values=pixel_values)
        else:
            outputs = self.vision_model(pixel_values=pixel_values)
        raw = outputs.pooler_output
        return self.projection(raw), raw

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """`pixel_values` must be normalised with CLIP's statistics (§6)."""
        projected, raw = self.features(pixel_values)
        feature = self.norm(projected)
        evidence = self.head(feature)
        return {
            "evidence": evidence,          # (B, K), non-negative
            "feature": feature,            # (B, 64) — the gate may read this summary
            "raw_feature": raw,            # (B, 1024) CLIP pooled, for diagnostics
        }

    def dirichlet(self, pixel_values: torch.Tensor) -> DirichletState:
        return to_dirichlet(self.forward(pixel_values)["evidence"])


# ---------------------------------------------------------------------------
# recipe helpers (§10) — ported, so Stage B does not re-derive them
# ---------------------------------------------------------------------------


def dicome_param_groups(model: nn.Module, weight_decay: float = 0.01,
                        warn_unmatched_norms: bool = True) -> list[dict]:
    """AdamW parameter groups exactly as DiCoME builds them.

    Ported from `src/model/dicome_module.py:503-517`. §10 says to exclude bias/norm from weight
    decay *only if the reproduced code really does* — it does, matching on "bias", "norm" or
    "bn" in the parameter NAME, so that rule is inherited rather than assumed.

    The name rule has a failure mode worth knowing about, so it is reported rather than fixed:
    it catches a normalisation layer only when the layer's attribute name contains "norm". A
    LayerNorm held in an `nn.Sequential` is named `3.weight`, matches nothing, and quietly
    receives weight decay on its gain. DiCoME's own modules are named `semantic_norm` /
    `artifact_norm` so the rule works there, and ours uses `self.norm` for the same reason.
    Silently switching to a type-based rule would change the reproduced recipe — the exact thing
    §10 warns against — so the grouping stays faithful and any missed norm parameter is logged.
    """
    norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm,
                  nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    norm_param_ids = {id(p) for m in model.modules() if isinstance(m, norm_types)
                      for p in m.parameters(recurse=False)}

    decay, no_decay, unmatched = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "bn" in name:
            no_decay.append(param)
        else:
            decay.append(param)
            if id(param) in norm_param_ids:
                unmatched.append(name)

    if unmatched and warn_unmatched_norms:
        logger.warning(
            "%d normalisation parameter(s) will receive weight decay because DiCoME's name-based "
            "rule does not match their names (%s). This reproduces the ported recipe; rename the "
            "module to contain 'norm' if that is not intended.", len(unmatched), unmatched[:5])

    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


# Ported verbatim from DiCoME `src/dataset/base.py:67-78`, including the ORDER. Applied to every
# training image — blur and jitter are unconditional there, not probability-gated, which is the
# recipe delta §10 flags against our 10%-of-batches version.
DICOME_AUGMENTATION = {
    "random_horizontal_flip": {"p": 0.5},
    "random_affine": {"degrees": 10, "translate": (0.1, 0.1), "scale": (0.9, 1.1)},
    "gaussian_blur": {"kernel_size": (3, 7), "sigma": (0.1, 2.0)},
    "color_jitter": {"brightness": 0.2, "contrast": 0.2},
    "order": ["random_horizontal_flip", "random_affine", "gaussian_blur", "color_jitter"],
    "source": "DiCoME src/dataset/base.py:67-78",
}


def dicome_train_transform():
    """The ported training augmentation pipeline, in DiCoME's order.

    Returned as a torchvision Compose so Stage B can use the reproduced recipe directly instead
    of approximating it from the numbers above.
    """
    import torchvision.transforms as T

    a = DICOME_AUGMENTATION
    return T.Compose([
        T.RandomHorizontalFlip(p=a["random_horizontal_flip"]["p"]),
        T.RandomAffine(degrees=a["random_affine"]["degrees"],
                       translate=a["random_affine"]["translate"],
                       scale=a["random_affine"]["scale"]),
        T.GaussianBlur(kernel_size=a["gaussian_blur"]["kernel_size"],
                       sigma=a["gaussian_blur"]["sigma"]),
        T.ColorJitter(brightness=a["color_jitter"]["brightness"],
                      contrast=a["color_jitter"]["contrast"]),
    ])


DICOME_OPTIMIZER = {
    "optimizer": "AdamW",
    "lr": 1e-4,
    "betas": (0.9, 0.999),
    "weight_decay": 0.01,
    "exclude_from_weight_decay": ["bias", "norm", "bn"],
    "scheduler": "CosineAnnealingLR",
    "scheduler_interval": "step",
    "min_lr": 1e-6,
    "source": "DiCoME src/config.py:70-77, src/model/dicome_module.py:503-538",
}
