"""Teacher-student FS-VFM with a depth-resolved adaptation profile (brief Stage 1).

    frozen teacher  E_T  ---.
                             >-- d_l^TS(x) = 1 - cos(h_T^(l), h_S^(l))  ->  D(x)
    LoRA student    E_S  ---'                                               patch map

One model, not two
------------------
`E_T` and `E_S` are the SAME module: a single FS-VFM ViT-L with LoRA adapters, read twice — once
with the adapters disabled (the teacher, the frozen prior) and once with them active (the
student). This is not a memory optimisation, it is what makes the design's premise true by
construction rather than by bookkeeping:

* the two encoders provably share base weights and initialization — no pair of checkpoints can
  drift apart, and no loading bug can leave the "teacher" holding different weights;
* at initialization `h_T == h_S` **bit-identically** (verified: max |diff| = 0.0, because LoRA's
  B matrix is zero-initialised), which is exactly the flat region the brief's two-stage training
  exists to avoid;
* the teacher pass is deterministic. Every dropout in this ViT is p = 0.0 and `drop_path` is
  `Identity`, so two teacher passes on one input agree exactly, and LoRA's own dropout cannot
  reach the teacher because its adapters are off.

Readout: mean-pooled patches, not CLS
-------------------------------------
`h^(l)` is the **mean of the patch tokens** at layer `l`. The brief's notation says CLS, but this
checkpoint is loaded with `global_pool=True`, and that constructor deletes `self.norm` and adds
`fc_norm` (`fsvfm/models_vit.py:37-42`) — the representation FS-VFM was pretrained and evaluated
through is the patch mean. CLS is carried along but was never the trained readout, so a cosine
distance on it would measure a token the model had little reason to organise.

CLS is still computed, as `D_cls`, and reported beside the primary. It costs one extra cosine.

Cosine distance is scale-invariant, so the absence of a per-layer `norm` (deleted by
`global_pool=True`) does not bias the profile across depth.

Consistency requirement, enforced here
--------------------------------------
`L_preserve` must be computed on the SAME representation the delta is read from, so that the
quantity being preserved is the quantity being measured. `forward()` therefore returns the
per-layer deltas ONCE, as `D`, and the trainer masks them by label rather than recomputing a
second distance with its own pooling choice. There is deliberately no separate `preserve` output
that could diverge from `D`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..discern_v2.fsvfm_encoder import DEFAULT_CHECKPOINT, load_normalization

logger = logging.getLogger(__name__)

# ViT-L/24. The brief writes the layer set 1-indexed as {4, 8, 12, 16, 20, 24}; these are the
# ZERO-BASED block indices that correspond, and the convention is recorded in the artifact so an
# off-by-one cannot silently shift the whole depth profile.
DEFAULT_LAYERS = (3, 7, 11, 15, 19, 23)
LAYERS_ONE_INDEXED = tuple(i + 1 for i in DEFAULT_LAYERS)

# LoRA on the attention and MLP projections. `q_proj`/`v_proj` from the DiCoME recipe are
# HuggingFace CLIP names and do not exist here; this ViT has a FUSED `attn.qkv`, so adapting
# "attention" necessarily adapts q, k and v together. peft cannot address slices of a fused
# Linear. Recorded as the stated deviation from DiCoME's q,v-only recipe; rank and alpha carry
# over unchanged.
FPAD_LORA = {
    "target_modules": ["qkv", "proj", "fc1", "fc2"],
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bias": "none",
}


def _cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - cos, over the last dimension, clamped at 0.

    Cosine distance is mathematically in [0, 2], but `1 - cos(v, v)` in float32 lands within
    ~1e-7 either side of zero because the dot product and the norms accumulate separately. Left
    unclamped, `L_preserve` reports a small NEGATIVE loss at initialization — a number that
    cannot mean anything and that a reader would reasonably stop to investigate. Clamping costs
    nothing in gradient terms: at `D = 0` the cosine gradient is already zero, which is precisely
    the flat region the two-stage training exists to route around.
    """
    return (1.0 - F.cosine_similarity(a, b, dim=-1, eps=eps)).clamp_min(0.0)


class _BlockCapture:
    """Capture selected block outputs for one forward pass.

    Hooks are registered and removed around each pass rather than kept alive, so a teacher pass
    and a student pass can never write into each other's buffers — the failure that would make
    `d_l^TS` a comparison of one pass with itself, and would look like a perfectly plausible
    all-zero trajectory.
    """

    def __init__(self, blocks: nn.ModuleList, layers: tuple[int, ...]):
        self.blocks = blocks
        self.layers = layers
        self.out: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self) -> "_BlockCapture":
        self.out = {}
        for i in self.layers:
            self._handles.append(
                self.blocks[i].register_forward_hook(self._make_hook(i)))
        return self

    def _make_hook(self, index: int):
        def hook(_module, _inputs, output):
            self.out[index] = output
        return hook

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def stacked(self) -> torch.Tensor:
        """(B, L, 1 + N, D) in the configured layer order."""
        missing = [i for i in self.layers if i not in self.out]
        if missing:
            raise RuntimeError(
                f"blocks {missing} produced no output — the hooks did not fire. Either the layer "
                f"indices are out of range or forward() did not run the block loop.")
        return torch.stack([self.out[i] for i in self.layers], dim=1)


class FPADTeacherStudent(nn.Module):
    """One FS-VFM ViT-L, read twice: adapters off (teacher) and on (student)."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT,
                 layers: tuple[int, ...] = DEFAULT_LAYERS,
                 lora: dict | None = None, img_size: int = 224,
                 pooling: str = "global_pool"):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        from ..discern_v2.fsvfm_encoder import FrozenFSVFM

        # Reuse the validated loader: it reads the checkpoint, verifies provenance, applies the
        # FS-VFM normalization statistics from file, and FAILS on missing encoder tensors rather
        # than warning. Nothing about that changes because we are about to adapt it.
        base = FrozenFSVFM(checkpoint=checkpoint, pooling=pooling)
        self.provenance = base.provenance()
        self.mean, self.std = base.mean, base.std
        self.img_size = img_size
        vit = base.model
        if img_size != 224:
            self._resize_position_embedding(vit, img_size)

        self.layers = tuple(int(i) for i in layers)
        depth = len(vit.blocks)
        bad = [i for i in self.layers if not 0 <= i < depth]
        if bad:
            raise ValueError(f"layer indices {bad} are outside [0, {depth}) for this backbone")
        self.embed_dim = int(vit.blocks[0].norm1.normalized_shape[0])

        cfg = LoraConfig(task_type=None, **(lora or FPAD_LORA))
        self.peft = get_peft_model(vit, cfg)
        self.lora_config = dict(lora or FPAD_LORA)

        # Everything that is not a LoRA parameter is the frozen prior, including the classifier
        # head timm builds and we never use.
        for name, param in self.peft.named_parameters():
            param.requires_grad_("lora_" in name)
        self.assert_teacher_frozen()

    # ------------------------------------------------------------------ plumbing

    @staticmethod
    def _resize_position_embedding(vit: nn.Module, img_size: int) -> None:
        """Bicubic-interpolate `pos_embed` for a different input size.

        A larger input gives a finer patch grid (256 -> 16x16 instead of 14x14), which the brief
        notes would help localization. It is OFF by default: interpolating the position embedding
        moves the model away from the resolution it was pretrained at, and validating that it
        costs nothing is a separate experiment rather than a free win.
        """
        patch = vit.patch_embed.patch_size[0]
        new_grid = img_size // patch
        cls_pos, grid_pos = vit.pos_embed[:, :1], vit.pos_embed[:, 1:]
        old_grid = int(grid_pos.shape[1] ** 0.5)
        if old_grid == new_grid:
            return
        dim = grid_pos.shape[-1]
        grid = grid_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(new_grid, new_grid), mode="bicubic",
                             align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
        vit.pos_embed = nn.Parameter(torch.cat([cls_pos, grid], dim=1))
        vit.patch_embed.img_size = (img_size, img_size)
        vit.patch_embed.grid_size = (new_grid, new_grid)
        vit.patch_embed.num_patches = new_grid * new_grid
        logger.warning("interpolated pos_embed %dx%d -> %dx%d for img_size=%d; the model is now "
                       "off its pretraining resolution", old_grid, old_grid, new_grid, new_grid,
                       img_size)

    @property
    def _vit(self) -> nn.Module:
        return self.peft.base_model.model

    def assert_teacher_frozen(self) -> None:
        """Only LoRA trains. Checked from the parameters, not promised in a docstring."""
        leaked = [n for n, p in self.peft.named_parameters()
                  if p.requires_grad and "lora_" not in n]
        if leaked:
            raise RuntimeError(
                f"{len(leaked)} non-LoRA parameters are trainable (e.g. {leaked[:3]}). The "
                f"teacher is the frozen prior; if the backbone moves, `d_l^TS` stops measuring "
                f"adaptation away from the prior and starts measuring two moving targets.")

    def trainable_parameters(self) -> dict:
        train = sum(p.numel() for p in self.peft.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.peft.parameters())
        return {"lora": train, "frozen": total - train, "total": total}

    def normalize(self, pixels: torch.Tensor) -> torch.Tensor:
        """[0,1] pixels -> FS-VFM's own statistics. Never ImageNet's, never CLIP's."""
        mean = torch.as_tensor(self.mean, device=pixels.device).view(1, 3, 1, 1)
        std = torch.as_tensor(self.std, device=pixels.device).view(1, 3, 1, 1)
        if pixels.shape[-1] != self.img_size or pixels.shape[-2] != self.img_size:
            pixels = F.interpolate(pixels, size=(self.img_size, self.img_size),
                                   mode="bilinear", align_corners=False)
        return (pixels - mean) / std

    # ------------------------------------------------------------------ the two passes

    def _pass(self, x: torch.Tensor, adapters: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward. Returns (block outputs (B, L, 1+N, D), pooled final feature (B, D))."""
        capture = _BlockCapture(self._vit.blocks, self.layers)
        if adapters:
            with capture:
                pooled = self._vit.forward_features(x)
        else:
            # adapters off AND no grad: the teacher is a constant with respect to the loss
            with torch.no_grad(), self.peft.disable_adapter(), capture:
                pooled = self._vit.forward_features(x)
        return capture.stacked(), pooled

    def forward(self, pixels: torch.Tensor, normalized: bool = False) -> dict:
        """`pixels` in [0,1] unless `normalized`; returns the trajectory and the patch map."""
        x = pixels if normalized else self.normalize(pixels)

        t_blocks, t_pooled = self._pass(x, adapters=False)
        s_blocks, s_pooled = self._pass(x, adapters=True)

        # CLS is index 0; the patch tokens are the rest. `h^(l)` is the patch MEAN — the
        # representation this checkpoint was trained through (see the module docstring).
        t_cls, t_patch = t_blocks[:, :, 0], t_blocks[:, :, 1:]
        s_cls, s_patch = s_blocks[:, :, 0], s_blocks[:, :, 1:]

        D = _cosine_distance(t_patch.mean(dim=2), s_patch.mean(dim=2))      # (B, L)  PRIMARY
        D_cls = _cosine_distance(t_cls, s_cls)                              # (B, L)  diagnostic
        patch_delta = _cosine_distance(t_patch, s_patch)                    # (B, L, N)

        n_patches = patch_delta.shape[-1]
        side = int(round(n_patches ** 0.5))
        if side * side != n_patches:
            raise RuntimeError(f"{n_patches} patch tokens is not a square grid")
        patch_map = patch_delta.mean(dim=1).reshape(-1, side, side)         # (B, g, g)

        return {
            # `D` is returned once and used for BOTH the trajectory readout and L_preserve, so
            # the quantity preserved is provably the quantity measured.
            "D": D,
            "D_cls": D_cls,
            "patch_delta": patch_delta,
            "patch_map": patch_map,
            "grid": (side, side),
            "h_student": s_pooled,      # the direct readout (rung B1)
            "h_teacher": t_pooled,      # the frozen readout (rung B0)
            "layers": self.layers,
        }

    # ------------------------------------------------------------------ diagnostics

    # `1 - cos(v, v)` is not exactly 0 in float32: the dot product and the norms are accumulated
    # separately, so the ratio lands within an epsilon of 1 and the distance within ~1e-7 of 0.
    # Measured 1.79e-07 on ViT-L/1024-d at init. The tolerance is a floating-point floor, NOT a
    # slack for real differences — the bit-identity of the block outputs is what
    # `teacher_student_identical` actually establishes, and it is checked separately below.
    COSINE_ZERO = 1e-6

    @torch.no_grad()
    def teacher_student_identical(self, pixels: torch.Tensor,
                                  tol: float | None = None) -> bool:
        """At initialization this must be True: the two passes compute the same function.

        The brief's two-stage training exists because of it. `d_l^TS = 0` everywhere means the
        gradient of a cosine-distance objective is zero too, so driving LoRA through
        `D -> H_traj -> L_cls` alone starts in a flat region. Exposed so a run can assert the
        premise instead of assuming it.

        Checks the underlying representations for BIT identity, not just the distance, because a
        cosine distance near zero is a weaker claim than "the passes agree" — two genuinely
        different vectors can be collinear.
        """
        x = self.normalize(pixels)
        t_blocks, t_pooled = self._pass(x, adapters=False)
        s_blocks, s_pooled = self._pass(x, adapters=True)
        bit_identical = (torch.equal(t_blocks, s_blocks) and torch.equal(t_pooled, s_pooled))
        distance = float(_cosine_distance(t_blocks[:, :, 1:].mean(dim=2),
                                          s_blocks[:, :, 1:].mean(dim=2)).abs().max())
        return bool(bit_identical and distance <= (tol if tol is not None else self.COSINE_ZERO))


def build_teacher_student(cfg: dict | None = None) -> FPADTeacherStudent:
    """Build from a config block. Layer set and LoRA targets are recorded, never implicit."""
    cfg = cfg or {}
    model = FPADTeacherStudent(
        checkpoint=cfg.get("checkpoint", DEFAULT_CHECKPOINT),
        layers=tuple(cfg.get("layers", DEFAULT_LAYERS)),
        lora=cfg.get("lora"),
        img_size=int(cfg.get("img_size", 224)),
        pooling=cfg.get("pooling", "global_pool"))
    counts = model.trainable_parameters()
    logger.info("FPAD teacher-student: layers %s (1-indexed %s), LoRA %s, "
                "trainable %s / frozen %s",
                model.layers, tuple(i + 1 for i in model.layers),
                model.lora_config["target_modules"], f"{counts['lora']:,}",
                f"{counts['frozen']:,}")
    return model
