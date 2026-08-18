"""FrozenFSVFM — Branch B's coordinate system: the FS-VFM ViT-L/16 real-face prior, frozen.

V1 spec §4. FS-VFM is pretrained self-supervised on VGGFace2 (~3M real faces) and is the broad
bona-fide-face prior; FF++ reals are *not* that prior, they are a lightweight reference
calibration inside this frozen space (§4.1). Nothing here is ever fine-tuned: the encoder is
frozen at construction and stays frozen for every stage.

The artifact
------------
`weights/FS-VFM/checkpoint-599.pth` from HF `Wolowolo/fsfm-3c`,
`pretrained_models/FS-VFM_ViT-L_VF2_600e/`. Verified locally:

    epoch 599 · args.model fsfm_vit_large_patch16 · input_size 224 · mask_ratio 0.75
    norm_pix_loss True · data VGG-Face2 · sha256 3fd99324c998ee06e4daa7f9d845e73f65f1628dee741b2c98664d566526bc87

The directory says `600e` and the file says `599` because the epoch index is 0-based, exactly as
§4 anticipates. Worth recording: the pretraining `args.epochs` is 800, so the released artifact
is epoch 599 of a longer configured schedule, not the end of a 600-epoch run.

Two pooling rules, and why the default is not free
--------------------------------------------------
The official downstream default is `global_pool=True` (`main_linearprobe_DfD.py` sets it), which
mean-pools the patch tokens and applies `fc_norm`. But the ViT class *deletes* `norm` in that
mode, and the pretrained checkpoint has no `fc_norm` — so `fc_norm` stays at LayerNorm's default
init. Frozen, that is a plain parameter-free LayerNorm, which is deterministic and reproducible
but does discard the pretrained final norm. `cls` pooling instead keeps the pretrained `norm` and
reads the CLS token.

`global_pool` is the default here because it is what the authors' downstream path uses and what
the released fine-tuned checkpoints were trained under, so it is the configuration their numbers
describe. `cls` is exposed so the parity work can measure the difference rather than assume it.

Loading is ported from the official `load_pretrained_encoder`, with one guard added: the official
version prints the load message, this one *fails* if any encoder tensor is missing. A silently
partial load leaves a partly-random "pretrained" prior whose residuals look plausible and mean
nothing — the same class of failure as Phase 1's projector.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

from .fsvfm import models_vit
from .fsvfm.pos_embed import interpolate_pos_embed

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = REPO_ROOT / "weights" / "FS-VFM" / "checkpoint-599.pth"
DEFAULT_MEAN_STD = REPO_ROOT / "weights" / "FS-VFM" / "pretrain_ds_mean_std.txt"
EXPECTED_SHA256 = "3fd99324c998ee06e4daa7f9d845e73f65f1628dee741b2c98664d566526bc87"

# From the shipped pretrain_ds_mean_std.txt. Held here only to CHECK the file, never as a
# substitute for reading it: FS-VFM does not use ImageNet statistics, and normalising a frozen
# encoder's input with the wrong constants degrades it silently rather than failing.
EXPECTED_MEAN = (0.5482207536697388, 0.42340534925460815, 0.3654651641845703)
EXPECTED_STD = (0.2789176106452942, 0.2438540756702423, 0.23493893444538116)

POOLINGS = ("global_pool", "cls")
FEATURE_DIM = 1024


def load_normalization(path: Path = DEFAULT_MEAN_STD) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """FS-VFM's own mean/std, read from the artifact shipped beside the checkpoint.

    The file repeats one JSON object per line (one per pretraining shard); every line is
    identical in the released ViT-L artifact, and a disagreement between lines would mean the
    file does not describe a single normalisation, so it is checked rather than assumed.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Download it beside the checkpoint:\n"
            f"  curl -L -o {path} 'https://hf.co/Wolowolo/fsfm-3c/resolve/main/"
            f"pretrained_models/FS-VFM_ViT-L_VF2_600e/pretrain_ds_mean_std.txt?download=true'")
    entries = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"{path} is empty")
    mean = tuple(entries[0]["mean"])
    std = tuple(entries[0]["std"])
    for e in entries[1:]:
        if tuple(e["mean"]) != mean or tuple(e["std"]) != std:
            raise ValueError(f"{path} contains more than one normalisation; refusing to guess")
    for got, want, name in ((mean, EXPECTED_MEAN, "mean"), (std, EXPECTED_STD, "std")):
        if max(abs(a - b) for a, b in zip(got, want)) > 1e-6:
            raise ValueError(
                f"FS-VFM {name} in {path} is {got}, expected {want}. This is either a different "
                f"model's statistics file or a different release; resolve it before training, "
                f"because the wrong constants degrade a frozen encoder silently.")
    return mean, std


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class FrozenFSVFM(nn.Module):
    """The frozen FS-VFM encoder. `z_ref = E_FSVFM_frozen(x)`, shape (B, 1024)."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT,
                 pooling: str = "global_pool",
                 mean_std_path: str | Path = DEFAULT_MEAN_STD,
                 verify_checksum: bool = False):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        checkpoint = Path(checkpoint)
        # os.path.isfile, not Path.is_file(): on Python 3.10 the latter propagates
        # PermissionError for an unreadable parent instead of answering "no", turning a clear
        # "checkpoint missing, here is the download" into an unrelated traceback.
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"FS-VFM checkpoint not found at {checkpoint}. This is the pretrained Online "
                f"Network, NOT a fine-tuned downstream model:\n"
                f"  curl -L -o {checkpoint} 'https://hf.co/Wolowolo/fsfm-3c/resolve/main/"
                f"pretrained_models/FS-VFM_ViT-L_VF2_600e/checkpoint-599.pth?download=true'")

        self.pooling = pooling
        self.checkpoint_path = str(checkpoint)
        self.mean, self.std = load_normalization(Path(mean_std_path))

        # off by default: hashing 4.4 GB costs ~20 s, which is noise once per run but not once
        # per unit test. The build report records it; set verify_checksum=True to re-check.
        self.checksum = file_sha256(checkpoint) if verify_checksum else None
        if self.checksum and self.checksum != EXPECTED_SHA256:
            raise ValueError(
                f"{checkpoint} sha256 {self.checksum} != expected {EXPECTED_SHA256}. A different "
                f"artifact than the one this branch was validated against.")

        # num_classes=2 keeps the class shape the official code builds; the head is unused here
        # (we read forward_features) and is deleted below so it cannot be trained by accident.
        model = models_vit.vit_large_patch16(
            num_classes=2, drop_path_rate=0.0, global_pool=(pooling == "global_pool"))
        self.load_report = self._load_pretrained(model, checkpoint)
        model.head = nn.Identity()
        self.model = model

        self.freeze()

    def _load_pretrained(self, model: nn.Module, checkpoint: Path) -> dict:
        """Ported from FSFM-CVPR25 `load_pretrained_encoder`, with a completeness guard added."""
        blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        state = blob.get("model", blob.get("state_dict", blob))
        # the released pretrain artifact has bare keys; student./teacher. prefixes appear in some
        # SSL checkpoints and are stripped exactly as the official loader does
        state = {k[len("student."):] if k.startswith("student.")
                 else k[len("teacher."):] if k.startswith("teacher.") else k: v
                 for k, v in state.items()}
        model_state = model.state_dict()
        dropped = [k for k in list(state)
                   if k in model_state and state[k].shape != model_state[k].shape]
        for k in dropped:
            del state[k]
        interpolate_pos_embed(model, state)
        msg = model.load_state_dict(state, strict=False)

        # The official loader prints this; we fail on it. `fc_norm`/`head` are legitimately
        # absent from a pretraining checkpoint (§ docstring), anything else means the encoder is
        # partly randomly initialised — a "pretrained prior" that is nothing of the kind.
        encoder_missing = [k for k in msg.missing_keys
                           if not k.startswith(("fc_norm.", "head."))]
        if encoder_missing:
            raise RuntimeError(
                f"{len(encoder_missing)} FS-VFM encoder tensors missing from {checkpoint}: "
                f"{encoder_missing[:8]}. Refusing to run a partly-random 'pretrained' prior.")
        report = {
            "checkpoint": str(checkpoint),
            "epoch": blob.get("epoch"),
            "pretrain_args_model": getattr(blob.get("args"), "model", None),
            "dropped_shape_mismatch": dropped,
            "missing_keys": list(msg.missing_keys),
            "n_unexpected_keys": len(msg.unexpected_keys),
            "pooling": self.pooling,
        }
        logger.info(f"  FS-VFM loaded: epoch={report['epoch']} pooling={self.pooling} "
                    f"missing={report['missing_keys']} unexpected={report['n_unexpected_keys']}")
        return report

    # ------------------------------------------------------------------ freezing

    def freeze(self) -> "FrozenFSVFM":
        """No parameter of this encoder may ever receive gradient (§19).

        `.grad` is cleared as well as `requires_grad`: a stale gradient left on a parameter would
        let an optimizer step it once, and would make the "did it receive gradient?" guard unable
        to tell a leak from a leftover.
        """
        for p in self.parameters():
            p.requires_grad_(False)
            p.grad = None
        self.eval()
        return self

    def train(self, mode: bool = True):
        """Stay in eval regardless of the enclosing model's mode.

        A frozen ViT in train mode would still run dropout/drop-path, so the same image would
        produce different `z_ref` on different steps and the reference residual would be noise
        on top of the quantity it is supposed to measure.
        """
        super().train(False)
        return self

    def assert_frozen(self) -> None:
        live = [n for n, p in self.named_parameters() if p.requires_grad]
        if live:
            raise RuntimeError(f"FS-VFM has trainable parameters: {live[:8]}")
        if self.training:
            raise RuntimeError("FS-VFM is in train mode; it must stay in eval")

    # ------------------------------------------------------------------ forward

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) normalised with FS-VFM statistics -> (B, 1024).

        `no_grad` is structural, not an optimisation: it makes "FS-VFM receives zero gradient"
        true by construction rather than by configuration.
        """
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(x.shape)}")
        return self.model.forward_features(x)

    def fingerprint(self) -> str:
        """sha256 over the encoder's parameters, for cache/artifact provenance."""
        h = hashlib.sha256()
        for name, p in sorted(self.model.named_parameters(), key=lambda kv: kv[0]):
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().float().numpy().tobytes())
        return h.hexdigest()

    def provenance(self) -> dict:
        return {**self.load_report, "mean": list(self.mean), "std": list(self.std),
                "sha256": self.checksum, "feature_dim": FEATURE_DIM,
                "fingerprint": self.fingerprint()}
