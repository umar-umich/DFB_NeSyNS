"""Source-paired batching with matched augmentation (brief Stage 4).

The V1 audit found a provenance shortcut: branches partly learned "which corpus is this" rather
than "is this manipulated". Source-paired sampling attacks that shortcut **at the sampler** — if
every batch contains a fake beside the real video it was made from, then corpus, identity,
lighting, compression and background are held constant across the label, and nothing about the
source can separate the classes.

What already existed, and why it is not enough
----------------------------------------------
`NeSyDeFakeDataset` has `paired_training` (default on). It is GenD-style **pair inclusion at
data-preparation time**: the training list contains both members of a source pair, and shuffled
batching is relied on to co-occur them. Two gaps:

1. **Co-occurrence is not guaranteed.** With ~22k frames and a batch of 32, a given pair lands in
   the same batch by chance. Pairing that holds only in expectation does not hold for the gradient
   of any particular step.
2. **Augmentation is drawn independently per image** (`train_v1.py`: `torch.stack([transform(img)
   for img in raw])`). The two members of a pair therefore receive different flips, affines, blurs
   and jitters — reintroducing a nuisance difference exactly where the pairing removed one. A model
   can separate the pair on its augmentation.

`PairedBatchSampler` fixes the first, `MatchedAugment` the second.

The pair key is the source video id, not an index
-------------------------------------------------
`MatchedAugment` groups by the **source-video id derived from the frame path**, rather than by a
pair index threaded through the DataLoader. That is deliberate: a batch assembled by worker
processes arrives as tensors and paths, and any index-based side channel has to survive collation,
worker boundaries and shuffling. Keying on something already in the batch cannot desynchronise.

It also means a *group* rather than a couple: FF++ has one source real per several manipulations,
so `802`, `802_885` and `802_123` all share a key and all share one augmentation draw. That is the
intended behaviour — the nuisance is held constant across the whole source group, not just one
couple.

The brief asks for matched augmentation "where possible". Frames with no partner in the batch get
an ordinary independent draw rather than being dropped.
"""

from __future__ import annotations

import random
from typing import Iterator, Sequence

import torch
import torchvision.transforms.functional as TF

from networks.discern_v2.semantic_branch import DICOME_AUGMENTATION


class PairedBatchSampler(torch.utils.data.Sampler):
    """Yield index batches in which source-matched real/fake frames sit together.

    Each batch is filled with `batch_size // 2` pairs. Frames with no partner fill whatever room
    is left, so an epoch still covers the full training set — a sampler that silently dropped
    unpaired frames would shrink the training data without changing anything a reader could see.
    """

    def __init__(self, labels: Sequence[int], source_ids: Sequence[str | None],
                 batch_size: int, seed: int = 42, drop_last: bool = True):
        if batch_size % 2:
            raise ValueError(f"batch_size must be even for pairing, got {batch_size}")
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        by_source: dict[str, dict[int, list[int]]] = {}
        for i, (label, source) in enumerate(zip(labels, source_ids)):
            if source is None:
                continue
            sides = by_source.setdefault(str(source), {0: [], 1: []})
            sides.setdefault(int(bool(label)), []).append(i)

        self.pairs: list[tuple[int, int]] = []
        paired: set[int] = set()
        for sides in by_source.values():
            reals, fakes = sides.get(0, []), sides.get(1, [])
            if not reals or not fakes:
                continue
            # One real partner per fake, cycling the reals: FF++ ships one source real per several
            # manipulations, so pairing strictly one-to-one would discard most fakes.
            for k, fake in enumerate(fakes):
                real = reals[k % len(reals)]
                self.pairs.append((real, fake))
                paired.add(real)
                paired.add(fake)
        self.unpaired = [i for i in range(len(labels)) if i not in paired]

        if not self.pairs:
            raise ValueError(
                "no source-matched real/fake pairs were found. Source-paired sampling is the "
                "PRIMARY arm of Stage 4, and running it with zero pairs would silently be the "
                "random-sampling control under another name. Check that the training set is FF++ "
                "and that the dataset exposes per-sample source ids.")

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle deterministically per epoch, so a resumed run repeats the same order."""
        self.epoch = epoch

    def pair_fraction(self) -> float:
        """Fraction of emitted samples that have a partner. Logged, never assumed to be 1.0."""
        total = 2 * len(self.pairs) + len(self.unpaired)
        return 2 * len(self.pairs) / total if total else 0.0

    def __len__(self) -> int:
        total = 2 * len(self.pairs) + len(self.unpaired)
        return total // self.batch_size if self.drop_last else -(-total // self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        pairs = list(self.pairs)
        singles = list(self.unpaired)
        rng.shuffle(pairs)
        rng.shuffle(singles)

        batch: list[int] = []
        pi = si = 0
        while pi < len(pairs) or si < len(singles):
            if pi < len(pairs) and len(batch) + 2 <= self.batch_size:
                batch.extend(pairs[pi])
                pi += 1
            elif si < len(singles) and len(batch) < self.batch_size:
                batch.append(singles[si])
                si += 1
            else:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch


class MatchedAugment:
    """DiCoME's augmentation pipeline, with the random draw shared inside a source group.

    Implemented functionally rather than as a `Compose`, because a `Compose` samples its own
    parameters and cannot be asked to reuse another image's. The operations, their order and their
    ranges are read from `DICOME_AUGMENTATION` — the same dictionary `dicome_train_transform()`
    builds from — so the paired and unpaired pipelines cannot drift into applying different
    augmentation.
    """

    def __init__(self, seed: int = 42, config: dict | None = None):
        self.seed = seed
        self.cfg = config or DICOME_AUGMENTATION
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _params(self, gen: torch.Generator, height: int, width: int) -> dict:
        c = self.cfg

        def uniform(lo: float, hi: float) -> float:
            return float(torch.empty(1).uniform_(lo, hi, generator=gen))

        degrees = c["random_affine"]["degrees"]
        tx, ty = c["random_affine"]["translate"]
        smin, smax = c["random_affine"]["scale"]
        k = c["gaussian_blur"]["kernel_size"]
        if isinstance(k, (list, tuple)) and len(k) == 2:
            # torchvision samples an ODD kernel size in [k0, k1]; reproduce that, don't fix it to
            # one end, or the paired arm would blur differently from the control
            lo, hi = int(k[0]) // 2, int(k[1]) // 2
            kernel = int(torch.randint(lo, hi + 1, (1,), generator=gen)) * 2 + 1
        else:
            kernel = int(k[0] if isinstance(k, (list, tuple)) else k)
        sigma_lo, sigma_hi = c["gaussian_blur"]["sigma"]
        return {
            "flip": float(torch.rand(1, generator=gen)) < c["random_horizontal_flip"]["p"],
            "angle": uniform(-degrees, degrees),
            "translate": [int(uniform(-tx, tx) * width), int(uniform(-ty, ty) * height)],
            "scale": uniform(smin, smax),
            "kernel": max(1, kernel),
            "sigma": uniform(sigma_lo, sigma_hi),
            "brightness": uniform(1 - c["color_jitter"]["brightness"],
                                  1 + c["color_jitter"]["brightness"]),
            "contrast": uniform(1 - c["color_jitter"]["contrast"],
                                1 + c["color_jitter"]["contrast"]),
        }

    def _apply(self, img: torch.Tensor, p: dict) -> torch.Tensor:
        if p["flip"]:
            img = TF.hflip(img)
        img = TF.affine(img, angle=p["angle"], translate=p["translate"], scale=p["scale"],
                        shear=[0.0, 0.0])
        img = TF.gaussian_blur(img, kernel_size=[p["kernel"], p["kernel"]],
                               sigma=[p["sigma"], p["sigma"]])
        img = TF.adjust_brightness(img, p["brightness"])
        img = TF.adjust_contrast(img, p["contrast"])
        return img

    def __call__(self, images: torch.Tensor, pair_keys: Sequence[str | None],
                 step: int = 0) -> torch.Tensor:
        """Augment a batch, sharing the draw among samples with the same non-None `pair_key`."""
        if len(pair_keys) != len(images):
            raise ValueError(f"{len(pair_keys)} pair keys for {len(images)} images")
        height, width = images.shape[-2], images.shape[-1]
        out = torch.empty_like(images)
        cache: dict[str, dict] = {}
        for i, key in enumerate(pair_keys):
            if key is not None and key in cache:
                params = cache[key]                       # the partner's exact draw
            else:
                token = key if key is not None else f"__solo_{i}"
                gen = torch.Generator().manual_seed(
                    abs(hash((self.seed, self.epoch, step, token))) % (2 ** 31))
                params = self._params(gen, height, width)
                if key is not None:
                    cache[key] = params
            out[i] = self._apply(images[i], params)
        return out

    def matched_fraction(self, pair_keys: Sequence[str | None]) -> float:
        """Fraction of the batch that actually shared a draw with at least one other sample."""
        counts: dict[str, int] = {}
        for key in pair_keys:
            if key is not None:
                counts[key] = counts.get(key, 0) + 1
        shared = sum(n for n in counts.values() if n > 1)
        return shared / len(pair_keys) if len(pair_keys) else 0.0


def source_ids_for(dataset) -> list[str | None]:
    """The per-sample source-video id the dataset already computes.

    Read from the dataset rather than re-derived: `_extract_source_video` encodes FF++'s
    `<source>_<target>` naming and the `_aug` suffix convention, and a second implementation would
    drift from it silently.
    """
    ids = getattr(dataset, "_per_sample_source_id", None)
    if ids is not None:
        return [None if v is None else str(v) for v in ids]
    extract = getattr(dataset, "_extract_source_video", None)
    if extract is None:
        raise AttributeError(
            "the dataset exposes neither `_per_sample_source_id` nor `_extract_source_video`, so "
            "source pairs cannot be formed. Do NOT fall back to unpaired sampling silently — that "
            "would make the primary Stage 4 arm the control under another name.")
    return [extract(p if isinstance(p, str) else p[0]) for p in dataset.image_list]


def pair_keys_for_paths(paths: Sequence[str], extract) -> list[str | None]:
    """Pair keys for an already-collated batch, from the paths it carries."""
    return [extract(p if isinstance(p, str) else p[0]) for p in paths]
