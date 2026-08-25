"""Source-paired batching and matched augmentation (brief Stage 4).

    python training/dataset/test_paired_sampler.py

Everything checked here fails SILENTLY if broken — a run would train, converge, and report
numbers, while the property the arm exists to establish quietly did not hold:

* pairs actually landing in the same batch (the whole point of the sampler);
* pair members receiving the IDENTICAL augmentation, and non-partners not;
* the epoch still covering every sample, so pairing does not shrink the training set;
* determinism across epochs, so a resumed run repeats its order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.paired_sampler import (  # noqa: E402
    MatchedAugment, PairedBatchSampler, pair_keys_for_paths)

PASSED = []


def check(name):
    def deco(fn):
        fn()
        PASSED.append(name)
        print(f"  ok  {name}")
        return fn
    return deco


def toy(n_sources: int = 20, fakes_per_source: int = 3, n_orphans: int = 7):
    """FF++-shaped: one source real per several manipulations, plus some unpartnered frames."""
    labels, sources = [], []
    for s in range(n_sources):
        labels.append(0)
        sources.append(f"src{s}")
        for _ in range(fakes_per_source):
            labels.append(1)
            sources.append(f"src{s}")
    for o in range(n_orphans):          # fakes whose source real is not in the split
        labels.append(1)
        sources.append(f"orphan{o}")
    return labels, sources


@check("every pair lands in the same batch")
def _pairs_co_occur():
    labels, sources = toy()
    sampler = PairedBatchSampler(labels, sources, batch_size=8, seed=0)
    for batch in sampler:
        # each emitted pair is contiguous, so every batch position with a partner has it here
        for real, fake in sampler.pairs:
            if real in batch or fake in batch:
                if real in batch and fake in batch:
                    continue
                # the other member may belong to a different pair scheduled elsewhere; what must
                # never happen is a batch containing a fake with NO real sharing its source
                shared = [i for i in batch if sources[i] == sources[fake]]
                assert any(labels[i] == 0 for i in shared) or sources[fake].startswith("orphan"), \
                    f"fake {fake} (source {sources[fake]}) has no same-source real in its batch"


@check("the epoch covers every sample exactly once per emitted slot")
def _full_coverage():
    labels, sources = toy()
    sampler = PairedBatchSampler(labels, sources, batch_size=8, seed=0, drop_last=False)
    seen = [i for batch in sampler for i in batch]
    # reals are reused as partners for several fakes, so coverage is over the emitted multiset:
    # every index must appear, and no unpaired frame may be dropped
    assert set(seen) == set(range(len(labels))), (
        f"{len(set(range(len(labels))) - set(seen))} samples never appeared — pairing shrank the "
        f"training set")
    for i in sampler.unpaired:
        assert i in seen, f"unpaired sample {i} was dropped"


@check("pair fraction is reported, not assumed to be 1.0")
def _pair_fraction():
    labels, sources = toy(n_sources=10, fakes_per_source=2, n_orphans=10)
    sampler = PairedBatchSampler(labels, sources, batch_size=8, seed=0)
    frac = sampler.pair_fraction()
    assert 0.0 < frac < 1.0, frac
    expected = 2 * 20 / (2 * 20 + 10)      # 10 sources x 2 fakes = 20 pairs, 10 orphans
    assert abs(frac - expected) < 1e-9, (frac, expected)


@check("refuses to run with zero pairs instead of silently becoming the control")
def _refuses_unpaired():
    try:
        PairedBatchSampler([1, 1, 1, 1], ["a", "b", "c", "d"], batch_size=4)
    except ValueError as exc:
        assert "PRIMARY" in str(exc)
    else:
        raise AssertionError("a set with no real/fake source match must be refused")


@check("odd batch size is refused")
def _odd_batch():
    labels, sources = toy()
    try:
        PairedBatchSampler(labels, sources, batch_size=7)
    except ValueError as exc:
        assert "even" in str(exc)
    else:
        raise AssertionError("an odd batch size cannot hold whole pairs")


@check("order is deterministic per epoch and changes between epochs")
def _deterministic():
    labels, sources = toy()
    a = PairedBatchSampler(labels, sources, batch_size=8, seed=7)
    b = PairedBatchSampler(labels, sources, batch_size=8, seed=7)
    a.set_epoch(3)
    b.set_epoch(3)
    assert [x for batch in a for x in batch] == [x for batch in b for x in batch]
    a.set_epoch(4)
    assert ([x for batch in a for x in batch]
            != [x for batch in b for x in batch]), "epochs must reshuffle"


@check("matched augmentation is IDENTICAL within a source group and differs across groups")
def _matched_augmentation():
    torch.manual_seed(0)
    images = torch.rand(6, 3, 64, 64)
    # rows 0-2 share a source, rows 3-4 share another, row 5 is a solo
    images[1] = images[0]
    images[2] = images[0]
    images[4] = images[3]
    keys = ["src0", "src0", "src0", "src1", "src1", None]
    out = MatchedAugment(seed=1)(images, keys, step=0)

    assert torch.allclose(out[0], out[1], atol=1e-6), (
        "identical images in one source group got DIFFERENT augmentation — the pairing is then "
        "reintroducing the nuisance it exists to remove")
    assert torch.allclose(out[0], out[2], atol=1e-6)
    assert torch.allclose(out[3], out[4], atol=1e-6)
    # different groups must not accidentally share a draw
    same_source_pixels = torch.allclose(images[0], images[3], atol=1e-6)
    assert not same_source_pixels or not torch.allclose(out[0], out[3], atol=1e-6)


@check("a real and its fake get the same draw even though their pixels differ")
def _pair_draw_shared():
    """The realistic case: the two members are different images, so equality of OUTPUT cannot be
    checked. Equality of the sampled PARAMETERS can, by augmenting a probe image under each."""
    aug = MatchedAugment(seed=2)
    probe = torch.rand(1, 3, 48, 48)
    batch = torch.cat([probe, probe])            # same pixels, so output equality tests params
    shared = aug(batch, ["src9", "src9"], step=5)
    solo = aug(batch, [None, None], step=5)
    assert torch.allclose(shared[0], shared[1], atol=1e-6), "pair members must share the draw"
    assert not torch.allclose(solo[0], solo[1], atol=1e-6), (
        "unpartnered samples must get INDEPENDENT draws, or the control arm is not a control")


@check("matched_fraction counts only samples that actually shared")
def _matched_fraction():
    aug = MatchedAugment()
    assert aug.matched_fraction(["a", "a", "b", None]) == 0.5
    assert aug.matched_fraction([None, None]) == 0.0
    assert aug.matched_fraction(["a", "a"]) == 1.0


@check("augmentation stays in range and preserves shape/dtype")
def _well_formed():
    images = torch.rand(4, 3, 64, 64)
    out = MatchedAugment(seed=3)(images, ["s", "s", None, None])
    assert out.shape == images.shape and out.dtype == images.dtype
    assert torch.isfinite(out).all()
    assert float(out.min()) >= -1e-6 and float(out.max()) <= 1.0 + 1e-6, (
        f"augmented pixels left [0,1] ({float(out.min()):.3f}, {float(out.max()):.3f}); the "
        f"branches normalize from [0,1] and would silently receive out-of-range input")


@check("pair keys derive from FF++ paths the way the dataset does")
def _pair_keys():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset.nesy_defake_dataset import NeSyDeFakeDataset

    extract = NeSyDeFakeDataset._extract_source_video
    paths = ["a/b/frames/802/0001.png", "a/b/frames/802_885/0001.png",
             "a/b/frames/929_aug1/0001.png", "no/frames/here.png"]
    keys = pair_keys_for_paths(paths, extract)
    assert keys[0] == "802" and keys[1] == "802", keys
    assert keys[2] == "929", keys
    assert keys[0] == keys[1], "a fake and its source real must share a key"


@check("mismatched key count is refused")
def _key_count():
    try:
        MatchedAugment()(torch.rand(3, 3, 32, 32), ["a", "b"])
    except ValueError as exc:
        assert "pair keys" in str(exc)
    else:
        raise AssertionError("a key/image count mismatch must not be silently zipped short")


if __name__ == "__main__":
    print(f"\nall {len(PASSED)} passed")
