#!/usr/bin/env python3
"""Branch P with an off-manifold FRONT END: PE-Core per frame -> reconstruction -> transformer.

    CUDA_VISIBLE_DEVICES=3 <temporal_vjepa2 python> upgrade/branchfusion/train_om_pecore.py \
        --operator bvae --model pecore_l14 --lora --augment B --seed 42

UMAR's architecture, applied to the branch Dr Kutub trained: PE-Core sees all 16 frames, the
reconstruction is applied to what it produces per frame, and the temporal transformer attends over
the RESIDUALS rather than the raw frame embeddings. The same shape as the S variant in
`seq_branch.py --om`, on the other PE spatio-temporal branch.

WHERE IT GOES, and why not after the clip embedding. `Branch.forward` for a pecore model does

    frames -> PE-Core per frame -> f (B, T, 1536) -> TemporalMixer -> z (512) -> head

and the residual is inserted between `f` and `TemporalMixer`, which is rebuilt to take the 64-d
residual as its input width. Putting it after `TemporalMixer` instead would ask the beta-VAE to
model a CLIP-level summary of a whole clip, which is not what "off-manifold" names and is also the
construction that already failed on the image trunks.

ENCODER IN THE LOOP. The LoRA adapters, the temporal mixer and the beta-VAE all train together, so
the residual is shaped by what the temporal stage can use. Arms 1 and 2 fitted a residual on frozen
features and found it could only ever be a function of an embedding that was never asked to make it
informative; that is the failure this file is built to avoid repeating.

THE TRAINER IS NOT COPIED. `tbiom/handoff/PECORE_L14_TemporalBranch/code/.../train_branch.py` is Dr
Kutub's run as delivered and stays the reference: its clip loader, arm-B processing, micro-batch
accumulation, selection rule and exports are untouched. This file patches the model's forward and
the loss, in the manner of `train_p_dense.py`, and records that it did.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
HANDOFF = ROOT / "tbiom/handoff/PECORE_L14_TemporalBranch/code"
sys.path[:0] = [str(HANDOFF), str(HANDOFF / "tbiom"), str(HANDOFF / "tbiom/temporal"),
                str(ROOT / "upgrade/branchfusion"), str(ROOT / "analysis/tbiom/temporal")]

import numpy as np                                                  # noqa: E402
import torch.nn as nn                                               # noqa: E402

import train_branch as TB                                           # noqa: E402
import residual_head as RH                                          # noqa: E402
import seq_branch as SB                                             # noqa: E402
from seq_branch import OMFrontEnd                                   # noqa: E402

STATE: dict = {}

#: Where Dr Kutub's trainer writes, and where ours must write instead.
KUTUB_RUNS = "/data/kutub/conferences/DFD/PECORE_L14_TemporalBranch/runs"
OUR_RUNS = ROOT / "logs/branchfusion/om_pecore"


def redirect_outputs() -> None:
    """Keep this campaign's runs out of Dr Kutub's directory.

    `train_branch.main` hardcodes its run directory under /data/kutub/..., because it was written
    to run from his checkout. Importing it as a library therefore writes OUR runs into HIS space,
    which is forbidden -- and it already happened: six directories and about 170 MB landed there
    before this was noticed. The trainer is his reference and is not edited, so the `Path` it calls
    is wrapped instead, rewriting only that one prefix and passing every other path through.
    """
    real = TB.Path

    def path(*a, **kw):
        if a and str(a[0]).startswith(KUTUB_RUNS):
            a = (str(a[0]).replace(KUTUB_RUNS, str(OUR_RUNS)),) + a[1:]
        return real(*a, **kw)

    TB.Path = path


def frame_index(key: str) -> int:
    """The frame's position in the SOURCE video, parsed from its h5 key.

    Keys look like `CDFv2/Celeb-real/id0_0001/009.png`, so the stored crops carry their true index
    and a clip's frames are typically 0, 9, 19 ... rather than consecutive. Branch S encodes that
    real spacing; P's mixer encodes the slot number, which asserts a constant frame rate that does
    not hold across corpora. Unparseable keys fall back to -1 and are repaired by the caller.
    """
    stem = str(key).rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return int(stem) if stem.isdigit() else -1


def tpos_table(clips) -> np.ndarray:
    """(N, T) real frame indices for every clip, computed the way `Clips.__getitem__` samples.

    Built in the PARENT process from the key lists alone -- no h5 reads -- so it costs nothing and
    cannot diverge from the sampler as long as both use `linspace(0, len(ks) - 1, T)`.
    """
    out = np.zeros((len(clips.items), clips.T), dtype=np.int64)
    for n, (ks, _y, _v) in enumerate(clips.items):
        idx = np.linspace(0, len(ks) - 1, clips.T).round().astype(int)
        pos = [frame_index(ks[j]) for j in idx]
        # A corpus that does not number its crops gets slot numbers, which is the old behaviour and
        # is still correct for whole-image sets where every "video" has one frame.
        out[n] = pos if all(q >= 0 for q in pos) else np.arange(clips.T)
    return out


class SeqTemporal(nn.Module):
    """Branch S's temporal stage: REAL-timestamp positions, two pre-norm blocks, mean over frames.

    The only thing that distinguishes S from P's `TemporalMixer` is this positional encoding --
    `TemporalPE` encodes relative time from the parsed frame indices, where the mixer adds a
    learned embedding per slot. Width, depth and head count are S's (`seq_branch.D_MODEL` etc.),
    so an S-vs-P comparison on the same trunk is a comparison of the time encoding and the
    transformer size, not of two unrelated designs.
    """

    def __init__(self, d_in: int, d_out: int, pe_scale: float = 10.0):
        super().__init__()
        self.proj = nn.Linear(d_in, SB.D_MODEL)
        self.pos = SB.TemporalPE(SB.D_MODEL, pe_scale)
        layer = nn.TransformerEncoderLayer(
            SB.D_MODEL, SB.N_HEADS, SB.FF, dropout=SB.DROPOUT, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, SB.N_BLOCKS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(SB.D_MODEL)
        self.out = nn.Linear(SB.D_MODEL, d_out)

    def forward(self, x):
        tp = STATE.get("tpos")
        if tp is None or tp.shape[0] != x.shape[0]:
            # No table for this loader (or a size mismatch after a partial batch): fall back to
            # slot numbers and say so once, rather than silently encoding the wrong times.
            if not STATE.get("warned"):
                print("  NOTE: no timestamp table for this batch; using slot numbers", flush=True)
                STATE["warned"] = True
            tp = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        # FLOAT, not the int64 the key parser produces: `TemporalPE` masks with +inf, which cannot
        # be represented in an integer tensor and raises "value cannot be converted to type int64_t
        # without overflow" at the first batch.
        tp = tp.to(x.device).float()[:, :x.shape[1]]
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        h = self.enc(self.proj(x) + self.pos(tp, mask))
        return self.out(self.norm(h).mean(1))


def install_tpos() -> None:
    """Carry each clip's real frame indices from the sampler to the temporal stage.

    `Branch.forward` receives only pixels, and the training loop that calls it lives in Dr Kutub's
    trainer and is not copied here, so the indices cannot be passed as an argument without editing
    his loop. The batch's THIRD element is the item index, and the DataLoader is the one place that
    sees it on the way past -- so the loader publishes `tpos` for the batch it is about to yield
    and the temporal stage reads it from the same STATE the VAE terms already use.

    Only datasets carrying a `_tpos` table are affected, so evaluation loaders built elsewhere, and
    any other DataLoader in the process, behave exactly as before.
    """
    orig_dl = torch.utils.data.DataLoader
    if getattr(orig_dl, "_publishes_tpos", False):
        return

    class TPosLoader(orig_dl):
        _publishes_tpos = True

        def __iter__(self):
            tab = getattr(self.dataset, "_tpos", None)
            for batch in super().__iter__():
                if tab is not None and len(batch) >= 3:
                    STATE["tpos"] = torch.as_tensor(tab[np.asarray(batch[2])])
                yield batch

    torch.utils.data.DataLoader = TPosLoader

    orig_init = TB.Clips.__init__

    def cinit(self, ds, *args, **kw):
        orig_init(self, ds, *args, **kw)
        self._tpos = tpos_table(self)

    TB.Clips.__init__ = cinit


def install(operator: str | None, block: int = 0, temporal: str = "mixer",
            keep_plain: bool = False) -> None:
    """Put the residual between PE-Core's per-frame output and the temporal mixer.

    `block` taps an intermediate PE-Core block instead of the final attention pool, per UMAR: the
    best visual embeddings are not at the output. The tap is the mean over patch tokens, with the
    model's own prefix tokens dropped rather than a hardcoded 1.
    """
    orig_init, orig_fwd = TB.Branch.__init__, TB.Branch.forward

    def init(self, name, *args, **kw):
        orig_init(self, name, *args, **kw)
        if not name.startswith("pecore"):
            raise SystemExit(f"{name} has no per-frame stage to put a residual in front of")
        d, T = TB.MODELS[name][1], self.T
        d_in = self.backbone.num_features
        self.om_block = block
        if block:
            blocks = self.backbone.blocks
            assert 1 <= block <= len(blocks), f"block {block} outside 1..{len(blocks)}"
            self._tap = {}
            npre = getattr(self.backbone, "num_prefix_tokens", 1)

            def hook(_m, _i, out):
                self._tap["z"] = out[:, npre:].mean(1)

            blocks[block - 1].register_forward_hook(hook)
            d_in = blocks[block - 1].norm1.normalized_shape[0]
        # `operator=None` is the CONTROL: the same block tap, the same rebuilt mixer, no residual.
        # Gate 1 for this branch is "beats its own no-OM control at the same depth", so without
        # this run a good OM number cannot be told apart from the block-16 tap simply being better.
        self.om = OMFrontEnd(d_in, operator, keep_plain=keep_plain) if operator else None
        # The mixer is rebuilt at the residual's width. Its weights were never trained at this
        # point -- the run starts from scratch -- so nothing is discarded by replacing it.
        d_t = self.om.d_out if self.om else d_in
        orphan = self.temporal
        if temporal == "seq":
            # Branch S's stage. The positional scale is the median spacing between the clip's real
            # frame indices, which is what `seq_branch` derives from the cache and passes as
            # `pe_scale`; taken from the training table so it is a property of the data.
            self.temporal = SeqTemporal(d_t, d, pe_scale=STATE.get("pe_scale", 10.0))
        else:
            self.temporal = TB.TemporalMixer(d_t, d, T)
        # PUT THE NEW MODULES WHERE THE TRAINER LOOKS. `orig_init` has already built its own
        # TemporalMixer and placed it in `trainable_blocks`, and that ModuleList -- not the
        # attributes -- is what `main` hands to AdamW and what `trainable_state` serialises.
        # Rebinding `self.temporal` therefore left the orphan in the list and BOTH new modules
        # outside it: the rebuilt temporal stage and the residual received gradients but no
        # optimiser step and no checkpoint entry, so every run of this file before 2026-10-10
        # trained LoRA and the head on top of a RANDOM temporal stage. The orphan is dropped by
        # identity rather than by index, because `inject_lora` decides how many entries precede it.
        keep = [m for m in self.trainable_blocks if m is not orphan]
        self.trainable_blocks = nn.ModuleList(
            keep + [self.temporal] + ([self.om] if self.om is not None else []))
        # Checked here rather than in a test, because what makes this hold is a property of DR
        # KUTUB'S trainer -- that it optimises and serialises `trainable_blocks` -- and a future
        # revision of his file could move the optimiser to `model.parameters()` and quietly make
        # the list the wrong place to be. A run that trains nothing it was built to train should
        # not start.
        reachable = {id(q) for q in self.trainable_blocks.parameters()}
        for tag, mod in (("temporal", self.temporal), ("om", self.om)):
            missing = [] if mod is None else [n for n, q in mod.named_parameters()
                                              if id(q) not in reachable]
            assert not missing, f"{tag} parameters outside trainable_blocks: {missing[:4]}"

    def forward(self, px):
        if not self.name.startswith("pecore"):
            return orig_fwd(self, px)
        B, T_ = px.shape[:2]
        x = torch.nn.functional.interpolate(
            px.flatten(0, 1), size=(self.pe_size, self.pe_size), mode="bicubic",
            align_corners=False)
        chunks = []
        for c in x.split(TB.FRAME_CHUNK):
            out = self.backbone(c)
            chunks.append(self._tap["z"] if self.om_block else out)
        f = torch.cat(chunks).view(B, T_, -1)
        if self.om is not None:
            mask = torch.ones(B, T_, dtype=torch.bool, device=f.device)  # clips are not padded here
            f = self.om(f, mask)
        STATE["om"] = self.om
        # SOFTPLUS, as in `Branch.forward`. The head emits EVIDENCE, and `opinions` forms
        # `alpha = evidence + 1`; without this the head emits logits, alpha can go negative and the
        # reported p leaves [0, 1] -- which it did, down to -0.186 in the seq control.
        return torch.nn.functional.softplus(self.head(self.temporal(f).float()))

    TB.Branch.__init__, TB.Branch.forward = init, forward

    orig_loss = TB.evidential_loss_discern_ext

    def loss(ev, y, K, ep, epochs):
        # The VAE term is added UNDIVIDED. The caller does `.mean() * (len(yc) / len(y))`, so the
        # chunk-share weighting that keeps accumulated gradients equal to the full-batch gradient
        # is applied to this term too. An earlier version divided by `base.shape[0]` and crashed:
        # `orig_loss` returns a SCALAR here, not the per-sample vector it returns elsewhere.
        base = orig_loss(ev, y, K, ep, epochs)
        om = STATE.get("om")
        return base if om is None else base + RH.LAMBDA_VAE * om.vae_loss(y)

    TB.evidential_loss_discern_ext = loss


def install_opinion_retry() -> None:
    """Retry a development scoring pass that failed the trainer's own coverage test.

    WHAT ACTUALLY CAUSED THE THREE DEATHS, recorded because two wrong diagnoses were committed
    before this one. Runs died at `assert t.p.notna().all()` -- "development videos without branch
    opinions" -- after five or six clean epochs. The first guess was a SHORT scoring pass under
    disk pressure, so the first version of this wrapper tested the row count; it never fired,
    because the count was always right. The assert does not test the count. It tests that every
    development video carries a usable `p`, and a row present with `p = NaN` fails it exactly as a
    missing row does.

    The NaN came from `install`'s forward, which had dropped the `softplus` that `Branch.forward`
    applies. The head then emitted logits rather than evidence, so `alpha = evidence + 1` could go
    negative -- opinions down to p = -0.186 were exported -- and a clip whose two alphas happened
    to sum near zero produced 0/0. That is why it struck mid-run, without pattern, and left the
    count intact. The softplus is back; this wrapper stays as the net.

    So it tests both things the assert cares about, where the dataset is still in hand: coverage
    against `videos_of`, and finiteness of `p`. A pass failing either is retried; one that keeps
    failing says WHICH of the two it is and names the dataset, which is what the assert alone
    cannot do. The assert is not disabled -- this only gives it a diagnosable failure.
    """
    orig = TB.opinions

    def check(out, want):
        """(ok, how it failed) for one scoring pass."""
        if len(out) < want:
            return False, f"{len(out)}/{want} videos"
        bad = int((~np.isfinite(out.p.to_numpy())).sum())
        if bad:
            return False, f"{bad}/{len(out)} opinions are NaN or inf"
        return True, ""

    def opinions(model, ds, dev, *args, **kw):
        want = len(TB.videos_of(ds)[1])
        out = why = None
        for attempt in (1, 2, 3):
            out = orig(model, ds, dev, *args, **kw)
            ok, why = check(out, want)
            if ok:
                return out
            print(f"  {ds}: {why} (attempt {attempt}/3) -- retrying", flush=True)
        print(f"  {ds}: STILL BAD after 3 attempts ({why}); the assert below will name this "
              f"dataset", flush=True)
        return out

    TB.opinions = opinions


def export_only(out_dir: Path, workers: int, model_name: str, lora: bool) -> int:
    """Write the exports of a run whose TRAINING finished but whose export stage crashed.

    Both reruns got through all their epochs and saved `best.pt`; the P control then hit the
    coverage assert and branch S hit a signature bug in the wrapper above. Retraining either would
    cost the two and four hours they already cost, and would train a DIFFERENT model -- the
    selected weights are on disk and are the thing the gates are about. This replays only the
    export tail of `TB.main`, in the same order and to the same filenames, against those weights.

    The model must be built AFTER `install`, exactly as in training, or `load_trainable` has no
    rebuilt mixer to load into.
    """
    ck_path = out_dir / "best.pt"
    if not ck_path.exists():
        print(f"no checkpoint at {ck_path}", flush=True)
        return 1
    import exp1_fusion_audit as A                                      # noqa: F401  (trainer's import)
    dev = "cuda"
    model = TB.Branch(model_name, lora=lora).to(dev)
    ck = torch.load(ck_path, map_location=dev)
    print(f"export-only: {out_dir.name}, epoch {ck['epoch']} ({TB.load_trainable(model, ck)})",
          flush=True)
    (out_dir / "opinions").mkdir(parents=True, exist_ok=True)
    for n in TB.DEV:
        TB.opinions(model, n, dev, workers=workers).to_csv(
            out_dir / "opinions" / f"{n}.csv", index=False)
        for c, spec in (("jpeg40", "jpeg:40"), ("rs06", "resize:0.6")):
            TB.opinions(model, n, dev, spec, workers).to_csv(
                out_dir / "opinions" / f"{n}_{c}.csv", index=False)
    for n in TB.TESTS:
        TB.opinions(model, n, dev, workers=workers).to_csv(
            out_dir / "opinions" / f"{n}.csv", index=False)
    for n in ("CDFv2", "DFDCP"):
        for c, spec in (("jpeg40", "jpeg:40"), ("rs06", "resize:0.6")):
            TB.opinions(model, n, dev, spec, workers).to_csv(
                out_dir / "opinions" / f"{n}_{c}.csv", index=False)
    (out_dir / "DONE").touch()
    TB.export_shuffled(model, dev, out_dir, workers)
    print("export-only: done", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", default="bvae", choices=["bvae", "bvae_real", "none"],
                    help="'none' is the no-OM control at the same depth, which gate 1 needs")
    ap.add_argument("--om-mode", default="residual", choices=["residual", "concat"],
                    help="residual = the temporal stage sees f_a alone; concat = [f_t ; f_a], "
                         "which is how DISCERN uses the transform (a second view, not a "
                         "replacement) and what the residual-only run shows is necessary")
    ap.add_argument("--temporal", default="mixer", choices=["mixer", "seq"],
                    help="mixer = P's learned slot positions; seq = S's real-timestamp encoding")
    ap.add_argument("--block", type=int, default=0,
                    help="tap this PE-Core block (1-indexed) instead of the final attention pool")
    ap.add_argument("--export-only", action="store_true",
                    help="training already finished and saved best.pt, but the export tail "
                         "crashed: replay only the exports against the selected weights")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="end-to-end rehearsal: N training clips, 1 epoch, 1 test export; the "
                         "DEVELOPMENT sets stay whole because the trainer asserts full coverage")
    a, rest = ap.parse_known_args()
    redirect_outputs()
    install_opinion_retry()
    op = None if a.operator == "none" else a.operator
    if a.temporal == "seq":
        install_tpos()
    install(op, a.block, a.temporal, keep_plain=(a.om_mode == "concat"))
    if a.smoke:
        TB.EPOCHS = 1
        orig = TB.Clips.__init__
        keep = set(TB.DEV)

        def cinit(self, ds, *args, **kw):
            orig(self, ds, *args, **kw)
            if ds not in keep and len(self.items) > a.smoke:
                step = max(1, len(self.items) // a.smoke)
                self.items = self.items[::step][:a.smoke]

        TB.Clips.__init__ = cinit
        TB.TESTS = TB.TESTS[:1]
    suffix = next((rest[i + 1] for i, x in enumerate(rest) if x == "--run-suffix"), "")
    rest = [x for i, x in enumerate(rest)
            if x != "--run-suffix" and (i == 0 or rest[i - 1] != "--run-suffix")]
    sys.argv = [sys.argv[0]] + rest + [
        "--run-suffix", f"{suffix}_om{a.operator}" + (f"_b{a.block}" if a.block else "")
        + ("_cat" if a.om_mode == "concat" else "")
        + ("_seq" if a.temporal == "seq" else "") + ("_smoke" if a.smoke else "")]
    print(f"OM-PE: residual {a.operator} between PE-Core's frames and the temporal mixer; "
          f"argv {' '.join(sys.argv[1:])}", flush=True)
    if a.export_only:
        # The run name the trainer would have built, from the same argv it would have parsed.
        def arg(flag, default):
            return next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == flag), default)

        model = arg("--model", "pecore_l14")
        name = (f"{model}{'_lora' if '--lora' in sys.argv else ''}"
                f"{'_B' if arg('--augment', None) == 'B' else ''}"
                f"{arg('--run-suffix', '')}_seed{arg('--seed', '0')}")
        return export_only(OUR_RUNS / name, int(arg("--workers", "4")),
                           model, "--lora" in sys.argv)
    TB.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
