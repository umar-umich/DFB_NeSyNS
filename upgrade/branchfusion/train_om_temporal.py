#!/usr/bin/env python3
"""Off-manifold on the TEMPORAL trunks, with the encoder in the loop.

    CUDA_VISIBLE_DEVICES=1 <temporal_vjepa2 python> upgrade/branchfusion/train_om_temporal.py \
        --model vjepa2 --lora --augment B --seed 42 --operator order

WHY THIS RE-OPENS A CLOSED DECISION, and says so out loud. UMAR's decision of 2026-09-17 lists
"run more residual-operator variants" under Do-not, on the evidence of EXP4B. But EXP4B's own line 3
says what it measured: "FROZEN X-CLIP, TimeSformer, VideoMAE and V-JEPA 2 video features ... the
same evidential probe". V-JEPA 2 went 0.7129 plain -> 0.7151 bvae -> 0.7166 bvae_real -> 0.7201
mrvae there, while the DEPLOYED V branch, whose LoRA actually trains, scores far above that probe.
So EXP4B tested a crippled encoder, exactly as arms 1 and 2 did on the image side. The ruling was
sound on the evidence it had; the evidence could not answer the question. Training the adapters
under the residual objective is a different experiment, not a re-run of a negative one.

THE OPERATORS, and why `order` is the one worth running first:

  order   NO VAE. f_c is the SAME projection applied to the SHUFFLED-frame feature, so f_a is the
          part of the ordered embedding orthogonal to its order-free counterpart -- literally what
          frame ORDER adds. This is the temporal-native off-manifold: the manifold it subtracts is
          defined by the data rather than learned, which is the thing a beta-VAE cannot do for a
          video. It costs a second encoder forward per batch, for the shuffled clip.
  bvae    the CLIP recipe transplanted, for comparison with the image-side arms.
  mrvae   DISCERN's FiLM multi-rate VAE; the residual view reads [residual (64); 5-point rate
          response] = 69-d.

`order` CARRIES ITS OWN FALSIFICATION TEST, and `--null-perm` runs it. With the identity
permutation the "shuffled" clip IS the ordered clip, so f_c == f_t and the residual f_a is ZERO by
construction. The residual head then sees nothing and MUST score at chance. If it scores above
chance, evidence is reaching the residual arm by some path other than frame order -- a leak in the
head, not a finding -- and nothing the real run reports can be believed. This control runs first.

(The obvious alternative control was `order` on P, which today's dense pilot showed is
order-invariant: mean |dp| under permutation V 0.0856, M 0.1084, P 0.0014. It is not available
here -- P is Dr Kutub's separate trainer and is not in this file's MODELS table -- and the
identity-permutation control is sharper anyway, because it forces f_a to zero exactly rather than
relying on an encoder happening to be insensitive.)

HOW IT IS BUILT. `train_branch.py` is the deployed trainer and is NOT copied here: its clip loader,
arm-B augmentation, sampler, scheduler, development table, selection rule and export are the
reference and must stay the reference. This file patches exactly two things into it, in the manner
of `train_p_dense.py`:

    head   Sequential(LayerNorm, Linear, GELU, Linear)  ->  TemporalResidualHead(d, operator)
    loss   EDL(head(z))                                 ->  head.loss(...) with the VAE/alignment terms

`forward` returns the RESIDUAL arm's evidence, so `opinions()` exports the off-manifold view and the
selection rule follows it without the selection code being touched at all -- the fused arm would
pick a checkpoint that is good because its plain temporal half is good, which is the branch we
already deploy as V.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
sys.path[:0] = [str(ROOT / "analysis/tbiom/temporal"), str(ROOT / "analysis/tbiom")]

import train_branch as TB                                          # noqa: E402
from residual_head import TemporalResidualHead                     # noqa: E402

#: Set by install(); the patched loss reads the live head and the stashed forward output from here.
STATE: dict = {}


def pooled(model, px):
    """The clip embedding, with `train_branch`'s per-family pooling and nothing else.

    Copied from `Branch.forward` deliberately: the pooling is the one place where the families
    genuinely differ, and a wrapper that called the original forward would get the HEAD's output
    back instead of the embedding we need to hand to the residual transform.
    """
    name = model.name
    if name.startswith("vjepa2"):
        return model.backbone(pixel_values_videos=px, skip_predictor=True).last_hidden_state.mean(1)
    if name == "xclip":
        B, T_ = px.shape[:2]
        cls = model.backbone.visual_projection(
            model.backbone.vision_model(pixel_values=px.flatten(0, 1))[1]).view(B, T_, -1)
        return model.backbone.mit(cls)[1]
    if name == "timesformer":
        return model.backbone(pixel_values=px).last_hidden_state[:, 0]
    return model.backbone(pixel_values=px).last_hidden_state.mean(1)


def install(operator: str, arm: str = "r", null_perm: bool = False) -> None:
    """Patch the residual head and its loss into `train_branch`, leaving everything else alone."""
    d_of = {k: v[1] for k, v in TB.MODELS.items()}
    orig_init = TB.Branch.__init__

    def init(self, name, lora=False):
        orig_init(self, name, lora=lora)
        self.head = TemporalResidualHead(d_of[name], operator)
        self.om_operator = operator

    def forward(self, px):
        z = pooled(self, px)
        if operator == "order":
            # The order operator's manifold is the SHUFFLED clip, so it needs a second forward.
            # The permutation is drawn per batch, not per sample, which matches how the shuffle
            # control in `temporal_controls.py` defines an order-free view of the same clip.
            perm = (torch.arange(px.shape[1], device=px.device) if null_perm
                    else torch.randperm(px.shape[1], device=px.device))
            out = self.head(z.float(), pooled(self, px[:, perm]).float())
        else:
            out = self.head(z.float())
        STATE["last"], STATE["head"] = out, self.head
        return out[arm] - 1.0                      # evidence; opinions() rebuilds alpha = e + 1

    TB.Branch.__init__, TB.Branch.forward = init, forward

    def loss(_ev, y, _K, ep, epochs):
        # train_branch calls this as evidential_loss_discern_ext(ev, y, 2, ep, EPOCHS).mean(); the
        # real loss needs the whole head output, not just the arm that was returned for scoring,
        # so it is read back from STATE. `.mean()` on the scalar below is a no-op.
        return STATE["head"].loss(STATE["last"], y, ep, epochs)

    TB.evidential_loss_discern_ext = loss


def smoke(n: int) -> None:
    """Shrink every split so the WHOLE trainer runs in minutes, end to end.

    The failures this campaign has had were all in code that runs AFTER an epoch -- the development
    merge, its coverage assert, the test exports -- so a smoke test that stops at the first batch
    proves nothing. This truncates the clip lists instead of short-circuiting the loop, so every one
    of those stages still executes on real data.
    """
    TB.EPOCHS = 1
    orig = TB.Clips.__init__
    # ONLY the training split and the one test export are truncated. The DEVELOPMENT sets are left
    # whole on purpose: `train_branch` asserts that every development video got an opinion, and a
    # thinned dev split fails that assert for a reason that has nothing to do with the code under
    # test -- which is exactly what the first version of this smoke did. Scoring full dev is also
    # the bulk of a real epoch, so this rehearses the expensive path rather than skipping it.
    keep_whole = set(TB.DEV)

    def init(self, ds, *args, **kw):
        orig(self, ds, *args, **kw)
        if ds not in keep_whole and len(self.items) > n:
            step = max(1, len(self.items) // n)
            self.items = self.items[::step][:n]

    TB.Clips.__init__ = init
    TB.TESTS = TB.TESTS[:1]          # one test export, so that path runs without costing an hour


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", default="order", choices=["order", "bvae", "bvae_real", "mrvae"])
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="end-to-end rehearsal: N clips per split, 1 epoch, 1 test export")
    ap.add_argument("--null-perm", action="store_true",
                    help="order operator only: identity permutation, so f_a is zero by construction "
                         "and the residual arm MUST score at chance. The falsification control.")
    ap.add_argument("--arm", default="r", choices=["r", "t", "fused"],
                    help="which arm is exported and selected on; r is the off-manifold view")
    a, rest = ap.parse_known_args()
    if a.null_perm and a.operator != "order":
        ap.error("--null-perm is only meaningful for the order operator")
    install(a.operator, a.arm, a.null_perm)
    if a.smoke:
        smoke(a.smoke)
    # The run name has to carry the operator or two operators overwrite each other's directory.
    suffix = next((rest[i + 1] for i, x in enumerate(rest) if x == "--run-suffix"), "")
    rest = [x for i, x in enumerate(rest)
            if x != "--run-suffix" and (i == 0 or rest[i - 1] != "--run-suffix")]
    sys.argv = [sys.argv[0]] + rest + [
        "--run-suffix", f"{suffix}_om{a.operator}" + ("_nullperm" if a.null_perm else "")
        + ("_smoke" if a.smoke else "")]
    print(f"OM-temporal: operator {a.operator}, exported arm '{a.arm}', "
          f"argv {' '.join(sys.argv[1:])}", flush=True)
    TB.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
