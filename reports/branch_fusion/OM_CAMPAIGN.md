# The off-manifold campaign beyond CLIP

Written unattended by `om_report.py`. Every number is read from a run record on disk;
nothing here is recalled from the session that set the campaign up.

## The question

The deployed Off-manifold view O is built on CLIP ViT-L/14 whose LoRA was trained WITH
the beta-VAE in the loop. The campaign asked whether that construction reproduces on
other trunks, and three successive answers were wrong for the same reason: the residual
was fitted on FROZEN features, so it could only ever be a function of an embedding that
was never asked to make it informative.

## Reference points

- frozen Y (arm 1, cached features): dev v3.1 0.8696
- A alone: dev v3.1 0.8736
- O, the CLIP off-manifold: dev v3.1 0.8252
- frozen Y vs A: +0.789
- in-loop Y16 vs A: +0.678
- O's highest phi with anything: +0.476

## Depth

PE spatial trunk, off-manifold with the encoder in the loop (full dev, residual arm):
  b8    0.7831
  b12   0.8053
  b16   0.8673
  last  0.8664
  b20   0.8613
  spread across depths: 0.0842

block 16: seed 0 0.8673, seed 1 0.8547, |difference| 0.0126
SEED NOISE (0.0126) EXCEEDS THE WHOLE DEPTH SPREAD (0.0060).
The depth ordering is not measurable at one seed per cell. No further depth run is
launched; reporting depth as indistinguishable is the honest result, and any future
depth claim needs several seeds per cell rather than another layer.

## Gates

Temporal off-manifold runs, PE-Core trunk, block 16 (development v3.1):

  ombvae_b16_cat                     off-manifold   0.8296  (epoch 9)
  ombvae_b16                         off-manifold   0.4942  (epoch 2)

P16om: ombvae_b16_cat
  GATE 1 cannot be evaluated -- its no-OM control has not finished. HELD.
P16om: ombvae_b16
  GATE 1 cannot be evaluated -- its no-OM control has not finished. HELD.

## How to read a negative here

A branch that fails gate 2 is not a weak branch; it is a DUPLICATE of one already in
the system. A+V+O was selected in all five nested folds without correlation ever being
shown to the search, so the fusion already prefers decorrelated views. A new PE-trunk
branch has to leave that cluster to be worth a row, and standalone accuracy cannot tell
you whether it did.
