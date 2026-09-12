# FreqBlender source audit

Audited 2026-09-12 against the local checkout, not against the paper or the brief's summary.

| field | value |
|---|---|
| repository | https://github.com/OUC-VAS/FreqBlender |
| local path | `/data/umar/Repos/FreqBlender` (pre-existing clone, sibling of `DFB_NeSyNS`) |
| revision | `7c8bc7f5a653a068d6dadcd2e64996f9e2805fc7` — **matches the pinned revision** |
| working tree | clean, no local edits (reused, not re-cloned) |
| **LICENSE** | **absent from the repository** |

The missing licence is recorded because it is load-bearing for the brief's instruction to
"preserve upstream attribution and applicable license notices when importing code." There are no
notices to preserve, and no stated grant. Importing code into our tree should therefore be
minimised in favour of *calling* the upstream checkout in place, with provenance recorded here.

## Paper-versus-code claims, independently verified

Every row below was checked against the source at the pinned revision. All of the brief's
claims held.

| component | brief's claim | verified in source |
|---|---|---|
| frequency masks | no final sigmoid/softmax | `net/aeNet/conv_autoencoder_pixelshuffle.py:55` — `return M1, M2, M3`, unconstrained |
| facial fidelity | clipped cosine, not squared L2 | `utils/loss_function.py:79` — `clip(dot/(norm*norm), 0, 1)` |
| authenticity | linear penalties on softmax class-1 | `utils/loss_function.py:66-71` — `softmax(1)[:, 1]`, so **index 1 is fake** |
| image scale | DCT on [0,1], losses on [0,255] | `utils/loss_function.py:33` — `clip(idct_2d(...)*255, 0, 255)` |
| schedule | batch 16, 100 epochs, StepLR(5, 0.5) | `config/base.json` epoch 100, batch 16, lr 1e-4; `main.py:59` StepLR step 5 gamma 0.5 |
| FPNet loader | reads FF++ Deepfakes, SBI commented out | `utils/dataloader.py:31,33,46` — SBI paths commented |

**Label convention.** The public loss treats softmax index 1 as *fake*. Our project convention
is also class 1 = fake (`binary_labels`), so the mapping is direct — but it is stated here
because the brief requires it be explicit rather than assumed.

## Dependency status — arm D is blocked, arms A-C are not

| dependency | required by | status |
|---|---|---|
| `mobilefacenet.pth` + MTCNN (P/R/O Net) | facial-fidelity teacher | **present**, shipped in the repo |
| `net/resnet/checkpoints/resnet34.tar` | authenticity teacher, `main.py:50` | **MISSING** — directory does not exist |
| FPNet released weights | skipping FPNet training | **MISSING** — none in the tree |
| modified SBI files (`base.json`, `sbi.py`) | official integration | **not retrieved** |

The three missing items are hosted on Baidu Pan. The host responds (HTTP 200) but share links
of the form `pan.baidu.com/s/...?pwd=` require an authenticated account and client, so they
cannot be retrieved non-interactively from this server.

**Consequence, stated precisely.** Arm D has no route that does not pass through the
authenticity teacher: the released FPNet weights are behind the same wall, and *training* FPNet
requires `resnet34.tar` at `main.py:50`. Arm D is therefore blocked on one named artefact.

Per the brief, we do not substitute a different generator and call it a reproduction. Arms A, B
and C proceed; D waits for the weights or for a deliberate, labelled decision to train a
replacement authenticity teacher on a split-safe source — which would be an adaptation, not the
official pipeline, and must be reported as such.

## Provenance not yet establishable

The brief requires recording training-data provenance for each released checkpoint before using
it in a clean OOD experiment. Since none of the released checkpoints could be retrieved, that
audit cannot be performed. If they are later obtained, teacher and parser training exposure must
be inspected before those weights touch a formal comparison; until then they would be usable
only for a labelled implementation diagnostic.
