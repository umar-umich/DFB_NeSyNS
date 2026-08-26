# VALmix — the diverse validation split

`/data/umar/Datasets/preprocessed/dataset_json/VALmix.json`  ·  label keys `VALmix_Real` = 0, `VALmix_Fake` = 1  ·  split key `val`

Rebuilt as a video-id manifest over `/data/umar/Datasets/preprocessed/`. The video LIST comes from the original HDF5, since that is where the split was defined; the FRAMES are ours. Those pixels were verified byte-identical, so selection sees the same images while the 5 GB cross-repo dependency goes away.

| domain | videos | frames |
|---|---:|---:|
| CDFv2val | 450 | 14395 |
| DFDCPval | 450 | 12725 |
| DFEval24val | 450 | 13022 |
| **total** | **1350** | **40142** |

Real / fake: 675 / 675.

## Why this and not each corpus's own `val`

For Celeb-DF-v2 and DFDCP the shipped `val` split IS the shipped `test` split (518/518 and 652/652), and Deepfake-Eval-2024 ships no `val` at all. Selecting on those would be selecting on test. Every video here was re-checked against its corpus's test list on the FINAL manifest: zero overlap in all three domains, and the builder refuses to write otherwise.

## The `test` key is a loader artifact

`abstract_dataset` supports only the modes `train` and `test` — it raises NotImplementedError on `val` — and `load_split` constructs in test mode before it can swap splits, so a val-only manifest is unloadable in this codebase. The `val` and `test` keys therefore hold IDENTICAL content. VALmix is a validation set: it selects checkpoints and nothing else, and a number computed on it must never be reported as a test result.

## What adopting it costs

| corpus | status |
|---|---|
| Celeb-DF-v2, DFDCP, Deepfake-Eval-2024 | **domain-seen** — not strict zero-shot |
| Celeb-DF-v1, Celeb-DF-v3, DFD, DFDC, UADFV | strict zero-shot |

Any zero-shot claim in the paper belongs to the second row.

## Per group

| group | label | videos | frames | directory |
|---|---|---:|---:|---|
| `CDFv2val-Celeb-real` | real | 160 | 5116 | `Celeb-DF-v2/Celeb-real/frames` |
| `CDFv2val-Celeb-synthesis` | fake | 225 | 7200 | `Celeb-DF-v2/Celeb-synthesis/frames` |
| `CDFv2val-YouTube-real` | real | 65 | 2079 | `Celeb-DF-v2/YouTube-real/frames` |
| `DFDCPval-method_A` | fake | 216 | 6247 | `DFDCP/method_A/frames` |
| `DFDCPval-method_B` | fake | 9 | 276 | `DFDCP/method_B/frames` |
| `DFDCPval-real` | real | 225 | 6202 | `DFDCP/original_videos/frames` |
| `DFEval24val-fake` | fake | 225 | 6477 | `Deepfake-Eval-2024/frames` |
| `DFEval24val-real` | real | 225 | 6545 | `Deepfake-Eval-2024/frames` |
