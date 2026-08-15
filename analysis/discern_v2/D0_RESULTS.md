# D0 — DISCERN v1 reproduction: reference numbers

Two runs of the same system, differing **only** in which datasets the trainer was allowed to
see when choosing the checkpoint. Both are reported, because they answer different questions
and the field does not agree on which to use.

Config: `nesy_defake_ablation4_ccv.yaml` (leaked selection) /
`nesy_defake_ablation4_ccv_d0.yaml` (leak-free). Verified: the two configs differ in
`test_dataset` and nothing else.

---

## Run A — leaked selection (standard practice)

`test_dataset` = all eight sets, so early stopping saw every OOD set. Early stop at epoch 24,
best at epoch 4, patience 20. Checkpoint:
`logs/train/nesy_defake_ablation4_ccv_2026-08-15-02-38-14/best_avg.pth`.
Full log: `results/discern_v2/D0_leaked_selection_run.log` (gitignored).

The trainer prints **two different tables**, and the difference matters:

| dataset | **A1: single checkpoint** (best-average epoch) | **A2: per-dataset best epoch** |
|---|---|---|
| FaceForensics++ | 0.9372 | 0.9509 |
| DeepFakeDetection | 0.8850 | 0.8851 |
| Celeb-DF-v1 | 0.9214 | 0.9214 |
| Celeb-DF-v2 | 0.8872 | 0.8880 |
| Celeb-DF-v3 | 0.6893 | 0.6893 |
| DFDC | 0.8072 | 0.8110 |
| DFDCP | 0.8222 | 0.8424 |
| UADFV | 0.9824 | 0.9855 |
| **average AUC** | **0.8665** | — |

Other averages at the best-average epoch: acc 0.7907, EER 0.1993, AP 0.9477,
**video AUC 0.9133**.

**A1 is the number to report; A2 is not.** A2 ("Each dataset best metric") takes each
dataset's best epoch independently, so it describes a *different model per dataset* — no
single checkpoint achieves that row. It is the trainer's diagnostic output, not a result.
A1 is one checkpoint evaluated everywhere, which is what a paper table means.

## Run B — leak-free selection

`test_dataset` = `[FaceForensics++, Celeb-DF-v2]` only; the other six held out entirely and
evaluated post-hoc. This follows the protocol commit `1668901` already established for
`config/detector/full/*`. Checkpoint directory:
`logs/train/nesy_defake_ablation4_ccv_d0_2026-08-15-16-34-38`.

`TODO(run)` — in progress, launched 2026-08-15 16:34.

Note that Celeb-DF-v2 is a **validation** set here, not a clean test set — the same caveat
`1668901` records.

---

## Which to use

Selecting the best epoch on test data is common in the deepfake-detection literature, and
many published numbers this work will be compared against were produced that way. Run A is
therefore the *comparable* number, and there is a legitimate case for leading with it.

The catch is specific to this paper rather than general: DISCERN's contribution is
**reliability**, and Table 4 in `README-discern-v2-integration.md` already records that
in-domain FF++ validation cannot rank cross-domain detectors — with the note that a leak-free
OOD selection protocol is *a candidate contribution in its own right*. A reliability paper
whose own checkpoints were picked by peeking at the test sets invites exactly the reviewer
question it is least well placed to answer.

**Reporting both is the strongest position, and costs nothing now that both runs exist:**

- **Run A** as the headline, for comparability with prior work, labelled as
  test-set-selected in the caption.
- **Run B** alongside, as the leak-free number.
- **The gap between them** as evidence for the selection-protocol argument the integration
  README already wants to make. If the gap is small, Run A is vindicated and the concern is
  dismissed with data. If it is large, that *is* the contribution.

Either way the D-ladder itself (D1–D5) should be measured against **one** protocol
consistently — mixing a leaked D0 with leak-free D1–D3 would put the deltas on different
footings and make every comparison ambiguous.
