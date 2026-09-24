#!/usr/bin/env python3
"""Render the two requested tables from EXP_ACQUISITION_INTERVALS.json and EXP_VIDEO_VIEW_TABLE.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

R = Path("tbiom/reports")
SETS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "KoDF", "CDFv3 reenactment"]
SHORT = {"CDFv2": "CDFv2", "CDFv3": "CDFv3", "DFD": "DFD", "DFDC": "DFDC", "DFDCP": "DFDCP",
         "DFEval24": "DFEval24", "KoDF": "KoDF", "CDFv3 reenactment": "CDFv3-reen"}
ORDER1 = [("no extra clip (k=0)", "no extra clip", 0),
          ("random extra clip (k=1)", "random extra clip", 1),
          ("fixed window w0 (k=1)", "fixed single window $w_0$", 1),
          ("evenly spaced extra clips w0+w2 (k=2)", "evenly spaced extra clips ($w_0{+}w_2$)", 2),
          ("quality-based choice (k=1)", "quality-based choice", 1),
          ("learned one-step choice (k=1)", "learned one-step choice", 1),
          ("learned two-step choice (k=2)", "learned two-step choice", 2),
          ("oracle, true label (k=1)", "oracle using the true label", 1),
          ("oracle, true label (k=2)", "oracle using the true label", 2),
          ("oracle, scores shuffled from other videos (k=1)",
           "oracle using scores shuffled from other videos", 1)]


def main():
    A = json.load(open(R / "EXP_ACQUISITION_INTERVALS.json"))
    V = json.load(open(R / "EXP_VIDEO_VIEW_TABLE.json"))
    S = A["splits"]
    m = lambda pol, k: float(np.mean([s["policies"][pol][k] for s in S]))
    ci = lambda pol, i: float(np.mean([s["policies"][pol]["ci"][i] for s in S]))
    nv = [s["n_videos"] for s in S]

    L = ["# Two requested result tables", "",
         "Built only from cached outputs by `analysis/tbiom/paper/two_tables.py`. Nothing was trained. Where a "
         "number was never computed it is marked **not computed**, never estimated.", "",
         "---", "", "## Table 1 — acquisition study", "",
         "**Split.** The five source-grouped v3.1 **selection folds** (the development pool: VALmix, VALmix-v2 and "
         f"FF++ val). Videos per fold: {', '.join(str(x) for x in nv)} (mean {np.mean(nv):.0f}); "
         f"{S[0]['n_source_groups']} source groups. Every row is scored on the same videos with the same "
         "extra-clip budget as every other row of its budget.", "",
         "**Fusion and threshold.** The deployed DS-R fusion with weights fitted on each fold's fitting half, and "
         "$\\tau$ = EER of the no-acquisition score on that fold's FF++ val videos. Acquired windows are aggregated "
         "within a branch (the `conf` rule) before fusion, so overlapping clips are never treated as independent "
         "sources.", "",
         "| policy | extra clips / video | AUROC | $\\Delta$ vs no extra clip | 95% CI on $\\Delta$ | FPR @ $\\tau$ |",
         "|---|---:|---:|---:|---|---:|"]
    for key, lab, k in ORDER1:
        L.append(f"| {lab} | {k} | {m(key, 'auroc'):.4f} | {m(key, 'delta'):+.4f} | "
                 f"[{ci(key, 0):+.4f}, {ci(key, 1):+.4f}] | {m(key, 'fpr'):.3f} |")
    L += ["", "Intervals are a paired source-group bootstrap, 2,000 draws per fold, resampling source groups with "
          "all of their linked videos, then averaged over the five folds. **These intervals did not exist in the "
          "original report** (`EXP_ACQUISITION_PILOT.md`, which gives point estimates only); they were computed here "
          "by reproducing the per-video policy scores from the cached window outputs.", "",
          "Reproduction check against the published report: eight of the ten policies match to $0.00000$ AUROC. The "
          "two that differ are the only stochastic ones — the random-window policy (the published value averages "
          "several random draws, this one uses a single draw, difference $0.0008$) and the permuted-window null "
          "oracle (a fresh permutation, difference $0.0019$).", "",
          "**Test sets: not computed.** The acquisition study was run on the development selection folds only. No "
          "acquisition policy was ever scored on CDFv2, CDFv3, DFD, DFDC, DFDCP, DFEval24, FFIW, KoDF, IDForge, "
          "DFDM or DF40, so there are no test-set rows to add.", "",
          "**Reading.** Every deployable policy is *worse* than not acquiring anything. The only rows that gain are "
          "the two oracles, and the null oracle — which picks among candidate scores borrowed from other videos and "
          "therefore cannot know anything about the video it is scoring — gains more (+0.0807) than the true-label "
          "oracle (+0.0314). Most of the apparent headroom is selection on labels among four correlated scores, not "
          "information a policy could exploit.", "",
          "---", "", "## Table 2 — video view contribution", "",
          f"**Fusion rule.** {V['fusion_rule']}. **Folds.** {V['folds']}, with the reliability weights refitted "
          "independently for every row by the same grid and objective, and $\\tau$ = EER on FF++ val computed per "
          "row. Only the view set, or the backbone supplying the video view, changes between rows.", "",
          "**Sets.** The six of the configuration slide (CDFv2, CDFv3, DFD, DFDC, DFDCP, DFEval24) plus KoDF and the "
          "reenactment subset of CDFv3. If the six intended on that slide were a different set, the rows are "
          "unaffected -- only the columns shown would change.", "",
          "**Why three FPR columns are identical.** CDFv2 and CDFv3 are scored against the *same 178 real videos* "
          "(the identifiers differ only by the dataset prefix; the per-video branch opinions match to $1.8$e$-7$), "
          "and the CDFv3 reenactment slice keeps all of CDFv3's reals. FPR is a property of the real videos alone, "
          "so it is necessarily identical across those three columns. Their AUROCs differ because the fake sets "
          "differ.", "", "### 2a. Video AUROC", "",
          "| row | video backbone | seeds | " + " | ".join(SHORT[s] for s in SETS) + " |",
          "|---|---|---|" + "---:|" * len(SETS)]

    def row(name, r, lab, bb):
        cells = []
        for s in SETS:
            d = r["per_set"].get(s)
            cells.append(f"{d['auroc']:.4f}" if d else "n/c")
        return f"| {lab} | {bb} | {r['seeds']} | " + " | ".join(cells) + " |"

    RN = [("APP+OM only", "Appearance + Off-manifold only", "--"),
          ("APP+OM+face-prior", "\\quad + face-prior view", "--"),
          ("T+APP+OM [V-JEPA 2 ViT-L, LoRA, B-augmented (reference)]",
           "\\quad + V-JEPA 2 video view (**reference**)", "V-JEPA 2 ViT-L, LoRA, B-aug"),
          ("V-JEPA 2 alone", "V-JEPA 2 alone", "V-JEPA 2 ViT-L, LoRA, B-aug"),
          ("T+APP+OM [X-CLIP, LoRA]", "three-view, X-CLIP as video view", "X-CLIP, LoRA"),
          ("T+APP+OM [VideoMAE, LoRA]", "three-view, VideoMAE as video view", "VideoMAE, LoRA"),
          ("T+APP+OM [TimeSformer, LoRA]", "three-view, TimeSformer as video view", "TimeSformer, LoRA"),
          ("T+APP+OM [V-JEPA 2 ViT-L, LoRA, NOT B-augmented]",
           "three-view, V-JEPA 2 *without* B-augmentation", "V-JEPA 2 ViT-L, LoRA"),
          ("T+APP+OM, reference with SHUFFLED frames", "reference with **shuffled** video frames",
           "V-JEPA 2 ViT-L, LoRA, B-aug")]
    for key, lab, bb in RN:
        if key in V["rows"]:
            L.append(row(key, V["rows"][key], lab, bb))
    L += ["", "### 2b. FPR at the frozen threshold", "",
          "| row | " + " | ".join(SHORT[s] for s in SETS) + " |", "|---|" + "---:|" * len(SETS)]
    for key, lab, _bb in RN:
        if key in V["rows"]:
            r = V["rows"][key]
            cells = [f"{r['per_set'][s]['fpr']:.3f}" if s in r["per_set"] else "n/c" for s in SETS]
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
    ref = V["rows"]["T+APP+OM [V-JEPA 2 ViT-L, LoRA, B-augmented (reference)]"]["per_set"]
    sh = V["rows"].get("T+APP+OM, reference with SHUFFLED frames", {}).get("per_set", {})
    L += ["", "`n/c` = **not computed**. The shuffled-frame control was exported for the reference branch on "
          f"{', '.join(V['shuffled_available_for'])} only; the remaining columns of that row were never produced and "
          "are not estimated here.", "",
          "**Seeds.** Every row uses spatial seed 42 (the deployed warm+gate checkpoint) and, where a video view is "
          "present, temporal seed 42. No row averages over seeds; there is one seed behind each.", "",
          "**A caveat on the backbone-swap rows.** The reference video view is trained with B-augmentation; the "
          "X-CLIP, VideoMAE and TimeSformer branches are not. Those three rows therefore differ from the reference "
          "in two things at once, not one. The row without B-augmentation is included as the matched anchor: "
          "comparing the swapped backbones against *it* isolates the backbone, and comparing the two V-JEPA 2 rows "
          "against each other isolates the augmentation.", ""]
    if sh:
        a = lambda d, k: d[k]["auroc"]
        L += ["**Shuffling the frames costs almost nothing, and that is a real qualification.** On the sets where "
              "the control exists the fused reference moves "
              + "; ".join(f"{a(ref, k):.4f} $\\to$ {a(sh, k):.4f} on {SHORT[k]} ({a(sh, k) - a(ref, k):+.4f})"
                          for k in sh)
              + ". Scored alone the video view moves $0.9263 \\to 0.9271$ on CDFv3 and $0.9656 \\to 0.9633$ on "
                "KoDF, i.e. within noise in both directions, even though the individual probabilities move a long "
                "way (mean $|\\Delta p|$ of $0.127$ and $0.041$).", "",
              "The reference video view samples 16 frames spread over the whole video, and at that sampling it is "
              "largely order-insensitive: it is reading multi-frame evidence, not frame order. The same pattern was "
              "measured independently at branch level in `EXP_TEMPORAL_SAMPLING.md` (sparse sampling: $0.8934$ "
              "ordered against $0.8873$ shuffled), where only the *dense* 32-consecutive-frame configurations showed "
              "a large order effect ($0.9045 \\to 0.8606$ pooled, $0.9049 \\to 0.6890$ with a GRU head). Any claim "
              "that this system exploits temporal *order* is therefore not supported by these numbers; what the "
              "evidence supports is that it exploits information spread across frames.", ""]
    (R / "EXP_TWO_TABLES.md").write_text("\n".join(L) + "\n")
    print("wrote", R / "EXP_TWO_TABLES.md")


if __name__ == "__main__":
    main()
