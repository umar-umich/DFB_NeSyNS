#!/usr/bin/env python3
"""Cross-manipulation results: our system and our trained DiCoME on every DF40 face-swap generator.

    OMP_NUM_THREADS=8 python analysis/tbiom/df40_analysis.py

Each generator is scored against the genuine pool DF40 ships for its source. Our system fuses the three reference
views with DS-R, the weights fitted per split on the development folds exactly as everywhere else, so the numbers are
comparable with every other table; the logistic comparator is reported beside it. DiCoME is our own trained
checkpoint (the same one in the baseline matrix), replayed through its own two-view DS fusion, so both sides are
scored on identical videos with identical aggregation.

Two generators DiCoME's published cross-manipulation table does not list — the classical graphics **FaceSwap** and
**DeepFaceLab** — are included and marked, together with the FF++-sourced version of every generator.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "temporal")]
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import finish_line as F  # noqa: E402
import symbolic_overlay as SO  # noqa: E402
from fusion_combos import attach  # noqa: E402
from video_id import video_id  # noqa: E402

S23, R, LOG = Path("logs/tbiom/stage23"), Path("tbiom/reports"), Path("logs/tbiom/df40")
BR = Path("logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/opinions")
VIEWS = ["T", "APP", "OM"]
NOT_LISTED = {"faceswap", "deepfacelab"}          # absent from DiCoME's published cross-manipulation table
PRETTY = {"blendface": "BlendFace", "e4s": "e4s", "facedancer": "FaceDancer", "faceswap": "FaceSwap",
          "fsgan": "FSGAN", "inswap": "InSwapper", "mobileswap": "MobileFaceSwap", "simswap": "SimSwap",
          "uniface": "UniFace", "deepfacelab": "DeepFaceLab", "danet": "DaNet", "facevid2vid": "FaceVid2Vid",
          "fomm": "FOMM", "hyperreenact": "HyperReenact", "lia": "LIA", "mcnet": "MCNet", "MRAA": "MRAA",
          "one_shot_free": "One-Shot Free-View", "pirender": "PIRender", "sadtalker": "SadTalker", "tpsm": "TPSM"}


def ours_table(name):
    d = A.load_frames(S23 / f"ours_{name}.csv")
    t = A.video_table(d, [1.0])
    return attach(t, [BR / f"{name}.csv"])


def dicome_score(name):
    """DiCoME's own two-view DS fusion: per-frame opinions averaged per video, then fused."""
    d = pd.read_csv(S23 / f"dicome_{name}.csv")
    d["v"] = video_id(d["key"])
    g = d.groupby("v", as_index=False).agg(p_semantic=("p_semantic", "mean"), u_semantic=("u_semantic", "mean"),
                                           p_artifact=("p_artifact", "mean"), u_artifact=("u_artifact", "mean"),
                                           y=("label", "max"))
    SO.BRANCHES[:] = ["semantic", "artifact"]; s = SO.fuse(g); SO.BRANCHES[:] = ["semantic", "artifact", "fsvfm"]
    return g, s


def main():
    names = [n for n in (LOG / "datasets.txt").read_text().split() if (S23 / f"ours_{n}.csv").exists()]
    dev = D.load("vjepa2_lora_B_seed42")[0]
    rules_per_split = []
    for k, Fd, _Sd in D.splits(dev):
        rules, params, w = F.rules_for(Fd, VIEWS); rules_per_split.append((rules, w))
    res = {}
    for n in names:
        t = ours_table(n); y = t.y.to_numpy()
        auc = lambda s: A.fast_auc(s[y == 0], s[y == 1])
        r = {"videos": int(len(t)), "fake": int((y == 1).sum()), "real": int((y == 0).sum()),
             "ours_dsr": float(np.mean([auc(rules["DS-R"](t)) for rules, _ in rules_per_split])),
             "ours_lr": float(np.mean([auc(rules["LR"](t)) for rules, _ in rules_per_split])),
             **{f"view_{v}": auc(t[f"p_{v}"].to_numpy()) for v in VIEWS}}
        if (S23 / f"dicome_{n}.csv").exists():
            g, s = dicome_score(n); yd = g.y.to_numpy()
            r["dicome"] = A.fast_auc(s[yd == 0], s[yd == 1])
        res[n] = r
        print(f"{n}: ours {r['ours_dsr']:.4f}" + (f", DiCoME {r['dicome']:.4f}" if "dicome" in r else ""), flush=True)
    json.dump(res, open(R / "EXP_DF40_CROSS_MANIPULATION.json", "w"), indent=1, default=float)

    meta = json.load(open(LOG / "meta.json"))

    def rows(src, family):
        return sorted((meta[n]["method"], r) for n, r in res.items()
                      if meta[n]["source"] == src and meta[n]["family"] == family)

    L = ["# Cross-manipulation results — DF40 face-swap generators", "", __doc__.split("\n\n", 1)[1].strip(), "", F.CITE, ""]
    for family, src, title in (("face swap", "cdf", "Face swap — Celeb-DF-sourced"),
                               ("face swap", "ff", "Face swap — FaceForensics++-sourced"),
                               ("reenactment", "cdf", "Reenactment — Celeb-DF-sourced"),
                               ("reenactment", "ff", "Reenactment — FaceForensics++-sourced")):
        rs = rows(src, family)
        if not rs:
            continue
        L += [f"## {title} generators", "",
              "| generator | videos (real / fake) | **ours, DS-R** | ours, logistic | DiCoME (our training) | Δ ours − DiCoME | temporal view | appearance | off-manifold |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m, r in rs:
            star = " ★" if m in NOT_LISTED else ""
            dic = r.get("dicome")
            L.append(f"| {PRETTY.get(m, m)}{star} | {r['real']} / {r['fake']} | **{r['ours_dsr']:.4f}** | {r['ours_lr']:.4f} | "
                     + (f"{dic:.4f} | {r['ours_dsr'] - dic:+.4f} | " if dic is not None else "TODO(run) | TODO(run) | ")
                     + f"{r['view_T']:.4f} | {r['view_APP']:.4f} | {r['view_OM']:.4f} |")
        ours = [r["ours_dsr"] for _, r in rs]; dics = [r["dicome"] for _, r in rs if "dicome" in r]
        L.append(f"| **mean over {len(rs)} generators** | | **{np.mean(ours):.4f}** | "
                 f"{np.mean([r['ours_lr'] for _, r in rs]):.4f} | " + (f"{np.mean(dics):.4f} | {np.mean(ours) - np.mean(dics):+.4f} | " if dics else "— | — | ")
                 + f"{np.mean([r['view_T'] for _, r in rs]):.4f} | {np.mean([r['view_APP'] for _, r in rs]):.4f} | "
                 f"{np.mean([r['view_OM'] for _, r in rs]):.4f} |")
        L.append("")
    L += ["★ = not listed in DiCoME's published cross-manipulation table. Every generator here is scored against the "
          "genuine pool DF40 ships for its source (`DF40-cdf-real`, `DF40-ff-real`, and DeepFaceLab's own pool), so a "
          "generator's AUROC is never computed against another source's genuine videos.", "",
          "Our rows are means over the five fitted splits (the fusion weights differ per split); DiCoME is a single "
          "checkpoint, so its rows carry no spread. Both sides use per-frame opinions averaged per video and then "
          "fused — the same aggregation as every other table. DF40 was never trained on by either system; its "
          "generators are unseen manipulations built on FF++ and Celeb-DF source videos, which both systems have seen "
          "as training or development content.", "", F.FOOT]
    (R / "EXP_DF40_CROSS_MANIPULATION.md").write_text("\n".join(L) + "\n")
    print("wrote", R / "EXP_DF40_CROSS_MANIPULATION.md")


if __name__ == "__main__":
    main()
