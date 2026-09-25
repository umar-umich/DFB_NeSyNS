#!/usr/bin/env python3
"""Task 8 of the leakage audit: the refit control. Cached opinions only, CPU only, no GPU.

    OMP_NUM_THREADS=8 python audit/leakage/refit_control.py

Tasks 1-5 found no video-level leak, but they did find that the development pool shares Celeb-DF **source videos and
identities** with CDFv2, CDFv3, DFDM and the DF40 evaluation sets. This control asks the only question that matters
for the reported numbers: if every development video whose source video or identity also appears in an evaluation set
is removed, and the trust weights and threshold are refitted on what is left, do the reported results move?

The clean subset is built at the strictest level available per dataset family: identity where the dataset encodes it,
source video otherwise.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path[:0] = [str(Path("analysis/tbiom/pcr6p").resolve()), str(Path("analysis/tbiom/grouped").resolve()),
                str(Path("analysis/tbiom").resolve()), str(Path("analysis/tbiom/temporal").resolve()),
                str(Path(__file__).resolve().parent)]
import numpy as np  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import arms as AR  # noqa: E402
import ops_pcr6p as P  # noqa: E402
import keys as K  # noqa: E402
from health import eer  # noqa: E402

OUT = Path("audit/leakage")
EXT = Path("/data/umar/Repos/DISCERN_Ext")
SPL = EXT / "eval_adaptation/data/splits"
TEN = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]


def eval_keys():
    """Every source video and identity that appears in any of the ten evaluation sets or in DF40 evaluation."""
    src, ids = set(), set()
    for ds in TEN + ["DF40"]:
        d = SPL / ds
        for p in sorted(d.glob("*.txt")) if d.exists() else []:
            _v, s, i = K.manifest_keys(p)
            src |= s; ids |= i
    return src, ids


def dev_video_keys():
    """Map each development-pool video id (as the cached tables name it) to its source/identity keys."""
    out = {}
    for name in ("VALmix", "VALmixV2", "FFpp_val"):
        d = SPL / name
        for p in sorted(d.glob("*.txt")) if d.exists() else []:
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                v = K.video_of(line)
                s, i = K.sources_and_identity(K.origin_of(line), v)
                e = out.setdefault(v, (set(), set()))
                e[0].update(s); e[1].update(i)
    return out


def main():
    esrc, eids = eval_keys()
    dvk = dev_video_keys()
    dev, _dproc, tests = D.load("vjepa2_lora_B_seed42")

    # a development video is contaminated if any of its source videos or identities appears in an evaluation set
    def contaminated(v):
        stem = str(v).split("/")[-1]
        s, i = dvk.get(stem, (set(), set()))
        return bool((s & esrc) or (i & eids))

    flag = np.array([contaminated(v) for v in dev.v])
    matched = sum(1 for v in dev.v if str(v).split("/")[-1] in dvk)
    print(f"development pool {len(dev)} videos | matched to a manifest key {matched} | "
          f"contaminated {int(flag.sum())} | clean {int((~flag).sum())}")
    if matched == 0:
        print("FLAG: no development video id could be matched to a manifest key; refit not attempted")
        (OUT / "refit_control.md").write_text(
            "# Task 8 — refit control\n\n**FLAG.** No development-pool video id in the cached tables could be "
            "matched to a split-manifest key, so the clean subset could not be constructed. The cached tables name "
            "videos differently from the manifests; reconciling the two naming schemes is the missing item.\n")
        return
    if not flag.any():
        (OUT / "refit_control.md").write_text(
            "# Task 8 — refit control\n\nNo development-pool video failed Tasks 1, 3, 4 or 5, so the clean subset "
            "equals the full pool and the refit is skipped, as the brief directs.\n")
        print("clean subset equals the full pool; refit skipped")
        return

    rows = []
    for label, keep in (("original", np.ones(len(dev), bool)), ("refit on clean subset", ~flag)):
        sub = dev[keep].reset_index(drop=True)
        per = {}
        for k, Fd, Sd in D.splits(sub):
            cl = A.cells(Fd)
            w = AR.fit_w(Fd, "PCR6P-R", cl)
            p_fit = P.p_of(AR.fuse(Fd, "PCR6P-R", w))
            ff = np.where(Fd.source.to_numpy() == "FFpp_val")[0]
            tau = float(eer(Fd.y.to_numpy()[ff], p_fit[ff])[1]) if len(ff) else 0.5
            for tn in TEN:
                T = tests[tn]; y = T.y.to_numpy()
                p = P.p_of(AR.fuse(T, "PCR6P-R", w))
                pred = (p >= tau).astype(int)
                d = per.setdefault(tn, {"auroc": [], "fpr": [], "tpr": []})
                d["auroc"].append(A.fast_auc(p[y == 0], p[y == 1]))
                d["fpr"].append(float((pred[y == 0] == 1).mean()))
                d["tpr"].append(float((pred[y == 1] == 1).mean()))
        for tn, d in per.items():
            rows.append({"version": label, "dataset": tn, "n_dev_videos": len(sub),
                         **{m: round(float(np.mean(d[m])), 6) for m in ("auroc", "fpr", "tpr")}})
    with open(OUT / "refit_control.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    o = {r["dataset"]: r for r in rows if r["version"] == "original"}
    n = {r["dataset"]: r for r in rows if r["version"] != "original"}
    L = ["# Task 8 — refit control", "",
         f"Development pool: {len(dev)} videos, of which **{int(flag.sum())}** share a Celeb-DF source video or "
         f"identity with an evaluation set and are removed, leaving **{int((~flag).sum())}**. Trust weights and the "
         "threshold are refitted on the clean subset with the same grid, folds, objective and threshold rule; the "
         "evaluation sets and the cached opinions are untouched.", "",
         "| dataset | AUROC orig | AUROC refit | Δ | FPR orig | FPR refit | Δ | TPR orig | TPR refit | Δ |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tn in TEN:
        a, b = o[tn], n[tn]
        L.append(f"| {tn} | {a['auroc']:.4f} | {b['auroc']:.4f} | {b['auroc'] - a['auroc']:+.4f} | "
                 f"{a['fpr']:.3f} | {b['fpr']:.3f} | {b['fpr'] - a['fpr']:+.3f} | "
                 f"{a['tpr']:.3f} | {b['tpr']:.3f} | {b['tpr'] - a['tpr']:+.3f} |")
    d = [n[t]["auroc"] - o[t]["auroc"] for t in TEN]
    L += ["", f"Mean AUROC change **{np.mean(d):+.4f}**, largest single move **{max(d, key=abs):+.4f}** "
          f"({TEN[int(np.argmax(np.abs(d)))]}).", ""]
    (OUT / "refit_control.md").write_text("\n".join(L) + "\n")
    print("wrote refit_control.md / .csv")
    for tn in TEN:
        print(f"  {tn:10s} {o[tn]['auroc']:.4f} -> {n[tn]['auroc']:.4f}  ({n[tn]['auroc'] - o[tn]['auroc']:+.4f})")


if __name__ == "__main__":
    main()
