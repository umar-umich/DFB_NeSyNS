#!/usr/bin/env python3
"""Task 1 of the leakage audit: split inventory. Read-only.

    OMP_NUM_THREADS=4 python audit/leakage/inventory.py

Every row cites the manifest file it was counted from. Nothing is inferred from a config name or a comment.
The deployed configuration is `DISCERN_Ext/eval_adaptation/configs/warm_dicome_gate_seed42.yaml`; the audit follows
the manifests that config names, and the manifests the temporal branch and the fusion actually read.
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

ROOT = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext")
OUT = ROOT / "audit/leakage"
SPL = EXT / "eval_adaptation/data/splits"

# label convention: a manifest whose stem names the real subset is real, everything else fake.
REAL_HINT = re.compile(r"real|Celeb-real|YouTube-real|-real$", re.I)


def read_manifest(p: Path):
    if not p.exists():
        return None
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def key_type(ds: str) -> str:
    if ds.startswith("FFpp"):
        return "video ID; FF++ fake AAA_BBB carries source videos AAA and BBB"
    if ds.startswith("CDF"):
        return "video ID; Celeb-DF idX_idY_NNNN carries identities idX and idY"
    if ds.startswith("DF40"):
        return "video ID; DF40 fakes carry their FF++ or Celeb-DF source video"
    return "video ID only"


def main():
    rows = []

    def add(split, component, path, items, note=""):
        real = sum(1 for x in items if REAL_HINT.search(Path(path).stem) or REAL_HINT.search(x))
        rows.append({"split": split, "component": component,
                     "manifest_path": str(Path(path).relative_to(EXT)) if str(path).startswith(str(EXT)) else str(path),
                     "videos": len(items), "real": real, "fake": len(items) - real,
                     "key_type": key_type(component if component else split), "note": note})

    cfg = (EXT / "eval_adaptation/configs/warm_dicome_gate_seed42.yaml").read_text()
    for blk, split in (("trn_files", "FFpp c23 train (spatial views)"),
                       ("val_files", "FFpp c23 validation (threshold data)")):
        m = re.search(rf"^{blk}:\n((?:- .*\n)+)", cfg, re.M)
        for line in (m.group(1).strip().split("\n") if m else []):
            p = EXT / line.lstrip("- ").strip()
            it = read_manifest(p)
            add(split, Path(p).stem, p, it or [], "" if it else "MANIFEST MISSING")

    # the temporal view's training manifest, from its own script's recorded recipe
    rec = ROOT / "logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/train_record.json"
    if rec.exists():
        j = json.load(open(rec))
        rows.append({"split": "FFpp c23 train (video view)", "component": "as recorded by train_branch.py",
                     "manifest_path": str(rec.relative_to(ROOT)), "videos": j.get("n_train", ""),
                     "real": "", "fake": "", "key_type": key_type("FFpp"),
                     "note": f"selected epoch {j.get('selected_epoch', '?')}; "
                             f"keys: {','.join(list(j)[:6])}"})

    # VALmix-v2 and the development pool
    for name in ("VALmix", "VALmixV2", "FFpp_val"):
        d = SPL / name
        if not d.exists():
            rows.append({"split": f"development pool component {name}", "component": "--",
                         "manifest_path": str(d), "videos": "", "real": "", "fake": "",
                         "key_type": "", "note": "SPLIT DIRECTORY MISSING"})
            continue
        for p in sorted(d.glob("*.txt")):
            add(f"development pool: {name}", p.stem, p, read_manifest(p) or [])

    # the ten test sets and the processed variants
    TEN = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]
    for ds in TEN:
        d = SPL / ds
        if not d.exists():
            rows.append({"split": f"test: {ds}", "component": "--", "manifest_path": str(d),
                         "videos": "", "real": "", "fake": "", "key_type": "", "note": "MISSING"})
            continue
        for p in sorted(d.glob("*.txt")):
            add(f"test: {ds}", p.stem, p, read_manifest(p) or [])
    d = SPL / "DF40"
    for p in sorted(d.glob("*.txt")) if d.exists() else []:
        add("DF40 evaluation", p.stem, p, read_manifest(p) or [])

    with open(OUT / "inventory.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["split", "component", "manifest_path", "videos", "real", "fake",
                                           "key_type", "note"])
        wr.writeheader(); wr.writerows(rows)
    miss = [r for r in rows if r["note"]]
    print(f"{len(rows)} inventory rows -> {OUT / 'inventory.csv'}")
    tot = {}
    for r in rows:
        if isinstance(r["videos"], int):
            tot[r["split"]] = tot.get(r["split"], 0) + r["videos"]
    for k, v in tot.items():
        print(f"  {k:46s} {v:6d} videos")
    if miss:
        print("\nrows carrying a note:")
        for r in miss:
            print(f"  {r['split']:46s} {r['note']}")


if __name__ == "__main__":
    main()
