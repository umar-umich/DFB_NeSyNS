#!/usr/bin/env python3
"""Section 2 of the PCR6+ brief: the reference manifest.

    OMP_NUM_THREADS=8 python analysis/tbiom/pcr6p/manifest.py

Reuses the grouped-fusion manifest where it is still accurate and says which parts were reused; everything specific
to this study -- the four detector configurations and their own development exports, the operator sources, the
historical PCR6 labelling -- is recomputed here from the live tree.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "grouped"), str(HERE.parent), str(HERE.parent / "temporal")]
import ops as G  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
sys.path.insert(0, str(HERE.parent / "grouped"))
import confirm as C  # noqa: E402

R, S23 = Path("tbiom/reports"), Path("logs/tbiom/stage23")
DEV_SETS = ("VALmix", "VALmixV2", "FFpp_val")


def sha(p, n=16):
    p = Path(p)
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def main():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    L = ["# PCR6+ reference manifest", "", __doc__.split("\n\n", 2)[2].strip(), "",
         f"Repository commit `{commit}`; working tree {'DIRTY' if dirty else 'clean'} at audit time"
         + (f" ({len(dirty.splitlines())} paths)" if dirty else "") + ".", "",
         "## 1. Reused from the grouped-fusion manifest", "",
         "`GROUPED_REFERENCE_MANIFEST.md` is reused unchanged for: the branch order and the shared CLIP path (§2), "
         "the opinion algebra verified on the caches — 0 rows outside u/2 <= p <= 1-u/2 on all three views (§3), the "
         "five source-grouped folds and the DF40 source-grouping rule (§4), and the exposure history (§5). Nothing "
         "in those sections changed between the two studies; the checkpoints and caches are the same files, and "
         "their hashes are re-verified below.", "",
         "## 2. The discount, stated exactly", "",
         "Reliability enters as **evidence scaling**, e'_k = r * e_k, applied in the evidence domain and converted "
         "back to an opinion. This is the discount the project deploys and it is kept throughout this study.", "",
         "It is **not** classical belief-mass (Shafer) discounting, m'(A) = r m(A) for A != Theta and "
         "m'(Theta) = 1 - r(1 - m(Theta)). The two agree at the endpoints r = 1 (both the identity) and r = 0 (both "
         "produce the vacuous opinion), and differ at every intermediate r. A worked comparison at r = 0.25 is in "
         "`REFERENCE_MANIFEST.md` §2, where the two differ by up to 0.29 in p.", "",
         "The shared r = 0 endpoint is what makes the PCR6+ neutrality property load-bearing here: a view discounted "
         "to zero is vacuous under either convention, and PCR6+ is the rule under which a vacuous view is exactly a "
         "removed view (`EXP_PCR6P_AUDIT.md` §3).", "",
         "## 3. The four detector configurations and their own development exports", "",
         "| tag | spatial views | temporal branch | development exports used to fit the discount |",
         "|---|---|---|---|"]
    rows = {"42": ("deployed warm+gate, epoch 19", "vjepa2_lora_B_seed42", "warmgate_e19"),
            "T1337": ("deployed warm+gate, epoch 19", "vjepa2_lora_B_seed1337", "warmgate_e19"),
            "armB42": ("SBI arm B seed 42, v3.1-selected epoch 8", "vjepa2_lora_B_seed42", "sbiarmB_e08"),
            "armB1337": ("SBI arm B seed 1337, v3.1-selected epoch 3", "vjepa2_lora_B_seed1337", "sbiarmBs1337_e03")}
    for tag, (sp, br, dv) in rows.items():
        have = all((S23 / f"{dv}_{n}.csv").exists() for n in DEV_SETS)
        L.append(f"| `{tag}` | {sp} | `{br}` | `{dv}_*` — {'present' if have else '**MISSING**'} |")
    L += ["", "**Arm B is a different training recipe, not merely a different seed.** Within arm B the seed is the "
          "only difference between seed 42 and seed 1337, but arm B differs from the deployed warm+gate recipe in "
          "more than a seed, and the four configurations are described that way wherever they are compared. The "
          "deployed recipe exists at one seed and this task forbids training another.", "",
          "A configuration's reliability weights are fitted on **its own** development exports. An earlier run of the "
          "grouped study fitted the arm-B weights on the deployed system's development exports, because "
          "`dsr_system.load` hardcodes the `warmgate_e19` spatial tag; that was corrected in commit `e58c751` and "
          "this study inherits the corrected loader.", "",
          "## 4. Checkpoint and cache hashes", "", "| artefact | sha256 (16) |", "|---|---|"]
    for k, p in {"T, V-JEPA 2 LoRA B seed 42": "logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt",
                 "T, V-JEPA 2 LoRA B seed 1337": "logs/tbiom/temporal/branches/vjepa2_lora_B_seed1337/best.pt",
                 "APP + OM, warm+gate epoch 19":
                 "/data/umar/Repos/DISCERN_Ext/runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/"
                 "discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt"}.items():
        L.append(f"| {k} | `{sha(p)}` |")
    for tag, arm in (("armB42", "sbiarmB"), ("armB1337", "sbiarmBs1337")):
        ck = json.load(open(S23 / f"{arm}_selection_v3.json"))["selected"]["checkpoint"]
        L.append(f"| spatial, {tag} (`{Path(ck).name}`) | `{sha(ck)}` |")

    L += ["", "## 5. Operator sources", "", "| operator | source | status |", "|---|---|---|",
          "| PCR6, direct S-source, eq. (15) | Dezert, Dezert & Smarandache, *Improvement of Proportional Conflict "
          "Redistribution Rules of Combination of Basic Belief Assignments*, JAIF 16(1), June 2021, pp. 48-73 | "
          "**obtained and read** — PDF sha256 `d87ae493e7c409d1b194ed701b6fe746`, "
          "https://isif.org/files/isif/2024-01/last.pdf |",
          "| PCR6+, direct S-source, eq. (26), with the binary keeping index of eqs. (22)-(24) | same paper | "
          "**obtained and read**; implemented from the equations, not from a summary |",
          "| Dempster, Yager, evidence averaging | as implemented in the grouped study | reused unchanged |", "",
          "PCR6+ is established prior work by Dezert, Dezert and Smarandache and is credited as such everywhere in "
          "this study. It is not presented as a new operator.", "",
          "## 6. Historical PCR6 labelling", ""]
    aud = json.load(open(R / "EXP_PCR6P_AUDIT.json"))
    dev = aud["notes"]["pairwise_vs_direct_max_dev"]
    L += [f"The existing `PCR6-R` rows in `EXP_FUSION_OPERATORS.md`, `EXP_DF40_OPERATORS.md` and the grouped reports "
          f"were produced by folding the two-source rule over the three views, not by the direct three-source "
          f"formula. PCR6 is not associative, so these are different operators; they differ by up to {dev:.3e} on "
          "random admissible opinions. Those reports are left unchanged, and every table in this study labels the "
          "fold **`PCR6-pairwise-R`** and the direct rule **`PCR6-R`**.", "",
          "## 7. Folds, threshold rule, grid", "",
          "Five existing source-grouped folds (`GroupShuffleSplit` on the source id, seeds 0-4) and the v3.1 macro "
          "objective over swap / reenactment / in-the-wild cells. Threshold: EER on FF++ val, computed per arm on "
          "its own scores — the frozen rule already in the protocol. Reliability grid: (0, 0.25, 0.5, 0.75, 1) per "
          "view with max = 1, the existing grid and candidate cap, unchanged.", "",
          "## 8. Exposure", "",
          "Unchanged from `GROUPED_REFERENCE_MANIFEST.md` §5 and repeated because it governs every claim here: "
          "CDFv2 and DeepfakeEval2024 informed development; DF40 has already informed operator design in two prior "
          "studies; FaceShifter was read once after the recipe was frozen. **No untouched confirmation set remains**, "
          "so outcome 6 of the brief — independent confirmation pending — applies to every result in this study "
          "regardless of what else it finds.", ""]
    (R / "PCR6P_REFERENCE_MANIFEST.md").write_text("\n".join(L) + "\n")
    print("wrote PCR6P_REFERENCE_MANIFEST.md")


if __name__ == "__main__":
    main()
