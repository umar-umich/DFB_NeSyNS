#!/usr/bin/env python3
"""`run1dce_<ds>.yaml` for datasets that do not have one yet, so O's CE counterpart can be
exported everywhere the evidential arm is scored.

BASED ON AN EXISTING CE CONFIG, never on the evidential one. The first version of this file
copied `run1auxedl_<ds>.yaml` and set `simple_ce: true`, which looks right and is wrong: the two
arms also differ in which FS-VFM checkpoint they load -- the evidential arm takes
`fsvfm_lora_checkpoint`, the CE arm `fsvfm_ft_checkpoint`. A config built the first way asks for
a LoRA-wrapped encoder that the simple-CE checkpoint does not contain, and every export dies in
`load_state_dict` with a wall of missing `base_model.model.*` keys.

It also OVERWROTE the eight correct configs that already existed. They were tracked in git and
were restored with `git checkout --`, but the rule that follows is cheaper than the recovery:
never regenerate a file that already exists without diffing against it first. This script now
refuses to touch one.
"""
from __future__ import annotations

from pathlib import Path

import yaml

EXT = Path("/data/umar/Repos/DISCERN_Ext")
CFG = EXT / "eval_adaptation/configs"
TEMPLATE = CFG / "run1dce_CDFv2.yaml"          # a known-good CE config, committed
SETS = ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM",
        "VALmix", "VALmixV2", "FFpp_val", "UADFV", "DeepSpeak-1.1", "DeepSpeak-2.0"]


def main() -> int:
    tpl = yaml.safe_load(open(TEMPLATE))
    made, kept, missing = [], [], []
    for ds in SETS:
        out = CFG / f"run1dce_{ds}.yaml"
        if out.exists():
            kept.append(ds); continue
        src = CFG / f"run1auxedl_{ds}.yaml"
        if not src.exists():
            missing.append(ds); continue
        ev = yaml.safe_load(open(src))
        c = dict(tpl)
        # ONLY the three fields that name the dataset. Everything else -- the heads, the losses,
        # the FS-VFM checkpoint -- stays as the CE template has it.
        c["run_name"] = f"run1dce_{ds}"
        c["tst_h5_path"] = ev["tst_h5_path"]
        c["tst_files"] = ev["tst_files"]
        diff = {k for k in set(tpl) | set(c) if tpl.get(k) != c.get(k)}
        assert diff <= {"run_name", "tst_h5_path", "tst_files"}, f"{ds}: extra differences {diff}"
        yaml.safe_dump(c, open(out, "w"), sort_keys=False)
        made.append(ds)
    print(f"wrote {len(made)}: {', '.join(made) or '-'}")
    print(f"kept existing {len(kept)}: {', '.join(kept) or '-'}")
    if missing:
        print(f"no evidential config (not packed?): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
