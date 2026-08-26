# Mirrored copies of the DiCoME-side adaptation code

These files RUN from `/data/umar/Repos/DiCoME/eval_adaptation/`, not from here. They are
mirrored into this repo because `eval_adaptation/` is gitignored in the DiCoME clone (existing
project convention — every adaptation script we have written lives there untracked), which
means the Step-1 exporter, the Step-2 driver and the ablation config would otherwise have no
version history anywhere.

The source-code change that the ablation depends on IS tracked, in the DiCoME clone itself:
commit `28a7ca9`, adding `detach_vae_from_encoder` to `src/config.py`,
`src/model/core_model.py` and `src/model/dicome_module.py`. Default False, so the published
recipe and the P0-DS reproduction are unchanged.

Keep these in sync by hand if the originals change; they are a record, not an import path.
