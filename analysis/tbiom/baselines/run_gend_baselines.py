"""Run the third-party baselines shipped in the GenD repo (GenD-CLIP/PE/DINO, Effort, ForAda, FSFM, FSFM-Adapter)
on test sets built from the GenD release, using exactly the frames in our DISCERN h5 files.

The GenD repo is imported read-only; every output goes under --run_dir (default: this repo's logs).
One run per (model, dataset): <run_dir>/<model>--<dataset>/.../test_predictions.csv (files, labels, prob_class_1).
"""
import argparse, glob, os, sys
from pathlib import Path

GEND = Path("/data/umar/Repos/GenD_NeSy")
os.chdir(GEND)                      # checkpoints in EVAL_MODELS are relative to the GenD repo
sys.path.insert(0, str(GEND))

from run import main                # noqa: E402
from src import config as C         # noqa: E402
from src.config import Config       # noqa: E402
from src.exp.examples import EVAL_MODELS  # noqa: E402


def done(run_dir: Path, name: str) -> bool:
    return bool(glob.glob(f"{run_dir}/{name}/**/test_predictions.csv", recursive=True))


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(EVAL_MODELS))
    ap.add_argument("--datasets", nargs="+", default=["FFIW", "KoDF", "IDForge", "DFDM"])
    ap.add_argument("--lists_root", type=Path, default=Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/baselines/gend_lists"))
    ap.add_argument("--run_dir", type=Path, default=Path("/data/umar/Repos/DFB_NeSyNS/logs/tbiom/baselines/gend_runs"))
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--limit_test_batches", type=int, default=None)
    a = ap.parse_args()
    for model in a.models:
        for ds in a.datasets:
            name = f"{model}--{ds}"
            if done(a.run_dir, name):
                print(f"skip {name}", flush=True); continue
            files = sorted(str(p) for p in (a.lists_root / ds).glob("*.txt"))
            assert files, f"no lists for {ds}"
            mod = Config(run_name=name, run_dir=str(a.run_dir), tst_files=files, batch_size=a.batch_size,
                         mini_batch_size=a.batch_size, max_epochs=1, wandb=False, devices=[0],
                         num_workers=a.num_workers, throw_exception_if_run_exists=False, **EVAL_MODELS[model])
            cfg = Config(**C.Config().model_copy(update=mod.model_dump(exclude_unset=True)).model_dump())
            if a.limit_test_batches:
                cfg.limit_test_batches = a.limit_test_batches
            cfg = Config(**cfg.model_dump())
            print(f"run {name}: {len(files)} lists", flush=True)
            main(cfg, False)


if __name__ == "__main__":
    main_cli()
