"""T20 — DPO training (LoRA, proposer only, AFTER the audit freeze).

The only trained component in CEC. Detectors, instruments and the gate stay
frozen (CURRENT_STATE rule 4); the proposer is preference-tuned on the gate's own
labels.

MULTIMODAL (Umar's decision): each preference pair is IMAGE-CONDITIONED — the
image is fed through the VLM processor alongside the prompt, so the model learns
"given THIS image, prefer the certified claim / the abstention" rather than a
text-only prior. TRL 1.9.0's DPOTrainer supports this via `processing_class` = the
model's processor and an `images` column in the dataset.

Splits are frozen in `cec/registration/dpo.yaml`: thresholds calibrated on TRAIN,
pairs from TEST, Pilot D evaluates on VAL (disjoint) — stated in config so the
eval split cannot be chosen after seeing results.

PREREQUISITE (T17): env `cec_dpo` (trl+peft), NEVER `GenD`. Import-guarded.

Run (in cec_dpo):
    <cec_dpo python> cec/dpo/train_dpo.py --prefs pooled_internvl_qwen \
        --model internvl3-8b --out cec/dpo/lora/internvl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CONFIG_PATH = REPO / "cec" / "registration" / "dpo.yaml"

# transformers class per proposer (multimodal path only).
_MODEL_CLASS = {
    "qwen2.5-vl-32b-instruct": "Qwen2_5_VLForConditionalGeneration",
}

# Training MODE per proposer (Umar's decision: train each as its arch allows).
#   multimodal  image-conditioned DPO via TRL's vision path (native processor).
#   text        DPO on the underlying causal LM only.
# InternVL3-8B is pinned to the `internvl_chat` checkpoint, which the native
# transformers class cannot load (embedding 151674x3584 vs native 151936x4096)
# and whose custom dynamic-tiling processor TRL's vision collator cannot drive.
# Its language model IS a standard Qwen2ForCausalLM, so we train that text-only
# and the LoRA adapter applies to `model.language_model` at inference.
_MODE = {
    "internvl3-8b": "text",
    "qwen2.5-vl-32b-instruct": "multimodal",
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"frozen DPO config missing: {CONFIG_PATH}")
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_pairs(prefs_name: str):
    data = REPO / "cec" / "dpo" / "data"
    for cand in (data / f"{prefs_name}.jsonl", data / f"{prefs_name}_prefs.jsonl"):
        if cand.exists():
            return [json.loads(l) for l in cand.read_text().splitlines() if l.strip()]
    raise FileNotFoundError(f"preference file missing: {prefs_name}[_prefs].jsonl in {data} "
                            f"(run build_prefs.py)")


def build_text_dataset(pairs, model_name):
    """Plain text DPO dataset (prompt -> chosen/rejected), this proposer's pairs only."""
    from datasets import Dataset
    rows = [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
            for p in pairs if p.get("proposer") == model_name]
    return Dataset.from_list(rows)


def build_vision_dataset(pairs, model_name):
    """TRL conversational vision dataset: images + prompt/chosen/rejected turns.

    Only pairs from THIS proposer are used (a pair reflects that proposer's own
    output distribution). Images are attached lazily as paths; the collator loads
    them via the processor.
    """
    from datasets import Dataset
    from PIL import Image
    from cec.data.pairing import resolve_image_path

    rows = []
    for p in pairs:
        if p.get("proposer") != model_name:
            continue
        img_path = resolve_image_path(p["image"])
        if not img_path.exists():
            continue
        rows.append({
            "images": [Image.open(img_path).convert("RGB")],
            "prompt": [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": p["prompt"]}]}],
            "chosen": [{"role": "assistant", "content": [{"type": "text", "text": p["chosen"]}]}],
            "rejected": [{"role": "assistant", "content": [{"type": "text", "text": p["rejected"]}]}],
        })
    return Dataset.from_list(rows)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefs", required=True)
    ap.add_argument("--model", default=cfg["model"]["order"][0], choices=list(_MODE))
    ap.add_argument("--out", default=None)
    ap.add_argument("--force-small-pool", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps (smoke test); -1 = full")
    ap.add_argument("--mode", choices=["multimodal","text"], default=None,
                    help="override the per-proposer default training mode")
    args = ap.parse_args()
    out = args.out or str(REPO / "cec" / "dpo" / "lora" / args.model)

    pairs = load_pairs(args.prefs)
    mine = [p for p in pairs if p.get("proposer") == args.model]
    mode_label = (args.mode or _MODE[args.model]).upper()
    print(f"[dpo] {len(pairs)} pooled pairs · {len(mine)} for {args.model} · {mode_label}")
    print(f"[dpo] splits — pairs '{cfg['splits']['train_pairs_split']}', held-out eval "
          f"'{cfg['splits']['heldout_eval_split']}' (pre-committed)")
    if len(pairs) < 500 and not args.force_small_pool:
        print("🟡 pool < 500 (pre-committed trigger). STOP (T18) or use --force-small-pool.")
        return 3
    if not mine:
        print(f"no pairs for proposer '{args.model}' in this pool.")
        return 4

    try:
        from peft import LoraConfig
        from transformers import AutoProcessor
        import transformers as tf
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        print(f"[trl not installed] {e} — create env '{cfg['env']['name']}' (T17), "
              f"NEVER '{cfg['env']['forbidden_env']}'.")
        return 2

    from cec.registration import load_pins
    pin = next(c for c in load_pins()["proposer"]["candidates"] if c["name"] == args.model)
    mode = args.mode or _MODE[args.model]

    if mode == "multimodal":
        model_cls = getattr(tf, _MODEL_CLASS[args.model])
        print(f"[dpo] MULTIMODAL · loading {args.model} ({model_cls.__name__}) + processor ...")
        processing_class = AutoProcessor.from_pretrained(
            pin["repo_id"], revision=pin.get("revision"), trust_remote_code=True)
        model = model_cls.from_pretrained(
            pin["repo_id"], revision=pin.get("revision"), torch_dtype="bfloat16",
            trust_remote_code=True, device_map="auto")
        ds = build_vision_dataset(pairs, args.model)
        print(f"[dpo] vision dataset: {len(ds)} image-conditioned pairs")
    else:  # text-only: train the underlying causal LM
        from transformers import AutoModel, AutoTokenizer
        print(f"[dpo] TEXT-ONLY · loading {args.model} and extracting its language model ...")
        processing_class = AutoTokenizer.from_pretrained(
            pin["repo_id"], revision=pin.get("revision"), trust_remote_code=True, use_fast=False)
        full = AutoModel.from_pretrained(
            pin["repo_id"], revision=pin.get("revision"), torch_dtype="bfloat16",
            trust_remote_code=True, low_cpu_mem_usage=True, device_map="auto")
        model = getattr(full, "language_model", None)
        if model is None:
            print("[dpo] could not reach .language_model on this checkpoint.")
            return 5
        print(f"[dpo] language model: {type(model).__name__}")
        ds = build_text_dataset(pairs, args.model)
        print(f"[dpo] text dataset: {len(ds)} pairs")

    lc, dc = cfg["lora"], cfg["dpo"]
    lora = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                      target_modules=lc["target_modules"], task_type="CAUSAL_LM")
    train_args = DPOConfig(
        output_dir=out, beta=dc["beta"], learning_rate=dc["learning_rate"],
        num_train_epochs=dc["num_train_epochs"],
        per_device_train_batch_size=dc["per_device_train_batch_size"],
        gradient_accumulation_steps=dc["gradient_accumulation_steps"],
        max_length=dc["max_length"],  # TRL 1.9.0 dropped max_prompt_length
        bf16=dc["bf16"], seed=dc["seed"], save_strategy=dc["save_strategy"],
        logging_steps=dc["logging_steps"],
        max_steps=args.max_steps, gradient_checkpointing=True,
        report_to="none",  # no wandb (headless); loss goes to stdout
    )
    trainer = DPOTrainer(model=model, args=train_args, train_dataset=ds,
                         processing_class=processing_class, peft_config=lora)
    trainer.train()
    trainer.save_model(out)
    print(f"[dpo] LoRA adapters saved to {out} (mode={mode}; base weights untouched)")
    print(f"[dpo] next: Pilot D on held-out '{cfg['splits']['heldout_eval_split']}' via the "
          f"UNTOUCHED gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
