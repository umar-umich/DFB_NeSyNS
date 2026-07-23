"""TASK 11 — DPO training (LoRA, proposer only, AFTER the audit freeze).

The only trained component in CEC. Detectors and instruments stay frozen
(CURRENT_STATE rule 4); the proposer is preference-tuned on the gate's labels.

Config is frozen in registration terms: TRL DPOTrainer, LoRA (r=16, attn+MLP
targets, beta≈0.1), bf16, one H200. Preference pairs from build_prefs.py.

PREREQUISITE: `trl` is not installed in any env (checked GenD/dfb_nesy/facebench).
Install into an ISOLATED env (not GenD — protect the frozen anchors), e.g.:
    python -m venv ~/envs/cec_dpo && ~/envs/cec_dpo/bin/pip install \
        trl peft transformers accelerate bitsandbytes
This module is import-guarded so the rest of CEC runs without trl.

Run (after freeze + isolated env):
    <cec_dpo python> cec/dpo/train_dpo.py --prefs audit_..._prefs --model internvl3-8b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Frozen DPO config (registration terms).
DPO_CONFIG = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "beta": 0.1,
    "learning_rate": 5e-6,
    "num_train_epochs": 1,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "bf16": True,
    "max_length": 1024,
    "seed": 17,
}


def load_pairs(prefs_name: str):
    path = REPO / "cec" / "dpo" / "data" / f"{prefs_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"preference file missing: {path} (run build_prefs.py)")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefs", required=True)
    ap.add_argument("--model", default="internvl3-8b")
    ap.add_argument("--out", default=str(REPO / "cec" / "dpo" / "lora"))
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        print(f"[trl not installed] {e}")
        print("Install trl+peft in an ISOLATED env (see module docstring); do NOT touch GenD.")
        return 2

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cec.registration import load_pins
    pin = next(c for c in load_pins()["proposer"]["candidates"] if c["name"] == args.model)

    pairs = load_pairs(args.prefs)
    print(f"[dpo] {len(pairs)} preference pairs; model {args.model} ({pin['repo_id']})")
    if len(pairs) < 50:
        print("🟡 scarce pool (<50) — confirm with Umar before spending training time.")

    ds = Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
                            for p in pairs])
    tok = AutoTokenizer.from_pretrained(pin["repo_id"], revision=pin.get("revision"), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pin["repo_id"], revision=pin.get("revision"), torch_dtype="bfloat16", trust_remote_code=True)

    lora = LoraConfig(r=DPO_CONFIG["lora_r"], lora_alpha=DPO_CONFIG["lora_alpha"],
                      target_modules=DPO_CONFIG["lora_targets"], task_type="CAUSAL_LM")
    cfg = DPOConfig(output_dir=args.out, beta=DPO_CONFIG["beta"],
                    learning_rate=DPO_CONFIG["learning_rate"],
                    num_train_epochs=DPO_CONFIG["num_train_epochs"],
                    per_device_train_batch_size=DPO_CONFIG["per_device_batch_size"],
                    gradient_accumulation_steps=DPO_CONFIG["gradient_accumulation_steps"],
                    bf16=DPO_CONFIG["bf16"], max_length=DPO_CONFIG["max_length"], seed=DPO_CONFIG["seed"])
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                         peft_config=lora)
    trainer.train()
    trainer.save_model(args.out)
    print(f"[dpo] LoRA saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
