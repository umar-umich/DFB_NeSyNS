"""Proposer inference — MLLM reads an image, emits candidate claims as JSON.

Frozen prompt (Type-B + codebook vocab + strict JSON schema), T=0 + fixed seed,
disk cache keyed on (image, model, prompt-hash). Interventions × models is the
cost centre, so every response is cached.

Two backends behind one interface (`Proposer.propose(image_path) -> str`):
  internvl3-8b            trust_remote_code InternVLChatModel, .chat() API
  qwen2.5-vl-32b-instruct Qwen2_5_VLForConditionalGeneration + processor

Model identities (repo + revision) come from cec/registration/pins.yaml; the
prompt + its hash from the frozen prompt. Nothing here invents a config.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

from cec.registration import load_params, load_pins, load_prompt

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "results" / "proposer_cache"


# --------------------------------------------------------------------------- InternVL image transform
def _internvl_transform(image_path, input_size=448, max_num=12):
    """InternVL dynamic-tiling preprocessing (standard from the model card)."""
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode

    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    def find_closest_ratio(ar, ratios, w, h):
        best, best_diff = (1, 1), float("inf")
        for r in ratios:
            diff = abs(ar - r[0] / r[1])
            if diff < best_diff:
                best_diff, best = diff, r
        return best

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    ar = w / h
    ratios = sorted({(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1)
                     for j in range(1, n + 1) if i * j <= max_num and i * j >= 1},
                    key=lambda x: x[0] * x[1])
    tw, th = find_closest_ratio(ar, ratios, w, h)
    resized = image.resize((input_size * tw, input_size * th))
    tiles = []
    for i in range(tw * th):
        box = ((i % tw) * input_size, (i // tw) * input_size,
               ((i % tw) + 1) * input_size, ((i // tw) + 1) * input_size)
        tiles.append(resized.crop(box))
    if len(tiles) != 1:
        tiles.append(image.resize((input_size, input_size)))  # thumbnail
    return torch.stack([transform(t) for t in tiles])


class Proposer:
    def __init__(self, name: str, device: str = "cuda:0", adapter: str | None = None):
        """adapter: optional LoRA adapter dir (T20 DPO-tuned proposer).

        The base weights are never modified — the adapter is applied on load, and
        the cache key includes it so base and tuned outputs never collide.
        """
        self.name = name
        self.device = device
        self.adapter = adapter
        self.params = load_params()
        self.prompt, self.prompt_hash = load_prompt("proposer_type_b")
        self._pin = self._resolve_pin(name)
        self._model = None
        self._tok = None
        self._proc = None
        CACHE.mkdir(parents=True, exist_ok=True)

    @property
    def variant(self) -> str:
        """'base' or 'tuned' — used in Pilot D reporting and the cache key."""
        return "base" if not self.adapter else "tuned"

    def _resolve_pin(self, name):
        for c in load_pins()["proposer"]["candidates"]:
            if c["name"] == name:
                if not c.get("on_disk"):
                    raise RuntimeError(f"proposer '{name}' is not on disk (see pins.yaml)")
                return c
        raise KeyError(f"unknown proposer '{name}'")

    # -- lazy load -----------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return
        torch.manual_seed(self.params.seed)
        if self.name.startswith("internvl"):
            from transformers import AutoModel, AutoTokenizer
            self._model = AutoModel.from_pretrained(
                self._pin["repo_id"], revision=self._pin["revision"],
                torch_dtype=torch.bfloat16, trust_remote_code=True,
                low_cpu_mem_usage=True).eval().to(self.device)
            self._tok = AutoTokenizer.from_pretrained(
                self._pin["repo_id"], revision=self._pin["revision"],
                trust_remote_code=True, use_fast=False)
        elif self.name.startswith("qwen"):
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self._pin["repo_id"], revision=self._pin["revision"],
                torch_dtype=torch.bfloat16, device_map=self.device).eval()
            self._proc = AutoProcessor.from_pretrained(
                self._pin["repo_id"], revision=self._pin["revision"])
        else:
            raise ValueError(f"no backend for proposer '{self.name}'")

        # T20: apply the DPO LoRA adapter, if any. Base weights stay untouched.
        if self.adapter:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter)
            self._model.eval()

    # -- cache ---------------------------------------------------------------
    def _cache_key(self, image_path):
        # adapter included so base and DPO-tuned outputs never collide in cache.
        key = f"{image_path}|{self.name}|{self.prompt_hash}|{self.adapter or 'base'}"
        return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"

    # -- inference -----------------------------------------------------------
    @torch.no_grad()
    def propose(self, image_path) -> str:
        image_path = str(image_path)
        ck = self._cache_key(image_path)
        if ck.exists():
            return json.loads(ck.read_text())["response"]

        self._load()
        if self.name.startswith("internvl"):
            pv = _internvl_transform(image_path).to(torch.bfloat16).to(self.device)
            gen = dict(max_new_tokens=self.params.raw["proposer"]["max_new_tokens"], do_sample=False)
            resp = self._model.chat(self._tok, pv, f"<image>\n{self.prompt}", gen)
        else:  # qwen
            from PIL import Image
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": Image.open(image_path).convert("RGB")},
                {"type": "text", "text": self.prompt}]}]
            text = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs = [Image.open(image_path).convert("RGB")]
            inputs = self._proc(text=[text], images=imgs, return_tensors="pt").to(self.device)
            out = self._model.generate(**inputs, max_new_tokens=self.params.raw["proposer"]["max_new_tokens"],
                                       do_sample=False)
            trimmed = out[:, inputs.input_ids.shape[1]:]
            resp = self._proc.batch_decode(trimmed, skip_special_tokens=True)[0]

        ck.write_text(json.dumps({"image": image_path, "model": self.name,
                                  "prompt_hash": self.prompt_hash, "response": resp}))
        return resp
