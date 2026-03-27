#!/usr/bin/env python3
"""
demo_semantic.py
================
Validate FaceBench Face-LLaVA on a single image.

Modes:
  --mode teacher_force  : Uses SemanticPrecomputer — the exact same batched
                          per-attribute teacher-forcing as the main pipeline.
                          This is the recommended validation mode.
  --mode generate       : Per-attribute yes/no via model.generate (211 passes).
                          Slow but matches FaceBench eval exactly.
  --mode describe       : Free-form facial description.
  --from_precomputed    : Load from an existing .pt file (no model needed).

Usage:
  # Validate using the exact same path as precompute_semantic_features.py:
  CUDA_VISIBLE_DEVICES=1 python preprocessing/demo_semantic.py \
      --demo /path/to/face.jpg --mode teacher_force \
      --detector_path training/config/detector/nesy_defake.yaml

  # Compare with slow generative mode (211 individual passes):
  CUDA_VISIBLE_DEVICES=1 python preprocessing/demo_semantic.py \
      --demo /path/to/face.jpg --mode generate

  # Load from precomputed .pt:
  python preprocessing/demo_semantic.py \
      --demo /path/to/face.jpg --from_precomputed
"""

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from demo_plot_helpers import report_and_plot, plot_description

FACE_LLAVA_REPO = '/data/umar/Repos/Face-LLaVA'


# ---------------------------------------------------------------------------
#  Raw Face-LLaVA model loader (for generate / describe modes only)
# ---------------------------------------------------------------------------

def _load_raw_facellava(args):
    """Load raw Face-LLaVA model for generative modes.
    Returns (model, tokenizer, image_processor, facellava constants)."""
    sys.path.insert(0, FACE_LLAVA_REPO)

    _ft = types.ModuleType('torchvision.transforms.functional_tensor')
    sys.modules.setdefault('torchvision.transforms.functional_tensor', _ft)
    _tv = types.ModuleType('torchvision.transforms._transforms_video')
    for _cls_name in ['NormalizeVideo', 'RandomCropVideo',
                      'RandomHorizontalFlipVideo', 'CenterCropVideo',
                      'RandomResizedCropVideo', 'UniformTemporalSubsample']:
        setattr(_tv, _cls_name, type(_cls_name, (), {}))
    sys.modules.setdefault(
        'torchvision.transforms._transforms_video', _tv)

    from facellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from facellava.conversation import conv_templates, SeparatorStyle
    from facellava.model.builder import load_pretrained_model
    from facellava.utils import disable_torch_init
    from facellava.mm_utils import (
        tokenizer_image_token, get_model_name_from_path,
        KeywordsStoppingCriteria, process_images)

    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), '..', 'training'))
    from networks.nesy_defake.semantic.facial_semantic_extractor import (
        FACEBENCH_ATTRIBUTES)

    # Patch LlavaConfig
    from facellava.model.language_model.llava_llama import LlavaConfig
    from transformers.models.auto.configuration_auto import (
        CONFIG_MAPPING_NAMES, CONFIG_MAPPING)
    CONFIG_MAPPING_NAMES["llava_llama"] = "LlavaConfig"
    CONFIG_MAPPING._extra_content["llava_llama"] = LlavaConfig

    _orig_getattr = LlavaConfig.__getattribute__

    def _patched_getattr(self, name):
        try:
            return _orig_getattr(self, name)
        except AttributeError:
            if name == 'mm_image_tower':
                return getattr(self, 'mm_vision_tower', None)
            if name == 'mm_video_tower':
                return None
            raise
    LlavaConfig.__getattribute__ = _patched_getattr

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if 'llava' not in model_name.lower():
        model_name = 'llava-v1.5-13b'

    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path, None, model_name,
        load_8bit=args.load_8bit, load_4bit=args.load_4bit,
        device=args.device)
    print(f"Model loaded: {model_name}, context_len={context_len}")

    return (model, tokenizer, processor['image'],
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            conv_templates, SeparatorStyle,
            tokenizer_image_token, KeywordsStoppingCriteria,
            process_images, FACEBENCH_ATTRIBUTES)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate FaceBench Face-LLaVA on a single face image')
    parser.add_argument('--demo', type=str, required=True)
    parser.add_argument('--mode',
                        choices=['teacher_force', 'generate', 'describe'],
                        default='teacher_force')
    parser.add_argument('--detector_path', type=str, default=None,
                        help='Required for --mode teacher_force')
    parser.add_argument('--model_path', type=str,
                        default='/data/umar/Repos/FaceBench')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load_4bit', action='store_true')
    parser.add_argument('--load_8bit', action='store_true')
    parser.add_argument('--from_precomputed', action='store_true')
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--attr_batch_size', type=int, default=128,
                        help='Attributes per forward pass (teacher_force)')
    return _run(parser.parse_args())


def _run(args):
    image_path = args.demo
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    # =======================================================================
    #  --from_precomputed: load from existing .pt file (no model needed)
    # =======================================================================
    if args.from_precomputed:
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from networks.nesy_defake.semantic.facial_semantic_extractor import (
            FACEBENCH_ATTRIBUTES)

        image_p = Path(image_path).resolve()
        frame_name = image_p.stem
        video_id = image_p.parent.name
        pt_path = (image_p.parent.parent.parent
                   / 'facellava_semantic' / f'{video_id}.pt')
        if not pt_path.exists():
            print(f"ERROR: Precomputed file not found: {pt_path}")
            sys.exit(1)

        data = torch.load(pt_path, map_location='cpu', weights_only=True)
        frame_idx = next(
            (i for i, fp in enumerate(data['frame_paths'])
             if Path(fp).stem == frame_name), None)
        if frame_idx is None:
            print(f"ERROR: Frame '{frame_name}' not found in {pt_path}")
            sys.exit(1)

        scores = data['features'][frame_idx].numpy()
        attr_names = list(FACEBENCH_ATTRIBUTES)
        img_pil = Image.open(image_path).convert('RGB')

        print("=" * 70)
        print(f"  Loading from precomputed: {pt_path}")
        print(f"  Frame idx: {frame_idx} / {len(data['frame_paths'])}")
        print("=" * 70)

        report_and_plot(img_pil, scores, attr_names, image_path,
                        save_path=args.save_path, top_k=args.top_k,
                        suffix='precomputed')
        return

    # =======================================================================
    #  --mode teacher_force: uses SemanticPrecomputer (same as main pipeline)
    # =======================================================================
    if args.mode == 'teacher_force':
        if not args.detector_path:
            print("ERROR: --detector_path is required for --mode teacher_force")
            sys.exit(1)
        from precompute_semantic_features import SemanticPrecomputer
        from config_utils import load_config

        config = load_config(args.detector_path)
        precomputer = SemanticPrecomputer(
            config, args.device,
            attr_batch_size=args.attr_batch_size,
            load_4bit=args.load_4bit, load_8bit=args.load_8bit)
        precomputer.demo(image_path, save_path=args.save_path,
                         top_k=args.top_k)
        return

    # =======================================================================
    #  Raw model modes: generate, describe
    # =======================================================================
    print("=" * 70)
    print("  FaceBench Face-LLaVA Validation Demo")
    print(f"  Image:    {image_path}")
    print(f"  Mode:     {args.mode}")
    print(f"  Model:    {args.model_path}")
    print(f"  Device:   {args.device}")
    quantize = ('4-bit' if args.load_4bit
                else '8-bit' if args.load_8bit else 'fp16')
    print(f"  Quantize: {quantize}")
    print("=" * 70)

    print("\nLoading Face-LLaVA model...")
    (model, tokenizer, image_processor,
     IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
     conv_templates, SeparatorStyle,
     tokenizer_image_token, KeywordsStoppingCriteria,
     process_images, FACEBENCH_ATTRIBUTES) = _load_raw_facellava(args)

    img_pil = Image.open(image_path).convert('RGB')

    # Prepare image tensor
    image_tensor = process_images([img_pil], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [
            img.squeeze(0).to(model.device, dtype=torch.float16)
            if img.ndim == 4
            else img.to(model.device, dtype=torch.float16)
            for img in image_tensor]
    elif image_tensor.ndim == 4:
        image_tensor = [
            image_tensor[i].to(model.device, dtype=torch.float16)
            for i in range(image_tensor.shape[0])]
    else:
        image_tensor = [image_tensor.to(model.device, dtype=torch.float16)]
    print(f"Image: {img_pil.size}, tensor: {image_tensor[0].shape}")

    attr_names = list(FACEBENCH_ATTRIBUTES)

    def run_generate(prompt_text):
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(model.device)
        stop_str = (conv.sep if conv.sep_style != SeparatorStyle.TWO
                    else conv.sep2)
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=image_tensor, do_sample=False,
                temperature=0, top_p=None, num_beams=1,
                max_new_tokens=1024, use_cache=True,
                stopping_criteria=[stopping_criteria])
        return tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True).strip()

    # -- describe --
    if args.mode == 'describe':
        prompts = [
            "Describe the person's face as seen in the photograph.",
            "Identify the visible features of the face in the image.",
            "Identify the active facial action units in the provided image.",
        ]
        print("\n" + "=" * 70)
        print("FREE-FORM FACIAL DESCRIPTION")
        print("=" * 70)
        descriptions = []
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            print("-" * 60)
            response = run_generate(prompt)
            print(response)
            descriptions.append((prompt, response))

        save_path = args.save_path or (
            os.path.splitext(image_path)[0] + '_facebench_describe.png')
        txt_path = os.path.splitext(save_path)[0] + '.txt'
        with open(txt_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FaceBench Face-LLaVA — Free-Form Description\n")
            f.write("=" * 70 + "\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Model: {args.model_path}\n\n")
            for prompt, response in descriptions:
                f.write(f"Prompt: {prompt}\n{'─' * 60}\n{response}\n\n")
        plot_description(img_pil, descriptions, save_path)
        print(f"\nPlot saved to: {save_path}")
        print(f"Text report saved to: {txt_path}")

    # -- generate --
    elif args.mode == 'generate':
        print(f"\nGenerative extraction: {len(attr_names)} attributes...")
        scores = np.zeros(len(attr_names), dtype=np.float32)
        raw_responses = {}

        for i, attr in enumerate(tqdm(attr_names, desc="Attributes")):
            prompt = (
                f"Does this person have {attr.replace('_', ' ')}?\n"
                f"Yes\nNo\nInformation not visible\n"
                f"Please directly select the appropriate option from the "
                f"given choices based on the image. "
                f"Please return the answer directly without additional "
                f"explanation.")
            response = run_generate(prompt)
            raw_responses[attr] = response
            r = response.strip().lower()
            if r.startswith('yes'):
                scores[i] = 1.0
            elif r.startswith('no'):
                scores[i] = 0.0
            else:
                scores[i] = 0.5

        n_yes = (scores > 0.5).sum()
        n_no = (scores < 0.5).sum()
        n_unsure = (scores == 0.5).sum()
        print(f"\nResults: {n_yes} Yes, {n_no} No, {n_unsure} Unsure")

        report_and_plot(img_pil, scores, attr_names, image_path,
                        save_path=args.save_path, top_k=args.top_k,
                        suffix='generate', raw_responses=raw_responses)

    print("Done.")


if __name__ == '__main__':
    main()
