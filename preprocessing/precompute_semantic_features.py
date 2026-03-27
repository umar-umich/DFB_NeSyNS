#!/usr/bin/env python3
"""
precompute_semantic_features.py
================================
Precompute Face-LLaVA semantic attributes for all frames and save to disk.

Loads the frozen Face-LLaVA 13B model ONCE, pre-tokenizes all 211 attribute
prompts ONCE, then runs batched per-attribute teacher-forcing over every frame
producing per-video .pt files with shape (n_frames, 211).

This uses the exact same batched teacher-forcing approach validated in
demo_semantic.py — each attribute gets its own clean yes/no prompt, run in
small padded batches through a single forward pass per group.

Output structure:
  .../original_sequences/youtube/c23/facellava_semantic/929.pt

Usage (batch precomputation):
  python preprocessing/precompute_semantic_features.py \
      --detector_path training/config/detector/nesy_defake.yaml \
      --batch_size 64 --output_dir facellava_semantic

Usage (single-image demo — validates the exact same extraction path):
  python preprocessing/precompute_semantic_features.py \
      --detector_path training/config/detector/nesy_defake.yaml \
      --demo /path/to/face.jpg

For alternative raw-model modes (generate, describe), use demo_semantic.py.
"""

import argparse
import os
import sys
import types
from pathlib import Path

# Set CUDA_VISIBLE_DEVICES before importing torch
if '--device' in sys.argv:
    _dev = sys.argv[sys.argv.index('--device') + 1]
    if _dev.startswith('cuda:'):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES',
                              os.environ.get('CUDA_VISIBLE_DEVICES', '0'))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config_utils import load_config, collect_videos_from_json

FACE_LLAVA_REPO = '/data/umar/Repos/Face-LLaVA'
DEFAULT_MODEL_PATH = '/data/umar/Repos/FaceBench'


class SemanticPrecomputer:
    """
    Loads Face-LLaVA 13B once, pre-tokenizes all 211 attribute prompts once,
    then extracts semantic features via batched per-attribute teacher-forcing.

    This is the same approach validated in demo_semantic.py's teacher_force
    mode — each attribute gets a clean yes/no prompt, batched through padded
    forward passes.
    """

    def __init__(self, config: dict, device: str = 'cuda:0',
                 attr_batch_size: int = 112,
                 load_4bit: bool = False, load_8bit: bool = False,
                 mode: str = 'combined'):
        """
        Args:
            mode: 'combined' (default) — single structured prompt with all 211
                      attributes, auto-chunked to fit context window.
                      ~2 LLM forward passes per image.
                  'per_attr' — individual yes/no prompt per attribute,
                      batched together. Original FaceBench-native approach.
                      ceil(211/attr_batch_size) passes per image.
        """
        self.device = device
        self.attr_batch_size = attr_batch_size
        self.mode = mode
        self.attr_names = self._get_attr_names()

        model_path = config.get('semantic_attributes', {}).get(
            'model_path', DEFAULT_MODEL_PATH)

        # Load Face-LLaVA model + tokenizer + image processor
        (self.model, self.tokenizer, self.image_processor,
         self.IMAGE_TOKEN_INDEX, self.DEFAULT_IMAGE_TOKEN,
         self.conv_templates, self.tokenizer_image_token_fn,
         self.process_images_fn) = self._load_model(
            model_path, device, load_4bit, load_8bit)

        # Visual offset: image token expands to num_patches visual tokens
        image_tower = self.model.get_image_tower()
        self.visual_offset = image_tower.num_patches - 1

        if mode == 'per_attr':
            # Pre-tokenize all 211 individual attribute prompts (CPU, done once)
            (self.yes_ids_list, self.no_ids_list,
             self.diff_pos_list) = self._pretokenize_prompts()
            self.img_tok_pos = (
                self.yes_ids_list[0] == self.IMAGE_TOKEN_INDEX
            ).nonzero(as_tuple=True)[0][0].item()
        else:
            # Combined mode: structured prompt, auto-chunked for context window
            self._build_combined_prompt()

    @staticmethod
    def _get_attr_names():
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from networks.nesy_defake.semantic.facial_semantic_extractor import (
            FACEBENCH_ATTRIBUTES)
        return list(FACEBENCH_ATTRIBUTES)

    @staticmethod
    def _load_model(model_path, device, load_4bit, load_8bit):
        """Load Face-LLaVA via the facellava native loader."""
        sys.path.insert(0, FACE_LLAVA_REPO)

        # Patch missing torchvision modules for compatibility
        _ft = types.ModuleType('torchvision.transforms.functional_tensor')
        sys.modules.setdefault('torchvision.transforms.functional_tensor', _ft)
        _tv = types.ModuleType('torchvision.transforms._transforms_video')
        for _cls_name in ['NormalizeVideo', 'RandomCropVideo',
                          'RandomHorizontalFlipVideo', 'CenterCropVideo',
                          'RandomResizedCropVideo',
                          'UniformTemporalSubsample']:
            setattr(_tv, _cls_name, type(_cls_name, (), {}))
        sys.modules.setdefault(
            'torchvision.transforms._transforms_video', _tv)

        from facellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from facellava.conversation import conv_templates
        from facellava.model.builder import load_pretrained_model
        from facellava.utils import disable_torch_init
        from facellava.mm_utils import (
            tokenizer_image_token, get_model_name_from_path, process_images)

        # Patch LlavaConfig for compat
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

        print(f"[SemanticPrecomputer] Loading Face-LLaVA from {model_path}...")
        disable_torch_init()

        # Patch Llama _init_weights to skip quantized (Byte/Char) tensors.
        # HF's from_pretrained calls model.apply(_initialize_weights) which
        # triggers normal_() on quantized weights — unsupported by PyTorch.
        if load_4bit or load_8bit:
            from transformers.models.llama.modeling_llama import LlamaForCausalLM
            _orig_init_weights = LlamaForCausalLM._init_weights

            def _safe_init_weights(self, module):
                try:
                    _orig_init_weights(self, module)
                except NotImplementedError:
                    pass  # skip quantized tensors
            LlamaForCausalLM._init_weights = _safe_init_weights

        model_name = get_model_name_from_path(model_path)
        if 'llava' not in model_name.lower():
            model_name = 'llava-v1.5-13b'

        tokenizer, model, processor, context_len = load_pretrained_model(
            model_path, None, model_name,
            load_8bit=load_8bit, load_4bit=load_4bit, device=device)

        print(f"[SemanticPrecomputer] Model loaded: {model_name}, "
              f"context_len={context_len}, device={device}")

        return (model, tokenizer, processor['image'],
                IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                conv_templates, tokenizer_image_token, process_images)

    def _pretokenize_prompts(self):
        """Pre-tokenize yes/no prompt pairs for all 211 attributes."""
        print(f"[SemanticPrecomputer] Pre-tokenizing {len(self.attr_names)} "
              f"attribute prompts...")
        yes_ids_list = []
        no_ids_list = []
        diff_pos_list = []

        for attr in self.attr_names:
            attr_display = attr.replace('_', ' ')
            question = (
                f"Does this person have {attr_display}?\n"
                "Yes\nNo\nInformation not visible\n"
                "Please directly select the appropriate option from the "
                "given choices based on the image. Please return the answer "
                "directly without additional explanation."
            )
            qs = self.DEFAULT_IMAGE_TOKEN + '\n' + question

            def _make_ids(answer_word, qs=qs):
                conv = self.conv_templates["llava_v1"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], answer_word)
                prompt = conv.get_prompt()
                return self.tokenizer_image_token_fn(
                    prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX,
                    return_tensors='pt')

            ids_yes = _make_ids("Yes")
            ids_no = _make_ids("No")

            # Find first position where yes/no differ (the answer token)
            min_len = min(len(ids_yes), len(ids_no))
            diff_pos = next(
                (p for p in range(min_len - 1, -1, -1)
                 if ids_yes[p] != ids_no[p]),
                len(ids_yes) - 1)

            yes_ids_list.append(ids_yes)
            no_ids_list.append(ids_no)
            diff_pos_list.append(diff_pos)

        print(f"[SemanticPrecomputer] Tokenization complete.")
        return yes_ids_list, no_ids_list, diff_pos_list

    # ── Combined-prompt mode ──────────────────────────────────────────

    def _build_combined_prompt(self):
        """
        Build combined prompts listing attributes with teacher-forced answers.
        Auto-chunks into groups that fit within the model's context window
        (2048 tokens after visual token expansion).

        Prompt structure per chunk (LLaVA-v1 conversation template):
          USER: <image>
          For each facial attribute, predict 1 if present or 0 if absent.
          ASSISTANT: black hair: 1\\nblonde hair: 1\\n...
        """
        context_limit = 2048

        attr_display = [a.replace('_', ' ') for a in self.attr_names]
        question = (
            "For each facial attribute, predict 1 if present or 0 if "
            "absent."
        )
        qs = self.DEFAULT_IMAGE_TOKEN + '\n' + question

        def _make_chunk_ids(attr_subset):
            answer_yes = '\n'.join(f' {a}: 1' for a in attr_subset)
            answer_no = '\n'.join(f' {a}: 0' for a in attr_subset)

            conv_y = self.conv_templates["llava_v1"].copy()
            conv_y.append_message(conv_y.roles[0], qs)
            conv_y.append_message(conv_y.roles[1], answer_yes)
            ids_y = self.tokenizer_image_token_fn(
                conv_y.get_prompt(), self.tokenizer,
                self.IMAGE_TOKEN_INDEX, return_tensors='pt')

            conv_n = self.conv_templates["llava_v1"].copy()
            conv_n.append_message(conv_n.roles[0], qs)
            conv_n.append_message(conv_n.roles[1], answer_no)
            ids_n = self.tokenizer_image_token_fn(
                conv_n.get_prompt(), self.tokenizer,
                self.IMAGE_TOKEN_INDEX, return_tensors='pt')
            return ids_y, ids_n

        # Estimate max attrs per chunk that fits context
        ids_1, _ = _make_chunk_ids(attr_display[:1])
        ids_10, _ = _make_chunk_ids(attr_display[:10])
        overhead = len(ids_1) + self.visual_offset
        tokens_per_attr = (len(ids_10) - len(ids_1)) / 9.0
        max_per_chunk = max(1, int((context_limit - overhead) / tokens_per_attr))

        # Build chunks
        self._combined_chunks = []
        offset = 0

        while offset < len(self.attr_names):
            size = min(max_per_chunk, len(self.attr_names) - offset)
            chunk_attrs = attr_display[offset:offset + size]
            ids_yes, ids_no = _make_chunk_ids(chunk_attrs)
            expanded = len(ids_yes) + self.visual_offset

            # Shrink if still over limit
            while expanded > context_limit and size > 1:
                size = max(1, size - 10)
                chunk_attrs = attr_display[offset:offset + size]
                ids_yes, ids_no = _make_chunk_ids(chunk_attrs)
                expanded = len(ids_yes) + self.visual_offset

            assert len(ids_yes) == len(ids_no)

            answer_positions = [
                i for i in range(len(ids_yes))
                if ids_yes[i] != ids_no[i]
            ]
            assert len(answer_positions) == size, (
                f"Found {len(answer_positions)} answer positions, "
                f"expected {size}")

            img_tok_pos = (
                ids_yes == self.IMAGE_TOKEN_INDEX
            ).nonzero(as_tuple=True)[0][0].item()

            self._combined_chunks.append({
                'ids': ids_yes,
                'answer_positions': answer_positions,
                'img_tok_pos': img_tok_pos,
                'yes_tok': ids_yes[answer_positions[0]].item(),
                'no_tok': ids_no[answer_positions[0]].item(),
                'attr_offset': offset,
                'chunk_size': size,
            })
            offset += size

        sizes = [c['chunk_size'] for c in self._combined_chunks]
        print(f"[SemanticPrecomputer] Combined prompts: "
              f"{len(self._combined_chunks)} chunk(s) of {sizes} attrs, "
              f"context_limit={context_limit}")

    def extract_attributes_combined(self, image_tensor) -> np.ndarray:
        """
        Combined-prompt extraction: auto-chunked LLM forward passes for
        all 211 attributes.

        Returns:
            (211,) float32 numpy array of P(attribute present) in [0, 1].
        """
        device = self.model.device
        scores = np.zeros(len(self.attr_names), dtype=np.float32)

        for chunk in self._combined_chunks:
            input_ids = chunk['ids'].unsqueeze(0).to(device)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    images=image_tensor,
                    return_dict=True,
                    use_cache=False,
                )
            logits = outputs.logits[0]  # (seq_len_expanded, vocab)

            for i, pos in enumerate(chunk['answer_positions']):
                if pos > chunk['img_tok_pos']:
                    logit_pos = pos + self.visual_offset - 1
                else:
                    logit_pos = pos - 1
                logit_pos = max(0, logit_pos)

                yes_logit = logits[logit_pos, chunk['yes_tok']].float()
                no_logit = logits[logit_pos, chunk['no_tok']].float()
                scores[chunk['attr_offset'] + i] = torch.sigmoid(
                    yes_logit - no_logit).item()

        return scores

    # ── Common helpers ────────────────────────────────────────────────

    def _prepare_image_tensor(self, img_pil):
        """Process a PIL image into the Face-LLaVA image tensor list."""
        image_tensor = self.process_images_fn(
            [img_pil], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            return [
                img.squeeze(0).to(self.model.device, dtype=torch.float16)
                if img.ndim == 4
                else img.to(self.model.device, dtype=torch.float16)
                for img in image_tensor]
        if image_tensor.ndim == 4:
            return [image_tensor[i].to(self.model.device, dtype=torch.float16)
                    for i in range(image_tensor.shape[0])]
        return [image_tensor.to(self.model.device, dtype=torch.float16)]

    def extract_attributes(self, image_tensor) -> np.ndarray:
        """
        Extract 211 attribute scores via teacher-forcing.
        Dispatches to combined or per-attribute mode based on self.mode.

        Returns:
            (211,) float32 numpy array of P(attribute present) in [0, 1].
        """
        if self.mode == 'combined':
            return self.extract_attributes_combined(image_tensor)

        # per_attr mode
        n_attrs = len(self.attr_names)
        scores = np.zeros(n_attrs, dtype=np.float32)
        device = self.model.device

        for batch_start in range(0, n_attrs, self.attr_batch_size):
            batch_end = min(batch_start + self.attr_batch_size, n_attrs)
            batch_ids_yes = self.yes_ids_list[batch_start:batch_end]
            batch_diff = self.diff_pos_list[batch_start:batch_end]

            # Pad to same length within this batch
            max_len = max(len(ids) for ids in batch_ids_yes)
            pad_id = self.tokenizer.pad_token_id or 0

            padded = torch.full(
                (len(batch_ids_yes), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros_like(padded)
            for bi, ids in enumerate(batch_ids_yes):
                padded[bi, :len(ids)] = ids
                attention_mask[bi, :len(ids)] = 1

            padded = padded.to(device)
            attention_mask = attention_mask.to(device)

            # Replicate image tensor for each attribute in this batch
            batch_images = image_tensor * len(batch_ids_yes)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=padded,
                    attention_mask=attention_mask,
                    images=batch_images,
                    return_dict=True,
                    use_cache=False,
                )
            logits = outputs.logits  # (batch, seq_len_expanded, vocab)

            # Extract yes/no logit at each attribute's answer position
            for bi, diff_pos in enumerate(batch_diff):
                # Map diff_pos from input-ids space to expanded-logits space
                if diff_pos > self.img_tok_pos:
                    logit_pos = diff_pos + self.visual_offset - 1
                else:
                    logit_pos = diff_pos - 1
                logit_pos = max(0, logit_pos)

                yes_tok = self.yes_ids_list[
                    batch_start + bi][diff_pos].item()
                no_tok = self.no_ids_list[
                    batch_start + bi][diff_pos].item()

                yes_logit = logits[bi, logit_pos, yes_tok].float()
                no_logit = logits[bi, logit_pos, no_tok].float()

                scores[batch_start + bi] = torch.sigmoid(
                    yes_logit - no_logit).item()

        return scores

    def process_single_image(self, image_path: str) -> np.ndarray:
        """
        Extract 211 semantic attributes for a single image.

        Returns:
            (211,) float32 numpy array of attribute probabilities.
        """
        img_pil = Image.open(image_path).convert('RGB')
        image_tensor = self._prepare_image_tensor(img_pil)
        return self.extract_attributes(image_tensor)

    def process_video(self, frame_paths: list) -> dict:
        """
        Extract 211 semantic attributes for all frames in a video.

        Returns:
            {'features': Tensor(n_frames, 211), 'frame_paths': List[str]}
        """
        all_scores = []
        valid_paths = []

        for fp in frame_paths:
            if not os.path.exists(fp):
                continue
            try:
                scores = self.process_single_image(fp)
                all_scores.append(scores)
                valid_paths.append(fp)
            except Exception as e:
                print(f"  Warning: failed on {fp}: {e}")

        if not all_scores:
            return None

        features = torch.from_numpy(np.stack(all_scores))
        return {
            'features': features,
            'frame_paths': valid_paths,
        }

    def demo(self, image_path: str, save_path: str = None,
             top_k: int = 30):
        """
        Run single-image demo using the exact same extraction path as
        batch precomputation, then generate plots and text report.
        """
        from demo_plot_helpers import report_and_plot

        print("=" * 70)
        print("  SemanticPrecomputer — Single Image Demo")
        print(f"  Image:      {image_path}")
        print(f"  Mode:       {self.mode}")
        if self.mode == 'per_attr':
            print(f"  Attr batch: {self.attr_batch_size} attrs/pass "
                  f"({-(-len(self.attr_names) // self.attr_batch_size)} "
                  f"passes)")
        else:
            n_chunks = len(self._combined_chunks)
            print(f"  Chunks:     {n_chunks} pass(es)")
        print("=" * 70)

        scores = self.process_single_image(image_path)
        img_pil = Image.open(image_path).convert('RGB')

        n_present = (scores > 0.5).sum()
        print(f"\nResults: {n_present}/{len(scores)} attributes present "
              f"(P>0.5)")
        print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}], "
              f"mean: {scores.mean():.3f}, std: {scores.std():.3f}")

        sorted_idx = np.argsort(scores)[::-1]
        print(f"\n  PRESENT (P > 0.5):")
        for i in sorted_idx:
            if scores[i] <= 0.5:
                break
            print(f"    {self.attr_names[i].replace('_', ' '):<35s} "
                  f"{scores[i]:.3f}")

        report_and_plot(
            img_pil, scores, self.attr_names, image_path,
            save_path=save_path, top_k=top_k, suffix='precomputer')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute Face-LLaVA semantic features')
    parser.add_argument('--detector_path', type=str, required=True)
    parser.add_argument('--attr_batch_size', type=int, default=128,
                        help='Number of attributes per forward pass '
                             '(higher=faster, more VRAM)')
    parser.add_argument('--output_dir', type=str, default='facellava_semantic')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--compression', type=str, default=None)
    parser.add_argument('--load_4bit', action='store_true')
    parser.add_argument('--load_8bit', action='store_true')
    parser.add_argument('--mode', type=str, default='combined',
                        choices=['combined', 'per_attr'],
                        help='combined: structured prompt auto-chunked '
                             '(~2 passes/image). '
                             'per_attr: individual yes/no prompts batched.')
    # Demo mode
    parser.add_argument('--demo', type=str, default=None,
                        help='Single image path for demo/validation mode')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Output path for demo plot (demo mode only)')
    parser.add_argument('--top_k', type=int, default=30,
                        help='Top-K attributes to show in demo plot')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.detector_path)
    if args.compression:
        config['compression'] = args.compression

    device = args.device if torch.cuda.is_available() else 'cpu'

    print(f"\nBuilding SemanticPrecomputer on {device}...")
    precomputer = SemanticPrecomputer(
        config, device,
        attr_batch_size=args.attr_batch_size,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        mode=args.mode)

    # --- Demo mode: single image validation ---
    if args.demo:
        if not os.path.exists(args.demo):
            print(f"ERROR: Image not found: {args.demo}")
            sys.exit(1)
        precomputer.demo(args.demo, save_path=args.save_path,
                         top_k=args.top_k)
        return

    # --- Batch precomputation mode ---
    print("=" * 60)
    print("Face-LLaVA Semantic Feature Precomputation")
    print("=" * 60)

    videos = collect_videos_from_json(config)
    if not videos:
        print("No videos found. Check your config and dataset JSONs.")
        return

    print(f"  Attr batch size: {args.attr_batch_size}")
    print(f"  Output subdir:   {args.output_dir}")

    n_skipped = 0
    n_processed = 0
    n_failed = 0

    for vid_key, vid_info in tqdm(videos.items(), desc="Videos"):
        output_base = vid_info['output_dir']
        video_id = vid_info['video_id']

        output_dir = os.path.join(output_base, args.output_dir)
        output_path = os.path.join(output_dir, f'{video_id}.pt')

        if args.skip_existing and os.path.exists(output_path):
            n_skipped += 1
            continue

        try:
            result = precomputer.process_video(vid_info['frames'])
            if result is None:
                n_failed += 1
                continue

            os.makedirs(output_dir, exist_ok=True)
            torch.save(result, output_path)
            n_processed += 1

        except Exception as e:
            print(f"  Error processing {vid_key}: {e}")
            n_failed += 1
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"Precomputation complete!")
    print(f"  Processed: {n_processed}")
    print(f"  Skipped:   {n_skipped}")
    print(f"  Failed:    {n_failed}")
    print(f"  Output:    */{args.output_dir}/<video_id>.pt")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
