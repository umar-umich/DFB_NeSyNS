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
                 load_4bit: bool = False, load_8bit: bool = False):
        """
        Per-attribute teacher-forcing: each of 211 attributes gets its own
        FaceBench-native yes/no prompt. Attributes are batched together
        (attr_batch_size per forward pass). Multiple images can be batched
        via process_video(image_batch_size=N).

        With attr_batch_size=112: ceil(211/112) = 2 LLM passes per image.
        With image_batch_size=N: processes N images in each pass.
        """
        self.device = device
        self.attr_batch_size = attr_batch_size
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

        # Pre-tokenize all 211 individual attribute prompts (CPU, done once)
        (self.yes_ids_list, self.no_ids_list,
         self.diff_pos_list) = self._pretokenize_prompts()
        self.img_tok_pos = (
            self.yes_ids_list[0] == self.IMAGE_TOKEN_INDEX
        ).nonzero(as_tuple=True)[0][0].item()

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

    def _encode_image(self, image_tensor_list) -> torch.Tensor:
        """
        Run the vision tower ONCE on a single image to get visual embeddings.

        Args:
            image_tensor_list: list of tensors from _prepare_image_tensor
                               (typically 1 tensor of shape (3, 336, 336))

        Returns:
            (num_patches, hidden_dim) visual embeddings, e.g. (576, 4096)
        """
        img = image_tensor_list[0]
        if img.ndim == 3:
            img = img.unsqueeze(0)  # (1, 3, H, W)
        with torch.inference_mode():
            visual_emb = self.model.encode_images(img)  # (1, 576, 4096)
        return visual_emb[0]  # (576, 4096)

    def _encode_images_batch(self, image_tensor_lists: list) -> list:
        """
        Batch-encode multiple images through the vision tower (CLIP ViT-L).

        CLIP ViT-L is small (~400M params) so batching is safe and fast.
        This avoids N sequential vision tower calls.

        Args:
            image_tensor_lists: list of N image tensor lists from
                                _prepare_image_tensor

        Returns:
            list of N tensors, each (num_patches, hidden_dim)
        """
        # Stack all images into (N, 3, H, W)
        imgs = []
        for itl in image_tensor_lists:
            img = itl[0]
            if img.ndim == 3:
                img = img.unsqueeze(0)
            imgs.append(img)
        batch = torch.cat(imgs, dim=0)  # (N, 3, H, W)

        with torch.inference_mode():
            visual_embs = self.model.encode_images(batch)  # (N, 576, 4096)

        return [visual_embs[i] for i in range(visual_embs.shape[0])]

    def _build_inputs_embeds(self, input_ids_batch, visual_emb):
        """
        Build inputs_embeds by merging text token embeddings with pre-computed
        visual embeddings. Replicates what prepare_inputs_labels_for_multimodal
        does, but reuses a single pre-computed visual embedding.

        Args:
            input_ids_batch: (N, seq_len) padded token IDs with IMAGE_TOKEN_INDEX
            visual_emb: (num_patches, hidden_dim) from _encode_image

        Returns:
            inputs_embeds: (N, expanded_seq_len, hidden_dim)
            attention_mask: (N, expanded_seq_len)
        """
        device = self.model.device
        N, seq_len = input_ids_batch.shape
        num_patches = visual_emb.shape[0]  # 576

        # Get text embeddings for all non-image tokens
        embed_fn = self.model.get_model().embed_tokens

        all_embeds = []
        all_masks = []

        for i in range(N):
            ids = input_ids_batch[i]
            # Find where the image token is
            img_positions = (ids == self.IMAGE_TOKEN_INDEX).nonzero(
                as_tuple=True)[0]

            if len(img_positions) == 0:
                # No image token — just embed text
                text_emb = embed_fn(ids)
                mask = (ids != (self.tokenizer.pad_token_id or 0)).long()
                all_embeds.append(text_emb)
                all_masks.append(mask)
                continue

            # Split around image token positions (typically just 1 image)
            segments = []
            masks = []
            prev_end = 0
            pad_id = self.tokenizer.pad_token_id or 0

            for img_pos in img_positions:
                img_pos = img_pos.item()
                # Text tokens before image
                if img_pos > prev_end:
                    text_ids = ids[prev_end:img_pos]
                    text_emb = embed_fn(text_ids)
                    segments.append(text_emb)
                    masks.append(
                        (text_ids != pad_id).long())

                # Insert visual embeddings (replacing the single image token)
                segments.append(visual_emb)
                masks.append(torch.ones(num_patches, dtype=torch.long,
                                        device=device))
                prev_end = img_pos + 1

            # Text tokens after last image token
            if prev_end < seq_len:
                text_ids = ids[prev_end:]
                text_emb = embed_fn(text_ids)
                segments.append(text_emb)
                masks.append(
                    (text_ids != pad_id).long())

            all_embeds.append(torch.cat(segments, dim=0))
            all_masks.append(torch.cat(masks, dim=0))

        # Pad to same length
        max_len = max(e.shape[0] for e in all_embeds)
        hidden_dim = visual_emb.shape[1]

        inputs_embeds = torch.zeros(
            N, max_len, hidden_dim, dtype=visual_emb.dtype, device=device)
        attention_mask = torch.zeros(
            N, max_len, dtype=torch.long, device=device)

        for i, (emb, mask) in enumerate(zip(all_embeds, all_masks)):
            cur_len = emb.shape[0]
            inputs_embeds[i, :cur_len] = emb
            attention_mask[i, :cur_len] = mask

        return inputs_embeds, attention_mask

    def extract_attributes(self, image_tensor) -> np.ndarray:
        """
        Extract 211 attributes for a single image.

        Encodes image ONCE through vision tower (CLIP ViT-L, small),
        then runs LLM-only forward passes for attribute batches.

        Returns:
            (211,) float32 numpy array of P(attribute present) in [0, 1].
        """
        visual_emb = self._encode_image(image_tensor)  # (576, 4096)
        return self._extract_from_visual_emb(visual_emb)

    def process_single_image(self, image_path: str) -> np.ndarray:
        """
        Extract 211 semantic attributes for a single image.

        Returns:
            (211,) float32 numpy array of attribute probabilities.
        """
        img_pil = Image.open(image_path).convert('RGB')
        image_tensor = self._prepare_image_tensor(img_pil)
        return self.extract_attributes(image_tensor)

    def _extract_from_visual_emb(self, visual_emb: torch.Tensor) -> np.ndarray:
        """
        Run LLM-only attribute extraction using a pre-computed visual embedding.

        This is the core LLM loop — called once per image. The vision tower
        has already run; this only does LLM forward passes.

        Args:
            visual_emb: (num_patches, hidden_dim) from _encode_image

        Returns:
            (211,) float32 numpy array of P(attribute present).
        """
        n_attrs = len(self.attr_names)
        scores = np.zeros(n_attrs, dtype=np.float32)
        device = self.model.device

        for batch_start in range(0, n_attrs, self.attr_batch_size):
            batch_end = min(batch_start + self.attr_batch_size, n_attrs)
            batch_ids_yes = self.yes_ids_list[batch_start:batch_end]
            batch_diff = self.diff_pos_list[batch_start:batch_end]

            max_len = max(len(ids) for ids in batch_ids_yes)
            pad_id = self.tokenizer.pad_token_id or 0

            padded = torch.full(
                (len(batch_ids_yes), max_len), pad_id, dtype=torch.long)
            for bi, ids in enumerate(batch_ids_yes):
                padded[bi, :len(ids)] = ids
            padded = padded.to(device)

            inputs_embeds, attention_mask = self._build_inputs_embeds(
                padded, visual_emb)

            with torch.inference_mode():
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    images=None,
                    return_dict=True,
                    use_cache=False,
                )
            logits = outputs.logits

            for bi, diff_pos in enumerate(batch_diff):
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

            del outputs, logits, inputs_embeds, attention_mask
            torch.cuda.empty_cache()

        return scores

    def process_video(self, frame_paths: list,
                      image_batch_size: int = 8) -> dict:
        """
        Extract 211 semantic attributes for all frames in a video.

        Two-stage pipeline that respects model sizes:
          1. Vision tower (CLIP ViT-L, ~400M): batch-encode image_batch_size
             images at once — fast and memory-safe.
          2. LLM (LLaMA 13B): loop one image at a time, running
             attr_batch_size attribute prompts per forward pass.

        image_batch_size controls the CLIP vision tower batch, NOT the
        LLM batch. The LLM always processes one image's attributes at
        a time to avoid OOM.

        Returns:
            {'features': Tensor(n_frames, 211), 'frame_paths': List[str]}
        """
        valid_pils = []
        valid_paths = []
        for fp in frame_paths:
            if not os.path.exists(fp):
                continue
            try:
                img = Image.open(fp).convert('RGB')
                valid_pils.append(img)
                valid_paths.append(fp)
            except Exception as e:
                print(f"  Warning: failed to load {fp}: {e}")

        if not valid_pils:
            return None

        all_scores = []

        # Process in chunks of image_batch_size
        for i in range(0, len(valid_pils), image_batch_size):
            batch_pils = valid_pils[i:i + image_batch_size]

            # Stage 1: Batch-encode through vision tower (CLIP, small model)
            batch_tensors = [
                self._prepare_image_tensor(p) for p in batch_pils]

            if len(batch_tensors) == 1:
                visual_embs = [self._encode_image(batch_tensors[0])]
            else:
                visual_embs = self._encode_images_batch(batch_tensors)

            # Stage 2: Loop LLM per image (LLaMA 13B, big model)
            for visual_emb in visual_embs:
                scores = self._extract_from_visual_emb(visual_emb)
                all_scores.append(scores[np.newaxis, :])

        features = torch.from_numpy(np.concatenate(all_scores, axis=0))
        return {
            'features': features,
            'frame_paths': valid_paths,
        }

    def demo(self, demo_path: str, save_path: str = None,
             top_k: int = 30, image_batch_size: int = 8):
        """
        Demo mode: validates extraction on a single image or a folder of
        frames (to test batched extraction before full dataset run).

        If demo_path is a directory, uses the two-stage pipeline
        (batch CLIP, loop LLM per image) — same path as process_video.

        If demo_path is a single image, processes it individually.
        """
        from demo_plot_helpers import report_and_plot

        n_passes = -(-len(self.attr_names) // self.attr_batch_size)

        # ── Folder mode: batch-process all frames ─────────────────────
        if os.path.isdir(demo_path):
            exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
            frame_paths = sorted([
                os.path.join(demo_path, f) for f in os.listdir(demo_path)
                if os.path.splitext(f)[1].lower() in exts
            ])
            if not frame_paths:
                print(f"ERROR: No images found in {demo_path}")
                return

            print("=" * 70)
            print("  SemanticPrecomputer — Batched Folder Demo")
            print(f"  Folder:     {demo_path}")
            print(f"  Frames:     {len(frame_paths)}")
            print(f"  Img batch:  {image_batch_size}")
            print(f"  Attr batch: {self.attr_batch_size} attrs/pass "
                  f"({n_passes} passes)")
            print("=" * 70)

            # Load all images
            pils, valid_paths = [], []
            for fp in frame_paths:
                try:
                    pils.append(Image.open(fp).convert('RGB'))
                    valid_paths.append(fp)
                except Exception as e:
                    print(f"  Warning: skipping {fp}: {e}")

            if not pils:
                print("ERROR: No valid images loaded.")
                return

            # Two-stage extraction (same code path as process_video)
            # Stage 1: batch CLIP vision tower, Stage 2: loop LLM per image
            import time
            t0 = time.time()
            all_scores = []
            for i in range(0, len(pils), image_batch_size):
                batch = pils[i:i + image_batch_size]
                tensors = [self._prepare_image_tensor(p) for p in batch]
                # Batch-encode through vision tower (CLIP, small)
                if len(tensors) == 1:
                    visual_embs = [self._encode_image(tensors[0])]
                else:
                    visual_embs = self._encode_images_batch(tensors)
                # Loop LLM per image (LLaMA 13B, big)
                for ve in visual_embs:
                    scores = self._extract_from_visual_emb(ve)
                    all_scores.append(scores[np.newaxis, :])
            scores_mat = np.concatenate(all_scores, axis=0)  # (N, 211)
            elapsed = time.time() - t0

            n_frames = scores_mat.shape[0]
            print(f"\nProcessed {n_frames} frames in {elapsed:.1f}s "
                  f"({elapsed/n_frames:.2f}s/frame)")

            # Per-frame summary
            print(f"\n{'Frame':<45s} {'P>0.5':>6s}  "
                  f"{'Mean':>6s}  {'Max':>6s}")
            print("-" * 70)
            for fi in range(n_frames):
                fname = os.path.basename(valid_paths[fi])
                s = scores_mat[fi]
                print(f"  {fname:<43s} {(s > 0.5).sum():>4d}   "
                      f"{s.mean():>.3f}  {s.max():>.3f}")

            # Average across frames
            avg_scores = scores_mat.mean(axis=0)
            n_present = (avg_scores > 0.5).sum()
            print(f"\n  Average across {n_frames} frames: "
                  f"{n_present}/{len(avg_scores)} attrs present (P>0.5)")

            sorted_idx = np.argsort(avg_scores)[::-1]
            print(f"\n  TOP ATTRIBUTES (avg P > 0.5):")
            for i in sorted_idx:
                if avg_scores[i] <= 0.5:
                    break
                print(f"    {self.attr_names[i].replace('_', ' '):<35s} "
                      f"{avg_scores[i]:.3f}")

            # Compare with single-image extraction of first frame
            print(f"\n  Validation: single-image vs batch for frame 0...")
            single_scores = self.process_single_image(valid_paths[0])
            diff = np.abs(single_scores - scores_mat[0])
            print(f"    Max diff:  {diff.max():.6f}")
            print(f"    Mean diff: {diff.mean():.6f}")
            if diff.max() < 1e-4:
                print(f"    PASS: batch matches single-image extraction")
            else:
                print(f"    WARNING: batch/single mismatch > 1e-4")

            # Plot ALL frames
            base_dir = save_path or demo_path
            if save_path and not os.path.isdir(save_path):
                base_dir = os.path.dirname(save_path) or demo_path
            plot_dir = os.path.join(base_dir, 'demo_plots')
            os.makedirs(plot_dir, exist_ok=True)

            for fi in range(n_frames):
                fname = os.path.splitext(
                    os.path.basename(valid_paths[fi]))[0]
                frame_save = os.path.join(
                    plot_dir, f'{fname}_facebench_batch_demo.png')
                report_and_plot(
                    pils[fi], scores_mat[fi], self.attr_names,
                    valid_paths[fi], save_path=frame_save,
                    top_k=top_k, suffix='batch_demo')

            print(f"\n  All {n_frames} frame plots saved to: {plot_dir}")
            return

        # ── Single image mode ────────────────────────────────────────
        print("=" * 70)
        print("  SemanticPrecomputer — Single Image Demo")
        print(f"  Image:      {demo_path}")
        print(f"  Attr batch: {self.attr_batch_size} attrs/pass "
              f"({n_passes} passes)")
        print("=" * 70)

        scores = self.process_single_image(demo_path)
        img_pil = Image.open(demo_path).convert('RGB')

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
            img_pil, scores, self.attr_names, demo_path,
            save_path=save_path, top_k=top_k, suffix='precomputer')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute Face-LLaVA semantic features')
    parser.add_argument('--detector_path', type=str, required=True)
    parser.add_argument('--attr_batch_size', type=int, default=211,
                        help='Number of attributes per forward pass '
                             '(higher=faster, more VRAM)')
    parser.add_argument('--output_dir', type=str, default='facellava_semantic')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--compression', type=str, default=None)
    parser.add_argument('--load_4bit', action='store_true')
    parser.add_argument('--load_8bit', action='store_true')
    parser.add_argument('--image_batch_size', type=int, default=8,
                        help='Images batched through CLIP vision tower. '
                             'LLM always processes one image at a time. '
                             'Higher = faster vision encoding, safe for '
                             'VRAM since CLIP is small (~400M params).')
    # Demo mode
    parser.add_argument('--demo', type=str, default=None,
                        help='Image path or frames folder for demo. '
                             'Folder mode tests batched extraction.')
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
        load_8bit=args.load_8bit)

    # --- Demo mode: single image or folder validation ---
    if args.demo:
        if not os.path.exists(args.demo):
            print(f"ERROR: Path not found: {args.demo}")
            sys.exit(1)
        precomputer.demo(args.demo, save_path=args.save_path,
                         top_k=args.top_k,
                         image_batch_size=args.image_batch_size)
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
    print(f"  Image batch:     {args.image_batch_size}")
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
            result = precomputer.process_video(
                vid_info['frames'],
                image_batch_size=args.image_batch_size,
            )
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
