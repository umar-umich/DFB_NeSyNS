#!/usr/bin/env python3
"""
precompute_semantic_features.py
================================
Precompute Face-LLaVA semantic attributes for all frames and save to disk.

Runs the frozen Face-LLaVA 13B model once over every frame in the dataset
JSONs, producing per-video .pt files with shape (n_frames, 211) containing
the 211 FaceBench attribute probabilities.

Output structure (mirrors frames/ directory):
  .../original_sequences/youtube/c23/facellava_semantic/929.pt
  .../manipulated_sequences/Deepfakes/c23/facellava_semantic/802_885.pt

Each .pt file contains:
  {'features': Tensor(n_frames, 211), 'frame_paths': List[str]}

Usage:
  python preprocessing/precompute_semantic_features.py \
      --detector_path training/config/detector/nesy_defake.yaml \
      --batch_size 32 \
      --output_dir facellava_semantic

This eliminates the need to run a 13B LLM during training (~5x speedup).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Must be set BEFORE importing torch / any CUDA call so that only
# the requested GPU is visible to the entire process (including
# HuggingFace accelerate auto-dispatch inside from_pretrained).
if '--device' in sys.argv:
    _dev = sys.argv[sys.argv.index('--device') + 1]
    if _dev.startswith('cuda:'):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES',
                              os.environ.get('CUDA_VISIBLE_DEVICES', '0'))

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute Face-LLaVA semantic features for all frames')
    parser.add_argument('--detector_path', type=str, required=True,
                        help='Path to detector YAML config')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Micro-batch size for Face-LLaVA inference')
    parser.add_argument('--output_dir', type=str, default='facellava_semantic',
                        help='Output subdirectory name (placed alongside frames/)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference on')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip videos that already have precomputed features')
    parser.add_argument('--compression', type=str, default=None,
                        help='Override compression level (e.g., c23)')
    return parser.parse_args()


def load_config(detector_path):
    """Load and merge detector + train configs."""
    with open(detector_path) as f:
        config = yaml.safe_load(f)
    train_cfg_path = os.path.join(
        os.path.dirname(detector_path), '..', 'config', 'train_config.yaml')
    if not os.path.exists(train_cfg_path):
        train_cfg_path = './training/config/train_config.yaml'
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path) as f:
            train_config = yaml.safe_load(f)
        if 'label_dict' in config:
            train_config['label_dict'] = config['label_dict']
        config.update(train_config)
    return config


def collect_videos_from_json(config):
    """
    Parse dataset JSONs and return a dict:
      video_id -> {'frames': [path1, path2, ...], 'output_dir': str}
    """
    json_folder = config['dataset_json_folder']
    compression = config.get('compression', 'c23')

    all_datasets = set()
    for key in ('train_dataset', 'test_dataset'):
        ds = config.get(key, [])
        if isinstance(ds, str):
            ds = [ds]
        all_datasets.update(ds)

    videos = {}
    total_frames = 0

    for dataset_name in sorted(all_datasets):
        json_path = os.path.join(json_folder, f'{dataset_name}.json')
        if not os.path.exists(json_path):
            print(f"  WARNING: JSON not found: {json_path}, skipping")
            continue

        with open(json_path) as f:
            data = json.load(f)

        for top_key, top_val in data.items():
            for label_key, label_val in top_val.items():
                for mode_key, mode_val in label_val.items():
                    first_val = next(iter(mode_val.values()), {})
                    if isinstance(first_val, dict) and 'frames' in first_val:
                        video_items = mode_val.items()
                    elif compression in mode_val:
                        video_items = mode_val[compression].items()
                    else:
                        continue

                    for video_id, video_data in video_items:
                        frames = video_data.get('frames', [])
                        if not frames:
                            continue

                        sample_frame = frames[0]
                        sep = '/' if '/' in sample_frame else '\\'
                        parts = sample_frame.split(sep)
                        frames_dir_idx = None
                        for pi, part in enumerate(parts):
                            if part == 'frames' or part.startswith('frames_aug_'):
                                frames_dir_idx = pi
                                break
                        if frames_dir_idx is not None:
                            base_dir = sep.join(parts[:frames_dir_idx])
                        else:
                            base_dir = os.path.dirname(
                                os.path.dirname(sample_frame))

                        vid_key = f"{base_dir}/{video_id}"
                        if vid_key not in videos:
                            videos[vid_key] = {
                                'frames': sorted(frames),
                                'output_dir': base_dir,
                                'video_id': video_id,
                            }
                            total_frames += len(frames)
                        else:
                            existing = set(videos[vid_key]['frames'])
                            for fp in frames:
                                if fp not in existing:
                                    videos[vid_key]['frames'].append(fp)
                                    total_frames += 1
                            videos[vid_key]['frames'].sort()

    print(f"\nCollected {len(videos)} videos, {total_frames} total frames")
    return videos


def build_extractor(config, device, batch_size):
    """Build the Face-LLaVA semantic extractor."""
    from copy import deepcopy

    extract_config = deepcopy(config)
    sem_cfg = extract_config.get('semantic_attributes', {})

    sem_cfg['backend'] = 'face_llava'
    sem_cfg['micro_batch_size'] = batch_size
    sem_cfg['use_llm'] = sem_cfg.get('use_llm', True)
    extract_config['semantic_attributes'] = sem_cfg

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
    from networks.nesy_defake.semantic import FacialSemanticExtractor

    extractor = FacialSemanticExtractor(extract_config)
    extractor = extractor.to(device)
    extractor.eval()

    return extractor


def precompute_all(args):
    """Main precomputation loop."""
    config = load_config(args.detector_path)
    if args.compression:
        config['compression'] = args.compression

    print("=" * 60)
    print("Face-LLaVA Semantic Feature Precomputation")
    print("=" * 60)

    videos = collect_videos_from_json(config)
    if not videos:
        print("No videos found. Check your config and dataset JSONs.")
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nBuilding Face-LLaVA extractor on {device}...")
    extractor = build_extractor(config, device, args.batch_size)

    use_llm = getattr(extractor, '_use_llm', False)
    print(f"  Mode: {'Full LLM (211 attributes)' if use_llm else 'Vision-only'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output subdir: {args.output_dir}")

    n_skipped = 0
    n_processed = 0
    n_failed = 0

    for vid_key, vid_info in tqdm(videos.items(), desc="Videos"):
        output_base = vid_info['output_dir']
        video_id = vid_info['video_id']
        frames = vid_info['frames']

        output_dir = os.path.join(output_base, args.output_dir)
        output_path = os.path.join(output_dir, f'{video_id}.pt')

        if args.skip_existing and os.path.exists(output_path):
            n_skipped += 1
            continue

        try:
            from PIL import Image
            import numpy as np

            frame_tensors = []
            valid_paths = []

            for fp in frames:
                if not os.path.exists(fp):
                    continue
                try:
                    img = Image.open(fp).convert('RGB')
                    img_np = np.array(img).astype(np.float32) / 255.0
                    img_t = torch.from_numpy(img_np).permute(2, 0, 1)
                    frame_tensors.append(img_t)
                    valid_paths.append(fp)
                except Exception as e:
                    print(f"  Warning: failed to load {fp}: {e}")

            if not frame_tensors:
                n_failed += 1
                continue

            batch = torch.stack(frame_tensors).to(device)

            with torch.no_grad():
                features = extractor(raw_images=batch)  # (N, 211)

            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'features': features.cpu(),
                'frame_paths': valid_paths,
            }, output_path)

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


# ---------------------------------------------------------------------------
#  Core fix: batched per-attribute teacher-forcing
# ---------------------------------------------------------------------------

def build_attr_prompt_ids(attr_display, tokenizer, image_token_index,
                          conv_templates, default_image_token,
                          tokenizer_image_token_fn):
    """
    Build (ids_yes, ids_no) token-id tensors for a SINGLE attribute.

    Prompt template (same as the working generate mode):
        <image>
        Does this person have {attr}?
        Yes
        No
        Information not visible
        Please directly select the appropriate option ...

    Answer forced to "Yes" / "No" — the logit difference at the answer
    position is the signal.  This is per-attribute so every prompt is
    short, well-formed, and matches what the model was trained on.
    """
    question = (
        f"Does this person have {attr_display}?\n"
        "Yes\nNo\nInformation not visible\n"
        "Please directly select the appropriate option from the given choices "
        "based on the image. Please return the answer directly without "
        "additional explanation."
    )
    qs = default_image_token + '\n' + question

    def _make_ids(answer_word):
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], answer_word)
        prompt = conv.get_prompt()
        ids = tokenizer_image_token_fn(
            prompt, tokenizer, image_token_index, return_tensors='pt')
        return ids

    return _make_ids("Yes"), _make_ids("No")


def batched_teacher_force(
        image_tensor,       # list of 1 processed image tensor (already on device)
        model,
        tokenizer,
        image_token_index,
        conv_templates,
        default_image_token,
        tokenizer_image_token_fn,
        attr_names,
        attr_prompt_batch_size=16,  # how many attributes per forward pass
        device='cuda',
):
    """
    Batched per-attribute teacher-forcing.

    For each attribute we build a short, clean yes/no prompt (matching the
    format used by the working generate mode), then run them in small
    batches through a SINGLE padded forward pass per group.

    Speed vs. original generate:   ~attr_prompt_batch_size x faster
    Accuracy vs. compact-prompt TF: much better — each prompt is short and
                                    well-formed, no token-alignment issues.

    Returns: np.ndarray of shape (211,) with P(present) in [0, 1].
    """
    import numpy as np

    # Pre-tokenise all attribute prompts once (CPU, cheap)
    yes_ids_list = []
    no_ids_list  = []
    diff_pos_list = []   # position in ids where yes vs no differ

    for attr in attr_names:
        attr_display = attr.replace('_', ' ')
        ids_yes, ids_no = build_attr_prompt_ids(
            attr_display, tokenizer, image_token_index,
            conv_templates, default_image_token, tokenizer_image_token_fn)

        # Find first position where the two token sequences differ
        # (that is the forced answer token).
        min_len = min(len(ids_yes), len(ids_no))
        diff_pos = None
        for p in range(min_len - 1, -1, -1):   # search from the end
            if ids_yes[p] != ids_no[p]:
                diff_pos = p
                break
        if diff_pos is None:
            # Fallback: last token of yes sequence
            diff_pos = len(ids_yes) - 1

        yes_ids_list.append(ids_yes)
        no_ids_list.append(ids_no)
        diff_pos_list.append(diff_pos)

    n_attrs = len(attr_names)
    scores = np.zeros(n_attrs, dtype=np.float32)

    # Visual offset: the <image> placeholder expands to num_patches tokens
    image_tower = model.get_image_tower()
    num_patches  = image_tower.num_patches   # typically 576
    visual_offset = num_patches - 1          # 575

    # Find image token position from the first prompt (same for all)
    img_tok_pos = (yes_ids_list[0] == image_token_index).nonzero(
        as_tuple=True)[0]
    assert len(img_tok_pos) == 1
    img_tok_pos = img_tok_pos[0].item()

    # Process in small batches of attributes
    for batch_start in range(0, n_attrs, attr_prompt_batch_size):
        batch_end = min(batch_start + attr_prompt_batch_size, n_attrs)
        batch_ids_yes = yes_ids_list[batch_start:batch_end]
        batch_diff    = diff_pos_list[batch_start:batch_end]

        # Pad to same length within this batch
        max_len = max(len(ids) for ids in batch_ids_yes)
        pad_id  = tokenizer.pad_token_id or 0

        padded = torch.full(
            (len(batch_ids_yes), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(padded)
        for bi, ids in enumerate(batch_ids_yes):
            padded[bi, :len(ids)] = ids
            attention_mask[bi, :len(ids)] = 1

        padded         = padded.to(device)
        attention_mask = attention_mask.to(device)

        # Replicate the single image tensor for each item in this batch
        batch_images = image_tensor * len(batch_ids_yes)

        with torch.inference_mode():
            outputs = model(
                input_ids=padded,
                attention_mask=attention_mask,
                images=batch_images,
                return_dict=True,
                use_cache=False,
            )
        logits = outputs.logits   # (batch, seq_len_expanded, vocab)

        # Extract yes/no logit at each attribute's answer position
        for bi, (ids_yes, diff_pos) in enumerate(
                zip(batch_ids_yes, batch_diff)):

            # Map diff_pos from input-ids space to expanded-logits space
            if diff_pos > img_tok_pos:
                logit_pos = diff_pos + visual_offset - 1
            else:
                logit_pos = diff_pos - 1
            logit_pos = max(0, logit_pos)

            # Get the token ids for "Yes" and "No" at this position
            yes_tok = yes_ids_list[batch_start + bi][diff_pos].item()
            no_tok  = no_ids_list [batch_start + bi][diff_pos].item()

            yes_logit = logits[bi, logit_pos, yes_tok].float()
            no_logit  = logits[bi, logit_pos, no_tok ].float()

            scores[batch_start + bi] = torch.sigmoid(
                yes_logit - no_logit).item()

    return scores


def demo_single_image():
    """
    Validate FaceBench Face-LLaVA on a single image.

    Modes:
      --mode describe       : Free-form facial description.
      --mode teacher_force  : PATCHED batched per-attribute teacher-forcing.
                              One padded forward pass per group of
                              --tf_batch_size attributes (default 16).
                              ~10-15× faster than generate, accurate results.
      --mode generate       : Per-attribute yes/no via model.generate (211 passes).
                              Slow but matches FaceBench eval exactly.

    Usage:
      CUDA_VISIBLE_DEVICES=1 python preprocessing/precompute_semantic_features.py \
          --demo /path/to/face.jpg --mode teacher_force

      CUDA_VISIBLE_DEVICES=1 python preprocessing/precompute_semantic_features.py \
          --demo /path/to/face.jpg --mode teacher_force --tf_batch_size 32
    """
    import numpy as np
    from PIL import Image

    FACE_LLAVA_REPO = '/data/umar/Repos/Face-LLaVA'
    sys.path.insert(0, FACE_LLAVA_REPO)

    import types

    _ft = types.ModuleType('torchvision.transforms.functional_tensor')
    sys.modules.setdefault('torchvision.transforms.functional_tensor', _ft)

    _tv = types.ModuleType('torchvision.transforms._transforms_video')
    for _cls_name in ['NormalizeVideo', 'RandomCropVideo', 'RandomHorizontalFlipVideo',
                      'CenterCropVideo', 'RandomResizedCropVideo', 'UniformTemporalSubsample']:
        setattr(_tv, _cls_name, type(_cls_name, (), {}))
    sys.modules.setdefault('torchvision.transforms._transforms_video', _tv)

    try:
        from facellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from facellava.conversation import conv_templates, SeparatorStyle
        from facellava.model.builder import load_pretrained_model
        from facellava.utils import disable_torch_init
        from facellava.mm_utils import (
            tokenizer_image_token, get_model_name_from_path,
            KeywordsStoppingCriteria,
        )
    except ImportError as e:
        print(f"ERROR: Cannot import facellava: {e}")
        print(f"Ensure Face-LLaVA repo exists at: {FACE_LLAVA_REPO}")
        sys.exit(1)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
    from networks.nesy_defake.semantic.facial_semantic_extractor import FACEBENCH_ATTRIBUTES

    parser = argparse.ArgumentParser(
        description='Validate FaceBench Face-LLaVA on a single face image')
    parser.add_argument('--demo', type=str, required=True)
    parser.add_argument('--mode', choices=['describe', 'teacher_force', 'generate'],
                        default='teacher_force')
    parser.add_argument('--model_path', type=str,
                        default='/data/umar/Repos/FaceBench')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load_4bit', action='store_true')
    parser.add_argument('--load_8bit', action='store_true')
    parser.add_argument('--from_precomputed', action='store_true')
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--save_path', type=str, default=None)
    # New: batch size for the patched teacher-forcing mode
    parser.add_argument('--tf_batch_size', type=int, default=128,  # two batches for 211 attrs
                        help='Number of attributes per forward pass in '
                             'teacher_force mode (default: 16). '
                             'Higher = faster but more VRAM.')
    args = parser.parse_args()

    image_path = args.demo
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    #  --from_precomputed: load from existing .pt file
    # ------------------------------------------------------------------
    if args.from_precomputed:
        image_p  = Path(image_path).resolve()
        frame_name = image_p.stem
        video_id   = image_p.parent.name
        frames_dir = image_p.parent.parent

        pt_path = frames_dir.parent / 'facellava_semantic' / f'{video_id}.pt'
        if not pt_path.exists():
            print(f"ERROR: Precomputed file not found: {pt_path}")
            sys.exit(1)

        data       = torch.load(pt_path, map_location='cpu', weights_only=True)
        features   = data['features']
        frame_paths = data['frame_paths']

        frame_idx = None
        for i, fp in enumerate(frame_paths):
            if Path(fp).stem == frame_name:
                frame_idx = i
                break

        if frame_idx is None:
            print(f"ERROR: Frame '{frame_name}' not found in {pt_path}")
            sys.exit(1)

        scores     = features[frame_idx].numpy()
        attr_names = list(FACEBENCH_ATTRIBUTES)
        img_pil    = Image.open(image_path).convert('RGB')

        print("=" * 70)
        print("  FaceBench — Loading from precomputed .pt file")
        print(f"  Image:      {image_path}")
        print(f"  PT file:    {pt_path}")
        print(f"  Frame idx:  {frame_idx} / {len(frame_paths)}")
        print("=" * 70)

        n_present = (scores > 0.5).sum()
        print(f"\nResults: {n_present}/{len(scores)} attributes present (P>0.5)")
        _report_and_plot(img_pil, scores, attr_names, args, image_path,
                         use_llm=True, suffix='precomputed')
        return

    print("=" * 70)
    print("  FaceBench Face-LLaVA Validation Demo")
    print(f"  Image:      {image_path}")
    print(f"  Mode:       {args.mode}")
    print(f"  Model:      {args.model_path}")
    print(f"  Device:     {args.device}")
    print(f"  Quantize:   {'4-bit' if args.load_4bit else '8-bit' if args.load_8bit else 'fp16'}")
    if args.mode == 'teacher_force':
        print(f"  TF batch:   {args.tf_batch_size} attrs/pass")
    print("=" * 70)

    # Load model
    print("\nLoading Face-LLaVA model...")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if 'llava' not in model_name.lower():
        model_name = 'llava-v1.5-13b'

    from facellava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    CONFIG_MAPPING_NAMES["llava_llama"] = "LlavaConfig"
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
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

    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path, None, model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    image_processor = processor['image']
    print(f"Model loaded: {model_name}, context_len={context_len}")

    img_pil = Image.open(image_path).convert('RGB')
    from facellava.mm_utils import process_images
    image_tensor = process_images([img_pil], image_processor, model.config)

    if isinstance(image_tensor, list):
        image_tensor = [
            img.squeeze(0).to(model.device, dtype=torch.float16)
            if img.ndim == 4
            else img.to(model.device, dtype=torch.float16)
            for img in image_tensor
        ]
    else:
        if image_tensor.ndim == 4:
            image_tensor = [
                image_tensor[i].to(model.device, dtype=torch.float16)
                for i in range(image_tensor.shape[0])
            ]
        else:
            image_tensor = [image_tensor.to(model.device, dtype=torch.float16)]
    print(f"Image: {img_pil.size}, tensor: {image_tensor[0].shape}")

    # -------------------------------------------------------------------
    #  Helper: generative (unchanged from original)
    # -------------------------------------------------------------------
    def run_generate(prompt_text):
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        return tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

    attr_names = list(FACEBENCH_ATTRIBUTES)

    # =====================================================================
    #  Mode: describe
    # =====================================================================
    if args.mode == 'describe':
        prompts = [
            "Describe the person's face as seen in the photograph.",
            "Identify the visible features of the face in the image.",
            "Identify the active facial action units in the provided image.",
        ]
        print("\n" + "=" * 70)
        print("FREE-FORM FACIAL DESCRIPTION (model.generate)")
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
                f.write(f"Prompt: {prompt}\n")
                f.write(f"{'-' * 60}\n")
                f.write(f"{response}\n\n")
        _plot_description(img_pil, descriptions, save_path)
        print(f"\nPlot saved to: {save_path}")
        print(f"Text report saved to: {txt_path}")

    # =====================================================================
    #  Mode: teacher_force  (PATCHED — batched per-attribute)
    # =====================================================================
    elif args.mode == 'teacher_force':
        print(f"\nBatched per-attribute teacher-forcing")
        print(f"  {len(attr_names)} attributes, {args.tf_batch_size} per pass "
              f"→ {-(-len(attr_names) // args.tf_batch_size)} forward passes")

        scores = batched_teacher_force(
            image_tensor=image_tensor,
            model=model,
            tokenizer=tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
            conv_templates=conv_templates,
            default_image_token=DEFAULT_IMAGE_TOKEN,
            tokenizer_image_token_fn=tokenizer_image_token,
            attr_names=attr_names,
            attr_prompt_batch_size=args.tf_batch_size,
            device=model.device,
        )

        n_present = (scores > 0.5).sum()
        print(f"\n{'=' * 70}")
        print(f"Results: {n_present}/{len(scores)} attributes present (P>0.5)")
        print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}], "
              f"mean: {scores.mean():.3f}, std: {scores.std():.3f}")
        print(f"{'=' * 70}")

        sorted_idx = np.argsort(scores)[::-1]
        print(f"\n  PRESENT (P > 0.5):")
        for i in sorted_idx:
            if scores[i] <= 0.5:
                break
            print(f"    {attr_names[i].replace('_', ' '):<35s} {scores[i]:.3f}")

        _report_and_plot(img_pil, scores, attr_names, args, image_path,
                         use_llm=True, suffix='teacher_force')

    # =====================================================================
    #  Mode: generate  (211 individual passes — reference / ground truth)
    # =====================================================================
    elif args.mode == 'generate':
        print(f"\nGenerative extraction: {len(attr_names)} attributes "
              f"(211 forward passes)...")

        scores = np.zeros(len(attr_names), dtype=np.float32)
        raw_responses = {}

        for i, attr in enumerate(tqdm(attr_names, desc="Attributes")):
            attr_display = attr.replace('_', ' ')
            prompt = (
                f"Does this person have {attr_display}?\n"
                f"Yes\nNo\nInformation not visible\n"
                f"Please directly select the appropriate option from the "
                f"given choices based on the image. "
                f"Please return the answer directly without additional "
                f"explanation."
            )
            response = run_generate(prompt)
            raw_responses[attr] = response

            r = response.strip().lower()
            if r.startswith('yes'):
                scores[i] = 1.0
            elif r.startswith('no'):
                scores[i] = 0.0
            else:
                scores[i] = 0.5

        n_yes    = (scores > 0.5).sum()
        n_no     = (scores < 0.5).sum()
        n_unsure = (scores == 0.5).sum()
        print(f"\n{'=' * 70}")
        print(f"Results: {n_yes} Yes, {n_no} No, {n_unsure} Unsure")
        print(f"{'=' * 70}")

        present = [(attr_names[i], raw_responses[attr_names[i]])
                   for i in range(len(scores)) if scores[i] > 0.5]
        print(f"\n  PRESENT ({len(present)}):")
        for name, resp in present:
            print(f"    {name.replace('_', ' '):<35s} -> {resp}")

        _report_and_plot(img_pil, scores, attr_names, args, image_path,
                         use_llm=True, suffix='generate',
                         raw_responses=raw_responses)

    print("Done.")


# ---------------------------------------------------------------------------
#  Shared report/plot helper
# ---------------------------------------------------------------------------

def _report_and_plot(img_pil, scores, attr_names, args, image_path,
                     use_llm, suffix, raw_responses=None):
    save_path = args.save_path or (
        os.path.splitext(image_path)[0] + f'_facebench_{suffix}.png')
    txt_path = os.path.splitext(save_path)[0] + '.txt'
    _plot_llm_attributes(img_pil, scores, attr_names, args.top_k, save_path)
    _save_text_report(scores, attr_names, image_path, use_llm, txt_path,
                      raw_responses=raw_responses)
    print(f"\nPlot saved to:        {save_path}")
    print(f"Text report saved to: {txt_path}")


# ---------------------------------------------------------------------------
#  Text report
# ---------------------------------------------------------------------------

def _save_text_report(scores, attr_names, image_path, use_llm, txt_path,
                      raw_responses=None):
    """Save a complete text report of all attributes and their values."""
    import numpy as np

    categories = [
        ('Hair', 0, 20), ('Forehead', 20, 23), ('Eyebrows', 23, 30),
        ('Eyes', 30, 45), ('Eyelashes', 45, 48), ('Nose', 48, 56),
        ('Mouth/Lips', 56, 66), ('Cheeks', 66, 70), ('Chin/Jaw', 70, 76),
        ('Face Shape', 76, 82), ('Ears', 82, 85), ('Skin', 85, 100),
        ('Facial Hair', 100, 106), ('Neck', 106, 108), ('Age', 108, 113),
        ('Other Appearance', 113, 116), ('Accessories', 116, 146),
        ('Makeup', 146, 159), ('Surrounding', 159, 171),
        ('Expression', 171, 179), ('Action Units', 179, 204),
        ('Identity', 204, 211),
    ]

    with open(txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FaceBench Face-LLaVA Attribute Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Image:      {image_path}\n")
        f.write(f"Method:     {'Generative' if use_llm else 'Vision-only'}\n")
        f.write(f"Num attrs:  {len(scores)}\n")

        if use_llm:
            n_yes    = (scores > 0.5).sum()
            n_no     = (scores < 0.5).sum()
            n_unsure = (scores == 0.5).sum()
            f.write(f"Present (Yes): {n_yes} / {len(scores)}\n")
            f.write(f"Absent  (No):  {n_no} / {len(scores)}\n")
            if n_unsure > 0:
                f.write(f"Unsure:        {n_unsure} / {len(scores)}\n")
        else:
            f.write(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]\n")
            f.write(f"Score mean:  {scores.mean():.4f}\n")
            f.write(f"Score std:   {scores.std():.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("ALL ATTRIBUTES BY CATEGORY\n")
        f.write("=" * 70 + "\n")

        if use_llm and len(attr_names) == 211:
            for cat_name, start, end in categories:
                cat_scores  = scores[start:end]
                n_cat_yes   = (cat_scores > 0.5).sum()
                f.write(f"\n--- {cat_name} (idx {start}-{end-1}, "
                        f"{n_cat_yes}/{end-start} present) ---\n")
                for i in range(start, end):
                    name         = attr_names[i]
                    display_name = name.replace('_', ' ')
                    val          = scores[i]
                    flag = "  YES" if val > 0.5 else ("  NO" if val < 0.5 else "  UNSURE")
                    raw  = ""
                    if raw_responses and name in raw_responses:
                        raw = f"  (raw: {raw_responses[name]})"
                    f.write(f"  [{i:3d}] {display_name:<35s}{flag}{raw}\n")
        else:
            for i, (name, val) in enumerate(zip(attr_names, scores)):
                f.write(f"  [{i:3d}] {name:<35s} {val:.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("ALL ATTRIBUTES RANKED (PRESENT FIRST, THEN UNSURE, THEN ABSENT)\n")
        f.write("=" * 70 + "\n")

        sorted_idx = np.argsort(scores)[::-1]
        for rank, i in enumerate(sorted_idx, 1):
            name         = attr_names[i]
            display_name = name.replace('_', ' ')
            val          = scores[i]
            flag = "YES" if val > 0.5 else ("NO" if val < 0.5 else "UNSURE")
            raw  = ""
            if raw_responses and name in raw_responses:
                raw = f"  (raw: {raw_responses[name]})"
            f.write(f"  {rank:3d}. [{i:3d}] {display_name:<35s} {flag}{raw}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Scores: 1.0 = Yes, 0.0 = No, 0.5 = Unsure/Not visible\n")


# ---------------------------------------------------------------------------
#  Plot helpers
# ---------------------------------------------------------------------------

def _plot_description(img_pil, descriptions, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             gridspec_kw={'width_ratios': [1, 2]})
    axes[0].imshow(img_pil)
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].axis('off')
    text_content = ""
    for i, (prompt, response) in enumerate(descriptions):
        text_content += f"Q{i+1}: {prompt}\n"
        text_content += f"{'─' * 60}\n"
        words = response.split()
        line  = ""
        for w in words:
            if len(line) + len(w) + 1 > 80:
                text_content += line + "\n"
                line = w
            else:
                line = line + " " + w if line else w
        if line:
            text_content += line + "\n"
        text_content += "\n\n"

    axes[1].text(0.02, 0.98, text_content, transform=axes[1].transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    axes[1].set_title('Face-LLaVA Responses (model.generate)',
                      fontsize=14, fontweight='bold')
    fig.suptitle('FaceBench Face-LLaVA — Free-Form Validation',
                 fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def _plot_llm_attributes(img_pil, scores, attr_names, top_k, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    sorted_idx = np.argsort(scores)[::-1]
    n_attrs    = len(scores)

    categories = [
        ('Hair', 0, 20), ('Forehead', 20, 23), ('Eyebrows', 23, 30),
        ('Eyes', 30, 45), ('Eyelashes', 45, 48), ('Nose', 48, 56),
        ('Mouth/Lips', 56, 66), ('Cheeks', 66, 70), ('Chin/Jaw', 70, 76),
        ('Face Shape', 76, 82), ('Ears', 82, 85), ('Skin', 85, 100),
        ('Facial Hair', 100, 106), ('Neck', 106, 108), ('Age', 108, 113),
        ('Other Appear.', 113, 116), ('Accessories', 116, 146),
        ('Makeup', 146, 159), ('Surrounding', 159, 171),
        ('Expression', 171, 179), ('Action Units', 179, 204),
        ('Identity', 204, 211),
    ]

    fig = plt.figure(figsize=(24, 16))
    gs  = gridspec.GridSpec(2, 3, width_ratios=[1, 1.5, 1.5],
                            height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img_pil)
    ax_img.set_title('Input Image', fontsize=14, fontweight='bold')
    ax_img.axis('off')

    ax_top   = fig.add_subplot(gs[0, 1])
    top_n    = min(top_k, n_attrs)
    top_idx  = sorted_idx[:top_n]
    top_names  = [attr_names[i].replace('_', ' ') for i in top_idx]
    top_scores = [scores[i] for i in top_idx]
    colors_top = ['#2ecc71' if s > 0.5 else '#e74c3c' for s in top_scores]
    y_pos = np.arange(top_n)
    ax_top.barh(y_pos, top_scores, color=colors_top, height=0.7)
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels(top_names, fontsize=8)
    ax_top.invert_yaxis()
    ax_top.set_xlim(0, 1)
    ax_top.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_top.set_xlabel('P(present)')
    ax_top.set_title(f'Top-{top_n} Attributes', fontsize=14, fontweight='bold')

    ax_bot    = fig.add_subplot(gs[0, 2])
    bot_n     = min(top_k, n_attrs)
    bot_idx   = sorted_idx[-bot_n:][::-1]
    bot_names  = [attr_names[i].replace('_', ' ') for i in bot_idx]
    bot_scores = [scores[i] for i in bot_idx]
    colors_bot = ['#2ecc71' if s > 0.5 else '#e74c3c' for s in bot_scores]
    y_pos_b = np.arange(bot_n)
    ax_bot.barh(y_pos_b, bot_scores, color=colors_bot, height=0.7)
    ax_bot.set_yticks(y_pos_b)
    ax_bot.set_yticklabels(bot_names, fontsize=8)
    ax_bot.invert_yaxis()
    ax_bot.set_xlim(0, 1)
    ax_bot.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_bot.set_xlabel('P(present)')
    ax_bot.set_title(f'Bottom-{bot_n} Attributes', fontsize=14, fontweight='bold')

    ax_cat    = fig.add_subplot(gs[1, 0])
    cat_names = []
    cat_means = []
    cat_maxs  = []
    for cat_name, start, end in categories:
        cat_names.append(cat_name)
        cat_means.append(scores[start:end].mean())
        cat_maxs.append(scores[start:end].max())
    y_cat = np.arange(len(cat_names))
    ax_cat.barh(y_cat, cat_means, color='#3498db', height=0.6, label='Mean')
    ax_cat.barh(y_cat, cat_maxs,  color='#3498db', height=0.6, alpha=0.3, label='Max')
    ax_cat.set_yticks(y_cat)
    ax_cat.set_yticklabels(cat_names, fontsize=8)
    ax_cat.invert_yaxis()
    ax_cat.set_xlim(0, 1)
    ax_cat.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_cat.set_xlabel('Score')
    ax_cat.set_title('Category Summary', fontsize=14, fontweight='bold')
    ax_cat.legend(fontsize=8)

    ax_heat      = fig.add_subplot(gs[1, 1:])
    heatmap_data = scores.reshape(1, -1)
    im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='RdYlGn',
                        vmin=0, vmax=1, interpolation='nearest')
    ax_heat.set_yticks([])
    ax_heat.set_xlabel('Attribute Index')
    ax_heat.set_title('All 211 Attributes Heatmap', fontsize=14, fontweight='bold')
    for cat_name, start, end in categories:
        mid = (start + end) / 2
        ax_heat.axvline(x=start - 0.5, color='white', linewidth=0.5, alpha=0.7)
        if (end - start) > 5:
            ax_heat.text(mid, 0.6, cat_name, ha='center', va='top',
                         fontsize=5, rotation=90, color='black', alpha=0.7)
    plt.colorbar(im, ax=ax_heat, label='P(present)', shrink=0.8)

    n_present = (scores > 0.5).sum()
    fig.suptitle(
        f'FaceBench Validation — {n_present}/{len(scores)} attributes present (P>0.5)\n'
        f'Score range: [{scores.min():.3f}, {scores.max():.3f}], '
        f'mean: {scores.mean():.3f}, std: {scores.std():.3f}',
        fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    if '--demo' in sys.argv:
        demo_single_image()
    else:
        args = parse_args()
        precompute_all(args)