"""
networks/nesy_defake/semantic/facial_semantic_extractor.py
==========================================================
Facial Semantic Attribute Extraction using Dedicated Face Analysis Models.

WHY THIS EXISTS (replacing CLIPFacialAttributeExtractor)
--------------------------------------------------------
The previous approach computed "semantic attributes" by taking the spatial
CLIP pooler_output and computing cosine similarity with text prompts. This
is theoretically circular: cosine(visual_proj(pooler_output), text_embed)
is just a linear projection W @ pooler_output. It adds ZERO new information
to the pipeline -- the causal module sees the same spatial features twice,
dressed up differently.

Proper semantic attributes must come from an INDEPENDENT model processing
actual face images, providing genuinely new information that the spatial
and frequency branches do not capture.

SUPPORTED BACKENDS
------------------
1. FaceBench Face-LLaVA (recommended):
   Face-LLaVA-v1.5-13B fine-tuned on FaceBench (Wang et al., CVPR 2025).
   HuggingFace: wxqlab/face-llava-v1.5-13b

   TWO MODES:
   a) Vision-only (default, use_llm=False):
      Loads ONLY the CLIP-ViT-L/14@336 vision tower + mm_projector (~1.2GB).
      Produces face-specialized 5120-d visual features per image.
      A trainable projection maps these to output_dim for the causal module.
      VRAM: ~1.2GB. Efficient for training.

   b) Full LLM (use_llm=True):
      Loads the FULL frozen model (vision tower + mm_projector + 13B LLM).
      Teacher-forced single-pass extraction gives P(present) for 211 named
      facial attributes. One forward pass per image.
      VRAM: ~28GB FP16. Requires >= 40GB GPU.

2. FaRL:
   Microsoft's Face Representation Learning (Zheng et al., CVPR 2022).
   ViT-B/16 pre-trained on 20M face images.

3. CelebA ViT:
   ViT fine-tuned on CelebA-40 facial attributes (multi-label).

4. Precomputed:
   Pre-extracted attributes loaded from disk by the dataloader.

ARCHITECTURE (FaceBench vision-only)
-------------------------------------
  raw_images (B, 3, H, W)          [0, 1] range
       |
  [resize to 336x336 + CLIP-normalize]
       |
  [CLIP-ViT-L/14 vision tower]     <- FROZEN, from FaceBench checkpoint
       |
  hidden_states[-2][:, 1:, :]      576 patch tokens, 1024-d
       |
  [mm_projector]                    <- FROZEN, mlp2x_gelu (1024 -> 5120)
       |
  average pool over 576 tokens
       |
  face_features (B, 5120)           <- face-specialized visual features
       |
  [trainable projection]            5120 -> output_dim
       |
  semantic_attrs (B, output_dim)    <- OUTPUT: for causal module

ARCHITECTURE (FaceBench full LLM)
----------------------------------
  raw_images (B, 3, H, W)          [0, 1] range
       |
  [resize to 336x336 + CLIP-normalize]
       |
  [CLIP-ViT-L/14 vision tower]     <- FROZEN, from FaceBench checkpoint
       |
  hidden_states[-2][:, 1:, :]      576 patch tokens, 1024-d
       |
  [mm_projector]                    <- FROZEN, mlp2x_gelu (1024 -> 5120)
       |
  visual_tokens (B, 576, 5120)
       |
  [Concatenate with teacher-forced attribute prompt embeddings]
       |
  [13B LLM forward]                <- FROZEN, single pass
       |
  logits at 211 answer positions -> P(present) per attribute
       |
  attribute_scores (B, 211)         <- OUTPUT: named, interpretable
"""

import logging
from typing import List, Optional
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CelebA-40 attribute names (standard ordering from the CelebA dataset)
# ---------------------------------------------------------------------------
CELEBA_ATTRIBUTES = [
    '5_o_clock_shadow', 'arched_eyebrows', 'attractive', 'bags_under_eyes',
    'bald', 'bangs', 'big_lips', 'big_nose', 'black_hair', 'blond_hair',
    'blurry', 'brown_hair', 'bushy_eyebrows', 'chubby', 'double_chin',
    'eyeglasses', 'goatee', 'gray_hair', 'heavy_makeup', 'high_cheekbones',
    'male', 'mouth_slightly_open', 'mustache', 'narrow_eyes', 'no_beard',
    'oval_face', 'pale_skin', 'pointy_nose', 'receding_hairline', 'rosy_cheeks',
    'sideburns', 'smiling', 'straight_hair', 'wavy_hair', 'wearing_earrings',
    'wearing_hat', 'wearing_lipstick', 'wearing_necklace', 'wearing_necktie',
    'young',
]

# ---------------------------------------------------------------------------
# FaceBench 211 facial attributes (Wang et al., CVPR 2025)
# Organized by 5 views: Appearance, Accessories, Surrounding, Psychology, ID
# Attribute names use underscores; converted to spaces in the LLM prompt.
# ---------------------------------------------------------------------------
FACEBENCH_ATTRIBUTES = [
    # -- Appearance: Hair (20) --
    'black_hair', 'blonde_hair', 'brown_hair', 'gray_hair', 'red_hair',
    'white_hair', 'long_hair', 'medium_length_hair', 'short_hair', 'bald',
    'straight_hair', 'wavy_hair', 'curly_hair', 'bangs',
    'receding_hairline', 'hair_parted', 'ponytail', 'bun_hair',
    'braided_hair', 'dyed_hair',
    # -- Appearance: Forehead (3) --
    'large_forehead', 'small_forehead', 'forehead_wrinkles',
    # -- Appearance: Eyebrows (7) --
    'arched_eyebrows', 'straight_eyebrows', 'thick_eyebrows',
    'thin_eyebrows', 'bushy_eyebrows', 'unibrow', 'sparse_eyebrows',
    # -- Appearance: Eyes (15) --
    'brown_eyes', 'blue_eyes', 'green_eyes', 'hazel_eyes', 'black_eyes',
    'large_eyes', 'small_eyes', 'narrow_eyes', 'wide_set_eyes',
    'double_eyelid', 'single_eyelid', 'hooded_eyes',
    'bags_under_eyes', 'dark_circles', 'puffy_eyes',
    # -- Appearance: Eyelashes (3) --
    'long_eyelashes', 'thick_eyelashes', 'sparse_eyelashes',
    # -- Appearance: Nose (8) --
    'large_nose', 'small_nose', 'pointed_nose', 'broad_nose',
    'upturned_nose', 'long_nose', 'crooked_nose', 'flat_nose',
    # -- Appearance: Mouth & Lips (10) --
    'full_lips', 'thin_lips', 'wide_mouth', 'small_mouth',
    'mouth_open', 'mouth_closed', 'smiling', 'frowning',
    'teeth_visible', 'white_teeth',
    # -- Appearance: Cheeks (4) --
    'high_cheekbones', 'round_cheeks', 'hollow_cheeks', 'rosy_cheeks',
    # -- Appearance: Chin & Jaw (6) --
    'pointed_chin', 'round_chin', 'double_chin', 'cleft_chin',
    'strong_jaw', 'narrow_jaw',
    # -- Appearance: Face Shape (6) --
    'oval_face', 'round_face', 'square_face', 'heart_shaped_face',
    'long_face', 'diamond_face',
    # -- Appearance: Ears (3) --
    'large_ears', 'small_ears', 'protruding_ears',
    # -- Appearance: Skin (15) --
    'fair_skin', 'medium_skin', 'dark_skin', 'olive_skin', 'pale_skin',
    'smooth_skin', 'wrinkled_skin', 'freckled_skin',
    'acne', 'moles', 'scars', 'skin_blemishes',
    'oily_skin', 'dry_skin', 'age_spots',
    # -- Appearance: Facial Hair (6) --
    'beard', 'mustache', 'goatee', 'sideburns', 'stubble', 'clean_shaven',
    # -- Appearance: Neck (2) --
    'long_neck', 'short_neck',
    # -- Appearance: Age (5) --
    'baby_face', 'young_looking', 'middle_aged', 'elderly_looking',
    'age_ambiguous',
    # -- Appearance: Other (3) --
    'attractive', 'symmetrical_face', 'asymmetrical_face',
    # -- Accessories (30) --
    'eyeglasses', 'sunglasses', 'reading_glasses',
    'round_glasses', 'rectangular_glasses', 'rimless_glasses',
    'thick_frame_glasses',
    'hat', 'baseball_cap', 'beanie', 'headband', 'turban', 'hood',
    'stud_earrings', 'hoop_earrings', 'dangling_earrings',
    'necklace', 'choker', 'pendant_necklace',
    'nose_piercing', 'lip_piercing', 'ear_piercing',
    'scarf', 'necktie', 'bowtie',
    'face_mask', 'headphones', 'hair_clip', 'hair_band', 'veil',
    # -- Makeup (13) --
    'heavy_makeup', 'light_makeup', 'no_makeup',
    'lipstick', 'red_lipstick', 'pink_lipstick', 'nude_lipstick',
    'eyeshadow', 'eyeliner', 'mascara',
    'blush', 'foundation', 'contour_makeup',
    # -- Surrounding (12) --
    'indoor_background', 'outdoor_background',
    'plain_background', 'complex_background',
    'bright_lighting', 'dim_lighting', 'natural_lighting',
    'artificial_lighting', 'side_lighting',
    'blurry_image', 'sharp_image', 'bokeh_background',
    # -- Psychology: Expression (8) --
    'neutral_expression', 'happy', 'sad', 'angry',
    'surprised', 'fearful', 'disgusted', 'contemptuous',
    # -- Psychology: Action Units (25) --
    'AU1_inner_brow_raise', 'AU2_outer_brow_raise', 'AU4_brow_lowerer',
    'AU5_upper_lid_raise', 'AU6_cheek_raise', 'AU7_lid_tightener',
    'AU9_nose_wrinkler', 'AU10_upper_lip_raiser',
    'AU11_nasolabial_deepener',
    'AU12_lip_corner_puller', 'AU13_sharp_lip_puller', 'AU14_dimpler',
    'AU15_lip_corner_depressor', 'AU16_lower_lip_depressor',
    'AU17_chin_raiser',
    'AU18_lip_pucker', 'AU20_lip_stretcher', 'AU22_lip_funneler',
    'AU23_lip_tightener', 'AU24_lip_pressor', 'AU25_lips_part',
    'AU26_jaw_drop', 'AU28_lip_suck', 'AU43_eyes_closed', 'AU45_blink',
    # -- Identity (7) --
    'male', 'female',
    'east_asian', 'african', 'caucasian', 'hispanic', 'south_asian',
]

assert len(FACEBENCH_ATTRIBUTES) == 211, \
    f"Expected 211 FaceBench attributes, got {len(FACEBENCH_ATTRIBUTES)}"


class FacialSemanticExtractor(nn.Module):
    """
    Extracts facial semantic attributes using a dedicated pre-trained face
    analysis model -- completely independent of the spatial/frequency backbone.

    For the FaceBench backend:
    - Vision-only mode (default): loads CLIP-ViT-L/14@336 + mm_projector
      from the FaceBench checkpoint (~1.2GB). Produces face-specialized
      5120-d features, projected to output_dim via a trainable head.
    - Full LLM mode (use_llm=True): loads the full 13B model and extracts
      211 named attribute probabilities via teacher-forced inference.
    """

    SUPPORTED_BACKENDS = ('face_llava', 'farl', 'celeba_vit', 'precomputed')

    def __init__(self, config: dict):
        super().__init__()

        sem_cfg = config.get('semantic_attributes', {})
        self.backend_name = sem_cfg.get('backend', 'face_llava')
        model_path = sem_cfg.get('model_path', '')
        self._output_dim = sem_cfg.get('output_dim', 128)

        if self.backend_name not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown semantic backend '{self.backend_name}'. "
                f"Supported: {self.SUPPORTED_BACKENDS}"
            )

        if self.backend_name == 'face_llava':
            self._build_face_llava(model_path, sem_cfg)
        elif self.backend_name == 'farl':
            self._build_farl(model_path, sem_cfg)
        elif self.backend_name == 'celeba_vit':
            self._build_celeba_vit(model_path, sem_cfg)
        elif self.backend_name == 'precomputed':
            self._build_precomputed(sem_cfg)

        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[FacialSemanticExtractor] backend={self.backend_name}, "
            f"output_dim={self._output_dim}, "
            f"params={n_total:,} (trainable={n_trainable:,})"
        )

    # ------------------------------------------------------------------ #
    #  Backend: FaceBench Face-LLaVA                                       #
    # ------------------------------------------------------------------ #

    def _build_face_llava(self, model_path: str, cfg: dict):
        """
        FaceBench Face-LLaVA-v1.5-13B (Wang et al., CVPR 2025).

        Dispatches to vision-only or full-LLM build based on use_llm config.
        """
        self._use_llm = cfg.get('use_llm', False)

        if self._use_llm:
            self._build_face_llava_with_llm(model_path, cfg)
        else:
            self._build_face_llava_vision_only(model_path, cfg)

    def _build_face_llava_vision_only(self, model_path: str, cfg: dict):
        """
        Vision tower + mm_projector ONLY (no 13B LLM). ~1.2GB VRAM.

        The mm_projector was trained jointly with the LLM during FaceBench
        fine-tuning, so it produces face-specialized 5120-d features that
        encode facial attribute information. We average-pool the 576 patch
        tokens and project to output_dim via a trainable head.

        This mode loads only the 1-2 safetensor shards containing vision
        tower and mm_projector weights (not all 6 shards).
        """
        import json

        self._llava_batch_size = cfg.get('micro_batch_size', 64)
        dtype = torch.float16 if cfg.get('fp16', True) else torch.bfloat16

        if not model_path or not Path(model_path).exists():
            raise ValueError(
                "face_llava backend requires 'model_path' pointing to the "
                "FaceBench Face-LLaVA checkpoint directory.\n"
                "Download: huggingface.co/wxqlab/face-llava-v1.5-13b\n"
                "Set semantic_attributes.model_path to that directory.")

        model_path = Path(model_path)
        logger.info(f"[FaceBench] Loading VISION-ONLY from: {model_path}")

        # --- Read model config ------------------------------------------------
        with open(model_path / 'config.json') as f:
            model_config = json.load(f)

        vision_tower_name = model_config.get(
            'mm_vision_tower', 'openai/clip-vit-large-patch14-336')
        mm_hidden_size = model_config.get('mm_hidden_size', 1024)
        hidden_size = model_config.get('hidden_size', 5120)
        self._vision_select_layer = model_config.get(
            'mm_vision_select_layer', -2)
        self._vision_select_feature = model_config.get(
            'mm_vision_select_feature', 'patch')

        # --- Load CLIP vision tower (skeleton from HuggingFace) ---------------
        from transformers import CLIPVisionModel, CLIPImageProcessor

        logger.info(f"[FaceBench] Loading vision tower: {vision_tower_name}")
        self.vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower_name, dtype=dtype)
        self.vision_tower.eval()

        # --- Build mm_projector (mlp2x_gelu) ----------------------------------
        self.mm_projector = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # --- Load vision tower + mm_projector from FaceBench checkpoint -------
        # Only loads the shards containing these weights (skips LLM shards).
        self._load_vision_weights(model_path, model_config, dtype)

        # --- Image processor for normalization --------------------------------
        try:
            image_processor = CLIPImageProcessor.from_pretrained(
                vision_tower_name)
            img_mean = list(image_processor.image_mean)
            img_std = list(image_processor.image_std)
            crop_size = image_processor.crop_size
            if isinstance(crop_size, dict):
                self._image_size = crop_size.get('height', 336)
            else:
                self._image_size = crop_size or 336
        except Exception:
            img_mean = [0.48145466, 0.4578275, 0.40821073]
            img_std = [0.26862954, 0.26130258, 0.27577711]
            self._image_size = 336

        self.register_buffer(
            'norm_mean', torch.tensor(img_mean).view(1, 3, 1, 1))
        self.register_buffer(
            'norm_std', torch.tensor(img_std).view(1, 3, 1, 1))

        # --- Freeze vision tower + mm_projector ───────────────────────────────
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        for p in self.mm_projector.parameters():
            p.requires_grad = False

        self.mm_projector = self.mm_projector.to(dtype)

        # --- Trainable projection: 5120 → output_dim ─────────────────────────
        # The mm_projector output is face-specialized but 5120-d is too large
        # for the causal module. This trainable head learns to extract the
        # causally relevant facial dimensions.
        self._backbone_dim = hidden_size  # 5120
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, self._output_dim * 2),
            nn.LayerNorm(self._output_dim * 2),
            nn.GELU(),
            nn.Linear(self._output_dim * 2, self._output_dim),
        )
        self._attr_names = [f'face_llava_{i}' for i in range(self._output_dim)]

        self._num_visual_tokens = (self._image_size // 14) ** 2  # 576

        logger.info(
            f"[FaceBench] Vision-only mode loaded. "
            f"backbone_dim={self._backbone_dim} -> output_dim={self._output_dim}, "
            f"image_size={self._image_size}, "
            f"micro_batch={self._llava_batch_size}, "
            f"visual_tokens={self._num_visual_tokens}")

    def _build_face_llava_with_llm(self, model_path: str, cfg: dict):
        """
        Full FaceBench model including 13B LLM. ~28GB VRAM.

        Teacher-forced attribute extraction:
          1. Image -> vision tower -> mm_projector -> visual tokens
          2. Construct structured prompt listing all 211 attributes
          3. Teacher-force all answers as "1" and run single LLM forward
          4. At each answer position, extract logit("1") - logit("0")
          5. sigmoid -> P(present) for each attribute
          6. Output: (B, 211) named attribute probabilities
        """
        import json

        self._llava_batch_size = cfg.get('micro_batch_size', 16)
        dtype = torch.float16 if cfg.get('fp16', True) else torch.bfloat16

        if not model_path or not Path(model_path).exists():
            raise ValueError(
                "face_llava backend requires 'model_path' pointing to the "
                "FaceBench Face-LLaVA checkpoint directory.\n"
                "Download: huggingface.co/wxqlab/face-llava-v1.5-13b\n"
                "Set semantic_attributes.model_path to that directory.")

        model_path = Path(model_path)
        logger.info(f"[FaceBench] Loading FULL model (with LLM) from: {model_path}")

        # --- Read model config ------------------------------------------------
        with open(model_path / 'config.json') as f:
            model_config = json.load(f)

        vision_tower_name = model_config.get(
            'mm_vision_tower', 'openai/clip-vit-large-patch14-336')
        mm_hidden_size = model_config.get('mm_hidden_size', 1024)
        hidden_size = model_config.get('hidden_size', 5120)
        self._vision_select_layer = model_config.get(
            'mm_vision_select_layer', -2)
        self._vision_select_feature = model_config.get(
            'mm_vision_select_feature', 'patch')

        # --- Load CLIP vision tower -------------------------------------------
        from transformers import CLIPVisionModel, CLIPImageProcessor

        logger.info(f"[FaceBench] Loading vision tower: {vision_tower_name}")
        self.vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower_name, dtype=dtype)
        self.vision_tower.eval()

        # --- Build mm_projector (mlp2x_gelu) ----------------------------------
        self.mm_projector = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # --- Load LLM from FaceBench checkpoint directly ---------------------
        # Use from_pretrained which handles sharded loading efficiently.
        # Vision tower and mm_projector weights are loaded separately below.
        from transformers import LlamaForCausalLM, AutoTokenizer

        logger.info(f"[FaceBench] Loading 13B LLM from: {model_path}")
        self.llm = LlamaForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )

        # --- Load tokenizer ---------------------------------------------------
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), use_fast=False)

        # --- Load vision tower + mm_projector from checkpoint -----------------
        # from_pretrained above only loaded LLM weights (Llama architecture).
        # Vision tower and mm_projector use different key prefixes and must
        # be loaded separately from the safetensors shards.
        self._load_vision_weights(model_path, model_config, dtype)

        # --- Image processor for normalization --------------------------------
        try:
            image_processor = CLIPImageProcessor.from_pretrained(
                vision_tower_name)
            img_mean = list(image_processor.image_mean)
            img_std = list(image_processor.image_std)
            crop_size = image_processor.crop_size
            if isinstance(crop_size, dict):
                self._image_size = crop_size.get('height', 336)
            else:
                self._image_size = crop_size or 336
        except Exception:
            img_mean = [0.48145466, 0.4578275, 0.40821073]
            img_std = [0.26862954, 0.26130258, 0.27577711]
            self._image_size = 336

        self.register_buffer(
            'norm_mean', torch.tensor(img_mean).view(1, 3, 1, 1))
        self.register_buffer(
            'norm_std', torch.tensor(img_std).view(1, 3, 1, 1))

        # --- Freeze everything ------------------------------------------------
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        for p in self.mm_projector.parameters():
            p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False

        self.mm_projector = self.mm_projector.to(dtype)
        self.llm = self.llm.to(dtype)
        self.llm.eval()

        # --- Build teacher-forced prompt template -----------------------------
        self._build_attribute_prompt(dtype)

        # --- Output config (211 named attributes, no trainable projection) ----
        self._output_dim = len(FACEBENCH_ATTRIBUTES)
        self._backbone_dim = len(FACEBENCH_ATTRIBUTES)
        self.projection = nn.Identity()
        self._attr_names = list(FACEBENCH_ATTRIBUTES)

        logger.info(
            f"[FaceBench] Full model loaded. "
            f"output_dim={self._output_dim} (named attributes), "
            f"image_size={self._image_size}, "
            f"micro_batch={self._llava_batch_size}")

    def _load_vision_weights(self, model_path: Path, model_config: dict,  # noqa: ARG002
                             dtype):
        """
        Load vision tower and mm_projector weights from FaceBench safetensors.

        Only reads the specific shards containing these weights, skipping
        the LLM shards (~24GB) entirely when in vision-only mode.
        """
        import json
        from safetensors import safe_open

        # Determine which shards contain vision tower / mm_projector keys
        index_path = model_path / 'model.safetensors.index.json'
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            needed_shards = set()
            for key, shard in index['weight_map'].items():
                if (key.startswith('model.vision_tower.')
                        or key.startswith('model.mm_projector.')):
                    needed_shards.add(shard)
            all_shards = sorted(needed_shards)
        else:
            # Fallback: scan all shards
            all_shards = sorted(
                p.name for p in model_path.glob('*.safetensors'))

        logger.info(f"[FaceBench] Loading vision weights from "
                    f"{len(all_shards)} shard(s) (of "
                    f"{len(list(model_path.glob('*.safetensors')))} total)")

        vt_state = {}
        proj_state = {}

        for shard_name in all_shards:
            shard_path = model_path / shard_name
            with safe_open(str(shard_path), framework='pt',
                           device='cpu') as f:
                for key in f.keys():
                    if key.startswith('model.vision_tower.vision_tower.'):
                        clean = key[len('model.vision_tower.vision_tower.'):]
                        vt_state[clean] = f.get_tensor(key).to(dtype)
                    elif key.startswith('model.mm_projector.'):
                        clean = key[len('model.mm_projector.'):]
                        proj_state[clean] = f.get_tensor(key).to(dtype)

        # Load vision tower
        if vt_state:
            missing, _ = self.vision_tower.load_state_dict(
                vt_state, strict=False)
            if missing:
                logger.warning(
                    f"[FaceBench] Vision tower missing keys "
                    f"({len(missing)}): {missing[:3]}...")
            logger.info(f"[FaceBench] Loaded {len(vt_state)} vision tower "
                        f"weight tensors")
        else:
            logger.warning("[FaceBench] No vision tower weights found in "
                          "checkpoint; using base CLIP weights.")

        # Load mm_projector
        if proj_state:
            self.mm_projector.load_state_dict(proj_state, strict=True)
            logger.info(f"[FaceBench] Loaded {len(proj_state)} mm_projector "
                        f"weight tensors")
        else:
            raise RuntimeError(
                "No mm_projector weights found in checkpoint.")

    def _build_attribute_prompt(self, dtype):
        """
        Pre-build the teacher-forced prompt and locate answer positions.

        Prompt structure (LLaVA-v1 conversation template):
          SYSTEM: A chat between ...
          USER: <image>\\nAssess facial attributes.
          ASSISTANT: black hair: 1\\nblonde hair: 1\\n...

        We tokenize the text before/after the image placeholder, and the
        answer section. At forward time, we replace <image> with visual
        tokens and run a single LLM forward pass.

        Answer positions are found by comparing tokenizations with "1"
        vs "0" answers -- positions that differ are the answer tokens.
        """
        # LLaVA-v1 conversation template
        system = (
            "A chat between a curious user and an artificial intelligence "
            "assistant. The assistant gives helpful, detailed, and polite "
            "answers to the user's questions. "
        )
        before_image = system + "USER: "
        after_image_question = (
            "\nFor each facial attribute, predict 1 if present or 0 if "
            "absent. ASSISTANT:"
        )

        # Build teacher-forced answer with "1" for all attributes
        attr_display = [a.replace('_', ' ') for a in FACEBENCH_ATTRIBUTES]
        answer_yes = '\n'.join(f' {a}: 1' for a in attr_display)
        answer_no = '\n'.join(f' {a}: 0' for a in attr_display)

        # Tokenize parts
        before_ids = self._tokenizer.encode(
            before_image, add_special_tokens=True)
        question_ids = self._tokenizer.encode(
            after_image_question, add_special_tokens=False)
        answer_yes_ids = self._tokenizer.encode(
            answer_yes, add_special_tokens=False)
        answer_no_ids = self._tokenizer.encode(
            answer_no, add_special_tokens=False)

        # Find answer positions: where "1" and "0" tokenizations differ
        assert len(answer_yes_ids) == len(answer_no_ids), (
            f"Tokenization length mismatch: yes={len(answer_yes_ids)} "
            f"vs no={len(answer_no_ids)}")

        answer_positions = [
            i for i in range(len(answer_yes_ids))
            if answer_yes_ids[i] != answer_no_ids[i]
        ]

        if len(answer_positions) != len(FACEBENCH_ATTRIBUTES):
            logger.warning(
                f"[FaceBench] Found {len(answer_positions)} answer positions, "
                f"expected {len(FACEBENCH_ATTRIBUTES)}. "
                f"Some attributes may have tokenization issues.")

        # Store token IDs for "1" and "0" at answer positions
        self._yes_token_id = answer_yes_ids[answer_positions[0]]
        self._no_token_id = answer_no_ids[answer_positions[0]]

        # Register prompt token IDs as buffers (move with model to device)
        self.register_buffer(
            '_before_ids',
            torch.tensor(before_ids, dtype=torch.long))
        self.register_buffer(
            '_question_ids',
            torch.tensor(question_ids, dtype=torch.long))
        self.register_buffer(
            '_answer_ids',
            torch.tensor(answer_yes_ids, dtype=torch.long))

        # Answer positions relative to the answer section start
        self.register_buffer(
            '_answer_positions_in_answer',
            torch.tensor(answer_positions, dtype=torch.long))

        # Number of visual tokens (576 for CLIP-ViT-L@336: 24x24 patches)
        self._num_visual_tokens = (
            (self._image_size // 14) ** 2)  # 336/14 = 24, 24^2 = 576

        total_text = len(before_ids) + len(question_ids) + len(answer_yes_ids)
        total_seq = total_text + self._num_visual_tokens
        logger.info(
            f"[FaceBench] Prompt built: {len(before_ids)} before + "
            f"{self._num_visual_tokens} visual + "
            f"{len(question_ids)} question + "
            f"{len(answer_yes_ids)} answer = {total_seq} total tokens, "
            f"{len(answer_positions)} answer positions found, "
            f"yes_id={self._yes_token_id}, no_id={self._no_token_id}")

    # ------------------------------------------------------------------ #
    #  Face-LLaVA feature extraction                                       #
    # ------------------------------------------------------------------ #

    def _extract_face_llava_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Vision-only: extract face-specialized 5120-d features.

        For each micro-batch:
          1. Vision tower -> hidden_states[-2] patch tokens (576 x 1024)
          2. mm_projector -> projected tokens (576 x 5120)
          3. Average pool over 576 tokens -> (5120,)

        Returns: (B, 5120) face-specialized visual features.
        """
        import torch.nn.functional as F_func

        B = images.shape[0]
        dtype = next(self.vision_tower.parameters()).dtype

        # Resize to vision tower resolution (336x336)
        if images.shape[-1] != self._image_size or \
                images.shape[-2] != self._image_size:
            images = F_func.interpolate(
                images, size=(self._image_size, self._image_size),
                mode='bilinear', align_corners=False)

        # Normalize with CLIP stats
        images = (images - self.norm_mean) / self.norm_std

        all_features = []
        mbs = self._llava_batch_size

        for i in range(0, B, mbs):
            batch = images[i:i + mbs].to(dtype)

            # 1. Vision tower forward
            vt_out = self.vision_tower(
                pixel_values=batch,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = vt_out.hidden_states[self._vision_select_layer]
            if self._vision_select_feature == 'patch':
                visual_tokens = hidden_states[:, 1:, :]  # skip CLS
            else:
                visual_tokens = hidden_states
            # visual_tokens: (mb, 576, 1024)

            # 2. MM projector
            projected = self.mm_projector(visual_tokens)
            # projected: (mb, 576, 5120)

            # 3. Average pool over spatial tokens
            pooled = projected.mean(dim=1)  # (mb, 5120)
            all_features.append(pooled.float())

        return torch.cat(all_features, dim=0)  # (B, 5120)

    def _extract_face_llava_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Full LLM: extract 211 named facial attribute scores via teacher-forced
        single-pass inference through the full FaceBench 13B model.

        For each micro-batch:
          1. Vision tower -> hidden_states[-2] patch tokens
          2. mm_projector -> projected visual tokens (mb, 576, 5120)
          3. Embed text prompt tokens via LLM embedding layer
          4. Concatenate: [before_img_embeds, visual_tokens, question_embeds,
                           answer_embeds]
          5. Single LLM forward pass (teacher-forced, no generation)
          6. Extract logits at answer positions -> sigmoid -> P(present)

        Returns: (B, 211) attribute probability scores.
        """
        import torch.nn.functional as F_func

        B = images.shape[0]
        dtype = next(self.vision_tower.parameters()).dtype

        # Resize to vision tower resolution (336x336)
        if images.shape[-1] != self._image_size or \
                images.shape[-2] != self._image_size:
            images = F_func.interpolate(
                images, size=(self._image_size, self._image_size),
                mode='bilinear', align_corners=False)

        # Normalize with CLIP stats
        images = (images - self.norm_mean) / self.norm_std

        all_scores = []
        mbs = self._llava_batch_size

        for i in range(0, B, mbs):
            batch = images[i:i + mbs].to(dtype)
            mb = batch.shape[0]

            # 1. Vision tower forward
            vt_out = self.vision_tower(
                pixel_values=batch,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = vt_out.hidden_states[self._vision_select_layer]
            if self._vision_select_feature == 'patch':
                visual_tokens = hidden_states[:, 1:, :]  # skip CLS
            else:
                visual_tokens = hidden_states
            # visual_tokens: (mb, 576, 1024)

            # 2. MM projector
            projected = self.mm_projector(visual_tokens)
            # projected: (mb, 576, 5120)

            # 3. Get text embeddings from LLM embedding layer
            embed_fn = self.llm.model.embed_tokens
            before_embeds = embed_fn(
                self._before_ids.unsqueeze(0).expand(mb, -1))
            question_embeds = embed_fn(
                self._question_ids.unsqueeze(0).expand(mb, -1))
            answer_embeds = embed_fn(
                self._answer_ids.unsqueeze(0).expand(mb, -1))

            # 4. Concatenate: [before, visual, question, answer]
            inputs_embeds = torch.cat([
                before_embeds,   # (mb, N_before, 5120)
                projected,       # (mb, 576, 5120)
                question_embeds, # (mb, N_question, 5120)
                answer_embeds,   # (mb, N_answer, 5120)
            ], dim=1)

            # 5. Single LLM forward pass (no generation, just logits)
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                return_dict=True,
            )
            logits = outputs.logits  # (mb, seq_len, vocab_size)

            # 6. Extract attribute scores at answer positions
            # The answer section starts at offset:
            #   len(before_ids) + num_visual_tokens + len(question_ids)
            offset = (len(self._before_ids)
                      + self._num_visual_tokens
                      + len(self._question_ids))

            # Logits at position p predict token at position p+1.
            # Answer token "1" is at position (offset + answer_pos).
            # So we read logits at position (offset + answer_pos - 1).
            logit_positions = offset + self._answer_positions_in_answer - 1

            attr_logits = logits[:, logit_positions, :]  # (mb, 211, vocab)

            yes_logits = attr_logits[:, :, self._yes_token_id]  # (mb, 211)
            no_logits = attr_logits[:, :, self._no_token_id]    # (mb, 211)

            scores = torch.sigmoid(yes_logits - no_logits)  # (mb, 211)
            all_scores.append(scores.float())

        return torch.cat(all_scores, dim=0)  # (B, 211)

    # ------------------------------------------------------------------ #
    #  Backend: FaRL                                                       #
    # ------------------------------------------------------------------ #

    def _build_farl(self, model_path: str, cfg: dict):
        """
        FaRL: Face Representation Learning (Zheng et al., CVPR 2022).

        Pre-trained on LAION-Face 20M with:
          - Image-text contrastive learning (face-specific pairs)
          - Masked image modeling (learns face structure)

        Architecture: identical to CLIP ViT-B/16, but trained specifically
        on face images -> produces face-specific embeddings that are
        fundamentally different from generic CLIP features.

        Outputs 512-d features (after visual projection in CLIP space).
        A trainable projection head maps these to output_dim.
        """
        import open_clip

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                f"FaRL weights not found at '{model_path}'.\n"
                f"Download from: https://github.com/FacePerceiver/FaRL\n"
                f"Recommended:   FaRL-Base-Patch16-LAIONFace20M-ep64.pth\n"
                f"Place at the path specified in config: "
                f"semantic_attributes.model_path"
            )

        logger.info(f"[FaRL] Loading ViT-B/16 face backbone from: {model_path}")

        # Create CLIP ViT-B/16 skeleton (FaRL shares this architecture)
        clip_model = open_clip.create_model('ViT-B-16', pretrained='')

        # Load FaRL pre-trained weights
        checkpoint = torch.load(
            model_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # FaRL checkpoints store full CLIP model -> extract visual encoder
        visual_keys = {k: v for k, v in state_dict.items()
                       if k.startswith('visual.')}

        if visual_keys:
            visual_state = {k[len('visual.'):]: v
                           for k, v in visual_keys.items()}
        else:
            visual_state = state_dict

        missing, unexpected = clip_model.visual.load_state_dict(
            visual_state, strict=False)
        if missing:
            logger.warning(f"[FaRL] Missing keys ({len(missing)}): "
                           f"{missing[:5]}...")
        if unexpected:
            logger.warning(f"[FaRL] Unexpected keys ({len(unexpected)}): "
                           f"{unexpected[:5]}...")

        self.backbone = clip_model.visual
        self._backbone_dim = clip_model.visual.output_dim  # 512
        del clip_model

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.register_buffer(
            'norm_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer(
            'norm_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        self.projection = nn.Sequential(
            nn.Linear(self._backbone_dim, self._output_dim * 2),
            nn.LayerNorm(self._output_dim * 2),
            nn.GELU(),
            nn.Linear(self._output_dim * 2, self._output_dim),
        )

        self._attr_names = [f'farl_{i}' for i in range(self._output_dim)]
        logger.info(
            f"[FaRL] Loaded successfully. "
            f"backbone_dim={self._backbone_dim} -> output_dim={self._output_dim}")

    # ------------------------------------------------------------------ #
    #  Backend: CelebA ViT                                                 #
    # ------------------------------------------------------------------ #

    def _build_celeba_vit(self, model_path: str, cfg: dict):
        """CelebA-40 facial attribute classifier (ViT fine-tuned)."""
        from transformers import ViTForImageClassification, ViTImageProcessor

        if not model_path:
            raise ValueError(
                "CelebA ViT backend requires 'model_path' in config.")

        logger.info(f"[CelebA-ViT] Loading from: {model_path}")

        self.backbone = ViTForImageClassification.from_pretrained(model_path)

        try:
            processor = ViTImageProcessor.from_pretrained(model_path)
            img_mean = list(processor.image_mean)
            img_std = list(processor.image_std)
            del processor
        except Exception:
            img_mean = [0.485, 0.456, 0.406]
            img_std = [0.229, 0.224, 0.225]

        self.register_buffer(
            'norm_mean', torch.tensor(img_mean).view(1, 3, 1, 1))
        self.register_buffer(
            'norm_std', torch.tensor(img_std).view(1, 3, 1, 1))

        for p in self.backbone.parameters():
            p.requires_grad = False

        num_labels = self.backbone.config.num_labels
        self._backbone_dim = num_labels

        if num_labels == 40:
            self._attr_names = list(CELEBA_ATTRIBUTES)
        else:
            self._attr_names = [f'attr_{i}' for i in range(num_labels)]

        if self._output_dim == num_labels:
            self.projection = nn.Identity()
            self._attr_names = self._attr_names[:self._output_dim]
        else:
            self.projection = nn.Sequential(
                nn.Linear(num_labels, self._output_dim * 2),
                nn.LayerNorm(self._output_dim * 2),
                nn.GELU(),
                nn.Linear(self._output_dim * 2, self._output_dim),
            )
            self._attr_names = [f'celeba_proj_{i}'
                                for i in range(self._output_dim)]

        logger.info(
            f"[CelebA-ViT] Loaded. num_labels={num_labels} -> "
            f"output_dim={self._output_dim}")

    # ------------------------------------------------------------------ #
    #  Backend: Precomputed                                                #
    # ------------------------------------------------------------------ #

    def _build_precomputed(self, cfg: dict):
        """Precomputed attributes loaded from disk by the dataset."""
        input_dim = cfg.get('precomputed_dim', self._output_dim)
        self._backbone_dim = input_dim

        if input_dim == self._output_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, self._output_dim * 2),
                nn.LayerNorm(self._output_dim * 2),
                nn.GELU(),
                nn.Linear(self._output_dim * 2, self._output_dim),
            )

        self.backbone = None

        self.register_buffer('norm_mean', torch.zeros(1))
        self.register_buffer('norm_std', torch.ones(1))

        self._attr_names = [f'precomputed_{i}'
                            for i in range(self._output_dim)]
        logger.info(
            f"[Precomputed] input_dim={input_dim} -> "
            f"output_dim={self._output_dim}")

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        raw_images: Optional[torch.Tensor] = None,
        precomputed_attrs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract facial semantic attributes.

        Args:
            raw_images:        (B, 3, H, W) in [0, 1].
            precomputed_attrs: (B, precomputed_dim) for 'precomputed' backend.

        Returns:
            (B, output_dim) semantic attribute vector.
            For face_llava vision-only: (B, output_dim) projected features.
            For face_llava with LLM: (B, 211) named attribute probabilities.
        """
        if self.backend_name == 'precomputed':
            if precomputed_attrs is None:
                raise ValueError(
                    "Precomputed backend requires precomputed_attrs tensor.")
            return self.projection(precomputed_attrs.float())

        if raw_images is None:
            raise ValueError(
                f"'{self.backend_name}' backend requires raw_images tensor.")

        with torch.no_grad():
            if self.backend_name == 'face_llava':
                if self._use_llm:
                    features = self._extract_face_llava_features(raw_images)
                else:
                    features = self._extract_face_llava_vision_features(
                        raw_images)
            elif self.backend_name == 'farl':
                x = (raw_images - self.norm_mean) / self.norm_std
                features = self.backbone(x)
            elif self.backend_name == 'celeba_vit':
                x = (raw_images - self.norm_mean) / self.norm_std
                features = torch.sigmoid(self.backbone(x).logits)

        return self.projection(features)

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def get_attribute_names(self) -> List[str]:
        return list(self._attr_names)

    @property
    def is_precomputed(self) -> bool:
        return self.backend_name == 'precomputed'

    @property
    def backbone_dim(self) -> int:
        return self._backbone_dim
