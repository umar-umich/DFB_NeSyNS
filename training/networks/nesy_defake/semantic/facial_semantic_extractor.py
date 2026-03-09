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
to the pipeline — the causal module sees the same spatial features twice,
dressed up differently.

Proper semantic attributes must come from an INDEPENDENT model processing
actual face images, providing genuinely new information that the spatial
and frequency branches do not capture.

SUPPORTED BACKENDS
------------------
1. Face-LLaVA (recommended for high-VRAM GPUs):
   Face-specific MLLM (WACV 2026) with Face-Region Guided Cross-Attention.
   Built on Video-LLaVA/LLaVA-Next (7B). We extract visual features from
   the vision tower + multi-modal projector WITHOUT running the LLM, giving
   rich face-aware features at ~1.5GB VRAM cost.
   https://github.com/ihp-lab/Face-LLaVA

2. FaRL:
   Microsoft's Face Representation Learning (CVPR 2022).
   ViT-B/16 pre-trained on 20M face images with face-specific contrastive
   and masked image modeling objectives.
   Download: https://github.com/FacePerceiver/FaRL
   Weights:  FaRL-Base-Patch16-LAIONFace20M-ep64.pth

3. CelebA ViT:
   ViT fine-tuned on CelebA-40 facial attributes (multi-label).
   40 interpretable binary attributes (smiling, eyeglasses, male, etc.)
   ideal for neuro-symbolic causal discovery.
   Requires user-provided fine-tuned model.

4. Precomputed:
   Pre-extracted attributes loaded from disk by the dataloader.
   Use with any offline face analysis pipeline.

ARCHITECTURE
------------
  raw_images (B,3,224,224)
       |
  [face backbone]  ← FROZEN (pre-trained face model, independent of CLIP)
       |
  features (B, backbone_dim)
       |
  [projection head]  ← TRAINABLE (maps to causal module's semantic space)
       |
  attributes (B, output_dim)
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


class FacialSemanticExtractor(nn.Module):
    """
    Extracts facial semantic attributes using a dedicated pre-trained face
    analysis model — completely independent of the spatial/frequency backbone.

    The face backbone is FROZEN (feature extraction only).
    The projection head is TRAINABLE (learns the most informative mapping
    from face features to the causal module's semantic input space).
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
    #  Backend: Face-LLaVA                                                 #
    # ------------------------------------------------------------------ #

    def _build_face_llava(self, model_path: str, cfg: dict):
        """
        Face-LLaVA (Chowdhury et al., WACV 2026).

        Face-specific MLLM with Face-Region Guided Cross-Attention that
        integrates face geometry with local visual features. Built on
        Video-LLaVA / LLaVA-Next (7B LLM + ViT-L vision encoder).

        Feature extraction modes (config key: feature_mode):
          'visual':  Vision tower + multi-modal projector only (~1.5 GB).
                     Extracts face-aware visual tokens projected into LLM
                     embedding space, then mean-pools. FAST — no LLM needed.
          'lm_hidden': Full forward through LLM, extracts last hidden states
                       at visual token positions (~16 GB FP16). RICHEST —
                       LLM contextualizes visual features. Needs >=24 GB.

        The model is loaded in FP16 (or BF16) and fully frozen.

        Setup:
          git clone https://github.com/ihp-lab/Face-LLaVA
          # Download checkpoint into ./checkpoints/FaceLLaVA
          # Set model_path in config to the checkpoint directory
        """
        self._feature_mode = cfg.get('feature_mode', 'visual')
        self._llava_batch_size = cfg.get('micro_batch_size', 64)
        dtype = torch.float16 if cfg.get('fp16', True) else torch.bfloat16

        if not model_path:
            raise ValueError(
                "Face-LLaVA backend requires 'model_path' in config.\n"
                "Clone: https://github.com/ihp-lab/Face-LLaVA\n"
                "Download checkpoint into checkpoints/FaceLLaVA\n"
                "Set semantic_attributes.model_path to that directory.")

        logger.info(
            f"[Face-LLaVA] Loading from: {model_path} "
            f"(mode={self._feature_mode}, dtype={dtype})")

        # Try loading as a standard HuggingFace LLaVA-family model first,
        # then fall back to Video-LLaVA / LLaVA-Next variants.
        model, image_processor = self._load_llava_model(model_path, dtype)

        # Store components we need for feature extraction
        self.vision_tower = model.get_vision_tower()
        if self.vision_tower is None:
            # Some LLaVA variants store it differently
            self.vision_tower = getattr(model, 'vision_tower',
                                        getattr(model.model, 'vision_tower',
                                                None))

        self.mm_projector = getattr(
            model, 'multi_modal_projector',
            getattr(model.model, 'mm_projector',
                    getattr(model, 'mm_projector', None)))

        if self._feature_mode == 'lm_hidden':
            # Keep the full language model for hidden state extraction
            self.language_model = getattr(model, 'language_model',
                                          getattr(model, 'model', None))
            self._backbone_dim = model.config.text_config.hidden_size
        else:
            # Visual-only mode: discard the LLM to save memory
            self.language_model = None
            # Projector output dim = LLM hidden size
            if hasattr(model.config, 'text_config'):
                self._backbone_dim = model.config.text_config.hidden_size
            else:
                # Detect from projector output
                self._backbone_dim = self._detect_projector_dim(model)

            # Free LLM memory
            if hasattr(model, 'language_model'):
                del model.language_model
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                del model.model.layers

        # Store image processor normalization
        if image_processor is not None:
            img_mean = getattr(image_processor, 'image_mean',
                               [0.48145466, 0.4578275, 0.40821073])
            img_std = getattr(image_processor, 'image_std',
                              [0.26862954, 0.26130258, 0.27577711])
            self._image_size = getattr(image_processor, 'size', {})
            if isinstance(self._image_size, dict):
                self._image_size = self._image_size.get(
                    'shortest_edge', self._image_size.get('height', 224))
        else:
            img_mean = [0.48145466, 0.4578275, 0.40821073]
            img_std = [0.26862954, 0.26130258, 0.27577711]
            self._image_size = 224

        self.register_buffer(
            'norm_mean', torch.tensor(img_mean).view(1, 3, 1, 1))
        self.register_buffer(
            'norm_std', torch.tensor(img_std).view(1, 3, 1, 1))

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

        # Trainable projection: backbone features → causal semantic space
        self.projection = nn.Sequential(
            nn.Linear(self._backbone_dim, self._output_dim * 2),
            nn.LayerNorm(self._output_dim * 2),
            nn.GELU(),
            nn.Linear(self._output_dim * 2, self._output_dim),
        )
        # projection is trainable (unfrozen by default after nn.Sequential init)

        self._attr_names = [f'face_llava_{i}'
                            for i in range(self._output_dim)]

        logger.info(
            f"[Face-LLaVA] Loaded. mode={self._feature_mode}, "
            f"backbone_dim={self._backbone_dim} → output_dim={self._output_dim}, "
            f"image_size={self._image_size}")

        # Clean up the full model reference
        del model
        torch.cuda.empty_cache()

    def _load_llava_model(self, model_path: str, dtype):
        """Load a LLaVA-family model. Tries multiple loading strategies."""
        image_processor = None

        # Strategy 1: Standard HuggingFace transformers LLaVA
        try:
            from transformers import (
                LlavaNextForConditionalGeneration,
                AutoProcessor,
            )
            logger.info("[Face-LLaVA] Trying LlavaNextForConditionalGeneration...")
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype, device_map='cpu',
                low_cpu_mem_usage=True)
            proc = AutoProcessor.from_pretrained(model_path)
            image_processor = proc.image_processor
            model.eval()
            return model, image_processor
        except Exception as e:
            logger.info(f"[Face-LLaVA] LlavaNext failed: {e}")

        # Strategy 2: Base LlavaForConditionalGeneration
        try:
            from transformers import (
                LlavaForConditionalGeneration,
                AutoProcessor,
            )
            logger.info("[Face-LLaVA] Trying LlavaForConditionalGeneration...")
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype, device_map='cpu',
                low_cpu_mem_usage=True)
            proc = AutoProcessor.from_pretrained(model_path)
            image_processor = proc.image_processor
            model.eval()
            return model, image_processor
        except Exception as e:
            logger.info(f"[Face-LLaVA] LlavaBase failed: {e}")

        # Strategy 3: AutoModel (handles custom model types)
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            logger.info("[Face-LLaVA] Trying AutoModelForCausalLM...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, device_map='cpu',
                trust_remote_code=True, low_cpu_mem_usage=True)
            try:
                proc = AutoProcessor.from_pretrained(model_path)
                image_processor = getattr(proc, 'image_processor', None)
            except Exception:
                pass
            model.eval()
            return model, image_processor
        except Exception as e:
            logger.info(f"[Face-LLaVA] AutoModel failed: {e}")

        raise RuntimeError(
            f"Could not load Face-LLaVA model from '{model_path}'.\n"
            f"Ensure the checkpoint directory contains a valid HuggingFace "
            f"model (config.json, model weights, etc.).\n"
            f"If Face-LLaVA uses a custom codebase, install it first:\n"
            f"  pip install -e /path/to/Face-LLaVA")

    def _detect_projector_dim(self, model) -> int:
        """Detect the multi-modal projector output dimension."""
        if self.mm_projector is not None:
            # Walk the projector to find the last Linear layer's out_features
            last_linear = None
            for module in self.mm_projector.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
            if last_linear is not None:
                return last_linear.out_features
        # Fallback: common LLaVA-7B hidden size
        logger.warning("[Face-LLaVA] Could not detect projector dim, "
                       "defaulting to 4096")
        return 4096

    def _extract_face_llava_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract face-aware visual features from Face-LLaVA.

        For 'visual' mode: vision_tower → mm_projector → mean_pool
        For 'lm_hidden' mode: full forward → last hidden at visual positions
        """
        import torch.nn.functional as F_func

        device = images.device
        B = images.shape[0]

        # Resize if the vision tower expects a different resolution
        if self._image_size and self._image_size != images.shape[-1]:
            images = F_func.interpolate(
                images, size=(self._image_size, self._image_size),
                mode='bilinear', align_corners=False)

        # Normalize
        images = (images - self.norm_mean) / self.norm_std

        # Process in micro-batches to control memory
        all_features = []
        mbs = self._llava_batch_size

        for i in range(0, B, mbs):
            batch = images[i:i + mbs]

            # Vision tower forward
            if hasattr(self.vision_tower, 'forward'):
                vision_out = self.vision_tower(batch)
            else:
                # Some LLaVA variants wrap the tower
                vision_out = self.vision_tower.image_processor_forward(batch)

            # Handle different output formats
            if hasattr(vision_out, 'last_hidden_state'):
                visual_tokens = vision_out.last_hidden_state
            elif isinstance(vision_out, (tuple, list)):
                visual_tokens = vision_out[0]
            else:
                visual_tokens = vision_out
            # visual_tokens: (mb, num_patches, hidden_dim)

            if self._feature_mode == 'visual':
                # Project to LLM space and mean-pool
                if self.mm_projector is not None:
                    projected = self.mm_projector(visual_tokens)
                else:
                    projected = visual_tokens
                # Mean-pool over spatial tokens → (mb, proj_dim)
                features = projected.mean(dim=1)

            elif self._feature_mode == 'lm_hidden':
                # Run through LLM to get contextualised hidden states
                if self.mm_projector is not None:
                    projected = self.mm_projector(visual_tokens)
                else:
                    projected = visual_tokens

                # Feed visual tokens as "inputs_embeds" to the LLM
                lm_out = self.language_model(
                    inputs_embeds=projected,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # Last hidden state, mean-pool over sequence
                features = lm_out.hidden_states[-1].mean(dim=1)

            all_features.append(features.float())

        return torch.cat(all_features, dim=0)  # (B, backbone_dim)

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
        on face images → produces face-specific embeddings that are
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

        # FaRL checkpoints store full CLIP model → extract visual encoder
        visual_keys = {k: v for k, v in state_dict.items()
                       if k.startswith('visual.')}

        if visual_keys:
            visual_state = {k[len('visual.'):]: v
                           for k, v in visual_keys.items()}
        else:
            # Assume state_dict is already visual-encoder-only
            visual_state = state_dict

        missing, unexpected = clip_model.visual.load_state_dict(
            visual_state, strict=False)
        if missing:
            logger.warning(f"[FaRL] Missing keys ({len(missing)}): "
                           f"{missing[:5]}...")
        if unexpected:
            logger.warning(f"[FaRL] Unexpected keys ({len(unexpected)}): "
                           f"{unexpected[:5]}...")

        # Keep only the visual encoder
        self.backbone = clip_model.visual
        self._backbone_dim = clip_model.visual.output_dim  # 512
        del clip_model

        # Freeze backbone entirely
        for p in self.backbone.parameters():
            p.requires_grad = False

        # FaRL uses the same normalization as CLIP
        self.register_buffer(
            'norm_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer(
            'norm_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        # Trainable projection: backbone features → causal semantic space
        self.projection = nn.Sequential(
            nn.Linear(self._backbone_dim, self._output_dim * 2),
            nn.LayerNorm(self._output_dim * 2),
            nn.GELU(),
            nn.Linear(self._output_dim * 2, self._output_dim),
        )

        self._attr_names = [f'farl_{i}' for i in range(self._output_dim)]
        logger.info(
            f"[FaRL] Loaded successfully. "
            f"backbone_dim={self._backbone_dim} → output_dim={self._output_dim}")

    # ------------------------------------------------------------------ #
    #  Backend: CelebA ViT                                                 #
    # ------------------------------------------------------------------ #

    def _build_celeba_vit(self, model_path: str, cfg: dict):
        """
        CelebA-40 facial attribute classifier.

        A ViT fine-tuned for multi-label classification on CelebA's 40
        binary facial attributes. Each dimension is interpretable:
        '5_o_clock_shadow', 'arched_eyebrows', ..., 'young'.

        This is ideal for neuro-symbolic reasoning — the causal module
        can discover real causal relationships like Male → No_Beard.

        Requires a user-provided model. Options:
          1. Fine-tune google/vit-base-patch16-224-in21k on CelebA-40
          2. Use github.com/clementapa/CelebFaces_Attributes_Classification
          3. Any ViTForImageClassification with problem_type='multi_label_classification'
        """
        from transformers import ViTForImageClassification, ViTImageProcessor

        if not model_path:
            raise ValueError(
                "CelebA ViT backend requires 'model_path' in config.\n"
                "Point it to a HuggingFace ViT fine-tuned on CelebA-40.\n"
                "See: github.com/clementapa/CelebFaces_Attributes_Classification")

        logger.info(f"[CelebA-ViT] Loading from: {model_path}")

        self.backbone = ViTForImageClassification.from_pretrained(model_path)

        # Extract normalization params from the processor
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

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        num_labels = self.backbone.config.num_labels
        self._backbone_dim = num_labels

        # Use named CelebA attributes if dimensionality matches
        if num_labels == 40:
            self._attr_names = list(CELEBA_ATTRIBUTES)
        else:
            self._attr_names = [f'attr_{i}' for i in range(num_labels)]

        # Projection: attribute logits → causal semantic space
        if self._output_dim == num_labels:
            # Direct pass-through: use raw sigmoid scores as attributes
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
            f"[CelebA-ViT] Loaded. num_labels={num_labels} → "
            f"output_dim={self._output_dim}")

    # ------------------------------------------------------------------ #
    #  Backend: Precomputed                                                #
    # ------------------------------------------------------------------ #

    def _build_precomputed(self, cfg: dict):
        """
        Precomputed attributes loaded from disk by the dataset.

        Use this with heavy models that cannot run per-batch, e.g.:
          - Face-LLaVA (WACV 2026): detailed facial descriptions → vectors
          - Any offline face analysis pipeline

        The dataset must provide 'semantic_attrs' in data_dict.
        An optional trainable projection maps to the causal input space.
        """
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

        # No image backbone needed
        self.backbone = None

        # Dummy normalization buffers (unused but kept for interface consistency)
        self.register_buffer('norm_mean', torch.zeros(1))
        self.register_buffer('norm_std', torch.ones(1))

        self._attr_names = [f'precomputed_{i}'
                            for i in range(self._output_dim)]
        logger.info(
            f"[Precomputed] input_dim={input_dim} → "
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
            raw_images:        (B, 3, H, W) in [0, 1]. Required for
                               'farl' and 'celeba_vit' backends.
            precomputed_attrs: (B, precomputed_dim). Required for
                               'precomputed' backend.

        Returns:
            (B, output_dim) semantic attribute vector.
        """
        if self.backend_name == 'precomputed':
            if precomputed_attrs is None:
                raise ValueError(
                    "Precomputed backend requires precomputed_attrs tensor. "
                    "Ensure dataset provides 'semantic_attrs' in data_dict.")
            return self.projection(precomputed_attrs.float())

        if raw_images is None:
            raise ValueError(
                f"'{self.backend_name}' backend requires raw_images tensor. "
                f"Ensure data_dict contains 'raw_frames'.")

        # Extract features from frozen backbone (no grad tracking for memory)
        with torch.no_grad():
            if self.backend_name == 'face_llava':
                features = self._extract_face_llava_features(raw_images)
            elif self.backend_name == 'farl':
                # Apply FaRL/CLIP normalization
                x = (raw_images - self.norm_mean) / self.norm_std
                features = self.backbone(x)              # (B, 512)
            elif self.backend_name == 'celeba_vit':
                x = (raw_images - self.norm_mean) / self.norm_std
                features = torch.sigmoid(
                    self.backbone(x).logits)              # (B, 40)

        # Trainable projection to causal semantic space
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
