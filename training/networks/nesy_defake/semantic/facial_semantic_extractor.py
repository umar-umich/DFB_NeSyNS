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
1. FaceBench Face-LLaVA (recommended):
   Face-LLaVA-v1.5-13B fine-tuned on FaceBench (Wang et al., CVPR 2025).
   Covers 211 facial attributes across 5 views: Appearance (hair, skin,
   eyes, face shape, age, race, emotion), Accessories, Surrounding,
   Psychology (action units), Identity.

   We load ONLY the vision tower (CLIP-ViT-L/14@336px) and the mm_projector
   (MLP) from the checkpoint — the 13B LLM is NEVER loaded, saving ~25GB.
   The mm_projector was jointly trained during FaceBench instruction tuning
   and has learned to emphasise face-relevant visual features.

   Total VRAM: ~1.2GB FP16 (vision tower 304M params + projector 10M).
   HuggingFace: wxqlab/face-llava-v1.5-13b

2. FaRL:
   Microsoft's Face Representation Learning (Zheng et al., CVPR 2022).
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

ARCHITECTURE (FaceBench backend)
--------------------------------
  raw_images (B, 3, H, W)       [0, 1] range
       |
  [resize to 336x336]
       |
  [CLIP normalization]
       |
  [vision tower]  <- FROZEN (CLIP-ViT-L/14 from FaceBench checkpoint)
       |
  hidden_states[-2][:, 1:, :]   (576 patch tokens, 1024-d)
       |
  [mm_projector]  <- FROZEN (mlp2x_gelu, fine-tuned on FaceBench)
       |                        211 facial attrs across 5 views
  mean_pool -> semantic features (B, 5120)   <- OUTPUT, used as-is
       |
  [causal module / classifier]  <- TRAINABLE (downstream modules learn
       |                           from frozen semantic features)
  detection output
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
    #  Backend: FaceBench Face-LLaVA                                       #
    # ------------------------------------------------------------------ #

    def _build_face_llava(self, model_path: str, cfg: dict):
        """
        FaceBench Face-LLaVA (Wang et al., CVPR 2025).

        Loads ONLY the vision tower (CLIP-ViT-L/14@336px) and mm_projector
        (mlp2x_gelu MLP) directly from the safetensors checkpoint. The 13B
        LLM is NEVER loaded — we extract its weights surgically.

        The mm_projector was jointly fine-tuned during FaceBench instruction
        tuning on 23,841 VQA pairs covering 211 facial attributes across:
          - Appearance: hair, skin, eyes, face shape, age, emotion, race, ...
          - Accessories: glasses, hat, earrings, necklace, ...
          - Surrounding: background, lighting, ...
          - Psychology: facial action units, expressions, ...
          - Identity: gender, cultural features, ...

        Feature flow:
          image -> resize(336) -> CLIP-normalize -> CLIP-ViT-L/14
          -> hidden_states[-2][:, 1:, :]   (patch tokens, skip CLS)
          -> mm_projector (Linear 1024->5120, GELU, Linear 5120->5120)
          -> mean_pool -> (B, 5120) -> trainable projection -> (B, output_dim)

        VRAM: ~1.2GB FP16 (vision tower + projector only).
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
        logger.info(f"[FaceBench] Loading vision tower + mm_projector from: "
                    f"{model_path}")

        # --- Read model config to get architecture details ----------------
        config_path = model_path / 'config.json'
        with open(config_path) as f:
            model_config = json.load(f)

        vision_tower_name = model_config.get(
            'mm_vision_tower', 'openai/clip-vit-large-patch14-336')
        mm_hidden_size = model_config.get('mm_hidden_size', 1024)
        hidden_size = model_config.get('hidden_size', 5120)
        self._vision_select_layer = model_config.get(
            'mm_vision_select_layer', -2)
        self._vision_select_feature = model_config.get(
            'mm_vision_select_feature', 'patch')

        # --- Load CLIP vision tower (HuggingFace CLIPVisionModel) ---------
        from transformers import CLIPVisionModel, CLIPImageProcessor

        logger.info(f"[FaceBench] Loading vision tower: {vision_tower_name}")
        self.vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower_name, dtype=dtype)
        self.vision_tower.eval()

        # --- Build mm_projector (mlp2x_gelu architecture) -----------------
        self.mm_projector = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # --- Load fine-tuned weights from checkpoint ----------------------
        self._load_facebench_weights(model_path, model_config, dtype)

        # --- Image processor for normalization ----------------------------
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

        self._backbone_dim = hidden_size  # 5120 for 13B

        # --- Freeze vision tower + mm_projector ---------------------------
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        for p in self.mm_projector.parameters():
            p.requires_grad = False

        # Cast mm_projector to same dtype as vision tower
        self.mm_projector = self.mm_projector.to(dtype)

        # --- No trainable projection: FaceBench features used as-is -------
        # The mm_projector already produces face-attribute-aware 5120-d
        # features. Downstream modules (causal, classifier) learn from these.
        self._output_dim = self._backbone_dim  # override config output_dim
        self.projection = nn.Identity()

        self._attr_names = [f'facebench_{i}'
                            for i in range(self._output_dim)]

        logger.info(
            f"[FaceBench] Loaded. vision_tower={vision_tower_name}, "
            f"output_dim={self._output_dim} (raw FaceBench features, no projection), "
            f"image_size={self._image_size}, "
            f"select_layer={self._vision_select_layer}, "
            f"select_feature={self._vision_select_feature}")

    def _load_facebench_weights(self, model_path: Path, model_config: dict,
                                dtype):
        """
        Load vision tower + mm_projector weights from safetensors checkpoint.

        Weight key mapping (from FaceBench checkpoint):
          model.vision_tower.vision_tower.vision_model.* -> self.vision_tower.*
          model.mm_projector.{0,2}.{weight,bias}         -> self.mm_projector.*
        """
        from safetensors import safe_open

        # Find which shard contains vision tower + projector weights
        index_path = model_path / 'model.safetensors.index.json'
        if index_path.exists():
            import json
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index['weight_map']

            # Collect unique shard files needed for vision + projector keys
            needed_shards = set()
            for key, shard in weight_map.items():
                if ('vision_tower' in key or 'mm_projector' in key):
                    needed_shards.add(shard)
        else:
            # Single safetensors file
            needed_shards = set()
            for p in model_path.glob('*.safetensors'):
                needed_shards.add(p.name)

        logger.info(f"[FaceBench] Loading weights from shards: {needed_shards}")

        # Collect all vision tower + projector tensors
        vt_state = {}
        proj_state = {}

        for shard_name in needed_shards:
            shard_path = model_path / shard_name
            with safe_open(str(shard_path), framework='pt',
                           device='cpu') as f:
                for key in f.keys():
                    tensor = f.get_tensor(key).to(dtype)
                    if key.startswith(
                            'model.vision_tower.vision_tower.'):
                        # Strip prefix to match CLIPVisionModel state dict
                        clean_key = key[len(
                            'model.vision_tower.vision_tower.'):]
                        vt_state[clean_key] = tensor
                    elif key.startswith('model.mm_projector.'):
                        clean_key = key[len('model.mm_projector.'):]
                        proj_state[clean_key] = tensor

        # Load vision tower weights (may override HF pretrained if
        # FaceBench fine-tuned the vision tower too)
        if vt_state:
            missing, unexpected = self.vision_tower.load_state_dict(
                vt_state, strict=False)
            if missing:
                logger.warning(
                    f"[FaceBench] Vision tower missing keys "
                    f"({len(missing)}): {missing[:3]}...")
            logger.info(f"[FaceBench] Loaded {len(vt_state)} vision tower "
                        f"weight tensors")
        else:
            logger.warning("[FaceBench] No vision tower weights found in "
                           "checkpoint; using HF pretrained weights")

        # Load mm_projector weights
        if proj_state:
            missing, unexpected = self.mm_projector.load_state_dict(
                proj_state, strict=True)
            logger.info(f"[FaceBench] Loaded {len(proj_state)} mm_projector "
                        f"weight tensors")
        else:
            raise RuntimeError(
                "No mm_projector weights found in checkpoint. "
                "Ensure model_path points to a valid LLaVA checkpoint.")

    def _extract_face_llava_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract face-aware visual features from FaceBench Face-LLaVA.

        Flow: resize(336) -> CLIP-normalize -> CLIP-ViT-L/14
              -> hidden_states[select_layer][:, 1:, :]  (patch tokens)
              -> mm_projector -> mean_pool -> (B, 5120)
        """
        import torch.nn.functional as F_func

        B = images.shape[0]

        # Resize to vision tower resolution (336x336)
        if images.shape[-1] != self._image_size or \
                images.shape[-2] != self._image_size:
            images = F_func.interpolate(
                images, size=(self._image_size, self._image_size),
                mode='bilinear', align_corners=False)

        # Normalize with CLIP stats
        images = (images - self.norm_mean) / self.norm_std

        # Process in micro-batches to control peak VRAM
        all_features = []
        mbs = self._llava_batch_size
        dtype = next(self.vision_tower.parameters()).dtype

        for i in range(0, B, mbs):
            batch = images[i:i + mbs].to(dtype)

            # Vision tower forward with hidden states
            vt_out = self.vision_tower(
                pixel_values=batch,
                output_hidden_states=True,
                return_dict=True,
            )

            # Select the right hidden layer (LLaVA uses second-to-last)
            hidden_states = vt_out.hidden_states[self._vision_select_layer]

            # Select feature type: 'patch' = skip CLS token at position 0
            if self._vision_select_feature == 'patch':
                visual_tokens = hidden_states[:, 1:, :]
            else:
                visual_tokens = hidden_states
            # visual_tokens: (mb, num_patches, 1024)

            # Project through mm_projector (fine-tuned on FaceBench)
            projected = self.mm_projector(visual_tokens)
            # projected: (mb, num_patches, 5120)

            # Mean-pool over spatial tokens
            features = projected.mean(dim=1)  # (mb, 5120)

            all_features.append(features.float())

        return torch.cat(all_features, dim=0)  # (B, 5120)

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
