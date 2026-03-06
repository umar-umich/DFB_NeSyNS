"""
networks/nesy_defake/semantic/clip_facial_attributes.py
=======================================================
On-the-fly CLIP-based facial attribute extraction (FaceLaVa/FaceBench-inspired).

Computes detailed facial semantic features during training by leveraging CLIP's
zero-shot classification capability. Text embeddings for ~68 facial attribute
prompts are precomputed once at init; at runtime, cosine similarity between
the spatial CLIP features and cached text embeddings produces attribute scores.

Advantages over precomputed DeepFace features:
  1. Augmentation-consistent: computed on the augmented image, not cached originals
  2. Differentiable: gradients flow through cosine similarity back to backbone
  3. Rich: ~68 fine-grained attributes vs 73 coarse DeepFace features
  4. Zero extra model: reuses the already-loaded CLIP backbone
  5. No preprocessing step: works on any new dataset without feature extraction

The attribute prompts are organized into categories:
  - Face structure (shape, symmetry, proportions)
  - Eyes and eyebrows
  - Nose and mouth
  - Skin quality and texture
  - Expression and micro-expressions
  - Demographics (age, gender)
  - Image quality indicators
  - Facial consistency cues (relevant for deepfake detection)
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Facial attribute prompt definitions (FaceLaVa/FaceBench-inspired)
# ---------------------------------------------------------------------------

FACIAL_ATTRIBUTE_PROMPTS = {
    # --- Face Structure ---
    'face_oval': 'a photo of a face with an oval shape',
    'face_round': 'a photo of a face with a round shape',
    'face_square': 'a photo of a face with a square jawline',
    'face_symmetric': 'a photo of a face with symmetric features',
    'face_asymmetric': 'a photo of a face with asymmetric features',
    'cheekbones_high': 'a photo of a face with high cheekbones',
    'chin_narrow': 'a photo of a face with a narrow chin',
    'forehead_high': 'a photo of a face with a high forehead',
    'face_proportional': 'a photo of a face with well-proportioned features',

    # --- Eyes & Eyebrows ---
    'eyes_large': 'a photo of a face with large open eyes',
    'eyes_narrow': 'a photo of a face with narrow eyes',
    'eyes_deep_set': 'a photo of a face with deep-set eyes',
    'eyes_direct_gaze': 'a photo of a face looking directly at the camera',
    'eyes_averted': 'a photo of a face with eyes looking sideways',
    'eyes_closed': 'a photo of a face with closed eyes',
    'eyebrows_thick': 'a photo of a face with thick eyebrows',
    'eyebrows_thin': 'a photo of a face with thin eyebrows',
    'eyebrows_arched': 'a photo of a face with arched eyebrows',
    'under_eye_bags': 'a photo of a face with bags under the eyes',

    # --- Nose & Mouth ---
    'nose_wide': 'a photo of a face with a wide nose',
    'nose_narrow': 'a photo of a face with a narrow pointed nose',
    'lips_full': 'a photo of a face with full lips',
    'lips_thin': 'a photo of a face with thin lips',
    'mouth_open': 'a photo of a face with an open mouth',
    'mouth_closed': 'a photo of a face with a closed mouth',
    'showing_teeth': 'a photo of a face showing teeth while smiling',

    # --- Expression ---
    'expr_smile': 'a photo of a face with a genuine smile',
    'expr_neutral': 'a photo of a face with a neutral expression',
    'expr_frown': 'a photo of a face with a frown',
    'expr_surprise': 'a photo of a face with a surprised expression',
    'expr_tense': 'a photo of a face with a tense expression',
    'expr_relaxed': 'a photo of a face with a relaxed expression',
    'expr_raised_brows': 'a photo of a face with raised eyebrows',
    'expr_squinting': 'a photo of a face with squinted eyes',
    'expr_pursed_lips': 'a photo of a face with pursed lips',

    # --- Skin & Texture ---
    'skin_smooth': 'a photo of a face with smooth clear skin',
    'skin_wrinkled': 'a photo of a face with wrinkled skin',
    'skin_pores': 'a photo of a face with visible skin pores',
    'skin_blemishes': 'a photo of a face with skin blemishes or acne',
    'skin_even_tone': 'a photo of a face with even skin tone',
    'skin_uneven_tone': 'a photo of a face with uneven skin tone',
    'facial_hair': 'a photo of a face with facial hair or beard',
    'no_facial_hair': 'a photo of a clean-shaven face',
    'wearing_makeup': 'a photo of a face wearing makeup',
    'natural_skin': 'a photo of a face with natural skin without makeup',
    'skin_texture_varied': 'a photo of a face with natural skin texture variation',

    # --- Demographics ---
    'young': 'a photo of a young person in their twenties',
    'middle_aged': 'a photo of a middle-aged person',
    'elderly': 'a photo of an elderly person with wrinkles',
    'male': 'a photo of a male face',
    'female': 'a photo of a female face',
    'glasses': 'a photo of a face wearing glasses',
    'no_glasses': 'a photo of a face without glasses',

    # --- Image Quality ---
    'sharp_image': 'a sharp high-quality photo of a face',
    'blurry_image': 'a blurry low-quality photo of a face',
    'harsh_shadows': 'a photo of a face with harsh shadows',
    'even_lighting': 'a photo of a face with even natural lighting',
    'overexposed': 'a photo of an overexposed face',
    'low_contrast': 'a photo of a face with low contrast',

    # --- Facial Consistency (deepfake-relevant texture cues) ---
    'natural_skin_texture': 'a photo of a face with natural detailed skin texture',
    'overly_smooth_skin': 'a photo of a face with unnaturally smooth airbrushed skin',
    'natural_eye_reflections': 'a photo of a face with natural specular eye reflections',
    'consistent_lighting_face': 'a photo of a face with consistent lighting across all regions',
    'natural_hair_boundary': 'a photo of a face with a natural hairline boundary',
    'natural_facial_contours': 'a photo of a face with natural shadow contours',
    'consistent_skin_across_face': 'a photo of a face with consistent skin texture everywhere',
    'natural_ear_detail': 'a photo of a face with natural detailed ear structure',
    'natural_teeth_detail': 'a photo of a face with natural detailed teeth',
}


class CLIPFacialAttributeExtractor(nn.Module):
    """
    On-the-fly facial attribute extraction using CLIP zero-shot classification.

    At init:
      1. Load CLIP text encoder (from the same model as spatial backbone)
      2. Compute text embeddings for all attribute prompts
      3. Extract visual projection weights (pooler_output -> shared space)
      4. Discard text encoder (only keep cached embeddings + projection)

    At forward:
      spatial_raw (B, 1024) -> visual_proj -> L2 normalize -> cosine sim
      with cached text embeddings -> attribute_scores (B, N_attrs)

    The attribute scores are differentiable w.r.t. spatial_raw when
    the visual projection is trainable (default: frozen for stability).
    """

    def __init__(self, config: dict):
        super().__init__()

        sem_cfg = config.get('semantic_attributes', {})
        clip_path = config['foundation_models']['spatial']['model_path']
        self.temperature = sem_cfg.get('temperature', 0.07)

        # Select which attribute categories to use
        categories = sem_cfg.get('categories', 'all')
        self.prompts, self.prompt_names = self._select_prompts(categories)
        self.num_attributes = len(self.prompts)

        # Load CLIP model to extract text embeddings and visual projection
        self._init_clip_embeddings(clip_path)

        logger.info(
            f"[CLIPFacialAttributes] {self.num_attributes} attributes, "
            f"clip_path={clip_path}, temperature={self.temperature}"
        )

    def _select_prompts(self, categories):
        """Select prompts based on configured categories."""
        if categories == 'all':
            prompts = list(FACIAL_ATTRIBUTE_PROMPTS.values())
            names = list(FACIAL_ATTRIBUTE_PROMPTS.keys())
            return prompts, names

        # Filter by category prefix
        if isinstance(categories, str):
            categories = [categories]

        category_prefixes = {
            'face_structure': ['face_', 'cheekbones_', 'chin_', 'forehead_'],
            'eyes': ['eyes_', 'eyebrows_', 'under_eye_'],
            'nose_mouth': ['nose_', 'lips_', 'mouth_', 'showing_'],
            'expression': ['expr_'],
            'skin': ['skin_', 'facial_hair', 'no_facial_', 'wearing_', 'natural_skin'],
            'demographics': ['young', 'middle_', 'elderly', 'male', 'female', 'glasses', 'no_glasses'],
            'quality': ['sharp_', 'blurry_', 'harsh_', 'even_light', 'overexposed', 'low_contrast'],
            'consistency': ['natural_', 'overly_', 'consistent_'],
        }

        selected_names = []
        for cat in categories:
            prefixes = category_prefixes.get(cat, [])
            for name in FACIAL_ATTRIBUTE_PROMPTS:
                if any(name.startswith(p) for p in prefixes) and name not in selected_names:
                    selected_names.append(name)

        prompts = [FACIAL_ATTRIBUTE_PROMPTS[n] for n in selected_names]
        return prompts, selected_names

    @torch.no_grad()
    def _init_clip_embeddings(self, clip_path: str):
        """Load CLIP, compute text embeddings, extract visual projection, discard the rest."""
        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"[CLIPFacialAttributes] Loading CLIP for text embeddings: {clip_path}")
        clip_model = CLIPModel.from_pretrained(clip_path)
        processor = CLIPProcessor.from_pretrained(clip_path)

        # Compute text embeddings for all prompts
        text_inputs = processor(
            text=self.prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
        )
        text_outputs = clip_model.text_model(**text_inputs)
        # Apply text projection to get shared-space embeddings
        text_embeds = clip_model.text_projection(text_outputs.pooler_output)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Cache text embeddings as buffer (not a parameter - no gradient needed)
        self.register_buffer('text_embeddings', text_embeds)  # (N_attrs, proj_dim)

        # Cache visual projection as a trainable linear layer
        # (frozen by default for stability, but can be unfrozen)
        proj_dim = clip_model.visual_projection.in_features
        out_dim = clip_model.visual_projection.out_features
        self.visual_projection = nn.Linear(proj_dim, out_dim, bias=False)
        self.visual_projection.weight.data.copy_(clip_model.visual_projection.weight.data)
        # Freeze by default - the zero-shot calibration is important
        self.visual_projection.weight.requires_grad = False

        logger.info(
            f"[CLIPFacialAttributes] Cached {self.num_attributes} text embeddings "
            f"({text_embeds.shape}), visual_proj: {proj_dim}->{out_dim}"
        )

        # Discard full CLIP model
        del clip_model, processor

    def forward(self, spatial_raw: torch.Tensor) -> torch.Tensor:
        """
        Compute facial attribute scores from spatial CLIP features.

        Args:
            spatial_raw: (B, backbone_dim) raw CLIP pooler_output from spatial extractor

        Returns:
            attribute_scores: (B, num_attributes) cosine similarity scores
        """
        # Project to CLIP shared embedding space
        image_embeds = self.visual_projection(spatial_raw)
        image_embeds = F.normalize(image_embeds, dim=-1)

        # Cosine similarity with cached text embeddings
        # (B, proj_dim) @ (proj_dim, N_attrs) -> (B, N_attrs)
        similarity = image_embeds @ self.text_embeddings.t()

        # Scale by temperature (higher temperature -> softer scores)
        attribute_scores = similarity / self.temperature

        return attribute_scores

    def get_attribute_names(self) -> List[str]:
        """Return human-readable names for each attribute dimension."""
        return list(self.prompt_names)

    @property
    def output_dim(self) -> int:
        return self.num_attributes
