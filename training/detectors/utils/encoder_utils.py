# ── Encoder registry ────────────────────────────────────────────────────────
# Central lookup for all supported encoders.
# Add new encoders here — extractors read from this registry.
import logging
logger = logging.getLogger(__name__)

ENCODER_REGISTRY = {
    # ── CLIP family (HuggingFace transformers) ───────────────────────────
    'openai/clip-vit-base-patch16':           {'dim': 768,  'size': 224, 'api': 'hf_clip'},
    'openai/clip-vit-base-patch32':           {'dim': 768,  'size': 224, 'api': 'hf_clip'},
    'openai/clip-vit-large-patch14':          {'dim': 1024, 'size': 224, 'api': 'hf_clip'},
    'openai/clip-vit-large-patch14-336':      {'dim': 1024, 'size': 336, 'api': 'hf_clip'},
    'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': {'dim': 1280, 'size': 224, 'api': 'hf_clip'},

    # ── Perception Encoder family (open_clip via timm hub) ───────────────
    # Requires: pip install open_clip_torch timm
    # PE-Core-L14-336: same ViT-L arch as CLIP-L but stronger pretraining
    # Outperforms SigLIP2 on image classification benchmarks (Apr 2025)
    'hf-hub:timm/PE-Core-L-14-336':          {'dim': 1024, 'size': 336, 'api': 'open_clip'},
    'hf-hub:timm/PE-Core-G-14-448':          {'dim': 1280, 'size': 448, 'api': 'open_clip'},

    # ── DINOv2 family (HuggingFace transformers) ─────────────────────────
    'facebook/dinov2-large':                  {'dim': 1024, 'size': 224, 'api': 'hf_dino'},
    'facebook/dinov2-giant':                  {'dim': 1536, 'size': 224, 'api': 'hf_dino'},
}


def build_encoder(model_path: str):
    """
    Build a vision encoder from the registry.
    Returns (encoder_module, hidden_dim, required_size, api_type).

    api_type drives which forward function to use in the extractor:
        'hf_clip'   — outputs.pooler_output
        'open_clip' — model.encode_image(x)
        'hf_dino'   — outputs.last_hidden_state[:, 0]
    """
    if model_path not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown model_path: '{model_path}'. "
            f"Add it to ENCODER_REGISTRY or choose from: "
            f"{list(ENCODER_REGISTRY.keys())}"
        )
    spec     = ENCODER_REGISTRY[model_path]
    api_type = spec['api']
    dim      = spec['dim']
    size     = spec['size']

    if api_type == 'hf_clip':
        from transformers import CLIPVisionModel
        logger.info(f"[EncoderRegistry] Loading HF CLIP: {model_path}")
        encoder = CLIPVisionModel.from_pretrained(model_path)

    elif api_type == 'open_clip':
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is required for Perception Encoder. "
                "Install with: pip install open_clip_torch timm"
            )
        logger.info(f"[EncoderRegistry] Loading PE via open_clip: {model_path}")
        # open_clip returns (model, preprocess_train, preprocess_val)
        # We only need the model — preprocessing is handled by our dataloader
        encoder, _, _ = open_clip.create_model_and_transforms(model_path)
        # Strip text tower to save memory — we only use visual encoder
        if hasattr(encoder, 'transformer'):
            del encoder.transformer   # text transformer
        if hasattr(encoder, 'token_embedding'):
            del encoder.token_embedding

    elif api_type == 'hf_dino':
        from transformers import AutoModel
        logger.info(f"[EncoderRegistry] Loading DINOv2: {model_path}")
        encoder = AutoModel.from_pretrained(model_path)

    else:
        raise ValueError(f"Unknown api_type: '{api_type}'")

    logger.info(
        f"[EncoderRegistry] Built encoder: {model_path} | "
        f"dim={dim}, size={size}, api={api_type}"
    )
    return encoder, dim, size, api_type