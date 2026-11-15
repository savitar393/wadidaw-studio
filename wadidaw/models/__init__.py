"""Multimodal audio-text gain staging models: M1 baseline and M4 variants."""

from wadidaw.losses import compute_multimodal_loss
from wadidaw.models.audio_cnn import (
    OpenL3TokenProjector,
    SharedStemAudioCNN,
    StemFrameTokenizer,
)
from wadidaw.models.fusion import (
    MODEL_REGISTRY,
    CrossAttentionGainModel,
    LateFusionScratchModel,
    build_model,
)
from wadidaw.models.text_encoder import (
    INDOBERT_EMBED_DIM,
    INDOBERT_MODEL_NAME,
    TEXT_EMBED_DIM,
    IndonesianSubwordTextEncoder,
    TextEncoderWrapper,
    get_text_embed_dim,
)

__all__ = [
    "CrossAttentionGainModel",
    "INDOBERT_EMBED_DIM",
    "INDOBERT_MODEL_NAME",
    "IndonesianSubwordTextEncoder",
    "LateFusionScratchModel",
    "MODEL_REGISTRY",
    "OpenL3TokenProjector",
    "SharedStemAudioCNN",
    "StemFrameTokenizer",
    "TEXT_EMBED_DIM",
    "TextEncoderWrapper",
    "build_model",
    "compute_multimodal_loss",
    "get_text_embed_dim",
]
