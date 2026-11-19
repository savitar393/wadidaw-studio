"""Multimodal audio-text gain staging model architectures (M1–M5 and ablations)."""

from wadidaw.models.audio_cnn import (
    MultiScaleSpectralBackbone,
    OpenL3TokenProjector,
    SharedStemAudioCNN,
    StemFrameTokenizer,
)
from wadidaw.models.fusion import (
    MODEL_REGISTRY,
    CrossAttentionGainModel,
    FiLMConditionedModel,
    LateFusionPretrainedModel,
    LateFusionScratchModel,
    build_model,
)
from wadidaw.models.structured import (
    StructuredTwoBranchModel,
    compute_multimodal_loss,
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
    "FiLMConditionedModel",
    "INDOBERT_EMBED_DIM",
    "INDOBERT_MODEL_NAME",
    "IndonesianSubwordTextEncoder",
    "LateFusionPretrainedModel",
    "LateFusionScratchModel",
    "MODEL_REGISTRY",
    "MultiScaleSpectralBackbone",
    "OpenL3TokenProjector",
    "SharedStemAudioCNN",
    "StemFrameTokenizer",
    "StructuredTwoBranchModel",
    "TEXT_EMBED_DIM",
    "TextEncoderWrapper",
    "build_model",
    "compute_multimodal_loss",
    "get_text_embed_dim",
]
