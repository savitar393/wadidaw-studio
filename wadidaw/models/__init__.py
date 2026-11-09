"""Multimodal audio-text gain staging model architectures (M1–M5 and ablations)."""

from wadidaw.models.audio_cnn import (
    MultiScaleSpectralBackbone,
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
    TEXT_EMBED_DIM,
    IndonesianSubwordTextEncoder,
    TextEncoderWrapper,
)

__all__ = [
    "CrossAttentionGainModel",
    "FiLMConditionedModel",
    "IndonesianSubwordTextEncoder",
    "LateFusionPretrainedModel",
    "LateFusionScratchModel",
    "MODEL_REGISTRY",
    "MultiScaleSpectralBackbone",
    "SharedStemAudioCNN",
    "StemFrameTokenizer",
    "StructuredTwoBranchModel",
    "TEXT_EMBED_DIM",
    "TextEncoderWrapper",
    "build_model",
    "compute_multimodal_loss",
]
