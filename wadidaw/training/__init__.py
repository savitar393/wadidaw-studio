"""Training and evaluation utilities for multimodal audio-text gain staging."""

from wadidaw.training.evaluate import (
    compute_regression_and_spectral_metrics,
    evaluate_model,
    make_baseline_predictor,
)
from wadidaw.training.train import (
    CHECKPOINT_DIR,
    TRAINING_DIR,
    TrainConfig,
    resolve_device,
    seed_everything,
    train_one_run,
)

__all__ = [
    "CHECKPOINT_DIR",
    "TRAINING_DIR",
    "TrainConfig",
    "compute_regression_and_spectral_metrics",
    "evaluate_model",
    "make_baseline_predictor",
    "resolve_device",
    "seed_everything",
    "train_one_run",
]
