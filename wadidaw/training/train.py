"""Supervised multimodal training loop with early stopping, AMP, and reproducibility tracking.

Implements Section 7 of dokumentasi.md:
- AdamW optimizer (`lr=1e-3`, `weight_decay=1e-4`), CosineAnnealingLR, gradient clipping (`1.0`).
- Automatic Mixed Precision (AMP) when running on CUDA GPUs.
- Early stopping on validation MAE (`patience=5`).
- Saves checkpoints, training curves (CSV), and full evaluation reports (JSON).
"""

from __future__ import annotations

import csv
import json
import platform
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wadidaw.data.dataset import MusdbMultimodalDataset, SAMPLES_ROOT
from wadidaw.data.musdb import ROOT
from wadidaw.models.fusion import MODEL_REGISTRY, build_model
from wadidaw.models.structured import compute_multimodal_loss
from wadidaw.training.evaluate import evaluate_model, make_baseline_predictor


CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"
TRAINING_DIR = ROOT / "outputs" / "training"


@dataclass
class TrainConfig:
    model_name: str = "m3"
    run_name: str = "m3_seed42"
    seed: int = 42
    batch_size: int = 16
    epochs: int = 30
    patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lambda_loudness: float = 0.1
    lambda_balance: float = 0.1
    text_backend: str = "subword"
    num_workers: int = 0
    device: str = "auto"
    use_amp: bool = True
    train_manifest: str = str(SAMPLES_ROOT / "train_samples.json")
    val_manifest: str = str(SAMPLES_ROOT / "validation_samples.json")
    test_human_manifest: str = str(SAMPLES_ROOT / "test_human_samples.json")
    test_synthetic_manifest: str = str(SAMPLES_ROOT / "test_synthetic_samples.json")
    overwrite: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def train_one_run(config: TrainConfig) -> dict[str, object]:
    """Train a single model run (`M1`..`M5` or ablation), evaluate on validation & test sets, and save artifacts."""
    if config.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name '{config.model_name}'. Expected one of {sorted(MODEL_REGISTRY)}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = CHECKPOINT_DIR / f"{config.run_name}_best.pt"
    csv_path = TRAINING_DIR / f"{config.run_name}_metrics.csv"
    json_path = TRAINING_DIR / f"{config.run_name}_results.json"

    if not config.overwrite:
        for path in (ckpt_path, csv_path, json_path):
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}. Pass overwrite=True or a new run_name.")

    seed_everything(config.seed)
    device = resolve_device(config.device)
    amp_enabled = bool(config.use_amp and device.type == "cuda")

    train_ds = MusdbMultimodalDataset(config.train_manifest)
    val_ds = MusdbMultimodalDataset(config.val_manifest)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = build_model(config.model_name, text_backend=config.text_backend).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
        eta_min=config.lr * 0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    best_val_metrics: dict[str, object] = {}
    epochs_no_improve = 0
    history: list[dict[str, object]] = []
    start_time = time.time()

    print(
        f"[{config.run_name}] Training {MODEL_REGISTRY[config.model_name]} on {device} "
        f"(train={len(train_ds)}, val={len(val_ds)}, params={sum(p.numel() for p in trainable_params):,})"
    )

    for epoch in range(1, config.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        total_loss = 0.0
        total_gain_mse = 0.0
        n_samples = 0

        for batch in train_loader:
            audio_key = getattr(model, "audio_input_key", "audio")
            audio = batch[audio_key].to(device=device, dtype=torch.float32)
            prompts = batch["prompt"]
            g_true = batch["gains_db"].to(device=device, dtype=torch.float32)
            l_true = batch["loudness_db"].to(device=device, dtype=torch.float32)
            r_true = batch["balance_target"].to(device=device, dtype=torch.float32)
            active_mask = batch["active_mask"].to(device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(audio, prompts, active_mask=active_mask, clamp_output=False)
                losses = compute_multimodal_loss(
                    outputs=outputs,
                    gains_true=g_true,
                    loudness_true=l_true,
                    balance_true=r_true,
                    active_mask=active_mask,
                    lambda_loudness=config.lambda_loudness,
                    lambda_balance=config.lambda_balance,
                )
                loss = losses["loss"]

            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, max_norm=config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bsz = len(audio)
            total_loss += float(loss.item()) * bsz
            total_gain_mse += float(losses["gain_mse"].item()) * bsz
            n_samples += bsz

        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device=device)
        val_mae = float(val_metrics["mae_db"])
        elapsed_epoch = time.time() - epoch_t0

        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(n_samples, 1), 6),
            "train_gain_mse": round(total_gain_mse / max(n_samples, 1), 6),
            "val_mae_db": val_mae,
            "val_rmse_db": float(val_metrics["rmse_db"]),
            "val_band_mae_db": float(val_metrics["band_mae_db"]),
            "val_success_1db": float(val_metrics["success_rate_1db"]),
            "val_adherence": float(val_metrics["instruction_adherence"]),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            "epoch_sec": round(elapsed_epoch, 2),
        }
        history.append(row)

        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            epochs_no_improve = 0
            marker = " *best*"
        else:
            epochs_no_improve += 1
            marker = ""

        print(
            f"  Epoch {epoch:02d}/{config.epochs:02d} | "
            f"train_mse={row['train_gain_mse']:.4f} | "
            f"val_MAE={val_mae:.4f} dB | "
            f"val_RMSE={row['val_rmse_db']:.4f} dB | "
            f"succ<1dB={row['val_success_1db']:5.1f}% | "
            f"adherence={row['val_adherence']:5.1f}% ({elapsed_epoch:.1f}s){marker}"
        )

        if config.patience > 0 and epochs_no_improve >= config.patience:
            print(f"  Early stopping triggered at epoch {epoch} (best epoch {best_epoch}: MAE={best_val_mae:.4f} dB)")
            break

    total_train_sec = round(time.time() - start_time, 2)
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    torch.save(
        {
            "model_name": config.model_name,
            "model_title": MODEL_REGISTRY[config.model_name],
            "text_backend": config.text_backend,
            "seed": config.seed,
            "best_epoch": best_epoch,
            "best_val_mae_db": best_val_mae,
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
        },
        ckpt_path,
    )

    # Evaluate on test_human and test_synthetic if manifests exist
    test_human_metrics = None
    if Path(config.test_human_manifest).is_file():
        test_human_ds = MusdbMultimodalDataset(config.test_human_manifest)
        test_human_loader = DataLoader(test_human_ds, batch_size=config.batch_size, shuffle=False)
        test_human_metrics = evaluate_model(model, test_human_loader, device=device)

    test_synthetic_metrics = None
    if Path(config.test_synthetic_manifest).is_file():
        test_syn_ds = MusdbMultimodalDataset(config.test_synthetic_manifest)
        test_syn_loader = DataLoader(test_syn_ds, batch_size=config.batch_size, shuffle=False)
        test_synthetic_metrics = evaluate_model(model, test_syn_loader, device=device)

    # Write CSV training curve
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    results = {
        "run_name": config.run_name,
        "model_name": config.model_name,
        "model_title": MODEL_REGISTRY[config.model_name],
        "seed": config.seed,
        "best_epoch": best_epoch,
        "total_train_sec": total_train_sec,
        "trainable_params": sum(p.numel() for p in trainable_params),
        "config": asdict(config),
        "versions": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "soundfile": version("soundfile"),
        },
        "validation": best_val_metrics,
        "test_human": test_human_metrics,
        "test_synthetic": test_synthetic_metrics,
        "history": history,
    }
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    return results
