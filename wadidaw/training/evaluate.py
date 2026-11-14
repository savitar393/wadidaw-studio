"""Comprehensive evaluation metrics and baseline evaluators for multimodal gain staging.

Implements Section 8 of dokumentasi.md:
- Parameter-level metrics (Section 8.1): MAE dB (overall & per stem), RMSE dB, Bias dB,
  R^2, Pearson r, Spearman rho, Max error dB.
- Spectral-level metrics (Section 8.2): 8-band MAE (dB), normalized 8-band MSE,
  64-bin Log-Spectral Distance (dB), Worst-band error (dB).
- Text / Instruction metrics (Section 8.4): Direction/Intent accuracy, Instruction adherence,
  Balance MAE (dB).
- Success thresholds (Section 8.6): `MAE < 1.0 dB` and `band_mse_norm < 0.01`.
- Baselines (Section 8.7): `no_op`, `random`, `dsp_equal_loudness` (audio-only DSP),
  and `rule_dsp_prompt` (audio DSP + rule-based prompt parser).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from wadidaw.data.features import GAIN_MAX_DB, GAIN_MIN_DB
from wadidaw.data.prompts import STEM_NAMES, parse_prompt_to_balance


def _compute_mix_8band_and_64mel_db(
    stem_mel_power_or_logmel: np.ndarray,
    gains_db: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute `(N, 8)` 8-band mix energy (dBFS) and `(N, 64)` mel profile from stem mel power.

    Args:
        stem_mel_power_or_logmel: `(N, 4, 64)` mean stem mel power or `(N, 4, 64, T)` log-mel dBFS.
        gains_db: `(N, 4)` applied stem gains in dB.
        active_mask: `(N, 4)` boolean mask of active stems.
    """
    if stem_mel_power_or_logmel.ndim == 4:
        stem_mel_power = np.power(10.0, np.clip(stem_mel_power_or_logmel, -100.0, 20.0) / 10.0).mean(axis=-1)
    else:
        stem_mel_power = stem_mel_power_or_logmel
    effective_gains = np.where(active_mask, gains_db, 0.0)
    gain_power = np.power(10.0, effective_gains / 10.0)[:, :, None]  # (N, 4, 1)

    mix_mel_power = np.sum(stem_mel_power * gain_power, axis=1)  # (N, 64)
    mix_mel_db = 10.0 * np.log10(np.maximum(mix_mel_power, 1e-12))

    # Group 64 mel bins into 8 spectral bands (8 mel bins per band)
    mix_8band_power = mix_mel_power.reshape(len(stem_mel_power), 8, 8).mean(axis=-1)  # (N, 8)
    mix_8band_db = 10.0 * np.log10(np.maximum(mix_8band_power, 1e-12))
    return mix_8band_db.astype(np.float32), mix_mel_db.astype(np.float32)


def compute_regression_and_spectral_metrics(
    gains_pred: np.ndarray,
    gains_true: np.ndarray,
    loudness_true: np.ndarray,
    balance_true: np.ndarray,
    active_mask: np.ndarray,
    band8_pred_db: np.ndarray,
    band8_true_db: np.ndarray,
    mel64_pred_db: np.ndarray,
    mel64_true_db: np.ndarray,
) -> dict[str, object]:
    """Compute all parameter, spectral, and instruction adherence metrics from arrays."""
    g_pred = np.asarray(gains_pred, dtype=np.float64)
    g_true = np.asarray(gains_true, dtype=np.float64)
    l_true = np.asarray(loudness_true, dtype=np.float64)
    r_true = np.asarray(balance_true, dtype=np.float64)
    mask = np.asarray(active_mask, dtype=bool)

    diff = (g_pred - g_true) * mask
    abs_diff = np.abs(diff)
    active_counts_per_sample = np.maximum(mask.sum(axis=1), 1)
    total_active = max(int(mask.sum()), 1)

    # 1. Parameter-level metrics (Section 8.1)
    mae_db = float(abs_diff.sum() / total_active)
    rmse_db = float(np.sqrt(np.square(diff).sum() / total_active))
    bias_db = float(diff.sum() / total_active)
    max_error_db = float(np.max(abs_diff))

    mae_per_stem: dict[str, float] = {}
    rmse_per_stem: dict[str, float] = {}
    for i, stem_name in enumerate(STEM_NAMES):
        s_mask = mask[:, i]
        s_denom = max(int(s_mask.sum()), 1)
        mae_per_stem[stem_name] = round(float(abs_diff[s_mask, i].sum() / s_denom), 4)
        rmse_per_stem[stem_name] = round(float(np.sqrt(np.square(diff[s_mask, i]).sum() / s_denom)), 4)

    flat_pred = g_pred[mask]
    flat_true = g_true[mask]
    ss_res = float(np.sum(np.square(flat_true - flat_pred)))
    ss_tot = float(np.sum(np.square(flat_true - np.mean(flat_true))))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    if len(flat_pred) > 2 and np.std(flat_pred) > 1e-8 and np.std(flat_true) > 1e-8:
        p_r = float(pearsonr(flat_pred, flat_true)[0])
        s_rho = float(spearmanr(flat_pred, flat_true).statistic)
    else:
        p_r = 0.0
        s_rho = 0.0

    # Sample-level MAE for success threshold (< 1.0 dB, Section 8.6)
    sample_mae = abs_diff.sum(axis=1) / active_counts_per_sample
    success_rate_1db = float(100.0 * np.mean(sample_mae < 1.0))

    # 2. Spectral-level metrics (Section 8.2)
    band_abs_diff = np.abs(band8_pred_db - band8_true_db)
    band_mae_db = float(np.mean(band_abs_diff))
    worst_band_error_db = float(np.mean(np.max(band_abs_diff, axis=1)))
    band_mse_norm_per_sample = np.mean(np.square((band8_pred_db - band8_true_db) / 12.0), axis=1)
    band_mse_norm = float(np.mean(band_mse_norm_per_sample))
    success_rate_band_001 = float(100.0 * np.mean(band_mse_norm_per_sample < 0.01))

    # 64-bin Log-Spectral Distance (LSD in dB)
    lsd_per_sample = np.sqrt(np.mean(np.square(mel64_pred_db - mel64_true_db), axis=1))
    log_spectral_distance_db = float(np.mean(lsd_per_sample))

    # 3. Text / Instruction Adherence metrics (Section 8.4)
    # Resulting post-gain loudness L*_pred = L_true + g_pred -> resulting relative balance r_hat
    l_final_pred = (l_true + g_pred) * mask
    l_final_mean = l_final_pred.sum(axis=1, keepdims=True) / active_counts_per_sample[:, None]
    r_achieved = (l_final_pred - l_final_mean) * mask

    balance_mae_db = float(np.abs((r_achieved - r_true) * mask).sum() / total_active)

    # Direction intent categories: boost (> +0.75 dB), neutral ([-0.75, 0.75] dB), cut (< -0.75 dB)
    dir_true = np.where(r_true > 0.75, 1, np.where(r_true < -0.75, -1, 0))
    dir_pred = np.where(r_achieved > 0.75, 1, np.where(r_achieved < -0.75, -1, 0))
    intent_direction_acc = float(100.0 * (dir_true[mask] == dir_pred[mask]).mean())

    # Instruction adherence: top-boosted and most-attenuated stems match target (accounting for tied targets)
    top_match = []
    for idx in range(len(g_pred)):
        r_t = r_true[idx]
        r_a = r_achieved[idx]
        if np.max(np.abs(r_t)) < 0.5:
            # Neutral balance prompt: check that max deviation is < 1.5 dB
            top_match.append(bool(np.max(np.abs(r_a)) < 1.5))
        else:
            top_ok = float(r_t[int(np.argmax(r_a))]) >= float(np.max(r_t)) - 0.5
            bot_ok = float(r_t[int(np.argmin(r_a))]) <= float(np.min(r_t)) + 0.5
            err_ok = float(np.mean(np.abs(r_a - r_t))) < 1.0
            top_match.append(bool(top_ok and bot_ok and err_ok))
    instruction_adherence = float(100.0 * np.mean(top_match))

    return {
        "sample_count": int(len(g_pred)),
        "mae_db": round(mae_db, 4),
        "rmse_db": round(rmse_db, 4),
        "bias_db": round(bias_db, 4),
        "max_error_db": round(max_error_db, 4),
        "r2": round(r2, 4),
        "pearson_r": round(p_r, 4),
        "spearman_rho": round(s_rho, 4),
        "success_rate_1db": round(success_rate_1db, 2),
        "mae_per_stem_db": mae_per_stem,
        "rmse_per_stem_db": rmse_per_stem,
        "band_mae_db": round(band_mae_db, 4),
        "band_mse_norm": round(band_mse_norm, 6),
        "worst_band_error_db": round(worst_band_error_db, 4),
        "log_spectral_distance_db": round(log_spectral_distance_db, 4),
        "success_rate_band_001": round(success_rate_band_001, 2),
        "balance_mae_db": round(balance_mae_db, 4),
        "intent_direction_acc": round(intent_direction_acc, 2),
        "instruction_adherence": round(instruction_adherence, 2),
    }


@torch.no_grad()
def evaluate_model(
    model_or_fn: nn.Module | Callable[[dict[str, object]], torch.Tensor],
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Evaluate a PyTorch model or baseline callable across a DataLoader.

    Returns overall metrics, per-stem breakdowns, and per-song aggregations.
    """
    is_module = isinstance(model_or_fn, nn.Module)
    if is_module:
        model_or_fn.eval()
        model_or_fn.to(device)

    all_g_pred, all_g_true = [], []
    all_l_true, all_r_true = [], []
    all_masks = []
    all_b8_pred, all_b8_true = [], []
    all_m64_pred, all_m64_true = [], []
    all_songs: list[str] = []

    for batch in dataloader:
        audio_key = getattr(model_or_fn, "audio_input_key", "audio") if is_module else "audio"
        audio = batch[audio_key].to(device=device, dtype=torch.float32)
        prompts = batch["prompt"]
        active_mask = batch["active_mask"].to(device=device)

        if is_module:
            outputs = model_or_fn(audio, prompts, active_mask=active_mask, clamp_output=True)
            g_pred = outputs["gains_pred"].detach().cpu().numpy()
        else:
            g_pred = model_or_fn(batch).detach().cpu().numpy()

        g_true = batch["gains_db"].numpy()
        l_true = batch["loudness_db"].numpy()
        r_true = batch["balance_target"].numpy()
        mask_np = batch["active_mask"].numpy()
        mel_power_np = (
            batch["stem_mel_power"].numpy()
            if "stem_mel_power" in batch
            else batch["logmel_db"].numpy()
        )

        b8_pred, m64_pred = _compute_mix_8band_and_64mel_db(mel_power_np, g_pred, mask_np)
        b8_true, m64_true = _compute_mix_8band_and_64mel_db(mel_power_np, g_true, mask_np)

        all_g_pred.append(g_pred)
        all_g_true.append(g_true)
        all_l_true.append(l_true)
        all_r_true.append(r_true)
        all_masks.append(mask_np)
        all_b8_pred.append(b8_pred)
        all_b8_true.append(b8_true)
        all_m64_pred.append(m64_pred)
        all_m64_true.append(m64_true)
        all_songs.extend(batch["song"])

    g_pred_arr = np.concatenate(all_g_pred, axis=0)
    g_true_arr = np.concatenate(all_g_true, axis=0)
    l_true_arr = np.concatenate(all_l_true, axis=0)
    r_true_arr = np.concatenate(all_r_true, axis=0)
    mask_arr = np.concatenate(all_masks, axis=0)
    b8_pred_arr = np.concatenate(all_b8_pred, axis=0)
    b8_true_arr = np.concatenate(all_b8_true, axis=0)
    m64_pred_arr = np.concatenate(all_m64_pred, axis=0)
    m64_true_arr = np.concatenate(all_m64_true, axis=0)

    overall = compute_regression_and_spectral_metrics(
        gains_pred=g_pred_arr,
        gains_true=g_true_arr,
        loudness_true=l_true_arr,
        balance_true=r_true_arr,
        active_mask=mask_arr,
        band8_pred_db=b8_pred_arr,
        band8_true_db=b8_true_arr,
        mel64_pred_db=m64_pred_arr,
        mel64_true_db=m64_true_arr,
    )

    song_indices: dict[str, list[int]] = defaultdict(list)
    for idx, s_name in enumerate(all_songs):
        song_indices[s_name].append(idx)

    per_song: dict[str, dict[str, object]] = {}
    for s_name, idxs in sorted(song_indices.items()):
        sel = np.asarray(idxs, dtype=int)
        s_metrics = compute_regression_and_spectral_metrics(
            gains_pred=g_pred_arr[sel],
            gains_true=g_true_arr[sel],
            loudness_true=l_true_arr[sel],
            balance_true=r_true_arr[sel],
            active_mask=mask_arr[sel],
            band8_pred_db=b8_pred_arr[sel],
            band8_true_db=b8_true_arr[sel],
            mel64_pred_db=m64_pred_arr[sel],
            mel64_true_db=m64_true_arr[sel],
        )
        per_song[s_name] = {
            "sample_count": s_metrics["sample_count"],
            "mae_db": s_metrics["mae_db"],
            "rmse_db": s_metrics["rmse_db"],
            "band_mae_db": s_metrics["band_mae_db"],
            "success_rate_1db": s_metrics["success_rate_1db"],
            "instruction_adherence": s_metrics["instruction_adherence"],
        }

    return {
        **overall,
        "per_song": per_song,
    }


def make_baseline_predictor(name: str, seed: int = 2026) -> Callable[[dict[str, object]], torch.Tensor]:
    """Create a baseline predictor function (`no_op`, `random`, `dsp_equal_loudness`) per Section 8.7."""
    rng = np.random.default_rng(seed)

    def _predict(batch: dict[str, object]) -> torch.Tensor:
        logmel_db = batch["logmel_db"].numpy()  # (B, 4, 64, 800)
        mask = batch["active_mask"].numpy().astype(bool)
        bsz = len(logmel_db)

        if name == "no_op":
            return torch.zeros((bsz, 4), dtype=torch.float32)

        if name == "random":
            rand_g = rng.uniform(-6.0, 6.0, size=(bsz, 4)).astype(np.float32)
            rand_g[~mask] = 0.0
            return torch.from_numpy(rand_g)

        if name in ("dsp_equal_loudness", "rule_dsp_prompt"):
            # Estimate stem loudness from log-mel power without touching labels
            if "stem_mel_power" in batch:
                stem_power = batch["stem_mel_power"].numpy().mean(axis=-1)
            else:
                stem_power = np.power(10.0, np.clip(logmel_db, -100.0, 20.0) / 10.0).mean(axis=(2, 3))
            l_est = 10.0 * np.log10(np.maximum(stem_power, 1e-10))
            preds = np.zeros((bsz, 4), dtype=np.float32)
            prompts = batch["prompt"]
            for i in range(bsz):
                act = mask[i]
                if not np.any(act):
                    continue
                l_ref = float(np.mean(l_est[i, act]))
                if name == "rule_dsp_prompt":
                    r_est = parse_prompt_to_balance(str(prompts[i]))["balance_target"]
                    r_act = r_est[act] - np.mean(r_est[act])
                    preds[i, act] = np.clip(l_ref + r_act - l_est[i, act], GAIN_MIN_DB, GAIN_MAX_DB)
                else:
                    preds[i, act] = np.clip(l_ref - l_est[i, act], GAIN_MIN_DB, GAIN_MAX_DB)
            return torch.from_numpy(preds)

        raise ValueError(f"Unknown baseline predictor: {name}")

    return _predict
