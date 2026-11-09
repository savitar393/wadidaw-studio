"""End-to-end multimodal gain staging inference pipeline (16 kHz model -> 44.1 kHz stereo mix).

Implements Section 3.1, Section 7.2, Section 7.3, and Section 9.2 of dokumentasi.md:
- Accepts 4 stereo/mono audio stems (`vocals`, `drums`, `bass`, `other`) + Indonesian text prompt.
- Extracts 16 kHz mono log-mel features `(1, 4, 64, 800)` and active stem mask (`> -45 dBFS`).
- Runs forward pass on a trained checkpoint (`M1`–`M5`) with automatic CPU fallback
  (or falls back to the structured DSP + prompt parser if no checkpoint file is provided).
- Renders the final stereo mix at the original sample rate (44.1 kHz) via `render_gain_staged_mix`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf
import torch

from wadidaw.data.features import (
    ACTIVE_THRESHOLD_DB,
    EXCERPT_DURATION_SEC,
    SAMPLE_RATE_RAW,
    active_stem_mask,
    compute_gain_targets,
    extract_logmel_stems,
    measure_loudness_db,
    normalize_logmel_for_model,
    resample_to_16k_mono,
)
from wadidaw.data.musdb import ROOT, render_gain_staged_mix
from wadidaw.data.prompts import STEM_NAMES, parse_prompt_to_balance
from wadidaw.models.fusion import build_model


CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"


def load_trained_model(
    checkpoint_path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Load a trained multimodal gain staging checkpoint onto `device` (CPU-safe)."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)
    model_name = str(payload.get("model_name", "m3"))
    text_backend = str(payload.get("text_backend", "subword"))

    model = build_model(model_name, text_backend=text_backend).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def load_stem_waveforms(
    stem_sources: Mapping[str, Path | str | np.ndarray],
    max_duration_sec: float | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """Load and frame-align 4 stems (`vocals`, `drums`, `bass`, `other`) as `(frames, 2)` float32 arrays."""
    loaded: dict[str, np.ndarray] = {}
    sample_rate: int | None = None

    for name in STEM_NAMES:
        if name not in stem_sources:
            raise ValueError(f"Missing required stem: '{name}'.")
        src = stem_sources[name]
        if isinstance(src, np.ndarray):
            arr = np.asarray(src, dtype=np.float32)
            sr = sample_rate or SAMPLE_RATE_RAW
        else:
            arr, sr = sf.read(src, dtype="float32", always_2d=True)

        if arr.ndim == 1:
            arr = np.column_stack((arr, arr))
        elif arr.ndim == 2 and arr.shape[1] == 1:
            arr = np.repeat(arr, 2, axis=1)
        elif arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Stem '{name}' must have 1 or 2 channels, got shape {arr.shape}.")

        if sample_rate is None:
            sample_rate = int(sr)
        elif int(sr) != sample_rate:
            raise ValueError(f"Sample rate mismatch on '{name}': expected {sample_rate} Hz, got {sr} Hz.")

        loaded[name] = arr.astype(np.float32)

    assert sample_rate is not None
    min_frames = min(len(a) for a in loaded.values())
    if max_duration_sec is not None and max_duration_sec > 0:
        min_frames = min(min_frames, int(round(max_duration_sec * sample_rate)))

    if min_frames < 2:
        raise ValueError("Stem audio is too short (< 2 frames).")

    for name in STEM_NAMES:
        loaded[name] = loaded[name][:min_frames]

    return loaded, sample_rate


@torch.no_grad()
def predict_and_render_mix(
    stem_sources: Mapping[str, Path | str | np.ndarray],
    prompt: str,
    checkpoint_path: Path | str | None = None,
    device: torch.device | str = "cpu",
    analysis_start_sec: float = 0.0,
    max_render_duration_sec: float | None = None,
) -> dict[str, object]:
    """Run multimodal gain staging inference and render the stereo mix.

    Returns a dictionary matching Section 3.1 of dokumentasi.md:
        - `gains_db`: `{vocals, drums, bass, other}` in dB
        - `loudness_measured_db`: measured per-stem loudness in dBFS
        - `loudness_pred_db`: model-predicted loudness (if M3)
        - `balance_pred_db`: model-predicted relative balance `r` (if M3)
        - `active_mask`: `{vocals, drums, bass, other}` booleans
        - `mix`: `(frames, 2)` stereo waveform at original sample rate
        - `sample_rate`: sample rate of `mix`
        - `metrics`: `{peak_linear, peak_dbfs, clipping_rate}`
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt teks Bahasa Indonesia tidak boleh kosong.")

    stems_stereo, sr = load_stem_waveforms(stem_sources, max_duration_sec=max_render_duration_sec)
    total_frames = len(stems_stereo["vocals"])
    win_frames = int(round(EXCERPT_DURATION_SEC * sr))
    start_frame = min(int(round(analysis_start_sec * sr)), max(0, total_frames - win_frames))

    stems_16k = []
    for name in STEM_NAMES:
        excerpt_slice = stems_stereo[name][start_frame : start_frame + win_frames]
        mono_16k = resample_to_16k_mono(excerpt_slice, orig_sr=sr)
        stems_16k.append(mono_16k)

    stems_16k_arr = np.stack(stems_16k, axis=0)
    loudness_dsp = measure_loudness_db(stems_16k_arr)
    mask_np = active_stem_mask(loudness_dsp, threshold_db=ACTIVE_THRESHOLD_DB)
    logmel_db = extract_logmel_stems(stems_16k_arr)
    audio_norm = normalize_logmel_for_model(logmel_db)

    parsed_prompt = parse_prompt_to_balance(prompt)
    loudness_pred_dict = None
    balance_pred_dict = None
    model_used = "dsp_prompt_fallback"

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        model, payload = load_trained_model(checkpoint_path, device=device)
        model_used = str(payload.get("model_name", "m3"))
        audio_tensor = torch.from_numpy(audio_norm).unsqueeze(0).to(device=device, dtype=torch.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(device=device)

        outputs = model(audio_tensor, [prompt], active_mask=mask_tensor, clamp_output=True)
        gains_vec = outputs["gains_pred"].squeeze(0).cpu().numpy()

        if "loudness_pred" in outputs:
            l_p = outputs["loudness_pred"].squeeze(0).cpu().numpy()
            loudness_pred_dict = {name: round(float(l_p[i]), 2) for i, name in enumerate(STEM_NAMES)}
        if "balance_pred" in outputs:
            r_p = outputs["balance_pred"].squeeze(0).cpu().numpy()
            balance_pred_dict = {name: round(float(r_p[i]), 2) for i, name in enumerate(STEM_NAMES)}
    else:
        targets = compute_gain_targets(
            loudness_raw_db=loudness_dsp,
            balance_target_db=parsed_prompt["balance_target"],
            g0_db=np.zeros(4, dtype=np.float32),
        )
        gains_vec = targets["gains_db"]
        balance_pred_dict = {
            name: round(float(targets["balance_target"][i]), 2) for i, name in enumerate(STEM_NAMES)
        }

    render_result = render_gain_staged_mix(
        stems_stereo=stems_stereo,
        gains_db=gains_vec,
        active_mask=mask_np,
    )

    return {
        "model_used": model_used,
        "prompt": prompt,
        "gains_db": render_result["effective_gains_db"],
        "loudness_measured_db": {name: round(float(loudness_dsp[i]), 2) for i, name in enumerate(STEM_NAMES)},
        "loudness_pred_db": loudness_pred_dict,
        "balance_pred_db": balance_pred_dict,
        "active_mask": {name: bool(mask_np[i]) for i, name in enumerate(STEM_NAMES)},
        "mix": render_result["mix"],
        "sample_rate": sr,
        "metrics": render_result["metrics"],
    }
