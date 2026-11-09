"""Audio preprocessing, loudness measurement, log-mel extraction, and gain target math.

Implements Section 4.3, Section 5.2, and Section 8.2 of dokumentasi.md:
- Resampling 44.1 kHz stereo stems to 16 kHz mono (exact polyphase ratio 160/441).
- 8-second excerpt framing (128,000 samples at 16 kHz -> 800 mel frames).
- 64-bin log-mel spectrogram extraction in dBFS (preserving absolute stem level).
- Loudness measurement (RMS dBFS) with active-stem silence gating (-45 dBFS).
- Target gain computation: L_ref = mean(L), g*_i = L_ref + r_i - L_i.
"""

from __future__ import annotations

from math import gcd
from typing import Sequence

import numpy as np
from scipy.signal import resample_poly


SAMPLE_RATE_RAW: int = 44100
SAMPLE_RATE_MODEL: int = 16000
EXCERPT_DURATION_SEC: float = 8.0
EXCERPT_FRAMES_16K: int = int(round(EXCERPT_DURATION_SEC * SAMPLE_RATE_MODEL))  # 128000
EXCERPT_FRAMES_44K: int = int(round(EXCERPT_DURATION_SEC * SAMPLE_RATE_RAW))    # 352800

N_FFT: int = 1024
HOP_LENGTH: int = 160
N_MELS: int = 64
MEL_FRAMES: int = 800
FMIN_HZ: float = 20.0
FMAX_HZ: float = 8000.0

ACTIVE_THRESHOLD_DB: float = -45.0
GAIN_MIN_DB: float = -18.0
GAIN_MAX_DB: float = 12.0
LOGMEL_MIN_DB: float = -80.0
LOGMEL_MAX_DB: float = 20.0

# 8 spectral evaluation bands (Hz) for Section 8.2 spectral metrics.
BAND_EDGES_HZ: tuple[int, ...] = (0, 80, 160, 320, 640, 1250, 2500, 5000, 8001)


def _hz_to_mel_slaney(frequencies_hz: np.ndarray | float) -> np.ndarray:
    """Convert Hz to Mel using Slaney's auditory toolbox formula (matches librosa/torchaudio)."""
    freq = np.asanyarray(frequencies_hz, dtype=np.float64)
    f_sp = 200.0 / 3.0
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_mask = freq >= min_log_hz
    if np.ndim(mels) == 0:
        if log_mask:
            mels = np.asarray(min_log_mel + np.log(freq / min_log_hz) / logstep)
        return mels
    mels[log_mask] = min_log_mel + np.log(freq[log_mask] / min_log_hz) / logstep
    return mels


def _mel_to_hz_slaney(mels: np.ndarray | float) -> np.ndarray:
    """Convert Mel to Hz using Slaney's formula."""
    mel_arr = np.asanyarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3.0
    freqs = f_sp * mel_arr
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_mask = mel_arr >= min_log_mel
    if np.ndim(freqs) == 0:
        if log_mask:
            freqs = np.asarray(min_log_hz * np.exp(logstep * (mel_arr - min_log_mel)))
        return freqs
    freqs[log_mask] = min_log_hz * np.exp(logstep * (mel_arr[log_mask] - min_log_mel))
    return freqs


def build_mel_filterbank(
    sample_rate: int = SAMPLE_RATE_MODEL,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = FMIN_HZ,
    fmax: float = FMAX_HZ,
) -> np.ndarray:
    """Construct a Slaney-normalized `(n_mels, n_fft // 2 + 1)` mel filterbank matrix."""
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mel_min = float(_hz_to_mel_slaney(fmin))
    mel_max = float(_hz_to_mel_slaney(fmax))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz_slaney(mel_points)

    ramps = np.subtract.outer(hz_points, fft_freqs)
    fdiff = np.diff(hz_points)

    weights = np.zeros((n_mels, len(fft_freqs)), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney area normalization
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


_MEL_BASIS_16K: np.ndarray = build_mel_filterbank()
_HANN_WINDOW_1024: np.ndarray = np.hanning(N_FFT + 1)[:-1].astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert `(frames,)` or `(frames, channels)` audio to a 1D float32 mono waveform."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] in (1, 2):
        return arr.mean(axis=1, dtype=np.float32)
    raise ValueError(f"Expected 1D mono or 2D (frames, 1|2) audio, got shape {arr.shape}.")


def resample_to_16k_mono(
    audio: np.ndarray,
    orig_sr: int = SAMPLE_RATE_RAW,
    target_sr: int = SAMPLE_RATE_MODEL,
    target_frames: int | None = EXCERPT_FRAMES_16K,
) -> np.ndarray:
    """Downmix stereo/mono audio and resample to 16 kHz mono using polyphase filtering."""
    mono = to_mono(audio)
    if orig_sr == target_sr:
        resampled = mono
    else:
        divisor = gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // divisor
        down = int(orig_sr) // divisor
        resampled = resample_poly(mono, up=up, down=down).astype(np.float32)

    if target_frames is not None:
        if len(resampled) >= target_frames:
            resampled = resampled[:target_frames]
        else:
            resampled = np.pad(resampled, (0, target_frames - len(resampled)))
    return resampled


def measure_loudness_db(stems_mono: np.ndarray | Sequence[np.ndarray], floor_db: float = -100.0) -> np.ndarray:
    """Measure per-stem RMS loudness in dBFS for 4 stems.

    Args:
        stems_mono: `(4, frames)` array or sequence of 4 waveforms.
        floor_db: minimum loudness floor in dBFS for digital silence.

    Returns:
        `(4,)` float32 array of loudness values in dBFS.
    """
    arr = np.asarray(stems_mono, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 4:
        raise ValueError(f"Expected (4, frames) stems array, got shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError("Stem audio contains non-finite values.")

    mean_sq = np.mean(np.square(arr), axis=1)
    min_power = 10.0 ** (floor_db / 10.0)
    loudness_db = 10.0 * np.log10(np.maximum(mean_sq, min_power))
    return loudness_db.astype(np.float32)


def active_stem_mask(loudness_db: Sequence[float] | np.ndarray, threshold_db: float = ACTIVE_THRESHOLD_DB) -> np.ndarray:
    """Return a boolean `(4,)` mask indicating which stems exceed `threshold_db`."""
    arr = np.asarray(loudness_db, dtype=np.float32)
    if arr.shape != (4,):
        raise ValueError(f"Expected 4 loudness values, got shape {arr.shape}.")
    return arr > float(threshold_db)


def extract_logmel_stem(
    mono_16k: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_Frames: int = MEL_FRAMES,
    min_db: float = -100.0,
) -> np.ndarray:
    """Compute a `(64, 800)` log-mel spectrogram in dBFS from a 16 kHz mono stem.

    Uses reflect padding (`center=True`) and normalizes FFT magnitude by window sum
    so that dB values stay calibrated to absolute waveform amplitude (dBFS).
    """
    x = np.asarray(mono_16k, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {x.shape}.")

    pad = n_fft // 2
    padded = np.pad(x, (pad, pad), mode="reflect")

    # Frame the signal with hop_length=160
    available_frames = 1 + (len(padded) - n_fft) // hop_length
    frames_count = min(available_frames, n_Frames)
    shape = (frames_count, n_fft)
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    windowed = frames * _HANN_WINDOW_1024[None, :]
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2) / float(n_fft)

    # Project onto 64 mel bins: (64, freq_bins) @ (freq_bins, time) -> (64, time)
    mel_power = _MEL_BASIS_16K @ power.T.astype(np.float32)
    min_power = 10.0 ** (min_db / 10.0)
    log_mel_db = 10.0 * np.log10(np.maximum(mel_power, min_power))

    if log_mel_db.shape[1] < n_Frames:
        log_mel_db = np.pad(
            log_mel_db,
            ((0, 0), (0, n_Frames - log_mel_db.shape[1])),
            mode="edge",
        )
    return log_mel_db[:, :n_Frames].astype(np.float32)


def extract_logmel_stems(stems_16k: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    """Compute `(4, 64, 800)` log-mel spectrogram in dBFS for all 4 stems."""
    arr = np.asarray(stems_16k, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 4:
        raise ValueError(f"Expected (4, frames) stems at 16 kHz, got shape {arr.shape}.")
    return np.stack([extract_logmel_stem(arr[i]) for i in range(4)], axis=0).astype(np.float32)


def apply_gain_to_logmel_db(
    logmel_db: np.ndarray,
    gains_db: Sequence[float] | np.ndarray,
    min_db: float = LOGMEL_MIN_DB,
    max_db: float = LOGMEL_MAX_DB,
) -> np.ndarray:
    """Apply per-stem gain `gains_db` (dB) directly in the log-mel dB domain.

    Because STFT and Mel projection are linear, scaling stem `i` by `10**(g_i / 20)`
    shifts its power log-mel spectrogram by `+g_i` dB on all active bins.
    """
    mel = np.asarray(logmel_db, dtype=np.float32)
    g = np.asarray(gains_db, dtype=np.float32)
    if mel.shape[0] != 4 or g.shape != (4,):
        raise ValueError(f"Shape mismatch: logmel {mel.shape}, gains {g.shape}.")
    shifted = mel + g[:, None, None]
    return np.clip(shifted, min_db, max_db).astype(np.float32)


def normalize_logmel_for_model(
    logmel_db: np.ndarray,
    min_db: float = LOGMEL_MIN_DB,
    max_db: float = LOGMEL_MAX_DB,
) -> np.ndarray:
    """Scale dBFS log-mel from `[min_db, max_db]` to `[-1.0, 1.0]` using fixed global constants.

    IMPORTANT: Never normalize per-stem or per-clip mean/std, because doing so
    erases the absolute stem loudness `L_i` needed for gain staging.
    """
    clipped = np.clip(np.asarray(logmel_db, dtype=np.float32), min_db, max_db)
    mid = 0.5 * (max_db + min_db)      # -30.0 dB
    half_span = 0.5 * (max_db - min_db)  # 50.0 dB
    return ((clipped - mid) / half_span).astype(np.float32)


def compute_gain_targets(
    loudness_raw_db: Sequence[float] | np.ndarray,
    balance_target_db: Sequence[float] | np.ndarray,
    g0_db: Sequence[float] | np.ndarray | None = None,
    active_threshold_db: float = ACTIVE_THRESHOLD_DB,
    clip_bounds: tuple[float, float] = (GAIN_MIN_DB, GAIN_MAX_DB),
) -> dict[str, object]:
    """Compute perturbed stem loudness `L` and target gain adjustment `g*` (dB).

    Following user design decision A2 & A3 and Section 5.2 of dokumentasi.md:
    - Input stems arrive with initial gain offset `g0_db` applied, so their observed
      input loudness is `L_i = L_raw_i + g0_i` (for active stems).
    - Reference loudness `L_ref = mean(L_active)`.
    - The target fader gain to apply to the input stems to reach `L_ref + r_i` is:
          g*_i = L_ref + r_i - L_i
      which brings final stem loudness to `L*_i = L_i + g*_i = L_ref + r_i`.
    - Stems below `active_threshold_db` are masked (`g*_i = 0.0` dB) so silent stems
      neither skew `L_ref` nor get boosted by +12 dB.
    """
    l_raw = np.asarray(loudness_raw_db, dtype=np.float64)
    r = np.asarray(balance_target_db, dtype=np.float64)
    g0 = np.zeros(4, dtype=np.float64) if g0_db is None else np.asarray(g0_db, dtype=np.float64)

    if l_raw.shape != (4,) or r.shape != (4,) or g0.shape != (4,):
        raise ValueError("Expected 4-element vectors for loudness_raw_db, balance_target_db, and g0_db.")
    if not (np.isfinite(l_raw).all() and np.isfinite(r).all() and np.isfinite(g0).all()):
        raise ValueError("Loudness, balance_target, and g0 must be finite.")

    active = l_raw > float(active_threshold_db)
    l_perturbed = np.where(active, l_raw + g0, l_raw)

    if not np.any(active):
        return {
            "g0_db": g0.astype(np.float32),
            "loudness_db": l_perturbed.astype(np.float32),
            "l_ref_db": float(np.mean(l_perturbed)),
            "balance_target": np.zeros(4, dtype=np.float32),
            "gains_unclipped_db": np.zeros(4, dtype=np.float32),
            "gains_db": np.zeros(4, dtype=np.float32),
            "active_mask": active,
            "within_bounds": True,
        }

    # Re-center r across active stems so sum_{active}(r_i) = 0
    r_active_centered = np.zeros(4, dtype=np.float64)
    r_active_centered[active] = r[active] - np.mean(r[active])

    l_ref = float(np.mean(l_perturbed[active]))
    g_unclipped = np.zeros(4, dtype=np.float64)
    g_unclipped[active] = l_ref + r_active_centered[active] - l_perturbed[active]

    low, high = float(clip_bounds[0]), float(clip_bounds[1])
    within_bounds = bool(np.all((g_unclipped[active] >= low) & (g_unclipped[active] <= high)))
    g_clipped = np.clip(g_unclipped, low, high)
    g_clipped[~active] = 0.0

    return {
        "g0_db": g0.astype(np.float32),
        "loudness_db": l_perturbed.astype(np.float32),
        "l_ref_db": l_ref,
        "balance_target": r_active_centered.astype(np.float32),
        "gains_unclipped_db": g_unclipped.astype(np.float32),
        "gains_db": g_clipped.astype(np.float32),
        "active_mask": active,
        "within_bounds": within_bounds,
    }


def compute_band_grams_16k(stems_16k: np.ndarray, sample_rate: int = SAMPLE_RATE_MODEL) -> np.ndarray:
    """Compute `(8, 4, 4)` cross-stem spectral Gram matrices across 8 frequency bands.

    Enables fast, exact calculation of 8-band mix energy for any stem gain vector `g`
    during evaluation (Section 8.2 spectral metrics).
    """
    arr = np.asarray(stems_16k, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 4:
        raise ValueError(f"Expected (4, frames) stems, got {arr.shape}.")
    frames = arr.shape[1]
    spectrum = np.fft.rfft(arr, axis=1)
    frequencies = np.fft.rfftfreq(frames, d=1.0 / sample_rate)

    grams = []
    for low, high in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        block = spectrum[:, mask]
        gram = np.real(block @ block.conj().T) / float(frames ** 2)
        grams.append(gram)
    return np.stack(grams, axis=0).astype(np.float64)


def band_energies_from_grams(band_grams: np.ndarray, gains_db: Sequence[float] | np.ndarray) -> np.ndarray:
    """Evaluate 8-band mix spectral energy (dBFS) from cached `(8, 4, 4)` Gram matrices."""
    amplitudes = np.power(10.0, np.asarray(gains_db, dtype=np.float64) / 20.0)
    power = np.einsum("i,bij,j->b", amplitudes, band_grams, amplitudes, optimize=True)
    db = 10.0 * np.log10(np.maximum(power, 1e-12))
    return np.clip(db, -120.0, 20.0).astype(np.float32)
