"""Multimodal dataset builder, log-mel disk cache, and PyTorch Dataset for MUSDB18-HQ.

Implements Section 4.4, Section 4.5, Section 4.6, and Section 5.6 of dokumentasi.md:
- Extracts and caches `(4, 64, 800)` log-mel spectrograms per excerpt as compact `.npy` files.
- Generates leak-free multimodal samples pairing `(audio, prompt) -> (g*, L, r)`:
    * 80 train songs x 10 excerpts x 4 prompts = 3,200 train samples
    * 20 validation songs x 10 excerpts x 4 prompts = 800 validation samples
      (total 4,000 samples across the 100 MUSDB train songs)
    * 50 test songs paired with the 200 human-written Indonesian prompts (`test_human`)
      plus synthetic test prompts (`test_synthetic`).
- Lazy-loading `MusdbMultimodalDataset` that applies initial gain `g0_db` directly to
  the input log-mel spectrogram so the model predicts `g*` from `(audio, prompt)` alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from wadidaw.data.features import (
    ACTIVE_THRESHOLD_DB,
    EXCERPT_DURATION_SEC,
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    SAMPLE_RATE_RAW,
    apply_gain_to_logmel_db,
    compute_gain_targets,
    extract_logmel_stems,
    measure_loudness_db,
    normalize_logmel_for_model,
    resample_to_16k_mono,
)
from wadidaw.data.musdb import (
    MUSDB_ROOT,
    ROOT,
    SPLIT_MANIFEST_PATH,
    load_split_manifest,
    select_active_excerpts_for_song,
)
from wadidaw.data.prompts import (
    MIXING_PROFILES,
    STEM_NAMES,
    MixingProfile,
    generate_human_test_prompts,
    generate_paraphrases,
)


CACHE_ROOT = ROOT / "data" / "cache" / "logmel"
SAMPLES_ROOT = ROOT / "data" / "prompts"


def extract_and_cache_song_excerpts(
    song_member: dict[str, object],
    split: str,
    song_number: int,
    musdb_root: Path | str = MUSDB_ROOT,
    cache_root: Path | str = CACHE_ROOT,
    n_excerpts: int = 10,
    dtype: str = "float16",
    overwrite: bool = False,
) -> list[dict[str, object]]:
    """Extract and cache `(4, 64, 800)` log-mel `.npy` files for one song's excerpts."""
    mroot = Path(musdb_root)
    croot = Path(cache_root) / split
    croot.mkdir(parents=True, exist_ok=True)

    song_dir = mroot / str(song_member["relative_dir"])
    excerpts = select_active_excerpts_for_song(
        song_dir,
        n_excerpts=n_excerpts,
        duration_sec=EXCERPT_DURATION_SEC,
    )

    cached_excerpts: list[dict[str, object]] = []
    for ex in excerpts:
        ex_idx = int(ex["excerpt_index"])
        stem_id = f"{split}_{song_number:04d}_ex{ex_idx:02d}"
        npy_path = (croot / f"{stem_id}.npy").resolve()
        try:
            rel_npy_path = npy_path.relative_to(ROOT).as_posix()
        except ValueError:
            rel_npy_path = npy_path.as_posix()

        start_frame = int(ex["start_frame_44k"])
        frames_44k = int(ex["frames_44k"])

        if not npy_path.exists() or overwrite:
            stems_16k = []
            for stem_name in STEM_NAMES:
                wav_path = song_dir / f"{stem_name}.wav"
                audio_44k, sr = sf.read(
                    wav_path,
                    start=start_frame,
                    frames=frames_44k,
                    dtype="float32",
                )
                mono_16k = resample_to_16k_mono(audio_44k, orig_sr=sr)
                stems_16k.append(mono_16k)

            stems_16k_arr = np.stack(stems_16k, axis=0)
            loudness_16k = measure_loudness_db(stems_16k_arr)
            logmel_db = extract_logmel_stems(stems_16k_arr)
            np.save(npy_path, logmel_db.astype(np.dtype(dtype)))
            loudness_raw = [round(float(x), 4) for x in loudness_16k]
        else:
            loudness_raw = list(ex["loudness_raw_db"])

        active_mask = [bool(float(x) > ACTIVE_THRESHOLD_DB) for x in loudness_raw]
        cached_excerpts.append({
            "excerpt_id": stem_id,
            "song": str(song_member["song"]),
            "artist": str(song_member["artist"]),
            "split": split,
            "excerpt_index": ex_idx,
            "excerpt_start_sec": float(ex["excerpt_start_sec"]),
            "duration_sec": float(ex["duration_sec"]),
            "audio_path": rel_npy_path,
            "loudness_raw_db": loudness_raw,
            "active_mask": active_mask,
            "all_stems_active": bool(all(active_mask)),
        })

    return cached_excerpts


def _choose_valid_profile_and_g0(
    loudness_raw_db: Sequence[float],
    preferred_profile_idx: int,
    rng: np.random.Generator,
    perturb_g0: bool = True,
) -> tuple[MixingProfile, dict[str, object]]:
    """Sample `g0 ~ Uniform(-3, 3)` and a `MixingProfile` such that `g*` lies in `[-18, 12]` dB."""
    n_profiles = len(MIXING_PROFILES)
    for attempt in range(n_profiles * 2):
        prof_idx = (preferred_profile_idx + attempt) % n_profiles
        profile = MIXING_PROFILES[prof_idx]
        if perturb_g0 and attempt < n_profiles:
            g0 = rng.uniform(-3.0, 3.0, size=4).astype(np.float32)
        else:
            g0 = np.zeros(4, dtype=np.float32)

        targets = compute_gain_targets(
            loudness_raw_db=loudness_raw_db,
            balance_target_db=profile.balance_target,
            g0_db=g0,
            clip_bounds=(GAIN_MIN_DB, GAIN_MAX_DB),
        )
        if targets["within_bounds"]:
            return profile, targets

    # Fallback to neutral balance profile with g0 = 0
    neutral_profile = MIXING_PROFILES[10]  # balanced_reference_v1
    targets = compute_gain_targets(
        loudness_raw_db=loudness_raw_db,
        balance_target_db=neutral_profile.balance_target,
        g0_db=np.zeros(4, dtype=np.float32),
        clip_bounds=(GAIN_MIN_DB, GAIN_MAX_DB),
    )
    return neutral_profile, targets


def build_split_samples(
    cached_excerpts: Sequence[dict[str, object]],
    split: str,
    n_prompts_per_excerpt: int = 4,
    seed: int = 2026,
    perturb_g0: bool = True,
) -> list[dict[str, object]]:
    """Build multimodal sample records conforming to Section 4.6 schema.

    Each excerpt is paired with `n_prompts_per_excerpt` distinct mixing profiles
    and independent `g0` perturbations so that the same audio excerpt requires
    different target gains depending on the text prompt and input level offset.
    """
    rng = np.random.default_rng(seed)
    samples: list[dict[str, object]] = []

    for ex_pos, ex in enumerate(cached_excerpts):
        for p_offset in range(n_prompts_per_excerpt):
            pref_idx = (ex_pos * n_prompts_per_excerpt + p_offset) % len(MIXING_PROFILES)
            profile, targets = _choose_valid_profile_and_g0(
                loudness_raw_db=ex["loudness_raw_db"],
                preferred_profile_idx=pref_idx,
                rng=rng,
                perturb_g0=perturb_g0,
            )
            paraphrases = generate_paraphrases(
                profile=profile,
                split=split,
                n_paraphrases=max(n_prompts_per_excerpt, 4),
                rng=rng,
            )
            p_info = paraphrases[(ex_pos + p_offset) % len(paraphrases)]

            g0_list = [round(float(x), 4) for x in targets["g0_db"]]
            loudness_list = [round(float(x), 4) for x in targets["loudness_db"]]
            balance_list = [round(float(x), 4) for x in targets["balance_target"]]
            gains_list = [round(float(x), 4) for x in targets["gains_db"]]
            active_list = [bool(x) for x in targets["active_mask"]]

            p_idx = p_offset + 1
            sample_id = f"{ex['excerpt_id']}_p{p_idx:02d}"
            samples.append({
                "id": sample_id,
                "song": ex["song"],
                "artist": ex["artist"],
                "split": split,
                "excerpt_start_sec": ex["excerpt_start_sec"],
                "duration_sec": ex["duration_sec"],
                "audio_path": ex["audio_path"],
                "g0_db": g0_list,
                "loudness_raw_db": ex["loudness_raw_db"],
                "loudness_db": loudness_list,
                "balance_target": balance_list,
                "gains_db": gains_list,
                "active_mask": active_list,
                "prompt": p_info["prompt"],
                "prompt_template": p_info["prompt_template"],
            })

    return samples


def build_human_test_samples(
    cached_test_excerpts: Sequence[dict[str, object]],
    seed: int = 2026,
    perturb_g0: bool = True,
) -> list[dict[str, object]]:
    """Pair the 200 human-written Indonesian prompts with excerpts across the 50 test songs."""
    if not cached_test_excerpts:
        raise ValueError("cached_test_excerpts must not be empty.")

    rng = np.random.default_rng(seed + 999)
    human_prompts = generate_human_test_prompts(count=200, seed=seed)
    samples: list[dict[str, object]] = []

    for idx, h_item in enumerate(human_prompts):
        ex = cached_test_excerpts[idx % len(cached_test_excerpts)]
        g0 = (
            rng.uniform(-3.0, 3.0, size=4).astype(np.float32)
            if perturb_g0
            else np.zeros(4, dtype=np.float32)
        )
        targets = compute_gain_targets(
            loudness_raw_db=ex["loudness_raw_db"],
            balance_target_db=h_item["balance_target"],
            g0_db=g0,
            clip_bounds=(GAIN_MIN_DB, GAIN_MAX_DB),
        )
        if not targets["within_bounds"]:
            g0 = np.zeros(4, dtype=np.float32)
            targets = compute_gain_targets(
                loudness_raw_db=ex["loudness_raw_db"],
                balance_target_db=h_item["balance_target"],
                g0_db=g0,
                clip_bounds=(GAIN_MIN_DB, GAIN_MAX_DB),
            )

        samples.append({
            "id": f"{ex['excerpt_id']}_{h_item['id']}",
            "song": ex["song"],
            "artist": ex["artist"],
            "split": "test_human",
            "excerpt_start_sec": ex["excerpt_start_sec"],
            "duration_sec": ex["duration_sec"],
            "audio_path": ex["audio_path"],
            "g0_db": [round(float(x), 4) for x in targets["g0_db"]],
            "loudness_raw_db": ex["loudness_raw_db"],
            "loudness_db": [round(float(x), 4) for x in targets["loudness_db"]],
            "balance_target": [round(float(x), 4) for x in targets["balance_target"]],
            "gains_db": [round(float(x), 4) for x in targets["gains_db"]],
            "active_mask": [bool(x) for x in targets["active_mask"]],
            "prompt": h_item["prompt"],
            "prompt_template": h_item["profile_hint"],
        })

    return samples


class MusdbMultimodalDataset(Dataset):
    """PyTorch Dataset yielding `(audio, prompt)` inputs and `(gains_db, loudness_db, balance_target)` labels.

    In accordance with anti-leakage rules (Section 5.7) and design decision A3:
    - `g0_db` is applied directly to the cached raw log-mel spectrogram in dBFS, so
      `item["audio"]` reflects the actual input stem levels `L = L_raw + g0`.
    - The model only needs `(item["audio"], item["prompt"])` at both training and inference.
    """

    def __init__(
        self,
        samples_or_path: Sequence[dict[str, object]] | Path | str,
        root_dir: Path | str = ROOT,
        cache_in_memory: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.cache_in_memory = cache_in_memory
        self._npy_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if isinstance(samples_or_path, (Path, str)):
            path = Path(samples_or_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset manifest not found: {path}")
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict) and "samples" in loaded:
                self.samples = list(loaded["samples"])
            elif isinstance(loaded, list):
                self.samples = loaded
            else:
                raise ValueError(f"Unrecognized dataset JSON format in {path}.")
        else:
            self.samples = list(samples_or_path)

        if not self.samples:
            raise ValueError("MusdbMultimodalDataset initialized with 0 samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.samples[index]
        rel_path = str(record["audio_path"])
        if self.cache_in_memory and rel_path in self._npy_cache:
            raw_arr, raw_mel_power = self._npy_cache[rel_path]
        else:
            npy_path = self.root_dir / rel_path
            raw_arr = np.load(npy_path)
            raw_mel_power = np.power(
                10.0,
                np.clip(raw_arr.astype(np.float32), -100.0, 20.0) / 10.0,
            ).mean(axis=-1).astype(np.float32)
            if self.cache_in_memory:
                self._npy_cache[rel_path] = (raw_arr, raw_mel_power)

        logmel_raw_db = raw_arr.astype(np.float32)
        if logmel_raw_db.shape != (4, 64, 800):
            raise ValueError(
                f"Expected cached log-mel shape (4, 64, 800), got {logmel_raw_db.shape} at {rel_path}"
            )

        g0_db = np.asarray(record["g0_db"], dtype=np.float32)
        active_mask = np.asarray(
            record.get("active_mask", [True, True, True, True]),
            dtype=bool,
        )
        # Apply g0 perturbation only to active stems
        effective_g0 = np.where(active_mask, g0_db, 0.0).astype(np.float32)
        logmel_perturbed_db = apply_gain_to_logmel_db(logmel_raw_db, effective_g0)
        audio_norm = (logmel_perturbed_db + 30.0) * 0.02
        stem_mel_power = raw_mel_power * np.power(10.0, effective_g0 / 10.0)[:, None]

        return {
            "id": str(record["id"]),
            "song": str(record["song"]),
            "split": str(record["split"]),
            "audio": torch.from_numpy(audio_norm.astype(np.float32)),
            "logmel_db": torch.from_numpy(logmel_perturbed_db),
            "stem_mel_power": torch.from_numpy(stem_mel_power.astype(np.float32)),
            "prompt": str(record["prompt"]),
            "prompt_template": str(record["prompt_template"]),
            "g0_db": torch.from_numpy(g0_db),
            "loudness_db": torch.tensor(record["loudness_db"], dtype=torch.float32),
            "balance_target": torch.tensor(record["balance_target"], dtype=torch.float32),
            "gains_db": torch.tensor(record["gains_db"], dtype=torch.float32),
            "active_mask": torch.from_numpy(active_mask),
        }
