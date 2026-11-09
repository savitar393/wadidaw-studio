"""MUSDB18-HQ dataset discovery, song-level split manifest, excerpt selection, and rendering.

Implements Section 4.1, Section 4.2, and Section 7.3 of dokumentasi.md:
- Validates all 100 train tracks and 50 test tracks in `data/raw/musdb18hq/`.
- Builds `data/splits/musdb18hq_v1.json` with 80 train / 20 validation / 50 test songs,
  enforcing both song-level and artist-level disjointness between train and validation.
- Selects active 8-second excerpts where stems are active (> -45 dBFS), falling back
  cleanly with an explicit `active_mask` when a song lacks a stem (e.g., bassless acoustic).
- Renders 44.1 kHz stereo mixes via `wadidaw.stereo_mixer.render_stereo`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch

from wadidaw.data.features import (
    ACTIVE_THRESHOLD_DB,
    EXCERPT_DURATION_SEC,
    SAMPLE_RATE_RAW,
)
from wadidaw.data.prompts import STEM_NAMES
from wadidaw.stereo_mixer import render_stereo


ROOT = Path(__file__).resolve().parents[2]
MUSDB_ROOT = ROOT / "data" / "raw" / "musdb18hq"
SPLIT_MANIFEST_PATH = ROOT / "data" / "splits" / "musdb18hq_v1.json"


def parse_artist_name(song_name: str) -> str:
    """Extract the artist prefix from a `'Artist - Title'` folder name."""
    if " - " in song_name:
        return song_name.split(" - ", 1)[0].strip()
    return song_name.strip()


def discover_musdb_tracks(musdb_root: Path | str = MUSDB_ROOT) -> dict[str, list[dict[str, object]]]:
    """Scan and validate `train/` (100 songs) and `test/` (50 songs) under `musdb_root`."""
    root = Path(musdb_root)
    if not root.exists():
        raise FileNotFoundError(f"MUSDB18-HQ directory not found: {root}")

    discovered: dict[str, list[dict[str, object]]] = {}
    for subset, expected_count in (("train", 100), ("test", 50)):
        subset_dir = root / subset
        if not subset_dir.is_dir():
            raise FileNotFoundError(f"Missing MUSDB18-HQ subset directory: {subset_dir}")

        song_dirs = sorted(p for p in subset_dir.iterdir() if p.is_dir())
        if len(song_dirs) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} songs in {subset_dir}, found {len(song_dirs)}."
            )

        records: list[dict[str, object]] = []
        for index, song_dir in enumerate(song_dirs):
            required_files = (*STEM_NAMES, "mixture")
            infos = []
            for name in required_files:
                wav_path = song_dir / f"{name}.wav"
                if not wav_path.is_file():
                    raise FileNotFoundError(f"Missing stem file: {wav_path}")
                infos.append(sf.info(wav_path))

            sr = infos[0].samplerate
            frames = infos[0].frames
            if sr != SAMPLE_RATE_RAW:
                raise ValueError(f"Expected {SAMPLE_RATE_RAW} Hz in {song_dir}, got {sr} Hz.")
            if any(info.samplerate != sr or info.channels != 2 for info in infos):
                raise ValueError(f"Expected stereo {SAMPLE_RATE_RAW} Hz WAV files in {song_dir}.")
            if any(info.frames != frames for info in infos):
                raise ValueError(f"Mismatched stem frame counts in {song_dir}.")

            records.append({
                "index": index,
                "song": song_dir.name,
                "artist": parse_artist_name(song_dir.name),
                "subset": subset,
                "relative_dir": f"{subset}/{song_dir.name}",
                "sample_rate": sr,
                "frames": frames,
                "duration_sec": round(frames / float(sr), 4),
            })

        discovered[subset] = records

    return discovered


def create_musdb_split_manifest(
    musdb_root: Path | str = MUSDB_ROOT,
    output_path: Path | str = SPLIT_MANIFEST_PATH,
    seed: int = 2026,
    val_count: int = 20,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create `data/splits/musdb18hq_v1.json` (80 train / 20 validation / 50 test).

    Groups the 100 official `train` songs by artist before selecting exactly `val_count`
    validation songs so that neither songs nor artists leak between train and validation.
    """
    discovered = discover_musdb_tracks(musdb_root)
    train_pool = discovered["train"]
    test_pool = discovered["test"]

    by_artist: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in train_pool:
        by_artist[str(item["artist"])].append(item)

    artists = sorted(by_artist.keys())
    rng = np.random.default_rng(seed)
    shuffled_artists = [artists[int(i)] for i in rng.permutation(len(artists))]

    # Select artist groups whose song counts sum to exactly `val_count` (20).
    val_artists: set[str] = set()
    current_val_count = 0
    for artist in shuffled_artists:
        group_size = len(by_artist[artist])
        if current_val_count + group_size <= val_count:
            val_artists.add(artist)
            current_val_count += group_size
            if current_val_count == val_count:
                break

    if current_val_count != val_count:
        raise RuntimeError(
            f"Could not form an exact {val_count}-song artist-disjoint validation split."
        )

    train_members = [item for item in train_pool if str(item["artist"]) not in val_artists]
    val_members = [item for item in train_pool if str(item["artist"]) in val_artists]

    assert len(train_members) == len(train_pool) - val_count
    assert len(val_members) == val_count

    manifest: dict[str, object] = {
        "dataset": "MUSDB18-HQ",
        "version": "musdb18hq_v1",
        "seed": seed,
        "stem_names": list(STEM_NAMES),
        "counts": {
            "train": len(train_members),
            "validation": len(val_members),
            "test": len(test_pool),
        },
        "splits": {
            "train": train_members,
            "validation": val_members,
            "test": test_pool,
        },
    }

    validate_split_manifest(manifest)

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        existing = json.loads(dest.read_text())
        if existing != manifest:
            raise RuntimeError(f"Incompatible split manifest already exists at {dest}.")
    else:
        dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    return manifest


def validate_split_manifest(manifest: dict[str, object]) -> None:
    """Verify split counts and strictly reject any song overlap across splits."""
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Manifest missing 'splits' dictionary.")

    for req in ("train", "validation", "test"):
        if req not in splits or not splits[req]:
            raise ValueError(f"Manifest missing or empty split: {req}")

    seen_keys: set[tuple[str, str]] = set()
    seen_song_names: set[str] = set()
    for split_name, members in splits.items():
        for member in members:
            key = (str(member["subset"]), str(member["song"]))
            song_name = str(member["song"])
            if key in seen_keys or song_name in seen_song_names:
                raise ValueError(f"Duplicate song detected across splits: {song_name} (in {split_name})")
            seen_keys.add(key)
            seen_song_names.add(song_name)

    train_artists = {str(m["artist"]) for m in splits["train"]}
    val_artists = {str(m["artist"]) for m in splits["validation"]}
    overlap_artists = train_artists & val_artists
    if overlap_artists:
        raise ValueError(f"Artist leakage between train and validation: {sorted(overlap_artists)}")


def load_split_manifest(manifest_path: Path | str = SPLIT_MANIFEST_PATH) -> dict[str, object]:
    """Load and validate `data/splits/musdb18hq_v1.json`."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {path}. Run `python -m scripts.create_song_splits` first."
        )
    manifest = json.loads(path.read_text())
    validate_split_manifest(manifest)
    return manifest


def select_active_excerpts_for_song(
    song_dir: Path | str,
    n_excerpts: int = 10,
    duration_sec: float = EXCERPT_DURATION_SEC,
    stride_sec: float = 2.0,
    active_threshold_db: float = ACTIVE_THRESHOLD_DB,
) -> list[dict[str, object]]:
    """Select `n_excerpts` 8-second windows across `song_dir` prioritizing 4-stem activity.

    Uses fast cumulative energy sums over mono-downmixed stems to evaluate all candidate
    windows spaced by `stride_sec`, then picks `n_excerpts` evenly spaced windows from
    the fully-active pool (`all 4 stems > active_threshold_db`). If a track has fewer than
    `n_excerpts` fully-active windows (e.g., an acoustic song without bass), supplements
    with the highest-activity windows and records the per-stem `active_mask`.
    """
    sdir = Path(song_dir)
    info = sf.info(sdir / "mixture.wav")
    sr = info.samplerate
    total_frames = info.frames
    win_frames = int(round(duration_sec * sr))
    stride_frames = max(1, int(round(stride_sec * sr)))

    if total_frames < win_frames:
        raise ValueError(f"Song {sdir.name} is shorter ({total_frames / sr:.2f}s) than {duration_sec}s.")

    starts = np.arange(0, total_frames - win_frames + 1, stride_frames, dtype=np.int64)
    min_candidates = max(n_excerpts * 3, 30)
    if len(starts) < min_candidates:
        starts = np.unique(
            np.linspace(0, total_frames - win_frames, min_candidates, dtype=np.int64)
        )

    # Compute exact RMS dBFS per stem for all candidate windows via cumulative sum
    stem_window_db = []
    for stem_name in STEM_NAMES:
        audio, _ = sf.read(sdir / f"{stem_name}.wav", dtype="float32")
        mono = audio.mean(axis=1, dtype=np.float64)
        cs = np.concatenate(([0.0], np.cumsum(np.square(mono))))
        mean_sq = (cs[starts + win_frames] - cs[starts]) / float(win_frames)
        db = 10.0 * np.log10(np.maximum(mean_sq, 1e-10))
        stem_window_db.append(db)

    # Shape: (num_candidates, 4)
    loudness_matrix = np.stack(stem_window_db, axis=1).astype(np.float32)
    active_matrix = loudness_matrix > float(active_threshold_db)
    active_counts = np.sum(active_matrix, axis=1)

    full_active_indices = np.flatnonzero(active_counts == 4)

    if len(full_active_indices) >= n_excerpts:
        chosen_positions = np.linspace(0, len(full_active_indices) - 1, n_excerpts)
        chosen_idx = [int(full_active_indices[int(round(p))]) for p in chosen_positions]
        # Ensure uniqueness if rounding collided
        if len(set(chosen_idx)) < n_excerpts:
            chosen_idx = [int(i) for i in full_active_indices[:n_excerpts]]
    else:
        # Take all 4-stem active windows plus best remaining windows ranked by
        # (number of active stems, minimum active stem loudness)
        remaining = [i for i in range(len(starts)) if i not in set(full_active_indices)]
        remaining.sort(
            key=lambda i: (
                int(active_counts[i]),
                float(np.min(loudness_matrix[i, active_matrix[i]])) if active_counts[i] > 0 else -100.0,
            ),
            reverse=True,
        )
        combined = list(full_active_indices) + remaining
        pool = sorted(combined[: max(n_excerpts, len(full_active_indices))])
        if len(pool) >= n_excerpts:
            chosen_positions = np.linspace(0, len(pool) - 1, n_excerpts)
            chosen_idx = []
            used = set()
            for p in chosen_positions:
                cand = int(pool[int(round(p))])
                if cand not in used:
                    chosen_idx.append(cand)
                    used.add(cand)
            for cand in pool:
                if len(chosen_idx) >= n_excerpts:
                    break
                if int(cand) not in used:
                    chosen_idx.append(int(cand))
                    used.add(int(cand))
            chosen_idx.sort()
        else:
            chosen_idx = [int(i) for i in pool]

    excerpts: list[dict[str, object]] = []
    for ex_id, cand_i in enumerate(chosen_idx):
        start_frame = int(starts[cand_i])
        l_db = loudness_matrix[cand_i]
        act = active_matrix[cand_i]
        excerpts.append({
            "excerpt_index": ex_id,
            "start_frame_44k": start_frame,
            "frames_44k": win_frames,
            "excerpt_start_sec": round(start_frame / float(sr), 4),
            "duration_sec": float(duration_sec),
            "loudness_raw_db": [round(float(x), 4) for x in l_db],
            "active_mask": [bool(x) for x in act],
            "all_stems_active": bool(np.all(act)),
        })

    return excerpts


def load_excerpt_stems_44k(
    song_dir: Path | str,
    start_sec: float,
    duration_sec: float = EXCERPT_DURATION_SEC,
) -> dict[str, np.ndarray]:
    """Load 4 aligned `(frames, 2)` float32 stereo stems at 44.1 kHz for an excerpt."""
    sdir = Path(song_dir)
    start_frame = int(round(start_sec * SAMPLE_RATE_RAW))
    frames = int(round(duration_sec * SAMPLE_RATE_RAW))
    stems: dict[str, np.ndarray] = {}
    for stem_name in STEM_NAMES:
        audio, sr = sf.read(
            sdir / f"{stem_name}.wav",
            start=start_frame,
            frames=frames,
            dtype="float32",
            always_2d=True,
        )
        if sr != SAMPLE_RATE_RAW:
            raise ValueError(f"Expected {SAMPLE_RATE_RAW} Hz, got {sr} Hz.")
        stems[stem_name] = audio
    return stems


def render_gain_staged_mix(
    stems_stereo: Sequence[np.ndarray] | dict[str, np.ndarray],
    gains_db: Sequence[float] | np.ndarray,
    active_mask: Sequence[bool] | np.ndarray | None = None,
) -> dict[str, object]:
    """Render a stereo mix at original 44.1 kHz resolution and compute audio safety metrics.

    Applies `active_mask` if provided (forcing silent stems to 0.0 dB gain as specified in A2),
    sums the scaled stereo stems via `wadidaw.stereo_mixer.render_stereo`, and reports
    `peak_linear`, `peak_dbfs`, and `clipping_rate`.
    """
    if isinstance(stems_stereo, dict):
        stem_arrays = [np.asarray(stems_stereo[name], dtype=np.float32) for name in STEM_NAMES]
    else:
        stem_arrays = [np.asarray(s, dtype=np.float32) for s in stems_stereo]

    if len(stem_arrays) != 4:
        raise ValueError("Expected exactly 4 stems (vocals, drums, bass, other).")

    effective_gains = np.asarray(gains_db, dtype=np.float32).copy()
    if effective_gains.shape != (4,):
        raise ValueError(f"Expected 4 gain values in dB, got {effective_gains.shape}.")

    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)
        effective_gains[~mask] = 0.0

    effective_gains = np.clip(effective_gains, -18.0, 12.0)

    tensors = [torch.from_numpy(s) for s in stem_arrays]
    gains_tensor = torch.from_numpy(effective_gains)
    pans_tensor = torch.zeros(4, dtype=torch.float32)

    with torch.no_grad():
        mix_tensor = render_stereo(tensors, gains_tensor, pans_tensor)
    mix = mix_tensor.numpy()

    abs_mix = np.abs(mix)
    peak_linear = float(np.max(abs_mix)) if abs_mix.size > 0 else 0.0
    peak_dbfs = float(20.0 * np.log10(max(peak_linear, 1e-12)))
    clipping_rate = float(np.mean(abs_mix > 1.0)) if abs_mix.size > 0 else 0.0

    return {
        "mix": mix,
        "effective_gains_db": {name: round(float(effective_gains[i]), 4) for i, name in enumerate(STEM_NAMES)},
        "metrics": {
            "peak_linear": round(peak_linear, 6),
            "peak_dbfs": round(peak_dbfs, 4),
            "clipping_rate": round(clipping_rate, 6),
        },
    }
