from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "raw" / "babyslakh" / "babyslakh_16k"
OUTPUT = ROOT / "outputs" / "real_audio_preview"

DURATION_SECONDS = 8
NUM_STEMS = 4


def main():
    # Find valid Slakh song directories.
    stem_dirs = sorted(
        path
        for path in DATASET.rglob("stems")
        if path.is_dir()
        and path.parent.name.startswith("Track")
    )

    if not stem_dirs:
        raise FileNotFoundError(
            f"No Slakh stem directories found beneath {DATASET}"
        )

    stems_dir = stem_dirs[0]

    files = sorted(
        path
        for path in stems_dir.iterdir()
        if path.suffix.lower() in {".wav", ".flac"}
    )

    if len(files) < NUM_STEMS:
        raise RuntimeError("Not enough audio stems.")

    infos = [sf.info(path) for path in files]

    sample_rate = infos[0].samplerate

    if any(
        info.samplerate != sample_rate
        for info in infos
    ):
        raise ValueError("Sample rates do not match.")

    # Read the same middle excerpt from each stem.
    common_frames = min(info.frames for info in infos)

    frames = min(
        DURATION_SECONDS * sample_rate,
        common_frames,
    )

    start = max(0, (common_frames - frames) // 2)

    candidates = []

    for path in files:
        audio, _ = sf.read(
            path,
            start=start,
            frames=frames,
            dtype="float32",
        )

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        rms = float(np.sqrt(
            np.mean(np.square(audio, dtype=np.float64))
        ))

        if rms > 1e-5:
            candidates.append((rms, path, audio))

    # Choose the four loudest active stems.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    if len(candidates) < NUM_STEMS:
        raise RuntimeError(
            "Too few active stems in this excerpt."
        )

    selected = candidates[:NUM_STEMS]

    tracks = np.stack([
        item[2] for item in selected
    ])

    # A deliberately obvious gain adjustment.
    gains_db = np.array(
        [0.0, 6.0, -6.0, -3.0],
        dtype=np.float32,
    )

    amplitudes = np.power(
        10.0, gains_db / 20.0
    )

    original = tracks.sum(axis=0)

    adjusted = np.sum(
        tracks * amplitudes[:, None],
        axis=0,
    )

    # Apply one shared safety scale to both mixes.
    # Do not normalize them independently.
    max_peak = max(
        float(np.max(np.abs(original))),
        float(np.max(np.abs(adjusted))),
    )

    scale = min(
        1.0,
        0.95 / max(max_peak, 1e-9),
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    sf.write(
        OUTPUT / "original.wav",
        original * scale,
        sample_rate,
    )

    sf.write(
        OUTPUT / "adjusted.wav",
        adjusted * scale,
        sample_rate,
    )

    print("Song:", stems_dir.parent.name)
    print("Sample rate:", sample_rate)
    print("Excerpt duration:", frames / sample_rate)

    for index, (rms, path, audio) in enumerate(selected):
        sf.write(
            OUTPUT / f"stem_{index + 1}_{path.stem}.wav",
            audio * scale,
            sample_rate,
        )

        print(
            f"Stem {index + 1}: {path.name} | "
            f"RMS={rms:.5f} | "
            f"Gain={gains_db[index]:+.1f} dB"
        )

    print("Shared safety scale:", round(scale, 4))
    print("Preview saved to:", OUTPUT)


if __name__ == "__main__":
    main()
