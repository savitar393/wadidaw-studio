
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
STEMS_DIR = ROOT / "data" / "demo" / "stems"
OUTPUT_DIR = ROOT / "outputs"

STEMS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 24000
DURATION = 4

# Temporary synthetic sounds for testing.
# We'll use real music stems later.
frequencies = {
    "tone_110": 110,
    "tone_220": 220,
    "tone_330": 330,
    "tone_440": 440,
}

t = np.arange(SAMPLE_RATE * DURATION) / SAMPLE_RATE

# Prevent abrupt edges in the generated sounds.
fade = np.ones(len(t), dtype=np.float32)
n = int(SAMPLE_RATE * 0.05)
fade[:n] = np.linspace(0, 1, n)
fade[-n:] = np.linspace(1, 0, n)

# Step 1: Generate four stems.
for name, frequency in frequencies.items():
    audio = (
        0.12 * np.sin(2 * np.pi * frequency * t) * fade
    ).astype(np.float32)

    sf.write(STEMS_DIR / f"{name}.wav", audio, SAMPLE_RATE)


def db_to_amplitude(db):
    return 10 ** (db / 20)


# Step 2: Mix stems using adjustable gains.
def mix_tracks(gains_db):
    result = np.zeros(len(t), dtype=np.float32)

    for name in frequencies:
        audio, sr = sf.read(
            STEMS_DIR / f"{name}.wav",
            dtype="float32",
        )
        assert sr == SAMPLE_RATE

        result += audio * db_to_amplitude(gains_db[name])

    if np.max(np.abs(result)) > 1.0:
        raise ValueError("Mix would exceed digital full scale!")

    return result


# Step 3: Render original and adjusted mixes.
original = mix_tracks({name: 0 for name in frequencies})

adjusted = mix_tracks({
    "tone_110": 6,
    "tone_220": -18,
    "tone_330": -12,
    "tone_440": -18,
})

sf.write(OUTPUT_DIR / "original.wav", original, SAMPLE_RATE)
sf.write(OUTPUT_DIR / "adjusted.wav", adjusted, SAMPLE_RATE)

print("Original peak:", np.max(np.abs(original)))
print("Adjusted peak:", np.max(np.abs(adjusted)))
print("Audio mixer test completed!")
