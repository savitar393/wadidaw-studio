
from pathlib import Path

import gymnasium as gym
import numpy as np
import soundfile as sf
from gymnasium import spaces


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/raw/babyslakh/babyslakh_16k"


class MusicMixingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, duration_seconds=8, song_index=0, excerpt_fraction=0.5):
        super().__init__()

        song_dirs = sorted(
            p for p in DATASET.rglob("stems")
            if p.is_dir()
            and p.parent.name.startswith("Track")
        )

        if not song_dirs:
            raise FileNotFoundError(
                f"No stems found under {DATASET}"
            )

        stems_dir = song_dirs[song_index]
        self.song_name = stems_dir.parent.name

        files = sorted(
            p for p in stems_dir.iterdir()
            if p.suffix.lower() in {".wav", ".flac"}
        )

        if len(files) < 4:
            raise RuntimeError("Fewer than four stem files.")

        infos = [sf.info(path) for path in files]
        self.sample_rate = infos[0].samplerate

        if self.sample_rate != 16000:
            raise ValueError("Expected BabySlakh 16 kHz audio.")

        if any(
            info.samplerate != self.sample_rate
            for info in infos
        ):
            raise ValueError("Inconsistent sample rates.")

        common_frames = min(info.frames for info in infos)
        frames = min(
            common_frames,
            round(duration_seconds * self.sample_rate),
        )

        if frames < 2:
            raise ValueError("Audio excerpt is too short.")

        if not np.isfinite(excerpt_fraction):
            raise ValueError("Excerpt fraction must be finite.")

        if not 0.0 <= excerpt_fraction <= 1.0:
            raise ValueError("Excerpt fraction must be in [0, 1].")

        max_start = common_frames - frames

        start = int(round(max_start * excerpt_fraction))

        self.excerpt_start_seconds = start / self.sample_rate
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
                np.mean(np.square(audio.astype(np.float64)))
            ))

            if rms > 1e-5:
                candidates.append((rms, path, audio))

        if len(candidates) < 4:
            raise RuntimeError("Fewer than four active stems.")

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected = candidates[:4]

        self.stem_names = [
            item[1].name for item in selected
        ]

        self.stems = np.stack([
            item[2] for item in selected
        ]).astype(np.float32)

        self.max_steps = 12

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # 4 gains + 8 mix bands + 8 target bands
        # + 32 individual stem bands + 1 remaining-time feature.
        
        self.observation_space = spaces.Box(
            low=np.array(
                [-1.0] * 4       # Gains
                + [-1.2] * 48    # Current, target, stem bands
                + [-1.0] * 48    # Cross-stem interactions
                + [0.0],         # Remaining time
                dtype=np.float32,
            ),
            high=np.array(
                [12.0 / 18.0] * 4
                + [0.2] * 48
                + [1.0] * 48
                + [1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )


        # Compute the Fourier transform of each stem once.
        spectrum = np.fft.rfft(self.stems, axis=1)
        frequencies = np.fft.rfftfreq(
            frames, d=1 / self.sample_rate
        )

        # Eight bands covering the 16 kHz signal.
        edges = [
            0, 80, 160, 320, 640,
            1250, 2500, 5000, 8001,
        ]

        grams = []

        for low, high in zip(edges[:-1], edges[1:]):
            mask = (
                (frequencies >= low)
                & (frequencies < high)
            )

            block = spectrum[:, mask]

            # Captures per-stem energy and
            # cross-stem interference.
            gram = np.real(
                block @ block.conj().T
            ) / (frames ** 2)

            grams.append(gram)

        self.band_grams = np.stack(grams)

        # Extract the spectral energy of each individual stem.
        # Shape of band_grams: (8 bands, 4 stems, 4 stems).
        # The diagonal contains each stem's own energy.
        stem_powers = np.diagonal(
            self.band_grams,
            axis1=1,
            axis2=2,
        ).T  # (4 stems, 8 frequency bands)

        self.stem_band_features = np.clip(
            10.0 * np.log10(
                np.maximum(stem_powers, 1e-12)
            ),
            -120.0,
            20.0,
        ).astype(np.float32)

        assert self.stem_band_features.shape == (4, 8)

        
        # Six unique pairs of four stems.
        self.stem_pairs = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3),
        ]

        cross_features = []

        for i, j in self.stem_pairs:
            # Per-band cross energy between stems i and j.
            cross_energy = self.band_grams[:, i, j]

            energy_i = np.maximum(
                self.band_grams[:, i, i], 0.0
            )
            energy_j = np.maximum(
                self.band_grams[:, j, j], 0.0
            )

            denominator = np.sqrt(energy_i * energy_j)

            # Normalized cross energy, bounded in [-1, 1].
            coherence = np.divide(
                cross_energy,
                denominator,
                out=np.zeros_like(cross_energy),
                where=denominator > 1e-20,
            )

            cross_features.append(
                np.clip(coherence, -1.0, 1.0)
            )

        self.cross_band_features = np.stack(
            cross_features
        ).astype(np.float32)

        assert self.cross_band_features.shape == (6, 8)


        self.gains = np.zeros(4, dtype=np.float32)
        self.target_gains = np.zeros(4, dtype=np.float32)
        self.target_features = np.zeros(
            8, dtype=np.float32
        )

        self.steps = 0
        self.done = False

    def render_mix(self, gains_db=None):
        """Render a waveform from four stem gains."""
        if gains_db is None:
            gains_db = self.gains

        amplitudes = np.power(
            10.0, np.asarray(gains_db) / 20.0
        )

        return np.sum(
            self.stems * amplitudes[:, None],
            axis=0,
        ).astype(np.float32)

    def _features(self, gains_db):
        """Return eight spectral-band energies in dB."""
        amplitudes = np.power(
            10.0, np.asarray(gains_db) / 20.0
        )

        power = np.einsum(
            "i,bij,j->b",
            amplitudes,
            self.band_grams,
            amplitudes,
            optimize=True,
        )

        db = 10.0 * np.log10(
            np.maximum(power, 1e-12)
        )

        return np.clip(
            db, -120.0, 20.0
        ).astype(np.float32)

    def _error(self, features):
        difference = (
            features - self.target_features
        ) / 12.0

        return float(np.mean(difference ** 2))

    
    def _observation(self, features):
        remaining = (
            self.max_steps - self.steps
        ) / self.max_steps

        return np.concatenate([
            self.gains / 18.0,                        # 4
            features / 100.0,                        # 8
            self.target_features / 100.0,            # 8
            self.stem_band_features.flatten() / 100, # 32
            self.cross_band_features.flatten(),      # 48
            np.array([remaining], dtype=np.float32), # 1
        ]).astype(np.float32)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.gains = np.zeros(4, dtype=np.float32)

        self.target_gains = self.np_random.uniform(
            -8.0, 8.0, size=4
        ).astype(np.float32)

        self.target_features = self._features(
            self.target_gains
        )

        self.steps = 0
        self.done = False

        features = self._features(self.gains)

        return self._observation(features), {
            "error": self._error(features),
            "gains_db": self.gains.copy(),
        }

    def step(self, action):
        if self.done:
            raise RuntimeError(
                "Episode ended. Call reset()."
            )

        action = np.asarray(
            action, dtype=np.float32
        )

        if (
            action.shape != (4,)
            or not np.isfinite(action).all()
        ):
            raise ValueError(
                "Expected four finite actions."
            )

        action = np.clip(action, -1.0, 1.0)

        self.gains = np.clip(
            self.gains + 2.0 * action,
            -18.0,
            12.0,
        ).astype(np.float32)

        self.steps += 1

        features = self._features(self.gains)
        error = self._error(features)

        audio = self.render_mix()
        peak = float(np.max(np.abs(audio)))

        clipping_penalty = (
            0.1 * max(0.0, peak - 1.0) ** 2
        )

        success = bool(error < 0.001)

        # The 12-action budget is part of the task.
        terminated = bool(success or self.steps >= self.max_steps)
        truncated = False

        self.done = terminated

        reward = (
            -error
            - clipping_penalty
            + (1.0 if success else 0.0)
        )

        return (
            self._observation(features),
            float(reward),
            terminated,
            truncated,
            {
                "error": error,
                "peak": peak,
                "success": success,
                "gains_db": self.gains.copy(),
            },
        )
