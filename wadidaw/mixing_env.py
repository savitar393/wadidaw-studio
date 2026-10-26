
from pathlib import Path

import gymnasium as gym
import numpy as np
import soundfile as sf
from gymnasium import spaces


ROOT = Path(__file__).resolve().parents[1]
STEMS_DIR = ROOT / "data" / "demo" / "stems"


class MixingEnv(gym.Env):
    """Four-track audio mixing reinforcement learning environment."""

    def __init__(self):
        super().__init__()

        self.names = [
            "tone_110",
            "tone_220",
            "tone_330",
            "tone_440",
        ]

        # Load our four audio stems.
        audio_tracks = []
        self.sample_rate = None

        for name in self.names:
            audio, sr = sf.read(
                STEMS_DIR / f"{name}.wav",
                dtype="float32",
            )

            if audio.ndim != 1:
                raise ValueError("Expected mono audio stems.")

            if self.sample_rate is None:
                self.sample_rate = sr
            elif sr != self.sample_rate:
                raise ValueError("Sample rates must match.")

            audio_tracks.append(audio)

        lengths = [len(track) for track in audio_tracks]
        if len(set(lengths)) != 1:
            raise ValueError("Stem lengths must match.")

        self.stems = np.stack(audio_tracks)

        # FFT bins corresponding to our four test tones.
        frequencies = np.array([110, 220, 330, 440])
        self.bins = np.rint(
            frequencies * lengths[0] / self.sample_rate
        ).astype(int)

        # Cache the four frequency bins for every stem.
        self.stem_fft_bins = np.fft.rfft(
            self.stems, axis=1
        )[:, self.bins]

        # The desired mix, kept inside the environment.
        self.target_gains = np.array(
            [-6, 6, -6, -6], dtype=np.float32
        )

        self.target_features = self._features_from_gains(self.target_gains)

        # Four continuous actions between -1 and +1.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Current gains (4), current features (4),
        # target features (4).
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 4 + [0.0] * 8, dtype=np.float32),
            high=np.array([12.0 / 18.0] * 4 + [4.1] * 8, dtype=np.float32),
            # shape=(12,),
            dtype=np.float32,
        )

        self.max_steps = 12
        self.gains = np.zeros(4, dtype=np.float32)
        self.steps = 0
        self.done = False

    def _mix(self, gains):
        """Render an audio mix using the supplied gains."""
        amplitudes = np.power(
            10.0, gains / 20.0
        ).astype(np.float32)

        return np.sum(
            self.stems * amplitudes[:, None],
            axis=0,
        )

    def _measure(self, audio):
        """Measure the amplitudes of our four test tones."""
        spectrum = np.abs(np.fft.rfft(audio))

        return (
            2.0 * spectrum[self.bins] / len(audio)
        ).astype(np.float32)

    def _features_from_gains(self, gains):
        """Calculate spectral features from cached FFT bins."""
        amplitudes = np.power(
            10.0, np.asarray(gains) / 20.0
        )

        mixed_bins = np.sum(
            self.stem_fft_bins * amplitudes[:, None],
            axis=0,
        )

        return (
            2.0 * np.abs(mixed_bins) / self.stems.shape[1]
        ).astype(np.float32)

    def _error(self, features):
        """Measure spectral difference from the target."""
        relative_error = (
            (features - self.target_features)
            / np.maximum(self.target_features, 0.02)
        )

        return float(np.mean(relative_error ** 2))

    def _observation(self, features):
        return np.concatenate([
            self.gains / 18.0,
            features / 0.12,
            self.target_features / 0.12,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Generate a different target mix for every episode.
        self.target_gains = self.np_random.uniform(
            low=-8.0,
            high=8.0,
            size=4,
        ).astype(np.float32)

        # Calculate the new target's audio features.
        self.target_features = self._features_from_gains(self.target_gains)

        self.gains = np.zeros(4, dtype=np.float32)
        self.steps = 0
        self.done = False

        features = self._features_from_gains(self.gains)

        return self._observation(features), {
            "error": self._error(features),
            "gains_db": self.gains.copy(),
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode ended. Call reset().")

        action = np.asarray(action, dtype=np.float32)

        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Expected four finite actions.")

        action = np.clip(action, -1.0, 1.0)

        # Measure the old mix.
        previous_features = self._features_from_gains(self.gains)
        previous_error = self._error(previous_features)

        # Apply the action: up to +/- 2 dB per track.
        self.gains = np.clip(
            self.gains + action * 2.0,
            -18.0,
            12.0,
        ).astype(np.float32)

        self.steps += 1

        # Render and evaluate the new mix.
        audio = self._mix(self.gains)
        features = self._features_from_gains(self.gains)

        error = self._error(features)
        peak = float(np.max(np.abs(audio)))

        clipping_penalty = 0.1 * max(0.0, peak - 1.0) ** 2

        terminated = bool(error < 0.001)
        truncated = bool(
            self.steps >= self.max_steps and not terminated
        )

        self.done = terminated or truncated

        # reward = (
        #     previous_error - error
        #     - clipping_penalty
        #     + (1.0 if terminated else 0.0)
        # )

        # Penalize remaining spectral error at every step.
        reward = (
            -error
            - clipping_penalty
            + (1.0 if terminated else 0.0)
        )

        return (
            self._observation(features),
            float(reward),
            terminated,
            truncated,
            {
                "error": error,
                "peak": peak,
                "gains_db": self.gains.copy(),
            },
        )


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = MixingEnv()
    check_env(env, skip_render_check=True)
    print("Gymnasium environment check passed!")

    observation, info = env.reset(seed=42)

    print("\nTarget gains:", env.target_gains)
    print("Initial error:", round(info["error"], 4))

    for i in range(env.max_steps):
        # Privileged test controller: knows the target.
        # This is NOT the learning agent!
        action = np.clip(
            (env.target_gains - env.gains) / 2.0,
            -1.0,
            1.0,
        ).astype(np.float32)

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        print(
            f"Step {i + 1}: "
            f"error={info['error']:.6f}, "
            f"reward={reward:.4f}"
        )

        if terminated or truncated:
            print("Success:", terminated)
            break

    env.close()
