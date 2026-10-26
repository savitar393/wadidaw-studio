import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from wadidaw.music_env import (
    DATASET,
    MusicMixingEnv,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/splits/babyslakh_v1.json"


class MultiSongMixingEnv(gym.Env):
    """Mixing tasks drawn from a fixed song partition.

    Multiple excerpts are opt-in via ``excerpt_fractions``. Reset samples
    a song and excerpt uniformly, unless ``song_position`` and/or
    ``excerpt_position`` are supplied in options. Fix both positions and
    the reset seed to repeat an evaluation task. Positions index members
    and configured fractions respectively; ``envs`` caches their Cartesian
    product in song-major order. Fractions locate the excerpt start within
    the available start range, rather than within the full song duration.
    """

    def __init__(
        self, split="train", *,
        excerpt_fractions=(0.5,), duration_seconds=8,
    ):
        super().__init__()

        self.excerpt_fractions = tuple(float(value) for value in excerpt_fractions)
        if (
            not self.excerpt_fractions
            or any(
                not np.isfinite(value) or not 0 <= value <= 1
                for value in self.excerpt_fractions
            )
            or len(set(self.excerpt_fractions)) != len(self.excerpt_fractions)
        ):
            raise ValueError("Excerpt fractions must be distinct finite values in [0, 1].")

        if not MANIFEST.exists():
            raise FileNotFoundError(
                "Run python -m scripts.create_song_splits first."
            )

        manifest = json.loads(MANIFEST.read_text())

        if split not in manifest["splits"]:
            raise ValueError(f"Unknown split: {split}")

        # Reject leakage by either dataset index or song identity before
        # loading any audio, including overlap with the held-out partition.
        songs, indices = set(), set()
        for members in manifest["splits"].values():
            for member in members:
                if member["song"] in songs or member["index"] in indices:
                    raise ValueError("Overlapping songs or indices in split manifest.")
                songs.add(member["song"])
                indices.add(member["index"])

        self.split = split
        self.members = manifest["splits"][split]
        if not self.members:
            raise ValueError(f"Empty song split: {split}")

        # Cache every configured (song, excerpt) once. Invalid excerpts fail
        # explicitly; silently replacing them would change the sampling pool.
        self.envs = []

        for member in self.members:
            for fraction in self.excerpt_fractions:
                env = MusicMixingEnv(
                    song_index=member["index"],
                    duration_seconds=duration_seconds,
                    excerpt_fraction=fraction,
                )

                # Detect unexpected changes in dataset ordering.
                if env.song_name != member["song"]:
                    raise RuntimeError(
                        f"Dataset mismatch: expected "
                        f"{member['song']}, got {env.song_name}"
                    )

                self.envs.append(env)

        self.observation_space = (
            self.envs[0].observation_space
        )
        self.action_space = self.envs[0].action_space

        self.current_env = None
        self.song_name = None
        self.song_position = None
        self.excerpt_position = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}

        if "song_position" in options:
            position = int(options["song_position"])

            if not 0 <= position < len(self.members):
                raise ValueError("Invalid song position.")
        else:
            position = int(
                self.np_random.integers(len(self.members))
            )

        if "excerpt_position" in options:
            excerpt_position = int(options["excerpt_position"])
            if not 0 <= excerpt_position < len(self.excerpt_fractions):
                raise ValueError("Invalid excerpt position.")
        elif len(self.excerpt_fractions) == 1:
            # Preserve the original target RNG sequence for midpoint runs.
            excerpt_position = 0
        else:
            excerpt_position = int(self.np_random.integers(len(self.excerpt_fractions)))

        self.song_position = position
        self.excerpt_position = excerpt_position
        self.current_env = self.envs[
            position * len(self.excerpt_fractions) + excerpt_position
        ]
        self.song_name = self.current_env.song_name

        # Generate an independent, reproducible target seed.
        target_seed = int(
            self.np_random.integers(0, 2**31 - 1)
        )

        observation, info = self.current_env.reset(
            seed=target_seed
        )

        return observation, self._task_info(info)

    def _task_info(self, info):
        return {
            **info,
            "song": self.song_name,
            "song_index": self.members[self.song_position]["index"],
            "song_position": self.song_position,
            "excerpt_position": self.excerpt_position,
            "excerpt_fraction": self.excerpt_fractions[self.excerpt_position],
            "excerpt_start_seconds": self.current_env.excerpt_start_seconds,
        }

    def step(self, action):
        if self.current_env is None:
            raise RuntimeError(
                "Call reset() before step()."
            )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.current_env.step(action)

        return (
            observation,
            reward,
            terminated,
            truncated,
            self._task_info(info),
        )

    def close(self):
        for env in self.envs:
            env.close()
