import json
from pathlib import Path

import gymnasium as gym

from wadidaw.music_env import (
    DATASET,
    MusicMixingEnv,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/splits/babyslakh_v1.json"


class MultiSongMixingEnv(gym.Env):
    """Mixing tasks drawn from a fixed song partition."""

    def __init__(self, split="train"):
        super().__init__()

        if not MANIFEST.exists():
            raise FileNotFoundError(
                "Run python -m scripts.create_song_splits first."
            )

        manifest = json.loads(MANIFEST.read_text())

        if split not in manifest["splits"]:
            raise ValueError(f"Unknown split: {split}")

        self.split = split
        self.members = manifest["splits"][split]

        # Cache each song's environment once.
        self.envs = []

        for member in self.members:
            env = MusicMixingEnv(
                song_index=member["index"]
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}

        if "song_position" in options:
            position = int(options["song_position"])

            if not 0 <= position < len(self.envs):
                raise ValueError("Invalid song position.")
        else:
            position = int(
                self.np_random.integers(len(self.envs))
            )

        self.current_env = self.envs[position]
        self.song_name = self.current_env.song_name

        # Generate an independent, reproducible target seed.
        target_seed = int(
            self.np_random.integers(0, 2**31 - 1)
        )

        observation, info = self.current_env.reset(
            seed=target_seed
        )

        info = {
            **info,
            "song": self.song_name,
            "song_index": self.members[position]["index"],
        }

        return observation, info

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

        info = {
            **info,
            "song": self.song_name,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        for env in self.envs:
            env.close()
