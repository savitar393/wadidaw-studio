import json

import numpy as np
from gymnasium.utils.env_checker import check_env

from wadidaw.multi_song_env import (
    MANIFEST,
    MultiSongMixingEnv,
)


def main():
    manifest = json.loads(MANIFEST.read_text())
    splits = manifest["splits"]

    # Check that the partitions are disjoint.
    song_sets = {
        name: {
            item["song"] for item in members
        }
        for name, members in splits.items()
    }

    assert len(song_sets["train"]) == 12
    assert len(song_sets["validation"]) == 4
    assert len(song_sets["test"]) == 4

    assert song_sets["train"].isdisjoint(
        song_sets["validation"]
    )
    assert song_sets["train"].isdisjoint(
        song_sets["test"]
    )
    assert song_sets["validation"].isdisjoint(
        song_sets["test"]
    )

    print("Song split isolation: PASSED")

    for split in ["train", "validation", "test"]:
        env = MultiSongMixingEnv(split=split)

        check_env(env, skip_render_check=True)

        successes = 0

        for position in range(len(env.envs)):
            # Force every song to be tested at least once.
            observation, info = env.reset(
                seed=1000 + position,
                options={"song_position": position},
            )

            assert observation.shape == (101,)
            assert env.observation_space.contains(
                observation
            )
            assert info["song"] in song_sets[split]

            for step in range(env.current_env.max_steps):
                # Privileged oracle for validation only.
                underlying = env.current_env

                action = np.clip(
                    (
                        underlying.target_gains
                        - underlying.gains
                    ) / 2.0,
                    -1.0,
                    1.0,
                ).astype(np.float32)

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                assert env.observation_space.contains(
                    observation
                )

                if terminated or truncated:
                    break

            assert info["success"]
            successes += 1

        print(
            f"{split.upper()}: "
            f"{successes}/{len(env.envs)} "
            f"songs solved by oracle"
        )

        env.close()

    print("\nMulti-song environment tests passed!")


if __name__ == "__main__":
    main()
