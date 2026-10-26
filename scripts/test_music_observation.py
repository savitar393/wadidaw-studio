import numpy as np

from gymnasium.utils.env_checker import check_env
from wadidaw.music_env import MusicMixingEnv


def main():
    for song_index in range(5):
        env = MusicMixingEnv(song_index=song_index)

        check_env(env, skip_render_check=True)

        obs, info = env.reset(seed=42)

        assert obs.shape == (101,)
        assert env.observation_space.contains(obs)

        assert env.stem_band_features.shape == (4, 8)

        # The observation must contain the individual
        # stem spectral features at indices 20 through 51.
        observed_stems = obs[20:52].reshape(4, 8)

        assert env.cross_band_features.shape == (6, 8)

        observed_cross = obs[52:100].reshape(6, 8)

        assert np.allclose(
            observed_cross,
            env.cross_band_features,
            atol=1e-6,
        )

        assert np.isfinite(obs).all()

        expected_stems = (
            env.stem_band_features / 100.0
        )

        assert np.allclose(
            observed_stems,
            expected_stems,
            atol=1e-6,
        )

        # Verify that the oracle can still solve the task.
        for step in range(env.max_steps):
            action = np.clip(
                (env.target_gains - env.gains) / 2.0,
                -1.0,
                1.0,
            ).astype(np.float32)

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            assert env.observation_space.contains(obs)

            if terminated or truncated:
                break

        assert info["success"]

        print(
            f"{env.song_name} | "
            f"obs={obs.shape} | "
            f"stem_features={env.stem_band_features.shape} | "
            f"error={info['error']:.6f} | "
            f"success={info['success']}"
        )

        env.close()

    print("\nAll music observation tests passed!")


if __name__ == "__main__":
    main()
