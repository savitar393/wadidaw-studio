
import argparse

import numpy as np

from wadidaw.music_env import MusicMixingEnv, DATASET


BAND_EDGES = [
    0, 80, 160, 320, 640,
    1250, 2500, 5000, 8001,
]


def verify_spectral_features(env):
    """Compare cached features against an actual waveform FFT."""
    gains = np.array(
        [-3.0, 2.0, 5.0, -7.0],
        dtype=np.float32,
    )

    cached = env._features(gains)
    audio = env.render_mix(gains)

    spectrum = np.fft.rfft(audio.astype(np.float64))
    frequencies = np.fft.rfftfreq(
        len(audio), d=1.0 / env.sample_rate
    )

    direct = []

    for low, high in zip(
        BAND_EDGES[:-1], BAND_EDGES[1:]
    ):
        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        power = np.sum(
            np.abs(spectrum[mask]) ** 2
        ) / len(audio) ** 2

        db = 10.0 * np.log10(
            max(float(power), 1e-12)
        )

        direct.append(np.clip(db, -120.0, 20.0))

    difference = np.max(
        np.abs(cached - np.asarray(direct))
    )

    if difference >= 0.05:
        raise AssertionError(
            f"Spectral mismatch: {difference:.6f} dB"
        )

    return float(difference)


def evaluate_episode(env, policy, seed):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    for step in range(env.max_steps):
        if policy == "no_op":
            action = np.zeros(4, dtype=np.float32)

        elif policy == "random":
            action = rng.uniform(
                -1, 1, size=4
            ).astype(np.float32)

        elif policy == "oracle":
            # Knows hidden target gains: validation only.
            action = np.clip(
                (env.target_gains - env.gains) / 2.0,
                -1.0,
                1.0,
            ).astype(np.float32)

        else:
            raise ValueError(policy)

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        if terminated or truncated:
            break

    return (
        float(info["success"]),
        float(info["error"]),
        step + 1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    song_dirs = sorted(
        p for p in DATASET.rglob("stems")
        if p.is_dir()
        and p.parent.name.startswith("Track")
    )

    results = {
        "no_op": [],
        "random": [],
        "oracle": [],
    }

    tested = 0
    skipped = 0

    for index in range(min(args.songs, len(song_dirs))):
        try:
            env = MusicMixingEnv(song_index=index)
        except (RuntimeError, ValueError) as exc:
            print(
                f"Skipping {song_dirs[index].parent.name}: {exc}"
            )
            skipped += 1
            continue

        difference = verify_spectral_features(env)

        print(
            f"{env.song_name}: "
            f"FFT agreement within {difference:.6f} dB"
        )

        for episode in range(args.episodes):
            seed = 30_000 + index * 1000 + episode

            for policy in results:
                result = evaluate_episode(
                    env, policy, seed
                )
                results[policy].append(result)

        tested += 1
        env.close()

    if tested == 0:
        raise RuntimeError("No valid songs were tested.")

    print("\nMUSIC BASELINE RESULTS")
    print("Songs tested:", tested)
    print("Songs skipped:", skipped)
    print("Episodes per controller:", tested * args.episodes)

    for name, values in results.items():
        values = np.asarray(values, dtype=np.float64)

        print(f"\n{name.upper()}")
        print(f"Success rate: {values[:, 0].mean() * 100:.1f}%")
        print(f"Mean final error: {values[:, 1].mean():.6f}")
        print(f"Mean steps: {values[:, 2].mean():.2f}")


if __name__ == "__main__":
    main()
