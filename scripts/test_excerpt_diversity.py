import json

import numpy as np

from wadidaw.multi_song_env import MANIFEST
from wadidaw.music_env import MusicMixingEnv


FRACTIONS = (0.2, 0.5, 0.8)


def main():
    manifest = json.loads(MANIFEST.read_text())

    total_valid = 0
    total_failed = 0
    spectral_shifts = []

    # Don't touch the held-out test partition.
    for split in ("train", "validation"):
        print(f"\n{split.upper()}")

        for member in manifest["splits"][split]:
            excerpts = []

            for fraction in FRACTIONS:
                try:
                    env = MusicMixingEnv(
                        song_index=member["index"],
                        excerpt_fraction=fraction,
                    )
                except (RuntimeError, ValueError) as exc:
                    print(
                        f"  {member['song']} "
                        f"at {fraction:.1f}: {exc}"
                    )
                    total_failed += 1
                    continue

                obs, info = env.reset(seed=42)

                assert obs.shape == (101)
                assert env.observation_space.contains(obs)

                # Record the zero-gain mix spectrum.
                features = env._features(
                    np.zeros(4, dtype=np.float32)
                )

                # Check that the oracle still solves it.
                for step in range(env.max_steps):
                    action = np.clip(
                        (
                            env.target_gains - env.gains
                        ) / 2.0,
                        -1.0,
                        1.0,
                    ).astype(np.float32)

                    obs, reward, terminated, truncated, info = (
                        env.step(action)
                    )

                    if terminated or truncated:
                        break

                assert info["success"]

                excerpts.append({
                    "start": env.excerpt_start_seconds,
                    "features": features,
                    "stems": tuple(env.stem_names),
                })

                total_valid += 1
                env.close()

            if len(excerpts) >= 2:
                # Compare broad spectral content.
                shifts = [
                    float(np.mean(np.abs(
                        a["features"] - b["features"]
                    )))
                    for i, a in enumerate(excerpts)
                    for b in excerpts[i + 1:]
                ]

                shift = max(shifts)
                spectral_shifts.append(shift)

                starts = [
                    round(x["start"], 1) for x in excerpts
                ]

                print(
                    f"  {member['song']}: "
                    f"starts={starts}s, "
                    f"max spectral shift={shift:.2f} dB"
                )

    print("\nSUMMARY")
    print("Valid excerpts:", total_valid)
    print("Rejected excerpts:", total_failed)

    if spectral_shifts:
        print(
            "Mean maximum spectral shift:",
            f"{np.mean(spectral_shifts):.2f} dB",
        )


if __name__ == "__main__":
    main()
