
import numpy as np
from scipy.optimize import least_squares

from wadidaw.multi_song_env import MultiSongMixingEnv


def solve_gains(music, observation):
    """Find gains matching observed target spectral features."""

    # Indices 12:20 contain the eight target bands.
    # Undo the /100 normalization.
    target = observation[12:20].astype(np.float64) * 100.0

    def residual(gains):
        predicted = music._features(gains)
        return (
            predicted.astype(np.float64) - target
        ) / 12.0

    def jacobian(gains):
        # Fixed-size central differences avoid precision
        # issues from the environment's float32 features.
        delta = 0.05
        jac = np.empty((8, 4), dtype=np.float64)

        for i in range(4):
            positive = gains.copy()
            negative = gains.copy()

            positive[i] += delta
            negative[i] -= delta

            jac[:, i] = (
                residual(positive) - residual(negative)
            ) / (2.0 * delta)

        return jac

    result = least_squares(
        residual,
        x0=np.zeros(4, dtype=np.float64),
        jac=jacobian,
        bounds=(-8.0, 8.0),
        max_nfev=200,
    )

    # This is feature-reconstruction error, not gain error.
    fit_error = float(np.mean(result.fun ** 2))

    return result.x.astype(np.float32), fit_error


def main():
    env = MultiSongMixingEnv(split="validation")

    all_errors = []
    all_fit_errors = []
    all_gain_differences = []
    all_successes = []

    print("MODEL-BASED INVERSE MIXING")
    print("Using validation songs only.\n")

    for position, member in enumerate(env.members):
        song_errors = []
        song_successes = []

        for episode in range(10):
            seed = 50_000 + position * 1000 + episode

            observation, info = env.reset(
                seed=seed,
                options={"song_position": position},
            )

            music = env.current_env

            # Infer gains using target spectral features.
            estimated_gains, fit_error = solve_gains(
                music, observation
            )

            # Hidden gains used ONLY for reporting
            # whether the inverse solution is unique.
            gain_difference = float(np.mean(
                np.abs(
                    estimated_gains - music.target_gains
                )
            ))

            # Apply inferred gains through the normal
            # four-action-per-step environment interface.
            for step in range(music.max_steps):
                action = np.clip(
                    (estimated_gains - music.gains) / 2.0,
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

                if terminated or truncated:
                    break

            song_errors.append(info["error"])
            song_successes.append(float(info["success"]))

            all_errors.append(float(info["error"]))
            all_fit_errors.append(fit_error)
            all_gain_differences.append(gain_difference)
            all_successes.append(float(info["success"]))

            if position == 0 and episode == 0:
                print("EXAMPLE")
                print(
                    "True gains (diagnostic only):",
                    music.target_gains.round(3)
                )
                print(
                    "Estimated gains:",
                    estimated_gains.round(3)
                )
                print(
                    "Feature fit error:",
                    round(fit_error, 8)
                )
                print(
                    "Final episode error:",
                    round(info["error"], 8)
                )
                print()

        print(
            f"{member['song']} | "
            f"success={np.mean(song_successes) * 100:.1f}% | "
            f"error={np.mean(song_errors):.6f}"
        )

    print("\nSUMMARY — 40 VALIDATION TASKS")
    print(
        "Success rate:",
        f"{np.mean(all_successes) * 100:.1f}%"
    )
    print(
        "Mean final spectral error:",
        f"{np.mean(all_errors):.6f}"
    )
    print(
        "Mean optimizer feature-fit error:",
        f"{np.mean(all_fit_errors):.6f}"
    )
    print(
        "Mean absolute gain difference:",
        f"{np.mean(all_gain_differences):.4f} dB"
    )

    env.close()


if __name__ == "__main__":
    main()
