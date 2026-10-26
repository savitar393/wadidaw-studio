import numpy as np

from wadidaw.mixing_env import MixingEnv


def evaluate(policy_name, num_episodes=100):
    env = MixingEnv()

    successes = []
    final_errors = []
    total_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        seed = 42 + episode

        obs, info = env.reset(seed=seed)
        env.action_space.seed(seed)

        total_reward = 0.0

        for step in range(env.max_steps):

            if policy_name == "no_op":
                action = np.zeros(4, dtype=np.float32)

            elif policy_name == "random":
                action = env.action_space.sample()

            elif policy_name == "oracle":
                action = np.clip(
                    (env.target_gains - env.gains) / 2.0,
                    -1.0,
                    1.0,
                ).astype(np.float32)

            else:
                raise ValueError("Unknown policy")

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

            if terminated or truncated:
                break

        successes.append(terminated)
        final_errors.append(info["error"])
        total_rewards.append(total_reward)
        episode_lengths.append(step + 1)

    env.close()

    return {
        "success_rate": np.mean(successes) * 100,
        "mean_error": np.mean(final_errors),
        "mean_reward": np.mean(total_rewards),
        "mean_steps": np.mean(episode_lengths),
    }


if __name__ == "__main__":
    for name in ["no_op", "random", "oracle"]:
        result = evaluate(name)

        print(f"\n{name.upper()}")
        print(f"Success rate: {result['success_rate']:.1f}%")
        print(f"Mean final error: {result['mean_error']:.6f}")
        print(f"Mean reward: {result['mean_reward']:.4f}")
        print(f"Mean steps: {result['mean_steps']:.2f}")
