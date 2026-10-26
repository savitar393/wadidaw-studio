from wadidaw.mixing_env import MixingEnv
from wadidaw.rl.replay_buffer import ReplayBuffer


def main():
    env = MixingEnv()

    buffer = ReplayBuffer(
        capacity=50_000,
        obs_dim=12,
        action_dim=4,
        seed=42,
    )

    env.action_space.seed(42)

    episode = 0
    observation, info = env.reset(seed=42)

    for step in range(1000):
        # Random exploration for now.
        action = env.action_space.sample()

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        # Save experience.
        buffer.add(
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
        )

        observation = next_observation

        if terminated or truncated:
            episode += 1
            observation, info = env.reset(
                seed=42 + episode
            )

    print("Transitions stored:", len(buffer))
    print("Completed episodes:", episode)

    # Retrieve a random batch.
    batch = buffer.sample(batch_size=32)

    print("\nSampled batch shapes:")
    for name, array in batch.items():
        print(f"{name}: {array.shape}")

    print("\nFirst sampled reward:", batch["rewards"][0])
    print("First sampled action:", batch["actions"][0])

    env.close()


if __name__ == "__main__":
    main()
