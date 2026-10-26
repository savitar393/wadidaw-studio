
import numpy as np
import torch

from wadidaw.mixing_env import MixingEnv
from wadidaw.rl.replay_buffer import ReplayBuffer
from wadidaw.rl.sac import SACAgent


def parameters_changed(before, network):
    return any(
        not torch.equal(old, new)
        for old, new in zip(before, network.parameters())
    )


def snapshot(network):
    return [
        param.detach().clone()
        for param in network.parameters()
    ]


def main():
    torch.manual_seed(42)

    env = MixingEnv()
    env.action_space.seed(42)

    buffer = ReplayBuffer(capacity=512)

    agent = SACAgent()

    observation, _ = env.reset(seed=42)

    # Collect random experiences.
    for step in range(256):
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        buffer.add(
            observation,
            action,
            reward,
            next_obs,
            terminated,
            truncated,
        )

        if terminated or truncated:
            observation, _ = env.reset()
        else:
            observation = next_obs

    # Record weights before learning.
    actor_before = snapshot(agent.actor)
    critic_before = snapshot(agent.q1)
    target_before = snapshot(agent.target_q1)

    # One actual SAC training update.
    batch = buffer.sample(batch_size=64)
    metrics = agent.update(batch)

    # Verify gradients changed the parameters.
    actor_changed = parameters_changed(
        actor_before, agent.actor
    )
    critic_changed = parameters_changed(
        critic_before, agent.q1
    )
    target_changed = parameters_changed(
        target_before, agent.target_q1
    )

    assert actor_changed
    assert critic_changed
    assert target_changed
    assert all(np.isfinite(v) for v in metrics.values())

    print("SAC update completed!")
    print(f"Critic loss: {metrics['critic_loss']:.6f}")
    print(f"Actor loss: {metrics['actor_loss']:.6f}")
    print(f"Mean target Q: {metrics['mean_target_q']:.6f}")

    print("\nActor parameters changed:", actor_changed)
    print("Critic parameters changed:", critic_changed)
    print("Target parameters changed:", target_changed)

    # Check that the agent can produce valid actions.
    action = agent.act(observation, deterministic=True)

    print("\nPredicted action:", action)
    assert action.shape == (4,)
    assert np.isfinite(action).all()
    assert np.all(np.abs(action) <= 1.0)

    env.close()


if __name__ == "__main__":
    main()
