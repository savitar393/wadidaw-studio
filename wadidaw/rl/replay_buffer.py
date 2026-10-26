import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity=50_000,
        obs_dim=12,
        action_dim=4,
        seed=42,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.observations = np.zeros(
            (capacity, obs_dim), dtype=np.float32
        )
        self.actions = np.zeros(
            (capacity, action_dim), dtype=np.float32
        )
        self.rewards = np.zeros(
            (capacity, 1), dtype=np.float32
        )
        self.next_observations = np.zeros(
            (capacity, obs_dim), dtype=np.float32
        )
        self.terminated = np.zeros(
            (capacity, 1), dtype=np.float32
        )
        self.truncated = np.zeros(
            (capacity, 1), dtype=np.float32
        )

        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminated,
        truncated,
    ):
        observation = np.asarray(
            observation, dtype=np.float32
        )
        action = np.asarray(action, dtype=np.float32)
        next_observation = np.asarray(
            next_observation, dtype=np.float32
        )

        if (
            observation.shape != (self.obs_dim,)
            or next_observation.shape != (self.obs_dim,)
            or action.shape != (self.action_dim,)
        ):
            raise ValueError("Invalid transition dimensions.")

        i = self.position

        self.observations[i] = observation
        self.actions[i] = action
        self.rewards[i, 0] = reward
        self.next_observations[i] = next_observation
        self.terminated[i, 0] = float(terminated)
        self.truncated[i, 0] = float(truncated)

        # Circular buffer: overwrite oldest experiences
        # after reaching maximum capacity.
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=32):
        if self.size == 0:
            raise ValueError("Replay buffer is empty.")

        indices = self.rng.integers(
            0, self.size, size=batch_size
        )

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "terminated": self.terminated[indices],
            "truncated": self.truncated[indices],
        }

    def __len__(self):
        return self.size
