import torch
import torch.nn as nn


class InverseGainNet(nn.Module):
    def __init__(self, obs_dim=101, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, observation):
        # Output four absolute gain settings in dB.
        # The environment's target gains lie in [-8, 8].
        # We allow [-10, 10] to avoid saturating at targets.
        return 10.0 * torch.tanh(
            self.network(observation)
        )
