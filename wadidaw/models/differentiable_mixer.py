import torch
import torch.nn as nn


class DifferentiableBandMixer(nn.Module):
    """Reconstruct eight-band mix energy using observations."""

    PAIRS = (
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3),
    )

    def forward(self, observation, gains_db):
        # Double precision helps reduce reconstruction error.
        obs = torch.as_tensor(
            observation, dtype=torch.float64
        )
        gains = torch.as_tensor(
            gains_db,
            dtype=torch.float64,
            device=obs.device,
        )

        single = obs.ndim == 1

        if single:
            obs = obs.unsqueeze(0)
            gains = gains.unsqueeze(0)

        if obs.ndim != 2 or obs.shape[1] != 101:
            raise ValueError("Expected 101 observation features.")

        if gains.shape != (obs.shape[0], 4):
            raise ValueError("Expected four gains per observation.")

        # Decode individual stem energies.
        stem_db = obs[:, 20:52].reshape(-1, 4, 8) * 100.0
        stem_energy = torch.pow(10.0, stem_db / 10.0)

        # Decode normalized cross-stem interactions.
        correlations = obs[:, 52:100].reshape(-1, 6, 8)

        # dB -> linear amplitude.
        amplitudes = torch.pow(10.0, gains / 20.0)

        # Sum each stem's individual contribution.
        powers = (
            amplitudes.unsqueeze(-1).square() * stem_energy
        ).sum(dim=1)

        # Add cross-stem energy contributions.
        for pair_index, (i, j) in enumerate(self.PAIRS):
            cross_energy = (
                correlations[:, pair_index, :]
                * torch.sqrt(
                    stem_energy[:, i, :]
                    * stem_energy[:, j, :]
                )
            )

            powers = powers + (
                2.0
                * amplitudes[:, i, None]
                * amplitudes[:, j, None]
                * cross_energy
            )

        # Match the audio environment's spectral units.
        band_db = (
            10.0 * torch.log10(
                torch.clamp(powers, min=1e-12)
            )
        ).clamp(-120.0, 20.0)

        return band_db.squeeze(0) if single else band_db
