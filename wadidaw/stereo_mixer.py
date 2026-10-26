"""Differentiable gain and panning for aligned mono/stereo source audio."""

import math

import torch


def render_stereo(
    stems: list[torch.Tensor] | tuple[torch.Tensor, ...],
    gains_db: torch.Tensor,
    pans: torch.Tensor,
) -> torch.Tensor:
    """Render aligned stems to an unclipped ``(frames, 2)`` stereo tensor.

    Args:
        stems: Nonempty list/tuple of floating-point tensors. Each stem is
            ``(frames, 1)`` for mono or ``(frames, 2)`` for stereo, with at
            least one frame. All stems must have the same frame count.
        gains_db: ``(len(stems),)`` tensor of finite gains in dB. Linear
            amplitudes are ``10 ** (gains_db / 20)``.
        pans: ``(len(stems),)`` tensor of finite positions in ``[-1, 1]``:
            -1 is left, 0 is center, and +1 is right. Values outside this
            range are rejected, rather than silently clamped.

    All inputs must share a device and dtype (float32 or float64). The
    output retains that dtype/device. Inputs are never modified. There is
    no resampling, time alignment, normalization, or output clipping.
    Nonfinite inputs and numeric overflow raise ValueError.

    Channel count is determined ONLY by shape. ``(frames, 2)`` with a
    silent right channel is still stereo. Batching and padded channel
    layouts are unsupported: slice each source to its original channel
    count before calling; do not duplicate mono into two channels.

    Mono uses constant-power sine/cosine panning. Stereo uses the W3C
    StereoPanner routing law, preserving the original channel pair exactly
    at center and unity gain. Routing correlated stereo channels can boost
    power or cause cancellation; constant total power is a mono guarantee.
    Reference: https://www.w3.org/TR/webaudio-1.0/#stereopannernode-panning

    Gain, pan, and audio retain their autograd history. Stereo panning is
    continuous but has a derivative kink at center; at zero PyTorch uses
    the left branch. Check stereo pan gradients away from zero and bounds.
    Eager validity checks can synchronize device execution; this function
    does not promise support for tracing or torch.compile.
    """
    if not isinstance(stems, (list, tuple)):
        raise TypeError("stems must be a list or tuple of channel-explicit tensors.")
    if not stems:
        raise ValueError("At least one stem is required.")

    tensors = (*stems, gains_db, pans)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("Stems, gains_db, and pans must all be PyTorch tensors.")
    if any(tensor.dtype not in (torch.float32, torch.float64) for tensor in tensors):
        raise TypeError("Use float32 or float64 tensors for all inputs.")
    dtype, device = stems[0].dtype, stems[0].device
    if any(tensor.dtype != dtype or tensor.device != device for tensor in tensors):
        raise ValueError("All inputs must share a dtype and device.")
    if any(tensor.layout != torch.strided for tensor in tensors):
        raise TypeError("Only dense audio and parameter tensors are supported.")
    if gains_db.shape != (len(stems),) or pans.shape != (len(stems),):
        raise ValueError("Expected one gain and one pan per stem, each shaped (stems,).")
    if any(stem.ndim != 2 or stem.shape[1] not in (1, 2) or stem.shape[0] < 1 for stem in stems):
        raise ValueError("Each stem must have shape (frames, 1) or (frames, 2), with frames > 0.")
    frames = stems[0].shape[0]
    if any(stem.shape[0] != frames for stem in stems):
        raise ValueError("Stems must have aligned, equal frame counts.")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("Audio, gains, and pans must be finite.")
    if torch.any(pans.abs() > 1):
        raise ValueError("Pan positions must be in [-1, 1].")

    amplitudes = torch.pow(10.0, gains_db / 20.0)
    if not torch.isfinite(amplitudes).all():
        raise ValueError("Gain conversion overflows the input dtype.")

    mix = torch.zeros((frames, 2), dtype=dtype, device=device)
    for index, stem in enumerate(stems):
        pan = pans[index]
        if stem.shape[1] == 1:
            angle = (pan + 1) * (math.pi / 4)
            left = stem[:, 0] * torch.cos(angle)
            right = stem[:, 0] * torch.sin(angle)
        else:
            angle = pan * (math.pi / 2)
            sine, cosine = torch.sin(angle), torch.cos(angle)
            source_left, source_right = stem[:, 0], stem[:, 1]
            # For pan <= 0, cos((pan+1)*pi/2) = -sin(pan*pi/2).
            # This equivalent form gives sin(0)=0 and cos(0)=1 exactly,
            # avoiding cross-channel leakage from floating-point cos(pi/2).
            left = torch.where(pan <= 0, source_left - source_right * sine, source_left * cosine)
            right = torch.where(pan <= 0, source_right * cosine, source_right + source_left * sine)
        mix = mix + torch.stack((left, right), dim=1) * amplitudes[index]

    if not torch.isfinite(mix).all():
        raise ValueError("Stereo rendering overflows the input dtype.")
    return mix
