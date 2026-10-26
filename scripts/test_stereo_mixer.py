"""Synthetic DSP tests; no audio files, models, or experiment outputs are used."""

import unittest

import numpy as np
import torch

from wadidaw.stereo_mixer import render_stereo


def numpy_render_stereo(stems, gains_db, pans):
    """Independent float64 reference using the W3C's original angle mapping."""
    output = np.zeros((len(stems[0]), 2), dtype=np.float64)
    for stem, gain, pan in zip(stems, gains_db, pans):
        source = np.asarray(stem, dtype=np.float64)
        amplitude = np.power(10.0, gain / 20.0)
        if source.shape[1] == 1:
            angle = (pan + 1.0) * np.pi / 4.0
            left = source[:, 0] * np.cos(angle)
            right = source[:, 0] * np.sin(angle)
        elif pan <= 0:
            angle = (pan + 1.0) * np.pi / 2.0
            left = source[:, 0] + source[:, 1] * np.cos(angle)
            right = source[:, 1] * np.sin(angle)
        else:
            angle = pan * np.pi / 2.0
            left = source[:, 0] * np.cos(angle)
            right = source[:, 1] + source[:, 0] * np.sin(angle)
        output += amplitude * np.column_stack((left, right))
    return output


class StereoMixerTests(unittest.TestCase):
    def render_one(self, source, gain=0.0, pan=0.0, dtype=torch.float64):
        return render_stereo(
            [torch.as_tensor(source, dtype=dtype)],
            torch.tensor([gain], dtype=dtype),
            torch.tensor([pan], dtype=dtype),
        )

    def test_mono_left_center_and_right(self):
        mono = np.array([[1.0], [-0.5], [0.25], [0.0]])
        for pan, coefficients in ((-1, (1, 0)), (0, (2**-0.5, 2**-0.5)), (1, (0, 1))):
            with self.subTest(pan=pan):
                actual = self.render_one(mono, pan=pan)
                expected = mono * np.array(coefficients)
                np.testing.assert_allclose(actual.numpy(), expected, atol=1e-15, rtol=0)

    def test_mono_constant_power_across_pan_and_gain(self):
        source = torch.linspace(-0.7, 0.9, 31, dtype=torch.float64)[:, None]
        for gain in (-12.0, 0.0, 6.0):
            expected = (source.square().sum() * 10 ** (gain / 10)).numpy()
            for pan in np.linspace(-1, 1, 17):
                with self.subTest(gain=gain, pan=pan):
                    actual = self.render_one(source, gain=gain, pan=pan)
                    np.testing.assert_allclose(actual.square().sum().numpy(), expected, rtol=1e-14)

    def test_stereo_center_is_exact_identity_in_both_dtypes(self):
        # A tiny cosine residual would leak the large right signal into the left.
        source = [[0, 1e20], [-0.25, 0.875], [1e20, 0], [0.0, 0.0]]
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                tensor = torch.tensor(source, dtype=dtype)
                actual = self.render_one(tensor, dtype=dtype)
                self.assertTrue(torch.equal(actual, tensor))

    def test_stereo_endpoints_route_both_channels(self):
        source = np.array([[0.2, 0.7], [-0.4, 0.1], [0.5, -0.8]])
        summed = source.sum(axis=1)
        for pan, expected in ((-1, np.column_stack((summed, np.zeros(3)))),
                              (1, np.column_stack((np.zeros(3), summed)))):
            with self.subTest(pan=pan):
                np.testing.assert_allclose(self.render_one(source, pan=pan).numpy(), expected, atol=1e-15, rtol=0)

    def test_stereo_intermediate_routing(self):
        source = np.array([[1.0, 2.0], [-3.0, 0.5]])
        diagonal = np.sqrt(0.5)
        expected_left = np.column_stack((source[:, 0] + diagonal * source[:, 1], diagonal * source[:, 1]))
        expected_right = np.column_stack((diagonal * source[:, 0], source[:, 1] + diagonal * source[:, 0]))
        np.testing.assert_allclose(self.render_one(source, pan=-0.5).numpy(), expected_left, atol=1e-15)
        np.testing.assert_allclose(self.render_one(source, pan=0.5).numpy(), expected_right, atol=1e-15)

    def test_stereo_channel_swap_and_pan_negation_symmetry(self):
        source = np.random.default_rng(42).normal(size=(19, 2))
        for pan in (-1.0, -0.7, -0.1, 0.0, 0.1, 0.7, 1.0):
            with self.subTest(pan=pan):
                actual = self.render_one(source[:, ::-1].copy(), gain=-3.0, pan=-pan)
                expected = self.render_one(source, gain=-3.0, pan=pan).flip(1)
                torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-14)

    def test_gain_conversion_and_no_output_clipping_or_normalization(self):
        source = torch.tensor([[0.5, -1.0], [1.0, 0.25]], dtype=torch.float64)
        for gain, amplitude in ((0, 1), (20, 10), (-20, 0.1), (20 * np.log10(2), 2)):
            with self.subTest(gain=gain):
                actual = self.render_one(source, gain=gain)
                torch.testing.assert_close(actual, source * amplitude, atol=1e-14, rtol=1e-14)
        doubled = render_stereo([source, source], torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64))
        self.assertTrue(torch.equal(doubled, 2 * source))

    def test_mixed_sources_sum_without_mutating_inputs(self):
        mono = torch.tensor([[0.2], [0.7], [-0.5]], dtype=torch.float64)
        stereo = torch.tensor([[0.8, -0.1], [0.4, 0.9], [0.2, -0.3]], dtype=torch.float64)
        gains = torch.tensor([4.0, -2.0], dtype=torch.float64)
        pans = torch.tensor([-0.4, 0.6], dtype=torch.float64)
        before = [value.clone() for value in (mono, stereo, gains, pans)]
        actual = render_stereo((mono, stereo), gains, pans)
        expected = numpy_render_stereo([mono.numpy(), stereo.numpy()], gains.numpy(), pans.numpy())
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-15, rtol=1e-14)
        self.assertEqual(actual.shape, (3, 2))
        for original, saved in zip((mono, stereo, gains, pans), before):
            self.assertTrue(torch.equal(original, saved))

    def test_channel_count_is_explicit_and_never_inferred_from_silence(self):
        mono = torch.tensor([[1.0], [-0.5]], dtype=torch.float64)
        padded = torch.cat((mono, torch.zeros_like(mono)), dim=1)
        actual_mono = self.render_one(mono)
        actual_stereo = self.render_one(padded)
        self.assertTrue(torch.equal(actual_stereo, padded))
        torch.testing.assert_close(actual_mono[:, 0], mono[:, 0] / np.sqrt(2))
        torch.testing.assert_close(actual_mono[:, 1], mono[:, 0] / np.sqrt(2))

    def test_silence_single_frame_and_output_dtype(self):
        for dtype in (torch.float32, torch.float64):
            for frames in (1, 15):
                with self.subTest(dtype=dtype, frames=frames):
                    sources = [torch.zeros(frames, 1, dtype=dtype), torch.zeros(frames, 2, dtype=dtype)]
                    actual = render_stereo(sources, torch.tensor([12.0, -6.0], dtype=dtype), torch.tensor([-0.4, 0.7], dtype=dtype))
                    self.assertEqual(actual.shape, (frames, 2))
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(actual.device, sources[0].device)
                    self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_correlated_stereo_can_gain_power_at_endpoints(self):
        signal = np.array([0.25, -0.5, 1.0])
        source = np.column_stack((signal, signal))
        centered = self.render_one(source)
        left = self.render_one(source, pan=-1)
        np.testing.assert_allclose(left[:, 0].numpy(), 2 * signal, atol=1e-15)
        self.assertAlmostEqual(float(left.square().sum()), 2 * float(centered.square().sum()))

    def test_opposite_polarity_stereo_is_preserved_until_routed(self):
        signal = np.array([0.25, -0.5, 1.0])
        source = np.column_stack((signal, -signal))
        self.assertTrue(torch.equal(self.render_one(source), torch.from_numpy(source)))
        for pan in (-1, 1):
            np.testing.assert_allclose(self.render_one(source, pan=pan).numpy(), 0, atol=1e-15, rtol=0)

    def test_numpy_agreement_across_channel_layouts_and_dtypes(self):
        rng = np.random.default_rng(2026)
        for dtype, numpy_dtype, tolerance in ((torch.float32, np.float32, 8e-6),
                                               (torch.float64, np.float64, 3e-14)):
            maximum = 0.0
            for channels in ((1,), (2,), (1, 2, 1, 2)):
                for _ in range(12):
                    sources = [rng.normal(0, 0.4, (47, count)).astype(numpy_dtype) for count in channels]
                    gains = rng.uniform(-36, 18, len(channels)).astype(numpy_dtype)
                    pans = rng.uniform(-1, 1, len(channels)).astype(numpy_dtype)
                    actual = render_stereo([torch.from_numpy(source) for source in sources], torch.from_numpy(gains), torch.from_numpy(pans))
                    expected = numpy_render_stereo(sources, gains.astype(np.float64), pans.astype(np.float64))
                    difference = np.max(np.abs(actual.numpy() - expected))
                    maximum = max(maximum, float(difference))
                    np.testing.assert_allclose(actual.numpy(), expected, atol=tolerance, rtol=0)
            print(f"NumPy agreement ({dtype}): max absolute sample difference = {maximum:.3e}")

    def test_noncontiguous_sources(self):
        source = torch.arange(28, dtype=torch.float64).reshape(2, 14).T
        self.assertFalse(source.is_contiguous())
        torch.testing.assert_close(self.render_one(source, pan=-0.3), self.render_one(source.contiguous(), pan=-0.3))

    def test_invalid_source_containers_shapes_and_parameters(self):
        mono = torch.ones(3, 1)
        gain, pan = torch.zeros(1), torch.zeros(1)
        invalid = [
            ([], torch.zeros(0), torch.zeros(0), ValueError),
            (torch.ones(1, 3, 1), gain, pan, TypeError),
            ([np.ones((3, 1))], gain, pan, TypeError),
            ([mono], [0.0], pan, TypeError),
            ([mono], gain, [0.0], TypeError),
            ([torch.ones(3)], gain, pan, ValueError),
            ([torch.ones(1, 3, 2)], gain, pan, ValueError),
            ([torch.ones(0, 1)], gain, pan, ValueError),
            ([torch.ones(3, 0)], gain, pan, ValueError),
            ([torch.ones(3, 3)], gain, pan, ValueError),
            ([mono, torch.ones(4, 2)], torch.zeros(2), torch.zeros(2), ValueError),
            ([mono], torch.tensor(0.0), pan, ValueError),
            ([mono], gain, torch.tensor(0.0), ValueError),
            ([mono], torch.zeros(1, 1), pan, ValueError),
            ([mono], gain, torch.zeros(2), ValueError),
            ([mono], gain, torch.tensor([-1.01]), ValueError),
            ([mono], gain, torch.tensor([1.01]), ValueError),
        ]
        for sources, gains, pans, error in invalid:
            with self.subTest(sources_type=type(sources), gains=gains, pans=pans):
                with self.assertRaises(error):
                    render_stereo(sources, gains, pans)

    def test_invalid_dtypes_and_devices_are_rejected_without_implicit_casts(self):
        mono, gain, pan = torch.ones(3, 1), torch.zeros(1), torch.zeros(1)
        invalid = [
            ([mono.long()], gain, pan, TypeError),
            ([mono.to(torch.complex64)], gain, pan, TypeError),
            ([mono.half()], gain.half(), pan.half(), TypeError),
            ([mono], gain.long(), pan, TypeError),
            ([mono], gain, pan.long(), TypeError),
            ([mono], gain.double(), pan, ValueError),
            ([mono, mono.double()], torch.zeros(2), torch.zeros(2), ValueError),
            ([mono], gain, pan.to(device="meta"), ValueError),
        ]
        for sources, gains, pans, error in invalid:
            with self.subTest(error=error, dtype=gains.dtype, device=pans.device):
                with self.assertRaises(error):
                    render_stereo(sources, gains, pans)

    def test_nonfinite_inputs_and_numeric_overflow_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            for position in range(3):
                sources = [torch.ones(3, 1)]
                gains, pans = torch.zeros(1), torch.zeros(1)
                (sources[0] if position == 0 else gains if position == 1 else pans).fill_(value)
                with self.subTest(value=value, position=position), self.assertRaises(ValueError):
                    render_stereo(sources, gains, pans)
        with self.assertRaises(ValueError):
            self.render_one([[0.0], [0.0]], gain=10000)
        maximum = torch.finfo(torch.float32).max
        source = torch.full((2, 2), maximum)
        with self.assertRaises(ValueError):
            render_stereo([source, source], torch.zeros(2), torch.zeros(2))

    def test_gain_and_pan_gradients_against_independent_numpy_differences(self):
        rng = np.random.default_rng(417)
        sources = [rng.normal(0, 0.3, (17, channels)) for channels in (1, 2, 2)]
        target = rng.normal(0, 0.4, (17, 2))
        gains = np.array([-5.2, 2.1, -1.3])
        pans = np.array([0.1, -0.4, 0.65])
        gain_tensor = torch.tensor(gains, requires_grad=True)
        pan_tensor = torch.tensor(pans, requires_grad=True)
        actual = render_stereo([torch.from_numpy(source) for source in sources], gain_tensor, pan_tensor)
        (actual - torch.from_numpy(target)).square().mean().backward()

        def reference_loss(candidate_gains, candidate_pans):
            return np.mean((numpy_render_stereo(sources, candidate_gains, candidate_pans) - target) ** 2)

        epsilon = 1e-5
        for name, parameters, gradient in (("gain", gains, gain_tensor.grad), ("pan", pans, pan_tensor.grad)):
            numerical = np.zeros_like(parameters)
            for index in range(len(parameters)):
                positive, negative = parameters.copy(), parameters.copy()
                positive[index] += epsilon
                negative[index] -= epsilon
                if name == "gain":
                    numerical[index] = (reference_loss(positive, pans) - reference_loss(negative, pans)) / (2 * epsilon)
                else:
                    numerical[index] = (reference_loss(gains, positive) - reference_loss(gains, negative)) / (2 * epsilon)
            analytical = gradient.detach().numpy()
            self.assertTrue(np.isfinite(analytical).all())
            self.assertTrue(np.all(np.abs(numerical) > 1e-5))
            np.testing.assert_allclose(analytical, numerical, atol=2e-9, rtol=2e-6)
            print(f"{name.capitalize()} gradient: max absolute finite-difference error = {np.max(np.abs(analytical - numerical)):.3e}")

    def test_torch_gradcheck_for_mixed_sources(self):
        generator = torch.Generator().manual_seed(123)
        sources = [torch.randn(4, channels, generator=generator, dtype=torch.float64) for channels in (1, 2, 2)]
        gains = torch.tensor([-3.0, 1.0, -2.0], dtype=torch.float64, requires_grad=True)
        pans = torch.tensor([0.0, -0.35, 0.6], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda gain, pan: render_stereo(sources, gain, pan),
            (gains, pans), eps=1e-6, atol=1e-6, rtol=1e-4,
        ))

    def test_nonleaf_parameters_and_source_audio_keep_autograd_history(self):
        source = torch.tensor([[0.1, 0.3], [-0.7, 0.2]], dtype=torch.float64, requires_grad=True)
        raw_gain = torch.tensor([1.2], dtype=torch.float64, requires_grad=True)
        raw_pan = torch.tensor([-0.4], dtype=torch.float64, requires_grad=True)
        output = render_stereo([source], raw_gain * 2, torch.tanh(raw_pan))
        weights = torch.tensor([[0.7, -0.3], [0.2, 0.8]], dtype=torch.float64)
        (output * weights).sum().backward()
        for tensor in (source, raw_gain, raw_pan):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0)

    def test_stereo_routing_is_continuous_at_center(self):
        source = np.array([[0.2, -0.6], [0.7, 0.1]])
        center = self.render_one(source)
        for pan in (-1e-8, 1e-8):
            torch.testing.assert_close(self.render_one(source, pan=pan), center, atol=2e-8, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
