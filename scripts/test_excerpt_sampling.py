import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from gymnasium.utils.env_checker import check_env

from wadidaw import multi_song_env, music_env
from wadidaw.multi_song_env import MultiSongMixingEnv
from wadidaw.music_env import MusicMixingEnv


class ExcerptSamplingTests(unittest.TestCase):
    """Small real audio fixtures; no dataset or model checkpoints required."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        time = np.arange(64_000) / 16_000
        frequencies = (50, 110, 220, 440, 900, 1800, 3600, 6500)
        base = sum(np.sin(2 * np.pi * f * time) for f in frequencies) / 8

        for song in range(3):
            stems = self.root / f"Track{song + 1:05d}" / "stems"
            stems.mkdir(parents=True)
            for stem in range(7):
                audio = (0.1 + 0.02 * stem) * (0.3 + 0.7 * time / 4) * (
                    0.7 * base
                    + 0.3 * np.sin(2 * np.pi * 300 * time + stem * np.pi / 3)
                )
                if stem == 2:
                    audio = np.column_stack((0.5 * audio, 1.5 * audio))
                elif stem == 4:
                    audio *= 1e-6  # Below the active-stem threshold.
                elif stem == 5:
                    audio *= 0
                elif stem == 6:
                    audio *= 0.01  # Active, but quieter than the four selected stems.
                sf.write(stems / f"S{stem:02d}.wav", audio, 16_000, subtype="FLOAT")

        self.manifest = {
            "seed": 2026,
            "splits": {
                split: [{"index": i, "song": f"Track{i + 1:05d}"}]
                for i, split in enumerate(("train", "validation", "test"))
            },
        }
        self.manifest_path = self.root / "splits.json"
        self.write_manifest()
        dataset_patch = patch.object(music_env, "DATASET", self.root)
        manifest_patch = patch.object(multi_song_env, "MANIFEST", self.manifest_path)
        dataset_patch.start()
        manifest_patch.start()
        self.addCleanup(dataset_patch.stop)
        self.addCleanup(manifest_patch.stop)

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def music(self, **kwargs):
        env = MusicMixingEnv(duration_seconds=kwargs.pop("duration_seconds", 1), **kwargs)
        self.addCleanup(env.close)
        return env

    def multi(self, **kwargs):
        kwargs.setdefault("duration_seconds", 1)
        kwargs.setdefault("excerpt_fractions", (0.2, 0.5, 0.8))
        env = MultiSongMixingEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def assert_observation(self, env, observation):
        self.assertEqual(observation.shape, (101,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.isfinite(observation).all())
        self.assertTrue(env.observation_space.contains(observation))

    def solve(self, env, music):
        for _ in range(music.max_steps):
            action = np.clip((music.target_gains - music.gains) / 2, -1, 1)
            observation, reward, terminated, truncated, info = env.step(action)
            self.assert_observation(env, observation)
            self.assertTrue(np.isfinite(reward))
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertLess(info["error"], 0.001)
        self.assertLessEqual(music.steps, 4)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(4, dtype=np.float32))
        return info

    def test_excerpt_offsets_and_four_loudest_active_stems(self):
        stems_dir = self.root / "Track00001" / "stems"
        positions = ((0, 0), (0.12345, 5926), (0.2, 9600), (0.5, 24000), (0.8, 38400), (1, 48000))
        for fraction, start in positions:
            with self.subTest(fraction=fraction):
                env = self.music(excerpt_fraction=fraction)
                self.assertEqual(env.excerpt_start_seconds, start / 16_000)
                self.assertEqual(env.stems.shape, (4, 16_000))
                candidates = []
                for path in sorted(stems_dir.glob("*.wav")):
                    audio, _ = sf.read(path, start=start, frames=16_000, dtype="float32")
                    if audio.ndim == 2:
                        audio = audio.mean(axis=1)
                    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
                    if rms > 1e-5:
                        candidates.append((rms, path.name, audio))
                candidates.sort(key=lambda item: item[0], reverse=True)
                self.assertEqual(env.stem_names, [item[1] for item in candidates[:4]])
                self.assertEqual(len(set(env.stem_names)), 4)
                np.testing.assert_array_equal(env.stems, np.stack([item[2] for item in candidates[:4]]))

    def test_common_length_and_overlong_excerpt(self):
        path = self.root / "Track00001" / "stems" / "S03.wav"
        audio, rate = sf.read(path, frames=56_000)
        sf.write(path, audio, rate, subtype="FLOAT")
        env = self.music(excerpt_fraction=1)
        self.assertEqual(env.excerpt_start_seconds, 2.5)
        env = self.music(duration_seconds=8, excerpt_fraction=0.8)
        self.assertEqual(env.excerpt_start_seconds, 0)
        self.assertEqual(env.stems.shape, (4, 56_000))

    def test_invalid_fractions_and_insufficient_active_stems(self):
        for fraction in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                self.music(excerpt_fraction=fraction)
        for name in ("S03.wav", "S06.wav"):
            sf.write(self.root / "Track00001" / "stems" / name, np.zeros(64_000), 16_000)
        with self.assertRaisesRegex(RuntimeError, "four active stems"):
            self.music()

    def test_observation_layout_and_oracle_at_every_excerpt(self):
        for fraction in (0.2, 0.5, 0.8):
            env = self.music(excerpt_fraction=fraction)
            for seed in (0, 42, 2026):
                with self.subTest(fraction=fraction, seed=seed):
                    observation, _ = env.reset(seed=seed)
                    self.assert_observation(env, observation)
                    self.assertEqual(env.stem_band_features.shape, (4, 8))
                    self.assertEqual(env.cross_band_features.shape, (6, 8))
                    np.testing.assert_allclose(observation[20:52], env.stem_band_features.ravel() / 100)
                    np.testing.assert_array_equal(observation[52:100], env.cross_band_features.ravel())
                    self.assertEqual(observation[-1], 1)
                    self.solve(env, env)

    def test_spectral_grams_match_fft_of_rendered_audio(self):
        for fraction in (0.2, 0.5, 0.8):
            env = self.music(excerpt_fraction=fraction)
            np.testing.assert_allclose(env.band_grams, env.band_grams.transpose(0, 2, 1), atol=1e-15)
            self.assertGreater(np.max(np.abs(env.cross_band_features)), 0.1)
            for gains in ((0, 0, 0, 0), (1, -3, 5, -2), (-18, 12, -8, 6)):
                with self.subTest(fraction=fraction, gains=gains):
                    gains = np.asarray(gains, dtype=np.float32)
                    rendered = env.render_mix(gains)
                    spectrum = np.fft.rfft(rendered.astype(np.float64))
                    frequencies = np.fft.rfftfreq(len(rendered), 1 / 16_000)
                    edges = (0, 80, 160, 320, 640, 1250, 2500, 5000, 8001)
                    power = np.array([
                        np.sum(np.abs(spectrum[(frequencies >= low) & (frequencies < high)]) ** 2) / len(rendered) ** 2
                        for low, high in zip(edges[:-1], edges[1:])
                    ])
                    expected = np.clip(10 * np.log10(np.maximum(power, 1e-12)), -120, 20)
                    np.testing.assert_allclose(env._features(gains), expected, atol=0.001, rtol=0)

    def test_explicit_excerpt_selection_and_step_metadata(self):
        env = self.multi()
        for position, start in enumerate((0.6, 1.5, 2.4)):
            observation, info = env.reset(seed=42, options={"song_position": 0, "excerpt_position": position})
            self.assert_observation(env, observation)
            self.assertEqual(info["song"], "Track00001")
            self.assertEqual(info["song_index"], 0)
            self.assertEqual(info["song_position"], 0)
            self.assertEqual(info["excerpt_position"], position)
            self.assertEqual(info["excerpt_fraction"], (0.2, 0.5, 0.8)[position])
            self.assertEqual(info["excerpt_start_seconds"], start)
            final_info = self.solve(env, env.current_env)
            for key in ("song", "song_index", "song_position", "excerpt_position", "excerpt_fraction", "excerpt_start_seconds"):
                self.assertEqual(final_info[key], info[key])

    def test_default_midpoint_preserves_existing_seeded_targets(self):
        env = MultiSongMixingEnv()
        self.addCleanup(env.close)
        self.assertEqual(len(env.envs), len(env.members))
        observation, info = env.reset(seed=42, options={"song_position": 0})
        target_seed = int(np.random.default_rng(42).integers(0, 2**31 - 1))
        expected, _ = self.music(duration_seconds=8).reset(seed=target_seed)
        np.testing.assert_array_equal(observation, expected)

    def test_seeded_sampling_is_reproducible_and_covers_excerpts(self):
        self.manifest["splits"]["train"].append(self.manifest["splits"]["validation"].pop())
        self.write_manifest()
        first, second = self.multi(), self.multi()
        visited = set()
        for episode in range(40):
            seed = 2026 if episode == 0 else None
            observation, info = first.reset(seed=seed)
            other, other_info = second.reset(seed=seed)
            np.testing.assert_array_equal(observation, other)
            np.testing.assert_array_equal(first.current_env.target_gains, second.current_env.target_gains)
            self.assertEqual(info["song"], other_info["song"])
            self.assertEqual(info["excerpt_position"], other_info["excerpt_position"])
            visited.add((info["song"], info["excerpt_position"]))
        self.assertEqual(len(visited), 6)
        # Unrelated resets must not affect a fixed evaluation task.
        options = {"song_position": 1, "excerpt_position": 2}
        expected, _ = first.reset(seed=50000, options=options)
        first.reset(seed=7)
        actual, _ = first.reset(seed=50000, options=options)
        np.testing.assert_array_equal(actual, expected)

    def test_training_never_loads_other_partition_audio(self):
        # If the loader touches a validation/test stem directory, it will fail.
        for song in ("Track00002", "Track00003"):
            for path in (self.root / song / "stems").glob("*.wav"):
                path.write_bytes(b"not audio")
        env = self.multi()
        for seed in range(10):
            _, info = env.reset(seed=seed)
            self.assertEqual(info["song"], "Track00001")

    def test_overlapping_song_names_or_indices_are_rejected(self):
        for key, value in (("song", "Track00001"), ("index", 0)):
            with self.subTest(key=key):
                original = self.manifest["splits"]["validation"][0][key]
                self.manifest["splits"]["validation"][0][key] = value
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "[Oo]verlap"):
                    self.multi()
                self.manifest["splits"]["validation"][0][key] = original

    def test_invalid_excerpt_configuration_and_positions(self):
        for fractions in ((), (np.nan,), (-0.1,), (1.1,), (0.5, 0.5)):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                self.multi(excerpt_fractions=fractions)
        env = self.multi()
        for options in ({"song_position": -1}, {"song_position": 1}, {"excerpt_position": -1}, {"excerpt_position": 3}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                env.reset(seed=42, options=options)
        self.manifest["splits"]["train"] = []
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.multi()

    def test_dataset_order_mismatch_is_rejected(self):
        self.manifest["splits"]["train"][0]["song"] = "Track99999"
        self.write_manifest()
        with self.assertRaisesRegex(RuntimeError, "Dataset mismatch"):
            self.multi()

    def test_gymnasium_contract(self):
        check_env(self.music(), skip_render_check=True)
        check_env(self.multi(), skip_render_check=True)


if __name__ == "__main__":
    unittest.main()
