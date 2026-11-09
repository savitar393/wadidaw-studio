"""Unit and integration tests for the MUSDB18-HQ multimodal data & prompt pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from wadidaw.data import (
    ACTIVE_THRESHOLD_DB,
    EXCERPT_FRAMES_16K,
    MIXING_PROFILES,
    MUSDB_ROOT,
    SPLIT_MANIFEST_PATH,
    STEM_NAMES,
    MusdbMultimodalDataset,
    apply_gain_to_logmel_db,
    build_human_test_samples,
    build_split_samples,
    compute_band_grams_16k,
    compute_gain_targets,
    create_musdb_split_manifest,
    extract_and_cache_song_excerpts,
    extract_logmel_stems,
    generate_human_test_prompts,
    generate_paraphrases,
    load_excerpt_stems_44k,
    measure_loudness_db,
    normalize_balance_target,
    normalize_logmel_for_model,
    parse_prompt_to_balance,
    render_gain_staged_mix,
    resample_to_16k_mono,
    validate_split_manifest,
)


class MusdbDataPipelineTests(unittest.TestCase):
    def test_split_manifest_counts_and_disjoint_artists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "musdb18hq_v1.json"
            manifest = create_musdb_split_manifest(
                musdb_root=MUSDB_ROOT,
                output_path=out_path,
                seed=2026,
            )
            validate_split_manifest(manifest)
            splits = manifest["splits"]
            self.assertEqual(len(splits["train"]), 80)
            self.assertEqual(len(splits["validation"]), 20)
            self.assertEqual(len(splits["test"]), 50)

            train_songs = {m["song"] for m in splits["train"]}
            val_songs = {m["song"] for m in splits["validation"]}
            test_songs = {m["song"] for m in splits["test"]}
            self.assertFalse(train_songs & val_songs)
            self.assertFalse(train_songs & test_songs)
            self.assertFalse(val_songs & test_songs)

            train_artists = {m["artist"] for m in splits["train"]}
            val_artists = {m["artist"] for m in splits["validation"]}
            self.assertFalse(train_artists & val_artists)

    def test_dokumentasi_section_5_4_worked_example(self) -> None:
        # Section 5.4 worked example in dokumentasi.md:
        # g0 = [0, 0, 0, 0], L = [-18, -12, -15, -20], r = [+3, 0, -2, -1]
        # -> L_ref = -16.25, g* = [4.75, -4.25, -3.25, 2.75]
        l_raw = np.array([-18.0, -12.0, -15.0, -20.0], dtype=np.float32)
        r = normalize_balance_target([3.0, 0.0, -2.0, -1.0])
        g0 = np.zeros(4, dtype=np.float32)

        res = compute_gain_targets(l_raw, r, g0)
        self.assertAlmostEqual(res["l_ref_db"], -16.25, places=5)
        np.testing.assert_allclose(
            res["gains_db"],
            np.array([4.75, -4.25, -3.25, 2.75], dtype=np.float32),
            atol=1e-5,
        )
        # Final loudness L* = L + g* has mean -16.25 and relative offsets r
        l_final = res["loudness_db"] + res["gains_db"]
        np.testing.assert_allclose(
            l_final,
            np.array([-13.25, -16.25, -18.25, -17.25], dtype=np.float32),
            atol=1e-5,
        )

    def test_silent_stem_is_masked_and_excluded_from_lref(self) -> None:
        # Vocals silent (-65 dB <= -45 dB), other 3 stems active at [-12, -15, -18]
        l_raw = np.array([-65.0, -12.0, -15.0, -18.0], dtype=np.float32)
        r = normalize_balance_target([3.0, 0.0, 0.0, 0.0])
        res = compute_gain_targets(l_raw, r)
        self.assertEqual(list(res["active_mask"]), [False, True, True, True])
        # Silent vocals stem gets 0.0 dB gain instead of +50 dB boost
        self.assertEqual(float(res["gains_db"][0]), 0.0)
        # L_ref is mean of active stems (-12, -15, -18) = -15.0 dB
        self.assertAlmostEqual(res["l_ref_db"], -15.0, places=5)

    def test_prompts_zero_sum_and_split_template_isolation(self) -> None:
        for profile in MIXING_PROFILES:
            r = profile.balance_target
            self.assertEqual(r.shape, (4,))
            self.assertAlmostEqual(float(np.sum(r)), 0.0, places=5)

            train_p = generate_paraphrases(profile, "train", n_paraphrases=4)
            val_p = generate_paraphrases(profile, "validation", n_paraphrases=4)
            test_p = generate_paraphrases(profile, "test", n_paraphrases=4)

            train_texts = {x["prompt"] for x in train_p}
            val_texts = {x["prompt"] for x in val_p}
            test_texts = {x["prompt"] for x in test_p}
            self.assertEqual(len(train_texts), 4)
            self.assertFalse(train_texts & val_texts)
            self.assertFalse(train_texts & test_texts)
            self.assertFalse(val_texts & test_texts)

        human_prompts = generate_human_test_prompts(count=200, seed=2026)
        self.assertEqual(len(human_prompts), 200)
        for item in human_prompts:
            self.assertAlmostEqual(float(np.sum(item["balance_target"])), 0.0, places=3)

        parsed = parse_prompt_to_balance("vokal paling menonjol, bass agak mundur, drum seimbang")
        self.assertEqual(parsed["raw_slots_db"]["vocals"], 5.0)
        self.assertEqual(parsed["raw_slots_db"]["bass"], -2.0)
        self.assertEqual(parsed["raw_slots_db"]["drums"], 0.0)
        self.assertAlmostEqual(float(np.sum(parsed["balance_target"])), 0.0, places=5)

    def test_logmel_shape_and_gain_shift_linearity(self) -> None:
        rng = np.random.default_rng(42)
        t = np.arange(EXCERPT_FRAMES_16K, dtype=np.float32) / 16000.0
        freqs = (440.0, 120.0, 80.0, 1000.0)
        amps = (0.2, 0.3, 0.25, 0.15)
        stems_16k = np.stack(
            [a * np.sin(2 * np.pi * f * t) + 0.005 * rng.normal(size=len(t)) for f, a in zip(freqs, amps)],
            axis=0,
        ).astype(np.float32)

        logmel_raw = extract_logmel_stems(stems_16k)
        self.assertEqual(logmel_raw.shape, (4, 64, 800))

        g0_db = np.array([3.0, -2.5, 1.5, -1.0], dtype=np.float32)
        linear_scale = np.power(10.0, g0_db / 20.0)[:, None]
        logmel_from_scaled_wave = extract_logmel_stems(stems_16k * linear_scale)
        logmel_from_db_shift = apply_gain_to_logmel_db(logmel_raw, g0_db)

        # On active bins (> -60 dB), shifting log-mel in dB matches scaling the waveform directly
        active_bins = logmel_raw > -60.0
        max_diff = float(np.max(np.abs(logmel_from_scaled_wave[active_bins] - logmel_from_db_shift[active_bins])))
        self.assertLess(max_diff, 1e-4)

        norm = normalize_logmel_for_model(logmel_from_db_shift)
        self.assertEqual(norm.shape, (4, 64, 800))
        self.assertTrue(np.all((norm >= -1.0) & (norm <= 1.0)))

        grams = compute_band_grams_16k(stems_16k)
        self.assertEqual(grams.shape, (8, 4, 4))

    def test_end_to_end_song_cache_dataset_and_stereo_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = create_musdb_split_manifest(
                musdb_root=MUSDB_ROOT,
                output_path=tmp_path / "splits.json",
                seed=2026,
            )
            first_song = manifest["splits"]["train"][0]
            cached = extract_and_cache_song_excerpts(
                song_member=first_song,
                split="train",
                song_number=1,
                musdb_root=MUSDB_ROOT,
                cache_root=tmp_path / "cache",
                n_excerpts=2,
            )
            self.assertEqual(len(cached), 2)

            samples = build_split_samples(
                cached_excerpts=cached,
                split="train",
                n_prompts_per_excerpt=4,
                seed=2026,
            )
            self.assertEqual(len(samples), 8)

            ds = MusdbMultimodalDataset(samples)
            item = ds[0]
            self.assertEqual(tuple(item["audio"].shape), (4, 64, 800))
            self.assertEqual(tuple(item["logmel_db"].shape), (4, 64, 800))
            self.assertEqual(tuple(item["loudness_db"].shape), (4,))
            self.assertEqual(tuple(item["balance_target"].shape), (4,))
            self.assertEqual(tuple(item["gains_db"].shape), (4,))
            self.assertEqual(tuple(item["active_mask"].shape), (4,))
            self.assertIsInstance(item["prompt"], str)

            # Verify 44.1 kHz stereo rendering on the same excerpt
            song_dir = MUSDB_ROOT / first_song["relative_dir"]
            stems_44k = load_excerpt_stems_44k(
                song_dir,
                start_sec=cached[0]["excerpt_start_sec"],
                duration_sec=1.0,
            )
            rendered = render_gain_staged_mix(
                stems_44k,
                gains_db=item["gains_db"].numpy(),
                active_mask=item["active_mask"].numpy(),
            )
            self.assertEqual(rendered["mix"].shape, (44100, 2))
            self.assertTrue(np.isfinite(rendered["mix"]).all())
            self.assertEqual(set(rendered["effective_gains_db"].keys()), set(STEM_NAMES))
            self.assertIn("clipping_rate", rendered["metrics"])
            self.assertIn("peak_dbfs", rendered["metrics"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
