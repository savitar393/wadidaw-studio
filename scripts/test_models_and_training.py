"""Unit and integration tests for multimodal models M1–M5, loss functions, and evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wadidaw.data import (
    MusdbMultimodalDataset,
    build_split_samples,
    normalize_balance_target,
)
from wadidaw.models import (
    MODEL_REGISTRY,
    StructuredTwoBranchModel,
    build_model,
    compute_multimodal_loss,
)
from wadidaw.training import (
    TrainConfig,
    evaluate_model,
    make_baseline_predictor,
    train_one_run,
)


class MultimodalModelsAndTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.batch_size = 4
        self.audio = torch.randn(self.batch_size, 4, 64, 800, dtype=torch.float32) * 0.25
        self.prompts = [
            "vokal paling menonjol, bass agak mundur, drum seimbang",
            "buat drum lebih maju, bass sedikit lebih maju, instrumen pengiring lebih mundur",
            "atur campuran dengan vokal seimbang, drum seimbang, bass seimbang",
            "tolong pastikan instrumen pengiring paling menonjol, vokal lebih mundur",
        ]
        self.gains_true = torch.tensor(
            [
                [4.75, -4.25, -3.25, 2.75],
                [-1.5, 3.5, 1.5, -3.5],
                [1.0, -2.0, 0.5, 0.5],
                [-3.0, 0.0, -1.0, 4.0],
            ],
            dtype=torch.float32,
        )
        self.loudness_true = torch.tensor(
            [
                [-18.0, -12.0, -15.0, -20.0],
                [-16.0, -18.0, -17.0, -17.0],
                [-19.0, -16.0, -18.5, -18.5],
                [-15.0, -18.0, -17.0, -22.0],
            ],
            dtype=torch.float32,
        )
        self.balance_true = torch.stack(
            [
                torch.from_numpy(normalize_balance_target([5, 0, -2, -3])),
                torch.from_numpy(normalize_balance_target([0, 3, 2, -3])),
                torch.from_numpy(normalize_balance_target([0, 0, 0, 0])),
                torch.from_numpy(normalize_balance_target([-3, 0, -1, 5])),
            ],
            dim=0,
        )
        self.active_mask = torch.ones((self.batch_size, 4), dtype=torch.bool)
        self.active_mask[2, 0] = False  # Test silent stem masking on sample 2

    def test_all_registered_models_forward_and_backward(self) -> None:
        for model_name in sorted(MODEL_REGISTRY.keys()):
            with self.subTest(model_name=model_name):
                model = build_model(model_name)
                model.train()
                out = model(self.audio, self.prompts, active_mask=self.active_mask)
                self.assertIn("gains_pred", out)
                g_pred = out["gains_pred"]
                self.assertEqual(tuple(g_pred.shape), (self.batch_size, 4))
                self.assertTrue(torch.isfinite(g_pred).all())
                # Inactive stem (sample 2, stem 0) must be masked to 0.0 dB
                self.assertEqual(float(g_pred[2, 0].item()), 0.0)

                losses = compute_multimodal_loss(
                    outputs=out,
                    gains_true=self.gains_true,
                    loudness_true=self.loudness_true,
                    balance_true=self.balance_true,
                    active_mask=self.active_mask,
                )
                losses["loss"].backward()
                grad_norms = [
                    float(p.grad.abs().sum().item())
                    for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                self.assertTrue(len(grad_norms) > 0)
                self.assertTrue(any(g > 0.0 for g in grad_norms))

    def test_m3_structured_two_branch_physics_identities(self) -> None:
        m3 = StructuredTwoBranchModel()
        m3.eval()
        with torch.no_grad():
            out = m3(self.audio, self.prompts, clamp_output=False)
            g_pred = out["gains_pred"]
            l_pred = out["loudness_pred"]
            r_pred = out["balance_pred"]
            l_ref = out["l_ref_pred"]

            # 1. r_pred must sum to 0 for every sample
            torch.testing.assert_close(
                r_pred.sum(dim=1),
                torch.zeros(self.batch_size),
                atol=1e-5,
                rtol=1e-5,
            )
            # 2. l_ref must equal mean(l_pred)
            torch.testing.assert_close(
                l_ref,
                l_pred.mean(dim=1),
                atol=1e-5,
                rtol=1e-5,
            )
            # 3. g_pred must satisfy g_pred = l_ref + r_pred - l_pred
            expected_g = l_ref.unsqueeze(1) + r_pred - l_pred
            torch.testing.assert_close(g_pred, expected_g, atol=1e-5, rtol=1e-5)

    def test_synthetic_end_to_end_training_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rng = np.random.default_rng(2026)

            # Create 4 synthetic cached log-mel files
            excerpts = []
            for idx in range(4):
                npy_file = tmp_path / f"syn_ex{idx:02d}.npy"
                base_levels = np.array([-18.0, -14.0, -16.0, -20.0], dtype=np.float32) + rng.uniform(-2, 2, size=4).astype(np.float32)
                mel = base_levels[:, None, None] + rng.normal(0, 1.5, size=(4, 64, 800)).astype(np.float32)
                np.save(npy_file, mel.astype(np.float16))
                excerpts.append({
                    "excerpt_id": f"syn_ex{idx:02d}",
                    "song": f"Artist {idx} - Song {idx}",
                    "artist": f"Artist {idx}",
                    "split": "train" if idx < 2 else "validation",
                    "excerpt_index": idx,
                    "excerpt_start_sec": 0.0,
                    "duration_sec": 8.0,
                    "audio_path": npy_file.as_posix(),
                    "loudness_raw_db": [round(float(x), 4) for x in base_levels],
                    "active_mask": [True, True, True, True],
                    "all_stems_active": True,
                })

            train_samples = build_split_samples(excerpts[:2], "train", n_prompts_per_excerpt=2, seed=42)
            val_samples = build_split_samples(excerpts[2:], "validation", n_prompts_per_excerpt=2, seed=43)

            train_json = tmp_path / "train.json"
            val_json = tmp_path / "val.json"
            import json
            train_json.write_text(json.dumps(train_samples))
            val_json.write_text(json.dumps(val_samples))

            val_loader = DataLoader(MusdbMultimodalDataset(val_json), batch_size=4, shuffle=False)
            for b_name in ("no_op", "random", "dsp_equal_loudness", "rule_dsp_prompt"):
                pred_fn = make_baseline_predictor(b_name)
                metrics = evaluate_model(pred_fn, val_loader)
                self.assertIn("mae_db", metrics)
                self.assertIn("band_mae_db", metrics)
                self.assertIn("instruction_adherence", metrics)

            cfg = TrainConfig(
                model_name="m3",
                run_name="unit_test_m3_temp",
                seed=42,
                batch_size=4,
                epochs=2,
                patience=2,
                train_manifest=str(train_json),
                val_manifest=str(val_json),
                test_human_manifest=str(tmp_path / "nonexistent.json"),
                test_synthetic_manifest=str(tmp_path / "nonexistent.json"),
                overwrite=True,
            )
            res = train_one_run(cfg)
            self.assertEqual(res["model_name"], "m3")
            self.assertIn("validation", res)
            self.assertEqual(len(res["history"]), 2)

            # Test inference pipeline using the saved checkpoint and fallback mode
            from wadidaw.app.inference import predict_and_render_mix
            from wadidaw.training.train import CHECKPOINT_DIR, TRAINING_DIR
            ckpt_file = CHECKPOINT_DIR / "unit_test_m3_temp_best.pt"
            dummy_stems = {
                name: rng.normal(0.0, 0.1, size=(44100 * 2, 2)).astype(np.float32)
                for name in ("vocals", "drums", "bass", "other")
            }
            inf_out = predict_and_render_mix(
                stem_sources=dummy_stems,
                prompt="vokal paling menonjol, bass agak mundur, drum seimbang",
                checkpoint_path=ckpt_file,
                device="cpu",
            )
            self.assertEqual(inf_out["model_used"], "m3")
            self.assertEqual(inf_out["mix"].shape, (44100 * 2, 2))
            self.assertEqual(set(inf_out["gains_db"].keys()), {"vocals", "drums", "bass", "other"})

            # Clean up unit test artifacts
            for p in (
                ckpt_file,
                TRAINING_DIR / "unit_test_m3_temp_metrics.csv",
                TRAINING_DIR / "unit_test_m3_temp_results.json",
            ):
                p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
