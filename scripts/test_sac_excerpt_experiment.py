import contextlib
import csv
import io
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from scripts import train_music_sac as trainer
from wadidaw import multi_song_env, music_env
from wadidaw.multi_song_env import MultiSongMixingEnv
from wadidaw.rl.replay_buffer import ReplayBuffer


class RecordingEnv(MultiSongMixingEnv):
    """Trace real fixture episodes without replacing SAC or audio processing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, duration_seconds=0.125, **kwargs)
        self.resets = []
        self.actions = []
        for env in self.envs:
            env.max_steps = 2

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.resets.append({
            "seed": seed, "options": dict(options or {}),
            "song": info["song"], "fraction": info["excerpt_fraction"],
            "target": self.current_env.target_gains.copy(),
            "observation": observation.copy(),
        })
        return observation, info

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return super().step(action)


class SACExcerptExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        time = np.arange(8000) / 16_000
        for song in range(7):
            stems = cls.root / f"Track{song + 1:05d}" / "stems"
            stems.mkdir(parents=True)
            for stem in range(4):
                audio = (0.05 + time / 10) * np.sin(
                    2 * np.pi * (90 + 170 * stem + 10 * song) * time
                )
                sf.write(stems / f"S{stem:02d}.wav", audio, 16000, subtype="FLOAT")
        cls.manifest = {
            "seed": 2026,
            "splits": {
                "train": [{"index": i, "song": f"Track{i + 1:05d}"} for i in (0, 1)],
                "validation": [{"index": i, "song": f"Track{i + 1:05d}"} for i in (2, 3, 4, 5)],
                "test": [{"index": 6, "song": "Track00007"}],
            },
        }
        cls.manifest_path = cls.root / "splits.json"
        cls.manifest_path.write_text(json.dumps(cls.manifest))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(music_env, "DATASET", self.root))
        self.stack.enter_context(patch.object(multi_song_env, "MANIFEST", self.manifest_path))
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.output = Path(temporary)
        self.stack.enter_context(patch.object(trainer, "CHECKPOINT_DIR", self.output / "checkpoints"))
        self.stack.enter_context(patch.object(trainer, "METRICS_DIR", self.output / "training"))
        self.created_envs = []
        self.replay_seeds = []

        def make_env(*args, **kwargs):
            env = RecordingEnv(*args, **kwargs)
            self.created_envs.append(env)
            return env

        def make_replay(**kwargs):
            self.replay_seeds.append(kwargs["seed"])
            return ReplayBuffer(**kwargs)

        self.stack.enter_context(patch.object(trainer, "MultiSongMixingEnv", make_env))
        self.stack.enter_context(patch.object(trainer, "ReplayBuffer", make_replay))

    def small_run(self, name, seed=42, fractions=(0.5,)):
        args = trainer.parse_args([
            "--steps", "6", "--warmup", "2", "--batch-size", "2",
            "--eval-every", "3", "--seed", str(seed), "--run-name", name,
            "--excerpt-fractions", *map(str, fractions),
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            result = trainer.train(args)
        checkpoint = torch.load(
            self.output / "checkpoints" / f"{name}_best.pt", weights_only=True,
        )
        return result, checkpoint, self.created_envs[-2:]

    def test_cli_defaults_preserve_midpoint_training_and_accept_controls(self):
        args = trainer.parse_args([])
        self.assertEqual(tuple(args.excerpt_fractions), (0.5,))
        self.assertEqual(args.seed, 42)
        self.assertEqual((args.steps, args.warmup, args.batch_size, args.eval_every, args.alpha),
                         (20000, 1000, 64, 2000, 0.1))
        args = trainer.parse_args([
            "--excerpt-fractions", "0.2", "0.5", "0.8", "--seed", "123",
            "--run-name", "diversity_seed123",
        ])
        self.assertEqual(args.excerpt_fractions, [0.2, 0.5, 0.8])
        self.assertEqual(args.seed, 123)
        self.assertEqual(args.run_name, "diversity_seed123")

    def test_validation_runs_all_120_fixed_tasks_without_consuming_training_rng(self):
        env = RecordingEnv(split="validation", excerpt_fractions=(0.2, 0.5, 0.8))
        self.addCleanup(env.close)
        torch.manual_seed(2026)
        np.random.seed(2026)
        torch_before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        result = trainer.evaluate(env, None, "no_op", 10)
        self.assertEqual(len(env.resets), 120)
        expected = [
            (song, excerpt, 50000 + song * 1000 + excerpt * 100 + target)
            for song in range(4) for excerpt in range(3) for target in range(10)
        ]
        actual = [(r["options"]["song_position"], r["options"]["excerpt_position"], r["seed"])
                  for r in env.resets]
        self.assertEqual(actual, expected)
        self.assertEqual(result["episode_count"], 120)
        self.assertEqual(len(result["episodes"]), 120)
        self.assertEqual(set(result["per_excerpt"]), {"0.2", "0.5", "0.8"})
        self.assertEqual(len(result["per_song"]), 4)
        for metrics in result["per_song"].values():
            self.assertEqual(metrics["episode_count"], 30)
            self.assertIn("mean_return", metrics)
            self.assertIn("mean_steps", metrics)
        for metrics in result["per_excerpt"].values():
            self.assertEqual(metrics["episode_count"], 40)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
        self.assertEqual(numpy_before[2:], np.random.get_state()[2:])

    def test_actual_sac_runs_respect_config_budget_isolation_and_common_validation(self):
        first, _, (train_a, val_a) = self.small_run("midpoint_42")
        second, _, (train_b, val_b) = self.small_run("three_123", 123, (0.2, 0.5, 0.8))
        self.assertEqual(train_a.excerpt_fractions, (0.5,))
        self.assertEqual(train_b.excerpt_fractions, (0.2, 0.5, 0.8))
        self.assertEqual((len(train_a.actions), len(train_b.actions)), (6, 6))
        for env in (train_a, train_b):
            self.assertTrue({r["song"] for r in env.resets} <= {"Track00001", "Track00002"})
            self.assertTrue({e.song_name for e in env.envs} <= {"Track00001", "Track00002"})
        self.assertEqual(first["validation_protocol"], second["validation_protocol"])
        self.assertEqual(len(val_a.resets), 480)  # Two baselines and two checkpoints.
        self.assertEqual(len(val_b.resets), 480)
        for a, b in zip(val_a.resets, val_b.resets):
            self.assertEqual((a["seed"], a["options"]), (b["seed"], b["options"]))
            np.testing.assert_array_equal(a["target"], b["target"])
            np.testing.assert_array_equal(a["observation"], b["observation"])
        self.assertEqual([r["step"] for r in first["history"]], [3, 6])
        self.assertIn(first["best_step"], (3, 6))
        stored = json.loads((self.output / "training" / "midpoint_42_results.json").read_text())
        self.assertEqual(stored["status"], "complete")
        self.assertEqual(stored["config"]["seed"], 42)
        self.assertEqual(stored["best_validation"]["episode_count"], 120)

    def test_training_seed_replays_initialization_actions_updates_and_targets(self):
        first, weights_a, (env_a, _) = self.small_run("repeat_a", 42)
        second, weights_b, (env_b, _) = self.small_run("repeat_b", 42)
        third, weights_c, (env_c, _) = self.small_run("different_seed", 123)
        self.assertEqual(self.replay_seeds, [42, 42, 123])
        np.testing.assert_array_equal(env_a.actions, env_b.actions)
        for a, b in zip(env_a.resets, env_b.resets):
            np.testing.assert_array_equal(a["target"], b["target"])
        self.assertEqual(first["history"], second["history"])
        for key in weights_a["actor_state_dict"]:
            self.assertTrue(torch.equal(weights_a["actor_state_dict"][key], weights_b["actor_state_dict"][key]))
        self.assertFalse(np.array_equal(env_a.actions, env_c.actions))
        self.assertTrue(any(not torch.equal(weights_a["actor_state_dict"][key], weights_c["actor_state_dict"][key])
                            for key in weights_a["actor_state_dict"]))
        self.assertEqual(third["config"]["seed"], 123)

    def test_evaluation_rejects_training_and_held_out_partitions(self):
        for split in ("train", "test"):
            with self.subTest(split=split), patch.object(RecordingEnv, "reset", side_effect=AssertionError("audio reset")):
                env = RecordingEnv(split=split)
                self.addCleanup(env.close)
                with self.assertRaisesRegex(ValueError, "validation"):
                    trainer.evaluate(env, None, "no_op", 10)

    def test_existing_outputs_and_duplicate_run_names_are_never_overwritten(self):
        for suffix, directory in (("_best.pt", "checkpoints"), ("_metrics.csv", "training"), ("_results.json", "training")):
            with self.subTest(suffix=suffix):
                path = self.output / directory / f"occupied{suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"preserve me")
                with self.assertRaises(FileExistsError):
                    self.small_run("occupied")
                self.assertEqual(path.read_bytes(), b"preserve me")
                path.unlink()
        self.assertEqual(self.created_envs, [])

    def test_training_rejects_nonstandard_validation_counts_and_unsafe_names(self):
        for arguments in (("--episodes-per-song", "1"), ("--seed", "-1"), ("--run-name", "../unsafe")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                trainer.train(trainer.parse_args(list(arguments)))
        self.assertEqual(self.created_envs, [])

    def test_comparison_commands_hold_every_control_constant(self):
        from scripts import compare_excerpt_diversity as comparison
        commands = comparison.experiment_commands("unit_comparison")
        self.assertEqual(len(commands), 6)
        self.assertEqual(Counter(item[1] for item in commands), {42: 2, 123: 2, 2026: 2})
        names = set()
        for condition, seed, command in commands:
            args = trainer.parse_args(command[3:])
            self.assertEqual((args.steps, args.warmup, args.alpha, args.batch_size, args.eval_every),
                             (20000, 1000, 0.1, 64, 2000))
            self.assertEqual(args.seed, seed)
            self.assertEqual(tuple(args.excerpt_fractions), comparison.CONDITIONS[condition])
            self.assertEqual(args.episodes_per_song, 10)
            names.add(args.run_name)
        self.assertEqual(len(names), 6)

    def test_comparison_default_only_prints_commands(self):
        from scripts import compare_excerpt_diversity as comparison
        with patch.object(comparison.subprocess, "run", side_effect=AssertionError("unexpected training")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                comparison.main(["--prefix", "unit_comparison"])
        self.assertEqual(output.getvalue().count("-m scripts.train_music_sac"), 6)
        self.assertFalse((self.output / "training").exists())

    def completed_report_fixtures(self):
        """Fabricated metrics test reporting arithmetic, never model quality."""
        from scripts import compare_excerpt_diversity as comparison
        prefix = "synthetic_report_fixture"
        val = RecordingEnv(split="validation", excerpt_fractions=(0.2, 0.5, 0.8))
        self.addCleanup(val.close)
        trainer.METRICS_DIR.mkdir(parents=True)
        for condition, seed, command in comparison.experiment_commands(prefix):
            args = trainer.parse_args(command[3:])
            train = RecordingEnv(split="train", excerpt_fractions=args.excerpt_fractions)
            self.addCleanup(train.close)
            metadata = trainer.experiment_metadata(args, train, val)
            error = (comparison.SEEDS.index(seed) + 1) * (1 if condition == "midpoint" else 0.5)
            metrics = {
                "success_rate": 0.0, "mean_error": error, "mean_initial_error": 4.0,
                "mean_return": -error, "mean_steps": 12.0,
            }
            validation = {
                **metrics, "episode_count": 120,
                "per_song": {m["song"]: {**metrics, "episode_count": 30} for m in val.members},
                "per_excerpt": {str(f): {**metrics, "episode_count": 40} for f in val.excerpt_fractions},
                "per_song_excerpt": {
                    m["song"]: {str(f): {**metrics, "episode_count": 10} for f in val.excerpt_fractions}
                    for m in val.members
                },
                "episodes": [
                    {**task, "success": 0, "error": error, "initial_error": 4,
                     "return": -error, "steps": 12}
                    for task in metadata["validation_protocol"]["tasks"]
                ],
            }
            results = {
                "status": "complete", **metadata, "best_step": 20000,
                "best_validation": validation, "final_validation": validation,
                "history": [{"step": step, **metrics, "mean_error": error + (20000 - step) / 20000}
                            for step in range(2000, 20001, 2000)],
            }
            trainer.output_paths(args.run_name)[2].write_text(json.dumps(results))
        return prefix

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Install optional Matplotlib to test curve rendering.")
    def test_report_uses_seed_level_sample_deviation_and_writes_curves(self):
        from scripts import compare_excerpt_diversity as comparison
        prefix = self.completed_report_fixtures()
        output = self.output / "report"
        with contextlib.redirect_stdout(io.StringIO()):
            report = comparison.write_report(prefix, output)
        overall = next(row for row in report["aggregate"] if row["condition"] == "midpoint"
                       and row["selection"] == "best" and row["group"] == "overall"
                       and row["metric"] == "mean_error")
        self.assertEqual((overall["mean"], overall["std"], overall["n_seeds"]), (2, 1, 3))
        difference = next(row for row in report["paired_differences"]
                          if row["selection"] == "best" and row["metric"] == "mean_error")
        self.assertEqual((difference["mean"], difference["std"]), (-1, 0.5))
        self.assertTrue((output / "learning_curves.svg").is_file())
        self.assertGreater((output / "learning_curves.png").stat().st_size, 10000)
        with (output / "per_seed.csv").open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual({row["group"] for row in rows}, {"overall", "song", "excerpt", "song_excerpt"})
        self.assertEqual({row["selection"] for row in rows}, {"best", "final"})
        original = (output / "comparison.json").read_bytes()
        with self.assertRaises(FileExistsError):
            comparison.write_report(prefix, output)
        self.assertEqual((output / "comparison.json").read_bytes(), original)

    def test_report_rejects_incomplete_or_different_experiments(self):
        from scripts import compare_excerpt_diversity as comparison
        prefix = self.completed_report_fixtures()
        path = trainer.output_paths(f"{prefix}_three_excerpts_seed2026")[2]
        original = json.loads(path.read_text())
        for mutation in ("seed", "task", "incomplete", "budget", "nonfinite"):
            with self.subTest(mutation=mutation):
                result = json.loads(json.dumps(original))
                if mutation == "seed":
                    result["config"]["seed"] = 99
                elif mutation == "task":
                    result["best_validation"]["episodes"][0]["seed"] += 1
                elif mutation == "incomplete":
                    result["status"] = "running"
                elif mutation == "budget":
                    result["config"]["steps"] = 21000
                else:
                    result["history"][5]["mean_error"] = float("nan")
                path.write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    comparison.write_report(prefix, self.output / "invalid_report")
                self.assertFalse((self.output / "invalid_report").exists())

    def test_runner_checks_all_output_collisions_before_launch(self):
        from scripts import compare_excerpt_diversity as comparison
        path = trainer.output_paths("preflight_three_excerpts_seed2026")[0]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"existing checkpoint")
        with patch.object(comparison.subprocess, "run", side_effect=AssertionError("unexpected training")):
            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                comparison.main(["--prefix", "preflight", "--run"])
        self.assertEqual(path.read_bytes(), b"existing checkpoint")

    def test_legacy_midpoint_evaluator_keeps_fixed_target_seeds(self):
        env = RecordingEnv(split="validation")
        self.addCleanup(env.close)
        result = trainer.evaluate(env, None, "no_op", 10)
        self.assertEqual(result["episode_count"], 40)
        self.assertEqual({row["fraction"] for row in env.resets}, {0.5})
        self.assertEqual(env.resets[10]["seed"], 51000)

    def test_learned_policy_validation_is_deterministic_and_preserves_torch_rng(self):
        env = RecordingEnv(split="validation", excerpt_fractions=(0.2, 0.5, 0.8))
        self.addCleanup(env.close)
        trainer.seed_training(42)
        agent = trainer.SACAgent(obs_dim=101, action_dim=4)
        before = torch.get_rng_state().clone()
        first = trainer.evaluate(env, agent, "sac", 10)
        second = trainer.evaluate(env, agent, "sac", 10)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
