import unittest
import tempfile
from pathlib import Path

from redim_repro.runner import (
    WESPEAKER_COMMIT,
    build_average_command,
    build_epoch_train_command,
    build_upstream_run_command,
    build_raw_score_command,
    default_num_avg,
    materialize_config,
    normalize_gpus,
    stage_lm_checkpoint,
)

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is installed on servers
    yaml = None


class RunnerTests(unittest.TestCase):
    def test_normalize_gpus_accepts_common_forms(self):
        self.assertEqual(normalize_gpus("0,1"), "[0,1]")
        self.assertEqual(normalize_gpus("[0,1]"), "[0,1]")
        self.assertEqual(normalize_gpus([2, 3]), "[2,3]")

    def test_build_command_contains_reproduction_parameters(self):
        command = build_upstream_run_command(
            upstream_dir=Path("/tmp/wespeaker"),
            data_dir=Path("/datasets/vox"),
            config_path=Path("/repo/configs/redimnet2_b3_lm.yaml"),
            gpus="0,1",
            stage=3,
            stop_stage=3,
            exp_dir=Path("/experiments/redimnet2_b3_lm"),
            checkpoint=Path("/experiments/redimnet2_b3_ptn/models/model_120.pt"),
        )

        self.assertEqual(command[0:2], ["bash", "/tmp/wespeaker/examples/voxceleb/v2/run.sh"])
        self.assertIn("--data", command)
        self.assertIn("/datasets/vox", command)
        self.assertIn("--config", command)
        self.assertIn("/repo/configs/redimnet2_b3_lm.yaml", command)
        self.assertIn("--checkpoint", command)
        self.assertIn("--HOST_NODE_ADDR", command)
        self.assertIn("localhost:29400", command)
        self.assertIn("--data_type", command)
        self.assertEqual(command[command.index("--data_type") + 1], "shard")
        self.assertTrue(WESPEAKER_COMMIT)

    def test_averaging_defaults_match_training_phase(self):
        self.assertEqual(default_num_avg("ptn"), 10)
        self.assertEqual(default_num_avg("lm"), 1)

    def test_raw_score_command_disables_mean_normalization(self):
        command = build_raw_score_command(
            upstream_dir=Path("/tmp/wespeaker"),
            data_dir=Path("/datasets/vox"),
            exp_dir=Path("/experiments/redimnet2_b3_lm"),
            trial_name="vox1_O_cleaned.kaldi",
        )

        self.assertIn("--cal_mean", command)
        self.assertEqual(command[command.index("--cal_mean") + 1], "False")
        self.assertIn(
            "/datasets/vox/vox1/trials/vox1_O_cleaned.kaldi",
            command,
        )

    def test_average_command_uses_requested_checkpoint_count(self):
        command = build_average_command(
            upstream_dir=Path("/tmp/wespeaker"),
            exp_dir=Path("/experiments/redimnet2_b3_ptn"),
            num_avg=10,
        )

        self.assertIn("--dst_model", command)
        self.assertIn(
            "/experiments/redimnet2_b3_ptn/models/avg_model.pt",
            command,
        )
        self.assertEqual(command[command.index("--num") + 1], "10")

    def test_epoch_train_command_uses_torchrun_and_server_validation_paths(self):
        command = build_epoch_train_command(
            project_root=Path("/repo"),
            upstream_dir=Path("/opt/wespeaker"),
            data_dir=Path("/datasets/vox2-prepared"),
            config_path=Path("/repo/configs/redimnet2_b3_ptn.yaml"),
            exp_dir=Path("/experiments/redimnet2_b3"),
            gpus="0,1",
            num_nodes=2,
            host_node_addr="10.0.0.1:29400",
            job_id="vox2-b3",
            data_type="raw",
            eval_data=Path("/datasets/vox2-prepared/vox2_test/raw.list"),
            eval_trials=Path(
                "/datasets/vox2-prepared/vox2_test/trials/vox2_test_fixed.kaldi"
            ),
            eval_interval=1,
            eval_num_workers=2,
        )

        self.assertEqual(command[:4], ["torchrun", "--nnodes", "2", "--nproc_per_node"])
        self.assertEqual(command[4], "2")
        self.assertIn("/repo/scripts/train_with_epoch_eval.py", command)
        self.assertEqual(command[command.index("--gpus") + 1], "[0,1]")
        self.assertEqual(
            command[command.index("--train_data") + 1],
            "/datasets/vox2-prepared/vox2_dev/raw.list",
        )
        self.assertEqual(
            command[command.index("--eval_trials") + 1],
            "/datasets/vox2-prepared/vox2_test/trials/vox2_test_fixed.kaldi",
        )
        self.assertEqual(command[command.index("--eval_interval") + 1], "1")

    def test_stage_lm_checkpoint_uses_model_zero_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ptn" / "models" / "avg_model.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"checkpoint")

            staged = stage_lm_checkpoint(source, root / "lm")

            self.assertEqual(staged, root / "lm" / "models" / "model_0.pt")
            self.assertEqual(staged.read_bytes(), b"checkpoint")

    @unittest.skipUnless(yaml is not None, "PyYAML is installed with project requirements")
    def test_materialize_config_keeps_recipe_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "recipe.yaml"
            source.write_text(
                "enable_amp: false\n"
                "dataloader_args:\n"
                "  batch_size: 256\n"
                "  num_workers: 4\n"
                "  prefetch_factor: 4\n"
                "  pin_memory: false\n"
            )

            destination = materialize_config(
                source,
                root / "runtime.yaml",
                batch_size=64,
                num_workers=8,
                prefetch_factor=16,
                pin_memory=True,
                amp=True,
            )

            config = yaml.safe_load(destination.read_text())
            self.assertEqual(config["dataloader_args"]["batch_size"], 64)
            self.assertEqual(config["dataloader_args"]["num_workers"], 8)
            self.assertEqual(config["dataloader_args"]["prefetch_factor"], 16)
            self.assertTrue(config["dataloader_args"]["pin_memory"])
            self.assertTrue(config["enable_amp"])
            self.assertIn("batch_size: 256", source.read_text())
