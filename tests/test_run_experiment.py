import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"


def load_cli_module():
    spec = importlib.util.spec_from_file_location("run_experiment_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - impossible for this file
        raise RuntimeError("could not load run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunExperimentParserTests(unittest.TestCase):
    def test_server_index_preparation_command_accepts_corpus_and_trial_options(self):
        cli = load_cli_module()

        args = cli.build_parser().parse_args(
            [
                "prepare-server-vox2",
                "--corpus",
                "/public/share/voxceleb2",
                "--data",
                "/work/prepared",
                "--pairs-per-speaker",
                "400",
                "--seed",
                "9",
            ]
        )

        self.assertEqual(args.command, "prepare-server-vox2")
        self.assertEqual(args.corpus, Path("/public/share/voxceleb2"))
        self.assertEqual(args.data, Path("/work/prepared"))
        self.assertEqual(args.pairs_per_speaker, 400)
        self.assertEqual(args.seed, 9)

    def test_server_training_accepts_epoch_evaluation_and_augmentation_paths(self):
        cli = load_cli_module()

        args = cli.build_parser().parse_args(
            [
                "train",
                "--server-vox2",
                "--data",
                "/work/prepared",
                "--data-type",
                "raw",
                "--eval-interval",
                "5",
                "--eval-num-workers",
                "3",
                "--musan-lmdb",
                "/public/musan/lmdb",
                "--rirs-lmdb",
                "/public/rirs/lmdb",
            ]
        )

        self.assertTrue(args.server_vox2)
        self.assertEqual(args.eval_interval, 5)
        self.assertEqual(args.eval_num_workers, 3)
        self.assertEqual(args.musan_lmdb, Path("/public/musan/lmdb"))
        self.assertEqual(args.rirs_lmdb, Path("/public/rirs/lmdb"))
