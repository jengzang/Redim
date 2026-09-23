import tempfile
import unittest
from pathlib import Path


class EpochEvaluationTests(unittest.TestCase):
    def test_validation_dataset_disables_ddp_partition_for_rank_zero_full_coverage(self):
        from redim_repro.epoch_eval import disable_distributed_partition

        class Sampler:
            partition = True

        class DataList:
            sampler = Sampler()

        class Processor:
            def __init__(self, source):
                self.source = source

        dataset = Processor(Processor(DataList()))

        disable_distributed_partition(dataset)

        self.assertFalse(dataset.source.source.sampler.partition)

    def test_score_trials_reports_zero_eer_for_separated_target_and_nontarget(self):
        from redim_repro.epoch_eval import score_trials
        from redim_repro.vox2_index import Trial

        result = score_trials(
            {
                "a": [1.0, 0.0],
                "b": [0.9, 0.1],
                "c": [0.0, 1.0],
            },
            [Trial("a", "b", "target"), Trial("a", "c", "nontarget")],
        )

        self.assertAlmostEqual(result.eer, 0.0)
        self.assertAlmostEqual(result.min_dcf, 0.0)
        self.assertEqual(result.target_trials, 1)
        self.assertEqual(result.nontarget_trials, 1)

    def test_epoch_result_keeps_lowest_eer_then_lowest_mindcf_checkpoint(self):
        from redim_repro.epoch_eval import EvaluationResult, write_epoch_result

        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp) / "experiment"
            models = exp_dir / "models"
            models.mkdir(parents=True)
            first_checkpoint = models / "model_1.pt"
            second_checkpoint = models / "model_2.pt"
            third_checkpoint = models / "model_3.pt"
            first_checkpoint.touch()
            second_checkpoint.touch()
            third_checkpoint.touch()

            write_epoch_result(
                exp_dir,
                1,
                EvaluationResult(0.04, 0.3, 3, 3),
                first_checkpoint,
            )
            write_epoch_result(
                exp_dir,
                2,
                EvaluationResult(0.05, 0.1, 3, 3),
                second_checkpoint,
            )
            best = write_epoch_result(
                exp_dir,
                3,
                EvaluationResult(0.04, 0.2, 3, 3),
                third_checkpoint,
            )

            self.assertEqual(best["epoch"], 3)
            self.assertEqual(
                (exp_dir / "best_model.pt").resolve(),
                third_checkpoint.resolve(),
            )
            self.assertEqual(
                len((exp_dir / "epoch_eval" / "metrics.jsonl").read_text().splitlines()),
                3,
            )
