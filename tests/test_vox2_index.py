import json
import tempfile
import unittest
from pathlib import Path


class ServerVox2IndexTests(unittest.TestCase):
    def _write_audio(self, root: Path, relative_path: str) -> None:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def _write_indexes(self, root: Path) -> None:
        dev_rows = []
        test_rows = []
        for speaker_index in range(3):
            speaker = f"vox2_id{speaker_index:05d}"
            speaker_dir = f"id{speaker_index:05d}"
            for utterance_index in range(3):
                dev_relative = (
                    f"dev/audio/{speaker_dir}/dev_video/{utterance_index:05d}.wav"
                )
                test_relative = (
                    f"test/audio/{speaker_dir}/test_video/{utterance_index:05d}.wav"
                )
                self._write_audio(root, dev_relative)
                self._write_audio(root, test_relative)
                dev_rows.append(f"{speaker} {dev_relative}")
                test_rows.append(f"{speaker} {test_relative}")
        (root / "dev.spk2utt").write_text("\n".join(dev_rows) + "\n")
        (root / "test.spk2utt").write_text("\n".join(test_rows) + "\n")

    def test_prepare_server_indexes_writes_manifests_raw_list_and_fixed_trials(self):
        from redim_repro.vox2_index import prepare_server_vox2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpus"
            output = Path(tmp) / "prepared"
            self._write_indexes(root)

            report = prepare_server_vox2(
                corpus_root=root,
                output_root=output,
                pairs_per_speaker=2,
                seed=17,
            )

            self.assertEqual(report["vox2_dev"]["utterances"], 9)
            self.assertEqual(report["vox2_dev"]["speakers"], 3)
            self.assertEqual(
                (output / "vox2_dev" / "utt2spk").read_text().splitlines()[0],
                "id00000/dev_video/00000 vox2_id00000",
            )
            raw_record = json.loads(
                (output / "vox2_test" / "raw.list").read_text().splitlines()[0]
            )
            self.assertEqual(raw_record["key"], "id00000/test_video/00000")
            self.assertEqual(raw_record["spk"], "vox2_id00000")
            self.assertEqual(
                Path(raw_record["wav"]),
                root.resolve() / "test/audio/id00000/test_video/00000.wav",
            )

            trials = [
                line.split()
                for line in (
                    output / "vox2_test" / "trials" / "vox2_test_fixed.kaldi"
                ).read_text().splitlines()
            ]
            self.assertEqual(len(trials), 12)
            self.assertEqual(sum(row[2] == "target" for row in trials), 6)
            self.assertEqual(sum(row[2] == "nontarget" for row in trials), 6)
            self.assertTrue(all(row[0] != row[1] for row in trials))
            metadata = json.loads(
                (
                    output / "vox2_test" / "trials" / "vox2_test_fixed.json"
                ).read_text()
            )
            self.assertEqual(metadata["seed"], 17)
            self.assertEqual(metadata["target_trials"], 6)
            self.assertEqual(metadata["nontarget_trials"], 6)

    def test_preparation_is_byte_identical_for_same_indexes_and_seed(self):
        from redim_repro.vox2_index import prepare_server_vox2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpus"
            first = Path(tmp) / "prepared-first"
            second = Path(tmp) / "prepared-second"
            self._write_indexes(root)

            prepare_server_vox2(root, first, pairs_per_speaker=2, seed=23)
            prepare_server_vox2(root, second, pairs_per_speaker=2, seed=23)

            self.assertEqual(
                (first / "vox2_test" / "trials" / "vox2_test_fixed.kaldi").read_bytes(),
                (second / "vox2_test" / "trials" / "vox2_test_fixed.kaldi").read_bytes(),
            )
            self.assertEqual(
                (first / "vox2_test" / "trials" / "vox2_test_fixed.json").read_bytes(),
                (second / "vox2_test" / "trials" / "vox2_test_fixed.json").read_bytes(),
            )

    def test_rejects_index_path_that_does_not_belong_to_its_split(self):
        from redim_repro.vox2_index import ServerIndexError, prepare_server_vox2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpus"
            root.mkdir()
            (root / "dev.spk2utt").write_text(
                "vox2_id00000 test/audio/id00000/video/00000.wav\n"
            )
            (root / "test.spk2utt").write_text(
                "vox2_id00001 test/audio/id00001/video/00000.wav\n"
            )

            with self.assertRaisesRegex(ServerIndexError, "expected dev/audio"):
                prepare_server_vox2(root, Path(tmp) / "prepared")
