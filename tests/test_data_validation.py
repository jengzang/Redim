import tempfile
import unittest
from pathlib import Path

from redim_repro.data_validation import ManifestError, validate_voxceleb_layout


class ValidateVoxCelebLayoutTests(unittest.TestCase):
    def _write_layout(self, root: Path, *, duplicate_train=False, missing_trial=False):
        vox2 = root / "vox2_dev"
        vox1 = root / "vox1"
        (vox2 / "trials").mkdir(parents=True)
        (vox1 / "trials").mkdir(parents=True)

        train_rows = [
            "id10001/clip1/00001 /audio/train-1.wav",
            "id10002/clip1/00001 /audio/train-2.wav",
        ]
        if duplicate_train:
            train_rows.append(train_rows[0])
        (vox2 / "wav.scp").write_text("\n".join(train_rows) + "\n")
        (vox2 / "utt2spk").write_text(
            "id10001/clip1/00001 id10001\n"
            "id10002/clip1/00001 id10002\n"
        )

        (vox1 / "wav.scp").write_text(
            "id10001/clip1/00001 /audio/test-1.wav\n"
            "id10002/clip1/00002 /audio/test-2.wav\n"
        )
        (vox1 / "utt2spk").write_text(
            "id10001/clip1/00001 id10001\n"
            "id10002/clip1/00002 id10002\n"
        )
        test_ref = "id10002/clip1/does-not-exist" if missing_trial else "id10002/clip1/00002"
        (vox1 / "trials" / "vox1_O_cleaned.kaldi").write_text(
            "target id10001/clip1/00001 id10001/clip1/00001\n"
            f"nontarget id10001/clip1/00001 {test_ref}\n"
        )

    def test_valid_layout_returns_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_layout(root)

            report = validate_voxceleb_layout(root)

            self.assertEqual(report["vox2_dev"]["utterances"], 2)
            self.assertEqual(report["vox2_dev"]["speakers"], 2)
            self.assertEqual(report["vox1"]["trials"]["vox1_O_cleaned.kaldi"], 2)

    def test_duplicate_manifest_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_layout(root, duplicate_train=True)

            with self.assertRaisesRegex(ManifestError, "duplicate key"):
                validate_voxceleb_layout(root)

    def test_missing_trial_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_layout(root, missing_trial=True)

            with self.assertRaisesRegex(ManifestError, "missing from wav.scp"):
                validate_voxceleb_layout(root)


class ValidateServerVox2LayoutTests(unittest.TestCase):
    def test_server_layout_requires_every_trial_utterance_to_be_indexed(self):
        from redim_repro.data_validation import validate_server_vox2_layout
        from redim_repro.vox2_index import prepare_server_vox2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpus"
            output = Path(tmp) / "prepared"
            dev_rows = []
            test_rows = []
            for speaker_index in range(2):
                speaker = f"vox2_id{speaker_index:05d}"
                speaker_dir = f"id{speaker_index:05d}"
                for utterance_index in range(2):
                    for split, rows in (("dev", dev_rows), ("test", test_rows)):
                        relative = (
                            f"{split}/audio/{speaker_dir}/video/{utterance_index:05d}.wav"
                        )
                        audio = root / relative
                        audio.parent.mkdir(parents=True, exist_ok=True)
                        audio.touch()
                        rows.append(f"{speaker} {relative}")
            (root / "dev.spk2utt").write_text("\n".join(dev_rows) + "\n")
            (root / "test.spk2utt").write_text("\n".join(test_rows) + "\n")
            prepare_server_vox2(root, output, pairs_per_speaker=1)
            trial_path = output / "vox2_test" / "trials" / "vox2_test_fixed.kaldi"
            trial_path.write_text("missing/utt id00000/video/00000 target\n")

            with self.assertRaisesRegex(ManifestError, "missing from wav.scp"):
                validate_server_vox2_layout(output)
