from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import h5py
import numpy as np

from holodoppler.Holodoppler import Holodoppler
from holodoppler.config import ProcessingParameters, available_builtin_settings, load_builtin_parameters
from holodoppler.runner import _prepare_output_bundle, default_output_root, output_bundle_path, process_inputs


class ProcessingParametersTests(unittest.TestCase):
    def test_default_setting_is_packaged(self) -> None:
        self.assertIn("default_parameters.json", available_builtin_settings())

    def test_json_roundtrip(self) -> None:
        parameters = load_builtin_parameters()
        with TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "parameters.json"
            parameters.save_json(json_path)
            loaded = ProcessingParameters.from_json_file(json_path)
        self.assertEqual(parameters.to_dict(), loaded.to_dict())

    def test_unknown_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ProcessingParameters.from_mapping({"unknown": 1})


class HolodopplerExportTests(unittest.TestCase):
    def test_process_moments_writes_explicit_h5_path(self) -> None:
        parameters = load_builtin_parameters().to_dict()
        parameters.update(
            {
                "batch_size": 1,
                "batch_stride": 1,
                "end_frame": 1,
                "image_registration": False,
                "shack_hartmann": False,
                "square": False,
                "debug": False,
            }
        )

        processor = Holodoppler(backend="numpy")
        processor.ext = ".holo"
        processor.file_header = {"num_frames": 1}
        processor.file_footer = {}
        processor.read_frames = lambda first_frame, frame_size: np.zeros((frame_size, 2, 3), dtype=np.float32)
        processor.render_moments = lambda parameters, frames=None: {
            "M0": np.ones((2, 3), dtype=np.float32),
            "M1": np.ones((2, 3), dtype=np.float32) * 2,
            "M2": np.ones((2, 3), dtype=np.float32) * 3,
        }

        with TemporaryDirectory() as temp_dir:
            h5_path = Path(temp_dir) / "output.h5"
            video = processor.process_moments_(parameters, h5_path=str(h5_path), return_numpy=True)

            self.assertIsNotNone(video)
            self.assertTrue(h5_path.is_file())
            with h5py.File(h5_path, "r") as output_file:
                self.assertIn("moment0", output_file)
                self.assertIn("HD_parameters", output_file)


class RunnerRoutingTests(unittest.TestCase):
    def test_default_output_root_uses_file_parent_and_folder_itself(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            holo_file = root / "sample.holo"
            holo_file.write_bytes(b"")
            folder = root / "dataset"
            folder.mkdir()

            self.assertEqual(default_output_root(holo_file), root)
            self.assertEqual(default_output_root(folder), folder)

    def test_output_bundle_path_uses_intermediate_folder_without_number(self) -> None:
        self.assertEqual(
            output_bundle_path(Path("out"), "sample"),
            Path("out") / "sample" / "sample_HD",
        )

    def test_prepare_output_bundle_overwrites_existing_bundle_folder(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_file = root / "sample" / "sample_HD" / "old.txt"
            old_file.parent.mkdir(parents=True)
            old_file.write_text("old", encoding="utf-8")

            output_dir = _prepare_output_bundle(root, "sample")

            self.assertEqual(output_dir, root / "sample" / "sample_HD")
            self.assertTrue(output_dir.is_dir())
            self.assertFalse(old_file.exists())

    def test_single_file_processing_defaults_to_input_parent(self) -> None:
        parameters = load_builtin_parameters()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "one.holo"
            source.write_bytes(b"")
            calls: list[Path] = []

            def fake_process_single_file(**kwargs):
                calls.append(kwargs["output_parent"])
                return output_bundle_path(kwargs["output_parent"], kwargs["source_file"].stem)

            with patch("holodoppler.runner._process_single_file", side_effect=fake_process_single_file):
                summary = process_inputs(source, parameters=parameters)

        self.assertEqual(len(summary.processed), 1)
        self.assertEqual(calls, [root])
        self.assertEqual(summary.processed[0].output_dir, root / "one" / "one_HD")

    def test_single_file_processing_accepts_parameters_as_second_argument(self) -> None:
        parameters = load_builtin_parameters()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "one.holo"
            source.write_bytes(b"")

            def fake_process_single_file(**kwargs):
                return output_bundle_path(kwargs["output_parent"], kwargs["source_file"].stem)

            with patch("holodoppler.runner._process_single_file", side_effect=fake_process_single_file):
                summary = process_inputs(source, parameters)

        self.assertEqual(summary.processed[0].output_dir, root / "one" / "one_HD")

    def test_folder_processing_preserves_relative_parents(self) -> None:
        parameters = load_builtin_parameters()
        with TemporaryDirectory() as source_dir, TemporaryDirectory() as output_dir:
            root = Path(source_dir)
            first = root / "one.holo"
            second = root / "nested" / "two.holo"
            second.parent.mkdir(parents=True)
            first.write_bytes(b"")
            second.write_bytes(b"")

            calls: list[tuple[Path, Path]] = []

            def fake_process_single_file(**kwargs):
                calls.append((kwargs["source_file"], kwargs["output_parent"]))
                return output_bundle_path(kwargs["output_parent"], kwargs["source_file"].stem)

            with patch("holodoppler.runner._process_single_file", side_effect=fake_process_single_file):
                summary = process_inputs(root, output_dir, parameters)

        self.assertEqual(len(summary.processed), 2)
        self.assertEqual(len(calls), 2)
        output_parents = {output_parent for _, output_parent in calls}
        self.assertEqual(output_parents, {Path(output_dir), Path(output_dir) / "nested"})
        output_dirs = {item.output_dir.relative_to(output_dir) for item in summary.processed}
        self.assertEqual(output_dirs, {Path("one") / "one_HD", Path("nested") / "two" / "two_HD"})

    def test_zip_processing_scopes_outputs_under_zip_stem(self) -> None:
        parameters = load_builtin_parameters()
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            zip_path = temp_root / "dataset.zip"
            output_dir = temp_root / "out"

            with ZipFile(zip_path, "w") as archive:
                archive.writestr("group/sample.holo", b"")
                archive.writestr("group/readme.txt", b"skip")

            calls: list[tuple[Path, Path]] = []

            def fake_process_single_file(**kwargs):
                calls.append((kwargs["source_file"], kwargs["output_parent"]))
                return output_bundle_path(kwargs["output_parent"], kwargs["source_file"].stem)

            with patch("holodoppler.runner._process_single_file", side_effect=fake_process_single_file):
                summary = process_inputs(zip_path, output_dir, parameters)

        self.assertEqual(len(summary.processed), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], output_dir / "dataset" / "group")

    def test_unsupported_input_type_is_rejected(self) -> None:
        parameters = load_builtin_parameters()
        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "input.txt"
            file_path.write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                process_inputs(file_path, Path(temp_dir) / "out", parameters)


if __name__ == "__main__":
    unittest.main()
