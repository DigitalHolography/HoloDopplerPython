from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from typing import Callable
from zipfile import ZipFile

import cv2
import matplotlib.pyplot as plt
import numpy as np

from holodoppler.Holodoppler import Holodoppler
from holodoppler.config import ProcessingParameters


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ProcessedItem:
    source_label: str
    output_dir: Path


@dataclass(frozen=True, slots=True)
class FailedItem:
    source_label: str
    error: str


@dataclass(frozen=True, slots=True)
class BatchProcessingSummary:
    processed: tuple[ProcessedItem, ...]
    failed: tuple[FailedItem, ...]


def _noop(_: str) -> None:
    return None


def _effective_end_frame(processor: Holodoppler, parameters: ProcessingParameters) -> int:
    if parameters.end_frame > 0:
        return parameters.end_frame
    return processor.file_header["num_frames"]


def _calculate_batch_count(processor: Holodoppler, parameters: ProcessingParameters) -> int:
    end_frame = _effective_end_frame(processor, parameters)
    frame_span = end_frame - parameters.first_frame
    if parameters.batch_stride >= frame_span:
        return 1 if parameters.batch_size <= frame_span else 0
    return int(frame_span / parameters.batch_stride)


def _next_output_bundle(output_parent: Path, stem: str) -> Path:
    output_parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = output_parent / f"{stem}_HD_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def _write_video(video_path: Path, frames: np.ndarray, fps: float, codec: str) -> None:
    height, width = frames.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
        isColor=False,
    )
    try:
        for index in range(frames.shape[2]):
            writer.write(np.ascontiguousarray(frames[:, :, index]))
    finally:
        writer.release()


def _export_bundle(
    processor: Holodoppler,
    parameters: ProcessingParameters,
    video: np.ndarray,
    output_dir: Path,
    bundle_name: str,
    raw_h5_path: Path,
) -> None:
    png_dir = output_dir / "png"
    mp4_dir = output_dir / "mp4"
    json_dir = output_dir / "json"
    raw_dir = output_dir / "raw"

    for directory in (png_dir, mp4_dir, json_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for moment_index in range(video.shape[2]):
        image = np.mean(video[:, :, moment_index, :], axis=2)
        plt.imsave(png_dir / f"moment_{moment_index}.png", image, cmap="gray")

    moment_zero = video[:, :, 0, :].astype(np.float32)
    dynamic_range = moment_zero.max() - moment_zero.min()
    if dynamic_range > 0:
        moment_zero = (moment_zero - moment_zero.min()) / dynamic_range
    else:
        moment_zero = np.zeros_like(moment_zero, dtype=np.float32)
    moment_zero = (moment_zero * 255).astype(np.uint8)

    end_frame = _effective_end_frame(processor, parameters)
    num_batch = _calculate_batch_count(processor, parameters)
    duration = (end_frame - parameters.first_frame) / parameters.sampling_freq
    fps = num_batch / duration if duration > 0 and num_batch > 0 else 1.0

    _write_video(mp4_dir / "moment_0.mp4", moment_zero, fps, "mp4v")
    _write_video(mp4_dir / "moment_0.avi", moment_zero, fps, "XVID")

    (json_dir / "parameters.json").write_text(
        json.dumps(parameters.to_dict(), indent=2),
        encoding="utf-8",
    )

    shutil.copy2(raw_h5_path, raw_dir / f"{bundle_name}_output.h5")

    version_lines = [
        "Python:",
        f"Holodoppler pipeline version: {processor.pipeline_version}",
        f"Holodoppler backend: {processor.backend}",
    ]
    (output_dir / "version.txt").write_text("\n".join(version_lines) + "\n", encoding="utf-8")


def _process_single_file(
    source_file: Path,
    source_label: str,
    output_parent: Path,
    parameters: ProcessingParameters,
    backend: str,
    pipeline_version: str,
) -> Path:
    output_dir = _next_output_bundle(output_parent, source_file.stem)

    processor = Holodoppler(backend=backend, pipeline_version=pipeline_version)
    try:
        processor.load_file(str(source_file))
        if source_file.suffix.lower() != ".holo":
            raise ValueError(f"Unsupported input file {source_file.name!r}. Only .holo files are supported.")

        with TemporaryDirectory(prefix="holodoppler-export-") as temp_dir:
            temp_h5_path = Path(temp_dir) / "output.h5"
            video = processor.process_moments_(
                parameters.to_dict(),
                h5_path=str(temp_h5_path),
                return_numpy=True,
            )
            if video is None:
                raise RuntimeError(f"Processing failed for {source_label}.")
            if not temp_h5_path.is_file():
                raise RuntimeError(f"Processing did not create expected H5 output: {temp_h5_path}")
            _export_bundle(processor, parameters, video, output_dir, output_dir.name, temp_h5_path)
        return output_dir
    finally:
        processor._close_file()


def _discover_folder_inputs(folder_path: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in folder_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() == ".holo"
    )


def _safe_zip_member_path(member_name: str) -> Path:
    relative_path = Path(*PurePosixPath(member_name).parts)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe zip entry path: {member_name!r}")
    return relative_path


def _extract_zip_member(archive: ZipFile, member_name: str, temp_root: Path) -> Path:
    relative_path = _safe_zip_member_path(member_name)
    target_path = temp_root / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member_name) as source_file, target_path.open("wb") as target_file:
        shutil.copyfileobj(source_file, target_file)
    return target_path


def process_inputs(
    input_path: str | Path,
    output_root: str | Path,
    parameters: ProcessingParameters,
    backend: str = "numpy",
    pipeline_version: str = "latest",
    progress: ProgressCallback | None = None,
) -> BatchProcessingSummary:
    source = Path(input_path)
    destination_root = Path(output_root)
    reporter = progress or _noop

    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    processed: list[ProcessedItem] = []
    failed: list[FailedItem] = []

    if source.is_file() and source.suffix.lower() == ".holo":
        reporter(f"Processing file: {source}")
        try:
            output_dir = _process_single_file(
                source_file=source,
                source_label=str(source),
                output_parent=destination_root,
                parameters=parameters,
                backend=backend,
                pipeline_version=pipeline_version,
            )
            processed.append(ProcessedItem(source_label=str(source), output_dir=output_dir))
            reporter(f"Created output: {output_dir}")
        except Exception as exc:
            failed.append(FailedItem(source_label=str(source), error=str(exc)))
            reporter(f"Failed: {source} ({exc})")
        return BatchProcessingSummary(processed=tuple(processed), failed=tuple(failed))

    if source.is_dir():
        files = _discover_folder_inputs(source)
        if not files:
            raise FileNotFoundError(f"No .holo files found in folder: {source}")

        reporter(f"Found {len(files)} .holo file(s) in {source}")
        for file_path in files:
            relative_path = file_path.relative_to(source)
            reporter(f"Processing file: {relative_path}")
            try:
                output_dir = _process_single_file(
                    source_file=file_path,
                    source_label=str(relative_path),
                    output_parent=destination_root / relative_path.parent,
                    parameters=parameters,
                    backend=backend,
                    pipeline_version=pipeline_version,
                )
                processed.append(ProcessedItem(source_label=str(relative_path), output_dir=output_dir))
                reporter(f"Created output: {output_dir}")
            except Exception as exc:
                failed.append(FailedItem(source_label=str(relative_path), error=str(exc)))
                reporter(f"Failed: {relative_path} ({exc})")
        return BatchProcessingSummary(processed=tuple(processed), failed=tuple(failed))

    if source.is_file() and source.suffix.lower() == ".zip":
        with ZipFile(source) as archive, TemporaryDirectory(prefix="holodoppler-zip-") as temp_dir:
            temp_root = Path(temp_dir)
            members = sorted(
                archive.infolist(),
                key=lambda item: item.filename,
            )
            holo_members = [
                member
                for member in members
                if not member.is_dir() and member.filename.lower().endswith(".holo")
            ]
            if not holo_members:
                raise FileNotFoundError(f"No .holo files found in zip archive: {source}")

            reporter(f"Found {len(holo_members)} .holo file(s) in {source.name}")
            for member in holo_members:
                relative_path = _safe_zip_member_path(member.filename)
                reporter(f"Processing zip entry: {relative_path.as_posix()}")
                extracted_file = _extract_zip_member(archive, member.filename, temp_root)
                try:
                    output_dir = _process_single_file(
                        source_file=extracted_file,
                        source_label=f"{source.name}:{relative_path.as_posix()}",
                        output_parent=destination_root / source.stem / relative_path.parent,
                        parameters=parameters,
                        backend=backend,
                        pipeline_version=pipeline_version,
                    )
                    processed.append(
                        ProcessedItem(
                            source_label=f"{source.name}:{relative_path.as_posix()}",
                            output_dir=output_dir,
                        )
                    )
                    reporter(f"Created output: {output_dir}")
                except Exception as exc:
                    failed.append(
                        FailedItem(
                            source_label=f"{source.name}:{relative_path.as_posix()}",
                            error=str(exc),
                        )
                    )
                    reporter(f"Failed: {relative_path.as_posix()} ({exc})")
                finally:
                    if extracted_file.exists():
                        extracted_file.unlink()
        return BatchProcessingSummary(processed=tuple(processed), failed=tuple(failed))

    raise ValueError("Unsupported input. Use a .holo file, a folder, or a .zip archive.")
