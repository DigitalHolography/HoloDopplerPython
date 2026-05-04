from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()

    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")

    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON file: {path}\n{exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"Config JSON must contain an object at top level: {path}")

    return data


def _cmd_preview(args: argparse.Namespace) -> int:
    input_path: Path = args.input
    config_path: Path = args.config
    config = _load_json(config_path)

    # Keep this import inside the command so that `holodoppler --help`
    # stays fast and does not import CUDA/CuPy-heavy modules immediately.
    from holodoppler.Holodoppler import preview

    preview(input_path, config)

    return 0


def _cmd_process(args: argparse.Namespace) -> int:
    input_path: Path = args.input
    config_path: Path = args.config
    config = _load_json(config_path)

    from holodoppler.Holodoppler import process

    process(input_path, config)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="holodoppler",
        description="HoloDoppler command-line tools.",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
    )

    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview a HoloDoppler input file using a JSON configuration.",
    )
    preview_parser.add_argument(
        "input",
        type=_existing_file,
        help="Input file path.",
    )
    preview_parser.add_argument(
        "config",
        type=_existing_file,
        help="JSON configuration file path.",
    )
    preview_parser.set_defaults(func=_cmd_preview)

    process_parser = subparsers.add_parser(
        "process",
        help="Process a HoloDoppler input file using a JSON configuration.",
    )
    process_parser.add_argument(
        "input",
        type=_existing_file,
        help="Input file path.",
    )
    process_parser.add_argument(
        "config",
        type=_existing_file,
        help="JSON configuration file path.",
    )
    process_parser.set_defaults(func=_cmd_process)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)