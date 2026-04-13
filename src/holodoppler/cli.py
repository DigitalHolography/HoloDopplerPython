from __future__ import annotations

import argparse
from typing import Sequence

from holodoppler.config import (
    available_builtin_settings,
    export_builtin_setting,
    load_builtin_parameters,
    ProcessingParameters,
)
from holodoppler.runner import process_inputs


def _load_parameters(parameter_file: str | None, builtin_setting: str | None) -> ProcessingParameters:
    if parameter_file and builtin_setting:
        raise ValueError("Use either --parameters or --setting, not both.")
    if parameter_file:
        return ProcessingParameters.from_json_file(parameter_file)
    if builtin_setting:
        return load_builtin_parameters(builtin_setting)
    return load_builtin_parameters()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="holodoppler",
        description="Holodoppler batch runner for .holo files, folders, and zip archives.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run processing on a file, folder, or zip archive.")
    run_parser.add_argument("--input", required=True, help="Path to a .holo file, folder, or .zip archive.")
    run_parser.add_argument("--output", required=True, help="Output directory for generated Holodoppler bundles.")
    run_parser.add_argument("--parameters", help="Path to a JSON parameter file.")
    run_parser.add_argument("--setting", help="Builtin setting name, for example 'default_parameters'.")
    run_parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    run_parser.add_argument("--pipeline-version", choices=("latest", "old"), default="latest")

    settings_parser = subparsers.add_parser("settings", help="Inspect packaged settings presets.")
    settings_subparsers = settings_parser.add_subparsers(dest="settings_command")

    settings_subparsers.add_parser("list", help="List builtin settings presets.")

    export_parser = settings_subparsers.add_parser("export", help="Export a builtin setting to a JSON file.")
    export_parser.add_argument("--name", default="default_parameters", help="Builtin setting name to export.")
    export_parser.add_argument("--output", required=True, help="Destination JSON file path.")

    subparsers.add_parser("ui", help="Launch the Tkinter desktop application.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            parameters = _load_parameters(args.parameters, args.setting)
            summary = process_inputs(
                input_path=args.input,
                output_root=args.output,
                parameters=parameters,
                backend=args.backend,
                pipeline_version=args.pipeline_version,
                progress=print,
            )
        except Exception as exc:
            parser.exit(status=1, message=f"Error: {exc}\n")

        print(f"Processed {len(summary.processed)} item(s).")
        if summary.failed:
            for failed_item in summary.failed:
                print(f"Failed: {failed_item.source_label} -> {failed_item.error}")
            return 1
        return 0

    if args.command == "settings":
        if args.settings_command == "list":
            for setting_name in available_builtin_settings():
                print(setting_name)
            return 0

        if args.settings_command == "export":
            output_path = export_builtin_setting(args.name, args.output)
            print(output_path)
            return 0

        parser.error("settings requires a subcommand. Use 'settings list' or 'settings export'.")

    if args.command == "ui":
        from holodoppler.ui import launch_app

        launch_app()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
