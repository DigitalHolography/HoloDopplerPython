from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
from holodoppler.Holodoppler import Holodoppler
from matlab_imresize.imresize import imresize
import os
from .plotting import DebugPlotterManager


def preview(holo_path, parameters: dict) -> None:
    HD = Holodoppler(backend = "cupyRAM", pipeline_version = "latest_old_reg")

    HD.load_file(holo_path)
    
    # if HD.ext == ".holo":
    #     print("file header :", HD.file_header)
    
    print(parameters)

    frames = HD.read_frames(0, 1)
    res = HD.render_moments(parameters)

    def plot_debug_safe(HD, res):
        debug_manager = DebugPlotterManager(parameters) if parameters.get("debug") else None
        
        out = debug_manager.plot_all(res) if parameters.get("debug") else {}

        return out

    import imageio.v3 as iio

    def save_debug_images(debug_dict, save_dir, prefix="debug"):
        os.makedirs(save_dir, exist_ok=True)

        for key, img in debug_dict.items():
            if img is None:
                continue

            img_np = HD.bm.to_numpy(img)

            if img_np.ndim == 2 and parameters["square"]:
                H, W = img_np.shape
                L = max(H, W)
                # --- Resize ---
                img_np = imresize(img_np, output_shape=(L, L))

            if img_np.dtype != np.uint8:
                img_min = np.min(img_np)
                img_max = np.max(img_np)

                if img_max > img_min:
                    img_np = (img_np - img_min) / (img_max - img_min + 1e-12)

                img_np = (img_np * 255).astype(np.uint8)

            filename = os.path.join(save_dir, f"{prefix}_{key}.png")

            iio.imwrite(filename, img_np)

            print(f"Saved: {filename} | shape={img_np.shape} dtype={img_np.dtype}")


    # --- Generate debug safely ---
    debug_imgs = plot_debug_safe(HD, res)

    if parameters["debug"] and parameters["shack_hartmann"] and parameters["shack_hartmann_zernike_fit"]:
        print("zernike_fit_coeffs (radians):", HD.bm.to_numpy(res["coefs"]) if "coefs" in res else "N/A")
        print("delta to true z in mm if coef[0] is defocus : ", 4* np.sqrt(3) * parameters["z"]**2 / ((min(frames.shape[1:])* parameters["pixel_pitch"])**2)  * parameters["wavelength"] / (2*np.pi) * (HD.bm.to_numpy(res["coefs"])[0] if "coefs" in res else 0) * 1e3)

    # --- Add M0 ---
    if "M0" in res:
        M0 = HD.bm.to_numpy(res["M0"])
        M0 = (M0 - np.min(M0)) / (np.max(M0) - np.min(M0) + 1e-12)
        debug_imgs["M0"] = (M0 * 255).astype(np.uint8)

    print("DEBUG KEYS:", list(debug_imgs.keys()))

    # --- Save ---
    save_dir = "./debug_outputs"
    save_debug_images(debug_imgs, save_dir)
    
    M0img = debug_imgs.get("M0")
    if M0img is not None:
        if M0img.ndim == 2 and parameters["square"]:
            H, W = M0img.shape
            L = max(H, W)
            # --- Resize ---
            M0img = imresize(M0img, output_shape=(L, L))
            
        return M0img


def process(holo_path, parameters: dict) -> None:
    HD = Holodoppler(backend = "cupyRAM", pipeline_version = "latest")

    HD.load_file(holo_path)

    if HD.file_reader.ext == ".holo":
        print("file header :", HD.file_reader.file_header)
        
    print("parameters : ", parameters)

    HD.process_moments(parameters, holodoppler_path = True)

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


def _load_json(path: Path) -> dict:
    """Load and parse JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def _get_debug_config() -> dict:
    """Load debug configuration if it exists."""
    debug_paths_file = Path(".debug_paths.json")
    if debug_paths_file.exists():
        with open(debug_paths_file, 'r') as f:
            return json.load(f)
    return {}

def _cmd_preview(args: argparse.Namespace) -> int:
    debug_config = _get_debug_config()
    
    # Determine input path
    if args.input is None:
        holofilepath = debug_config.get("HOLOFILEPATH")
        if not holofilepath:
            print("Error: No input file provided and HOLOFILEPATH not found in .debug_paths.json")
            return 1
        input_path = Path(holofilepath)
        if not input_path.exists():
            print(f"Error: HOLOFILEPATH '{input_path}' does not exist")
            return 1
    else:
        input_path = args.input
    
    # Determine config path
    if args.config is None:
        config_path = Path("parameters/default_parameters_debug.json")
        if not config_path.exists():
            print("Error: No config file provided and parameters/default_parameters_debug.json not found")
            return 1
    else:
        config_path = args.config
    
    config = _load_json(config_path)
    preview(input_path, config)
    return 0

def _cmd_process(args: argparse.Namespace) -> int:
    debug_config = _get_debug_config()
    
    # Determine input path
    if args.input is None:
        holofilepath = debug_config.get("HOLOFILEPATH")
        if not holofilepath:
            print("Error: No input file provided and HOLOFILEPATH not found in .debug_paths.json")
            return 1
        input_path = Path(holofilepath)
        if not input_path.exists():
            print(f"Error: HOLOFILEPATH '{input_path}' does not exist")
            return 1
    else:
        input_path = args.input
    
    # Determine config path
    if args.config is None:
        config_path = Path("parameters/default_parameters_debug.json")
        if not config_path.exists():
            print("Error: No config file provided and parameters/default_parameters_debug.json not found")
            return 1
    else:
        config_path = args.config
    
    config = _load_json(config_path)
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
        nargs="?",  # Make optional
        default=None,
        help="Input file path. If not provided, uses HOLOFILEPATH from .debug_paths.json",
    )
    preview_parser.add_argument(
        "config",
        type=_existing_file,
        nargs="?",  # Make optional
        default=None,
        help="JSON configuration file path. If not provided, uses parameters/default_parameters_debug.json",
    )
    preview_parser.set_defaults(func=_cmd_preview)

    process_parser = subparsers.add_parser(
        "process",
        help="Process a HoloDoppler input file using a JSON configuration.",
    )
    process_parser.add_argument(
        "input",
        type=_existing_file,
        nargs="?",
        default=None,
        help="Input file path. If not provided, uses HOLOFILEPATH from .debug_paths.json",
    )
    process_parser.add_argument(
        "config",
        type=_existing_file,
        nargs="?",
        default=None,
        help="JSON configuration file path. If not provided, uses parameters/default_parameters_debug.json",
    )
    process_parser.set_defaults(func=_cmd_process)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return args.func(args)