# Holodoppler

`holodoppler` now ships with:

- a CLI for single-file and batch `.holo` processing
- a Tkinter desktop UI using the Sun Valley ttk theme
- a packaged settings library with builtin JSON presets

Only `.holo` processing is exposed through the new UI and CLI for now.

## Install

```bash
python -m pip install uv
uv sync
uv pip install -e .
```

GPU acceleration uses the CUDA 12 CuPy wheel (`cupy-cuda12x`). If the CUDA 12 runtime DLLs are not available on your machine, keep the backend on `numpy`.

## Builtin Settings Library

The packaged settings library currently includes `default_parameters.json`.

List available presets:

```bash
uv run holodoppler settings list
```

Export the default preset to a local JSON file to create a new one:

```bash
uv run holodoppler settings export --name default_parameters --output .\parameters.json
```

## CLI Usage

Process a single `.holo` file:

```bash
uv run holodoppler run --input .\sample.holo --output .\output
```

Process a folder recursively with a custom JSON parameter file:

```bash
uv run holodoppler run --input .\dataset --output .\output --parameters .\parameters.json
```

Process a zip archive with the builtin default preset:

```bash
uv run holodoppler run --input .\dataset.zip --output .\output --setting default_parameters
```

Optional runtime controls:

```bash
uv run holodoppler run --input .\sample.holo --output .\output --backend cupy --pipeline-version old
```

For folder inputs, the output mirrors the input relative directory structure. For zip inputs, the output is mirrored under a top-level folder named after the zip file stem.

## UI Usage

Launch the desktop app:

```bash
uv run holodoppler ui
```

The UI only offers the `cupy` backend when the CUDA FFT runtime is available.

The `Run` tab lets you:

- select a `.holo` file, folder, or zip archive
- choose an output folder
- run with a builtin preset, a JSON file, or the values from the advanced form

The `Advanced` tab lets you:

- browse the builtin settings library
- load a preset into the editable form
- edit parameters manually without starting from JSON
- save the current form as a JSON file

## Output Layout

Each processed input generates a Holodoppler bundle:

- `png/`
- `mp4/`
- `json/parameters.json`
- `raw/<bundle_name>_output.h5`
- `version.txt`
