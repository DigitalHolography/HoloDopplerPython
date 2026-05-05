# Getting Started

Follow the steps below to install dependencies and run the example.

## 1. Install the Project

```bash
python -m venv .venv
source ./.venv/Scripts/activate
python -m pip install -e .
```

## 2. Run the Example

### Preview

```bash
holodoppler preview "D:\path\to\holo.holo" "./src/holodoppler/default_parameters.json"
```

### Process

```bash
holodoppler process "D:\path\to\holo.holo" "./src/holodoppler/default_parameters.json"
```

### Building app

```bash
python -m pip install -e .[build]
python build_installer.py
```
