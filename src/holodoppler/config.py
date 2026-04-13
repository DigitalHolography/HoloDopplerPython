from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from importlib import resources
import json
from pathlib import Path
from typing import Any, get_type_hints


def _bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean.")


def _coerce_value(field_name: str, value: Any, target_type: type[Any]) -> Any:
    if target_type is bool:
        return _bool_from_value(value)
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    raise TypeError(f"Unsupported parameter type for {field_name!r}: {target_type!r}")


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    name: str
    label: str
    value_type: type[Any]
    choices: tuple[str, ...]


@dataclass(slots=True)
class ProcessingParameters:
    batch_size: int = field(default=256, metadata={"label": "Batch size"})
    batch_stride: int = field(default=256, metadata={"label": "Batch stride"})
    image_registration: bool = field(default=True, metadata={"label": "Image registration"})
    registration_disc_ratio: float = field(default=0.8, metadata={"label": "Registration disc ratio"})
    registration_flatfield_gw: int = field(default=35, metadata={"label": "Registration flatfield GW"})
    batch_size_registration: int = field(default=512, metadata={"label": "Registration batch size"})
    first_frame: int = field(default=0, metadata={"label": "First frame"})
    end_frame: int = field(default=0, metadata={"label": "End frame (0 = full file)"})
    wavelength: float = field(default=8.52e-7, metadata={"label": "Wavelength"})
    pixel_pitch: float = field(default=2e-5, metadata={"label": "Pixel pitch"})
    spatial_propagation: str = field(
        default="Fresnel",
        metadata={"label": "Spatial propagation", "choices": ("Fresnel", "AngularSpectrum")},
    )
    z: float = field(default=0.489, metadata={"label": "Propagation distance"})
    sampling_freq: float = field(default=37037.0, metadata={"label": "Sampling frequency"})
    low_freq: float = field(default=6000.0, metadata={"label": "Low frequency"})
    high_freq: float = field(default=18300.0, metadata={"label": "High frequency"})
    svd_threshold: int = field(default=64, metadata={"label": "SVD threshold"})
    square: bool = field(default=True, metadata={"label": "Square output"})
    transpose: bool = field(default=False, metadata={"label": "Transpose"})
    flip_x: bool = field(default=False, metadata={"label": "Flip X"})
    flip_y: bool = field(default=False, metadata={"label": "Flip Y"})

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if self.batch_stride <= 0:
            raise ValueError("batch_stride must be greater than 0.")
        if self.batch_size_registration <= 0:
            raise ValueError("batch_size_registration must be greater than 0.")
        if self.first_frame < 0:
            raise ValueError("first_frame cannot be negative.")
        if self.end_frame < 0:
            raise ValueError("end_frame cannot be negative.")
        if self.end_frame > 0 and self.end_frame <= self.first_frame:
            raise ValueError("end_frame must be greater than first_frame when it is provided.")
        if self.wavelength <= 0:
            raise ValueError("wavelength must be greater than 0.")
        if self.pixel_pitch <= 0:
            raise ValueError("pixel_pitch must be greater than 0.")
        if self.z <= 0:
            raise ValueError("z must be greater than 0.")
        if self.sampling_freq <= 0:
            raise ValueError("sampling_freq must be greater than 0.")
        if self.low_freq < 0:
            raise ValueError("low_freq cannot be negative.")
        if self.high_freq <= self.low_freq:
            raise ValueError("high_freq must be greater than low_freq.")
        if self.registration_disc_ratio < 0 or self.registration_disc_ratio > 1:
            raise ValueError("registration_disc_ratio must be between 0 and 1.")
        if self.registration_flatfield_gw <= 0:
            raise ValueError("registration_flatfield_gw must be greater than 0.")
        if self.spatial_propagation not in {"Fresnel", "AngularSpectrum"}:
            raise ValueError("spatial_propagation must be 'Fresnel' or 'AngularSpectrum'.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "ProcessingParameters":
        field_defs = {item.name: item for item in fields(cls)}
        type_hints = get_type_hints(cls)
        unknown_keys = sorted(set(values) - set(field_defs))
        if unknown_keys:
            unknown = ", ".join(unknown_keys)
            raise ValueError(f"Unknown parameter keys: {unknown}")

        normalized: dict[str, Any] = {}
        for name, field_def in field_defs.items():
            if name in values:
                normalized[name] = _coerce_value(name, values[name], type_hints[name])
                continue
            if field_def.default is not MISSING:
                normalized[name] = field_def.default
                continue
            raise ValueError(f"Missing required parameter: {name}")

        instance = cls(**normalized)
        instance.validate()
        return instance

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ProcessingParameters":
        source = Path(path)
        return cls.from_mapping(json.loads(source.read_text(encoding="utf-8")))

    @classmethod
    def default(cls) -> "ProcessingParameters":
        return load_builtin_parameters()


def parameter_definitions() -> tuple[ParameterDefinition, ...]:
    type_hints = get_type_hints(ProcessingParameters)
    definitions = []
    for field_def in fields(ProcessingParameters):
        definitions.append(
            ParameterDefinition(
                name=field_def.name,
                label=field_def.metadata.get("label", field_def.name.replace("_", " ").title()),
                value_type=type_hints[field_def.name],
                choices=tuple(field_def.metadata.get("choices", ())),
            )
        )
    return tuple(definitions)


def _settings_root():
    return resources.files("holodoppler.settings")


def available_builtin_settings() -> tuple[str, ...]:
    settings = sorted(
        item.name
        for item in _settings_root().iterdir()
        if item.is_file() and item.name.endswith(".json")
    )
    return tuple(settings)


def _normalize_setting_name(name: str) -> str:
    return name if name.endswith(".json") else f"{name}.json"


def builtin_setting_path(name: str):
    file_name = _normalize_setting_name(name)
    setting_file = _settings_root().joinpath(file_name)
    if not setting_file.is_file():
        available = ", ".join(available_builtin_settings()) or "<none>"
        raise FileNotFoundError(f"Unknown builtin setting {file_name!r}. Available settings: {available}")
    return setting_file


def load_builtin_parameters(name: str = "default_parameters") -> ProcessingParameters:
    setting_file = builtin_setting_path(name)
    return ProcessingParameters.from_mapping(json.loads(setting_file.read_text(encoding="utf-8")))


def export_builtin_setting(name: str, destination: str | Path) -> Path:
    setting_file = builtin_setting_path(name)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(setting_file.read_text(encoding="utf-8"), encoding="utf-8")
    return target
