from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


JsonMapping = Mapping[str, Any]

IPI_TEMPLATE_URLS = {
    "00-NVE": "00-NVE/input.xml",
    "01-NVT-for-dynamics": "01-NVT-for-dynamics/input.xml",
    "02-NVT-Langevin": "02-NVT-Langevin/input.xml",
    "03-NVT-SVR": "03-NVT-SVR/input.xml",
    "04-NVT-GLE": "04-NVT-GLE/input.xml",
    "05-NPT-isotropic-BZP": "05-NPT-isotropic-BZP/input.xml",
    "06-flexible-MTTK": "06-flexible-MTTK/input.xml",
    "07-REMD-NVT": "07-REMD-NVT/input.xml",
    "08-REMD-NPT": "08-REMD-NPT/input.xml",
    "09-REPIMD-NVT-for-dynamics": "09-REPIMD-NVT-for-dynamics/input.xml",
}

SIMULATION_ATTR_OPTIONS = {
    "verbosity": {"quiet", "low", "medium", "high", "debug"},
    "threading": {True, False, "true", "false", "True", "False"},
    "mode": {"md", "static"},
}

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./~:+-]+$")
TIME_RE = re.compile(r"^\d{1,3}:\d{2}:\d{2}$")


def _require_text(name: str, value: Optional[str]) -> str:
    if value is None or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    _reject_control_chars(name, value)
    return value.strip()


def _optional_text(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value.strip():
        return None
    _reject_control_chars(name, value)
    return value.strip()


def _reject_control_chars(name: str, value: str) -> None:
    if any(ord(ch) < 32 for ch in value):
        raise ValueError(f"{name} must not contain control characters")


def _require_safe_name(name: str, value: Optional[str]) -> str:
    text = _require_text(name, value)
    if not SAFE_NAME_RE.fullmatch(text):
        raise ValueError(f"{name} may only contain letters, numbers, '.', '_' and '-'")
    return text


def _require_safe_path(name: str, value: Optional[str]) -> str:
    text = _require_text(name, value)
    if not SAFE_PATH_RE.fullmatch(text):
        raise ValueError(
            f"{name} may only contain letters, numbers, '_', '.', '/', '~', ':', '+' and '-'"
        )
    return text


def _positive_int(name: str, value: int, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be zero or greater")
    elif value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _validate_key(name: str, value: str) -> str:
    text = _require_text(name, value)
    if not SAFE_KEY_RE.fullmatch(text):
        raise ValueError(f"{name} must be a valid CLI or Slurm key")
    return text


def _normalize_mapping(
    name: str, value: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: Dict[str, Any] = {}
    for key, item in value.items():
        safe_key = _validate_key(f"{name} key", str(key))
        if isinstance(item, str):
            _reject_control_chars(f"{name}.{safe_key}", item)
        result[safe_key] = item
    return result


def _normalize_flags(name: str, value: Optional[List[str]]) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [_validate_key(f"{name} item", str(flag)) for flag in value]


@dataclass(frozen=True)
class WorkflowArtifact:
    kind: str
    path: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "WorkflowArtifact":
        _require_text("artifact kind", self.kind)
        _require_safe_path("artifact path", self.path)
        if not isinstance(self.content, str):
            raise ValueError("artifact content must be a string")
        return self


@dataclass(frozen=True)
class ClusterSpec:
    job_name: str
    time: str
    partition: str
    nodes: int = 1
    ntasks: int = 1
    gpus_per_node: int = 1
    memory: str = "32G"
    conda_env: Optional[str] = None
    conda_activation_path: Optional[str] = None
    mail_user: Optional[str] = None
    mail_type: Optional[str] = None
    additional_slurm_args: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, *, require_conda: bool = True) -> "ClusterSpec":
        _require_safe_name("job_name", self.job_name)
        if not isinstance(self.time, str) or not TIME_RE.fullmatch(self.time):
            raise ValueError("time must use HH:MM:SS format")
        _require_safe_name("partition", self.partition)
        _positive_int("nodes", self.nodes)
        _positive_int("ntasks", self.ntasks)
        _positive_int("gpus_per_node", self.gpus_per_node, allow_zero=True)
        _require_text("memory", self.memory)
        _optional_text("mail_user", self.mail_user)
        _optional_text("mail_type", self.mail_type)
        _normalize_mapping("additional_slurm_args", self.additional_slurm_args)
        if require_conda:
            _require_safe_name("conda_env", self.conda_env)
            _require_safe_path("conda_activation_path", self.conda_activation_path)
        return self


@dataclass(frozen=True)
class MACEJobSpec:
    cluster: ClusterSpec
    name: str
    train_file: str
    e0s: str
    max_num_epochs: int
    valid_file: Optional[str] = None
    valid_fraction: Optional[float] = None
    device: str = "cuda"
    additional_args: Mapping[str, Any] = field(default_factory=dict)
    additional_flag_args: List[str] = field(default_factory=list)

    def validate(self) -> "MACEJobSpec":
        self.cluster.validate(require_conda=True)
        _require_safe_name("name", self.name)
        _require_safe_path("train_file", self.train_file)
        if self.valid_file and self.valid_fraction is not None:
            raise ValueError("valid_file and valid_fraction are mutually exclusive")
        if self.valid_file:
            _require_safe_path("valid_file", self.valid_file)
        elif self.valid_fraction is None:
            raise ValueError("Either valid_file or valid_fraction is required")
        else:
            if not isinstance(self.valid_fraction, (int, float)):
                raise ValueError("valid_fraction must be a number")
            if not 0 < float(self.valid_fraction) < 1:
                raise ValueError("valid_fraction must be between 0 and 1")
        _require_text("E0s", self.e0s)
        _positive_int("max_num_epochs", self.max_num_epochs)
        _require_text("device", self.device)
        _normalize_mapping("additional_args", self.additional_args)
        _normalize_flags("additional_flag_args", self.additional_flag_args)
        return self


@dataclass(frozen=True)
class IPIInputSpec:
    template_type: str
    modifications: Mapping[str, Any] = field(default_factory=dict)
    simulation_attrs: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "IPIInputSpec":
        if self.template_type not in IPI_TEMPLATE_URLS:
            raise ValueError(
                f"Unknown template type: {self.template_type}. Available options: {list(IPI_TEMPLATE_URLS.keys())}"
            )
        if not isinstance(self.modifications, Mapping):
            raise ValueError("modifications must be a mapping")
        if not isinstance(self.simulation_attrs, Mapping):
            raise ValueError("simulation_attrs must be a mapping")
        for attr, value in self.simulation_attrs.items():
            attr_name = _validate_key("simulation attribute", str(attr))
            if attr_name in SIMULATION_ATTR_OPTIONS:
                allowed = SIMULATION_ATTR_OPTIONS[attr_name]
                if value not in allowed:
                    raise ValueError(
                        f"simulation_attrs.{attr_name} must be one of {sorted(map(str, allowed))}"
                    )
            elif attr_name not in {"safe_stride", "floatformat", "sockets_prefix"}:
                raise ValueError(f"Unsupported simulation attribute: {attr_name}")
        return self


@dataclass(frozen=True)
class IPIASEClientSpec:
    calculator: str
    unixsocket_name: str = "driver"
    init_xyz_path: str = "init.xyz"
    use_stress: bool = True
    calculator_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, *, supported_calculators: Mapping[str, str]) -> "IPIASEClientSpec":
        calc_name = _validate_key("calculator", self.calculator.lower())
        if calc_name not in supported_calculators:
            raise ValueError(
                f"Unsupported calculator: '{self.calculator}'. Supported: {list(supported_calculators.keys())}"
            )
        _require_safe_name("unixsocket_name", self.unixsocket_name)
        _require_safe_path("init_xyz_path", self.init_xyz_path)
        if not isinstance(self.use_stress, bool):
            raise ValueError("use_stress must be a boolean")
        _normalize_mapping("calculator_kwargs", self.calculator_kwargs)
        return self


@dataclass(frozen=True)
class IPISubmitSpec:
    cluster: ClusterSpec
    ipi_input_path: str
    ase_path: str
    ipi_env_path: str = "~/i-pi/env.sh"
    sleep_time: int = 30
    multi_run: bool = False
    num_force_providers: int = 1

    def validate(self) -> "IPISubmitSpec":
        self.cluster.validate(require_conda=True)
        _require_safe_path("ipi_input_path", self.ipi_input_path)
        _require_safe_path("ase_path", self.ase_path)
        _require_safe_path("ipi_env_path", self.ipi_env_path)
        _positive_int("sleep_time", self.sleep_time)
        if not isinstance(self.multi_run, bool):
            raise ValueError("multi_run must be a boolean")
        _positive_int("num_force_providers", self.num_force_providers)
        if not self.multi_run and self.num_force_providers != 1:
            raise ValueError("num_force_providers must be 1 when multi_run is False")
        return self
