from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

SIMULATION_ATTR_OPTIONS = {
    "verbosity": {"quiet", "low", "medium", "high", "debug"},
    "threading": {True, False, "true", "false", "True", "False"},
    "mode": {"md", "static"},
}

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./~:+-]+$")
# sbatch time forms: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S
TIME_RE = re.compile(r"^(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?$")
MEMORY_RE = re.compile(r"^\d+[KMGT]?$")
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
SLURM_VALUE_RE = re.compile(r"^[A-Za-z0-9_.,:=@/%+-]+$")
MAIL_TYPES = {
    "NONE", "BEGIN", "END", "FAIL", "REQUEUE", "ALL", "INVALID_DEPEND", "STAGE_OUT",
    "TIME_LIMIT", "TIME_LIMIT_90", "TIME_LIMIT_80", "TIME_LIMIT_50", "ARRAY_TASKS",
}  # fmt: skip
# Options the generators write themselves; additional Slurm arguments must not repeat them.
DEDICATED_SLURM_OPTIONS = {
    "job-name", "time", "partition", "nodes", "ntasks", "cpus-per-task", "mem", "mail-user", "mail-type", "gres",
    "chdir", "export",
}  # fmt: skip


def check(problems: List[str], func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a raising validator; record its message instead of stopping at the first problem."""
    try:
        return func(*args, **kwargs)
    except ValueError as error:
        problems.append(str(error))
        return None


def raise_problems(what: str, problems: List[str]) -> None:
    """Raise one ValueError listing every problem as a bullet (nothing if the list is empty)."""
    if problems:
        bullets = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"{what}: {len(problems)} problem(s); fix all of them and call again:\n{bullets}")


def parse_slurm_time(text: Any) -> int:
    """Seconds of a Slurm time limit given in any sbatch form (a bare number means minutes)."""
    match = TIME_RE.fullmatch(text.strip()) if isinstance(text, str) else None
    if not match:
        raise ValueError(
            "time must use a Slurm format: M, M:S, H:M:S, D-H, D-H:M or D-H:M:S "
            f"(for example '24:00:00' or '2-00:00:00'); got {text!r}"
        )
    days, *rest = match.groups()
    parts = [int(part) for part in rest if part is not None]
    if days is not None:
        hours, minutes, seconds = (parts + [0, 0])[:3]
    elif len(parts) == 1:
        hours, minutes, seconds = 0, parts[0], 0
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        hours, minutes, seconds = parts
    total = ((int(days or 0) * 24 + hours) * 60 + minutes) * 60 + seconds
    if total <= 0:
        raise ValueError("time must be greater than zero")
    return total


def parse_slurm_limit(text: str) -> Optional[int]:
    """Seconds of a partition limit as sinfo prints it; None for 'infinite' or anything unreadable."""
    try:
        return parse_slurm_time(text)
    except ValueError:
        return None


def _require_text(name: str, value: Optional[str]) -> str:
    if not isinstance(value, str) or not value.strip():
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


def _memory(value: Any) -> str:
    text = _require_text("memory", value)
    if not MEMORY_RE.fullmatch(text):
        raise ValueError(f"memory must be a number with an optional K, M, G or T suffix, e.g. '32G'; got {value!r}")
    return text


def _mail_user(value: Any) -> Optional[str]:
    text = _optional_text("mail_user", value)
    if text is not None and not EMAIL_RE.fullmatch(text):
        raise ValueError(f"mail_user must be one e-mail address; got {value!r}")
    return text


def _mail_type(value: Any) -> Optional[str]:
    text = _optional_text("mail_type", value)
    if text is not None:
        unknown = [item for item in text.split(",") if item not in MAIL_TYPES]
        if unknown:
            raise ValueError(f"mail_type must be from {sorted(MAIL_TYPES)} (comma-separated); got {unknown}")
    return text


def _slurm_arg_problems(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return ["additional Slurm arguments must be a mapping of option name to value"]
    problems: List[str] = []
    for key, item in value.items():
        key = str(key)
        if not SAFE_KEY_RE.fullmatch(key) or len(key) < 2:
            problems.append(f"additional Slurm option {key!r} must be a long option name such as 'account'")
        elif "_" in key:
            problems.append(f"additional Slurm option {key!r} must use '-' instead of '_' ({key.replace('_', '-')!r})")
        elif key in DEDICATED_SLURM_OPTIONS:
            problems.append(f"additional Slurm option '--{key}' is set by a dedicated argument of this tool; use that")
        if item is None or item == "":
            continue
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            problems.append(f"additional Slurm option {key!r} needs a single text or number value; got {item!r}")
        elif not SLURM_VALUE_RE.fullmatch(str(item)):
            problems.append(
                f"additional Slurm option {key!r} value {item!r} must be one token of letters, digits and _.,:=@/%+-"
            )
    return problems


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
    cpus_per_task: Optional[int] = None  # None: Slurm's default, one core per task

    def problems(self, *, require_conda: bool = True) -> List[str]:
        p: List[str] = []
        check(p, _require_safe_name, "job_name", self.job_name)
        check(p, parse_slurm_time, self.time)
        check(p, _require_safe_name, "partition", self.partition)
        check(p, _positive_int, "nodes", self.nodes)
        check(p, _positive_int, "ntasks", self.ntasks)
        if self.cpus_per_task is not None:
            check(p, _positive_int, "cpus_per_task", self.cpus_per_task)
        check(p, _positive_int, "gpus_per_node", self.gpus_per_node, allow_zero=True)
        check(p, _memory, self.memory)
        check(p, _mail_user, self.mail_user)
        check(p, _mail_type, self.mail_type)
        p.extend(_slurm_arg_problems(self.additional_slurm_args))
        if require_conda:
            check(p, _require_safe_name, "conda_env", self.conda_env)
            check(p, _require_safe_path, "conda_activation_path", self.conda_activation_path)
        return p

    def validate(self, *, require_conda: bool = True) -> "ClusterSpec":
        raise_problems("Slurm settings", self.problems(require_conda=require_conda))
        return self

    def sbatch_header(self, gpus: int) -> List[str]:
        """Shebang and #SBATCH lines of a job script; a GPU line only when `gpus` > 0.

        --export=NONE: the job runs the environment its script sets up, not the shell the agent was started from
        (a PATH inherited that way once supplied an i-pi the job's conda environment does not have).
        """
        lines = [
            "#!/bin/bash -l",
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks={self.ntasks}",
            *([f"#SBATCH --cpus-per-task={self.cpus_per_task}"] if self.cpus_per_task else []),
            f"#SBATCH --mem={self.memory}",
            "#SBATCH --export=NONE",
        ]
        if self.mail_user:
            lines.append(f"#SBATCH --mail-user={self.mail_user}")
            if self.mail_type:
                lines.append(f"#SBATCH --mail-type={self.mail_type}")
        if gpus:
            lines.append(f"#SBATCH --gres=gpu:{gpus}")
        lines += [f"#SBATCH --{key}={value}" for key, value in self.additional_slurm_args.items()
                  if value is not None and value != ""]
        return lines

    def environment(self, *scripts: str) -> List[str]:
        """Shell lines that source `scripts` and conda.sh and activate the environment, with set -u off meanwhile:
        such scripts read unset variables."""
        return ["set +u  # environment and conda activation scripts may read unset variables",
                *(f"source {script}" for script in scripts),
                f"source {self.conda_activation_path}", f"conda activate {self.conda_env}", "set -u"]


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

    def problems(self) -> List[str]:
        p = self.cluster.problems(require_conda=True)
        check(p, _require_safe_name, "name", self.name)
        check(p, _require_safe_path, "train_file", self.train_file)
        if self.valid_file and self.valid_fraction is not None:
            p.append("valid_file and valid_fraction are mutually exclusive")
        elif self.valid_file:
            check(p, _require_safe_path, "valid_file", self.valid_file)
        elif self.valid_fraction is None:
            p.append("Either valid_file or valid_fraction is required")
        elif not isinstance(self.valid_fraction, (int, float)):
            p.append("valid_fraction must be a number")
        elif not 0 < float(self.valid_fraction) < 1:
            p.append("valid_fraction must be between 0 and 1")
        check(p, _require_text, "E0s", self.e0s)
        check(p, _positive_int, "max_num_epochs", self.max_num_epochs)
        check(p, _require_text, "device", self.device)
        check(p, _normalize_mapping, "additional_args", self.additional_args)
        check(p, _normalize_flags, "additional_flag_args", self.additional_flag_args)
        return p

    def validate(self) -> "MACEJobSpec":
        raise_problems("MACE job", self.problems())
        return self


@dataclass(frozen=True)
class IPIInputSpec:
    template_type: str
    modifications: Mapping[str, Any] = field(default_factory=dict)
    simulation_attrs: Mapping[str, Any] = field(default_factory=dict)

    def problems(self, *, templates: Mapping[str, Any]) -> List[str]:
        p: List[str] = []
        if self.template_type not in templates:
            p.append(f"Unknown template type: {self.template_type}. Available options: {list(templates)} "
                     "(ipi_template_schema() describes them)")
        if not isinstance(self.modifications, Mapping):
            p.append("modifications must be a mapping")
        if not isinstance(self.simulation_attrs, Mapping):
            p.append("simulation_attrs must be a mapping")
            return p
        for attr, value in self.simulation_attrs.items():
            attr_name = check(p, _validate_key, "simulation attribute", str(attr))
            if attr_name in SIMULATION_ATTR_OPTIONS:
                allowed = SIMULATION_ATTR_OPTIONS[attr_name]
                if value not in allowed:
                    p.append(f"simulation_attrs.{attr_name} must be one of {sorted(map(str, allowed))}")
            elif attr_name == "sockets_prefix":
                p.append("sockets_prefix is not accepted: the ASE client always connects under /tmp/ipi_, "
                         "so the tool sets the socket itself")
            elif attr_name is not None and attr_name not in {"safe_stride", "floatformat"}:
                p.append(f"Unsupported simulation attribute: {attr_name}")
        return p

    def validate(self, *, templates: Mapping[str, Any]) -> "IPIInputSpec":
        raise_problems("i-PI input", self.problems(templates=templates))
        return self


CLIENT_DEVICES = ("cpu", "cuda")


@dataclass(frozen=True)
class IPIASEClientSpec:
    calculator: str
    unixsocket_name: str = "driver"
    init_xyz_path: str = "init.xyz"
    use_stress: bool = True
    calculator_kwargs: Mapping[str, Any] = field(default_factory=dict)
    device: str = "cpu"

    def problems(self, *, supported_calculators: Mapping[str, str]) -> List[str]:
        p: List[str] = []
        calc_name = check(p, _validate_key, "calculator", str(self.calculator or "").lower())
        if calc_name is not None and calc_name not in supported_calculators:
            p.append(f"Unsupported calculator: '{self.calculator}'. Supported: {list(supported_calculators.keys())}")
        check(p, _require_safe_name, "unixsocket_name", self.unixsocket_name)
        check(p, _require_safe_path, "init_xyz_path", self.init_xyz_path)
        if not isinstance(self.use_stress, bool):
            p.append("use_stress must be a boolean")
        if self.device not in CLIENT_DEVICES:
            p.append(f"device must be one of {list(CLIENT_DEVICES)}; got {self.device!r}")
        if check(p, _normalize_mapping, "calculator_kwargs", self.calculator_kwargs) is not None:
            try:
                json.dumps(dict(self.calculator_kwargs))
            except (TypeError, ValueError):
                p.append("calculator_kwargs values must be plain data: text, numbers, true/false, None, lists or dicts")
            if "device" in self.calculator_kwargs:
                p.append("set the device with the device argument, not in calculator_kwargs")
            for key in self.calculator_kwargs:
                if not str(key).isidentifier():
                    p.append(f"calculator_kwargs key {key!r} is not a Python keyword-argument name")
        return p

    def validate(self, *, supported_calculators: Mapping[str, str]) -> "IPIASEClientSpec":
        raise_problems("ASE client", self.problems(supported_calculators=supported_calculators))
        return self


@dataclass(frozen=True)
class IPISubmitSpec:
    cluster: ClusterSpec
    ipi_input_path: str
    ase_path: str
    socket_address: str = "driver"
    ipi_env_path: Optional[str] = None
    num_force_providers: int = 1
    max_force_providers: int = 1  # beads x replicas of the input
    device: str = "cpu"  # of the ASE client

    def problems(self) -> List[str]:
        p = self.cluster.problems(require_conda=True)
        check(p, _require_safe_path, "ipi_input_path", self.ipi_input_path)
        check(p, _require_safe_path, "ase_path", self.ase_path)
        check(p, _require_safe_name, "socket_address", self.socket_address)
        if self.ipi_env_path is not None:
            check(p, _require_safe_path, "ipi_env_path", self.ipi_env_path)
        if check(p, _positive_int, "num_force_providers", self.num_force_providers) is not None:
            if self.num_force_providers > self.max_force_providers == 1:
                p.append("num_force_providers must be 1: the input has one bead and one replica, "
                         "so a second client would have nothing to compute")
            elif self.num_force_providers > self.max_force_providers:
                p.append(f"num_force_providers must be at most {self.max_force_providers} "
                         "(beads x replicas of the input); more clients would wait idle")
        if self.device == "cuda" and self.cluster.gpus_per_node == 0:
            p.append("the ASE client runs on cuda but gpus_per_node is 0; request a GPU or generate a cpu client")
        return p

    def validate(self) -> "IPISubmitSpec":
        raise_problems("i-PI job", self.problems())
        return self
