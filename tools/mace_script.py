"""MACE_script: a Slurm job script for mace_run_train, validated against the options of the MACE that will run it."""

import ast
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from smolagents.tools import tool

from tools.mace_cli_schema import MACECliSchema, cli_text, describe_mace_envs, job_environment, load_mace_cli_schema
from tools.preflight import Problem, register
from tools.session import Artifact, Session, get_session
from tools.specs import SAFE_PATH_RE, ClusterSpec, MACEJobSpec, raise_problems

DEDICATED_MACE_OPTIONS = ("name", "train_file", "valid_file", "valid_fraction", "E0s", "max_num_epochs", "device")
# Options that name input files; the job runs in its bundle directory, so a relative path finds nothing.
INPUT_PATH_OPTIONS = {
    "train_file", "valid_file", "test_file", "test_dir", "pt_train_file", "pt_valid_file", "statistics_file",
    "les_arguments",
}  # fmt: skip
# key=value pairs of an extended XYZ comment line; values may be quoted, keys may hold brackets ("REF_energy(M)").
EXTXYZ_PAIR_RE = re.compile(r'([^\s=]+)=("[^"]*"|\S+)')
LABEL_KEY_OPTIONS = ("energy_key", "forces_key", "stress_key")
# MACE's DefaultKeys (mace/data/utils.py): recent versions give these as the defaults of energy_key and forces_key.
MACE_DEFAULT_KEYS = {"DefaultKeys.ENERGY.value": "REF_energy", "DefaultKeys.FORCES.value": "REF_forces"}


def is_input_path(option: str, value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if option == "E0s":
        return value.endswith(".json")
    if option == "foundation_model":  # otherwise a model name such as "medium"
        return "/" in value or value.endswith(".model")
    return option in INPUT_PATH_OPTIONS


def path_problem(option: str, value: Any) -> Optional[str]:
    if not is_input_path(option, value):
        return None
    if not value.startswith(("/", "~/")):
        return (f"{option} must be an absolute path or start with '~/': the job runs in its own bundle directory, "
                f"where {value!r} does not exist")
    if option not in ("train_file", "valid_file") and not SAFE_PATH_RE.fullmatch(value):
        return f"{option} may only contain letters, numbers, '_', '.', '/', '~', ':', '+' and '-'"
    return None


def e0s_problem(e0s: Any, foundation_model: Any) -> Optional[str]:
    """The forms MACE reads: tools/scripts_utils.py get_atomic_energies and cli/run_train.py (v0.3.17 lines 516-566)."""
    if not isinstance(e0s, str) or not e0s.strip():
        return None  # reported by the spec
    word = e0s.lower()
    if word in ("estimated", "foundation"):
        return None if foundation_model else f"E0s={e0s!r} needs a foundation_model in additional_args"
    if word == "average" or e0s.endswith(".json"):
        return None
    try:
        parsed = ast.literal_eval(e0s)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        parsed = None
    if isinstance(parsed, dict) and parsed and all(type(key) is int for key in parsed):
        return None  # atomic number -> energy; MACE looks energies up by atomic number
    return ("E0s must be 'average', a dict of atomic number to isolated-atom energy such as {1: -13.6, 8: -2041.8}, "
            f"the path of a .json file, or 'estimated' / 'foundation' with a foundation_model; got {e0s!r}")


def first_frame_keys(path: Path) -> Optional[List[str]]:
    """Label keys (info and per-atom properties) of the first frame of an extended XYZ file; None for plain XYZ."""
    with open(path, encoding="utf-8", errors="replace") as file:
        file.readline()
        pairs = dict(EXTXYZ_PAIR_RE.findall(file.readline()))
    if "Properties" not in pairs:
        return None
    columns = pairs.pop("Properties").split(":")[::3]
    return sorted({*pairs, *columns} - {"Lattice", "pbc", "species", "pos"})


def label_keys(spec: MACEJobSpec, table: MACECliSchema) -> Dict[str, Tuple[str, bool]]:
    """{option: (key, given)}: the label keys given in additional_args (in any spelling MACE accepts), and for
    energy_key and forces_key otherwise the default the job's MACE will use."""
    keys: Dict[str, Tuple[str, bool]] = {}
    for option, value in spec.additional_args.items() if isinstance(spec.additional_args, Mapping) else ():
        flag = table.get(option)
        if flag is not None and flag.name in LABEL_KEY_OPTIONS and isinstance(value, str):
            keys[flag.name] = (value, True)
    for option in ("energy_key", "forces_key"):
        flag = table.get(option)
        if option not in keys and flag is not None and isinstance(flag.default, str):
            keys[option] = (MACE_DEFAULT_KEYS.get(flag.default, flag.default), False)
    return keys


def label_key_check(spec: MACEJobSpec, table: MACECliSchema) -> Tuple[List[str], List[str]]:
    """(problems, notes) for a train_file readable here: each label key must be in its first frame (MACE does not
    stop on a missing label, it trains with that weight set to zero); a note lists the keys there."""
    if not isinstance(spec.train_file, str):
        return [], []  # reported by the spec
    path = Path(spec.train_file).expanduser()
    if path.suffix.lower() not in (".xyz", ".extxyz") or not path.is_file():
        return [], []
    found = first_frame_keys(path)
    if found is None:
        return [], []
    problems = []
    for option, (key, given) in label_keys(spec, table).items():
        if key not in found:
            what = f"{option} {key!r}" if given else f"{option} is not set, and MACE's default {key!r}"
            problems.append(f"{what} is not in the first frame of {spec.train_file}; its label keys: {found}")
    return problems, [f"label keys in the first frame of {spec.train_file}: {found}"]


def command_options(spec: MACEJobSpec) -> List[Tuple[str, Any]]:
    """(option, value) pairs of the mace_run_train command, dedicated options first; None values are left out."""
    pairs = [("name", spec.name), ("train_file", spec.train_file)]
    pairs.append(("valid_file", spec.valid_file) if spec.valid_file else ("valid_fraction", spec.valid_fraction))
    pairs += [("E0s", spec.e0s), ("max_num_epochs", spec.max_num_epochs), ("device", spec.device)]
    if isinstance(spec.additional_args, Mapping):
        pairs += [(str(key), value) for key, value in spec.additional_args.items()]
    return [(option, value) for option, value in pairs if value is not None]


def option_problems(spec: MACEJobSpec, table: MACECliSchema) -> List[str]:
    """Every option problem at once: names, value or switch, argparse types and choices, paths, E0s."""
    problems = []
    switches = spec.additional_flag_args if isinstance(spec.additional_flag_args, list) else []
    extra = list(spec.additional_args) if isinstance(spec.additional_args, Mapping) else []
    for option, value in command_options(spec):
        flag = table.get(option)
        for problem in (table.problem(option, value), path_problem(flag.name if flag else option, value)):
            if problem:
                problems.append(problem)
    for option in switches:
        problem = table.problem(option, switch=True)
        if problem:
            problems.append(problem)
    for option in [*extra, *switches]:
        flag = table.get(option)
        if flag is not None and flag.name in DEDICATED_MACE_OPTIONS:
            problems.append(f"--{flag.name} has a dedicated argument of MACE_script; set it there, not in additional_*")
    foundation = spec.additional_args.get("foundation_model") if isinstance(spec.additional_args, Mapping) else None
    problem = e0s_problem(spec.e0s, foundation)
    if problem:
        problems.append(problem)
    return problems


def cli_argument(name: str, value: Any) -> str:
    text = cli_text(value)
    if text.startswith("~/") and SAFE_PATH_RE.fullmatch(text):
        return f'--{name}="$HOME/{text[2:]}"'  # bash never expands a quoted ~, but expands $HOME in double quotes
    return f"--{name}={shlex.quote(text)}"


def render_mace_script(spec: MACEJobSpec, table: MACECliSchema) -> str:
    """The job script of a spec whose problems() and option_problems() are empty."""
    cluster = spec.cluster
    arguments = [cli_argument(table.get(option).name, value) for option, value in command_options(spec)]
    arguments += [f"--{table.get(switch).name}" for switch in spec.additional_flag_args]
    lines = [
        *cluster.sbatch_header(cluster.gpus_per_node if spec.device == "cuda" else 0),
        f"# MACE options checked against MACE {table.version} ({table.origin})",
        "set -euo pipefail",
        "",
        *cluster.environment(),
        "",
        " \\\n    ".join(["mace_run_train", *arguments]),
    ]
    return "\n".join(lines) + "\n"


def _conda_problems(problems: List[str], conda_activation_path: Optional[str], conda_env: Optional[str]) -> List[str]:
    """Spec problems with the cluster probe's facts added to the conda messages, plus an unknown environment."""
    cluster = get_session().cluster or {}
    envs = cluster.get("available_conda_envs") or []
    with_mace = describe_mace_envs(conda_activation_path)
    where = f"environments with MACE: {with_mace}" if with_mace else f"environments on this machine: {envs}"
    hints = {
        "conda_env is required": "conda_env is required: the conda environment with mace-torch on the cluster"
        + (f"; {where}" if envs else ""),
        "conda_activation_path is required": "conda_activation_path is required: the path of conda.sh on the "
        "cluster, e.g. ~/miniconda3/etc/profile.d/conda.sh (the cluster probe found none)",
    }
    problems = [hints.get(problem, problem) for problem in problems]
    if conda_env and envs and cluster.get("available") and conda_env not in envs:
        problems.append(f"conda_env {conda_env!r} is not a conda environment of this cluster; found: {envs}")
    return problems


@tool
def MACE_script(
    # Slurm parameters
    job_name: str = "mace_job",
    time: str = "24:00:00",
    partition: str = None,
    nodes: int = 1,
    ntasks: int = 1,
    cpus_per_task: Optional[int] = None,
    gpus_per_node: int = 1,
    mail_user: str = None,
    mail_type: str = "ALL",
    memory: str = "32G",
    additional_args_slurm: Optional[Dict[str, Any]] = None,
    conda_env: str = None,
    conda_activation_path: str = None,
    # Required MACE parameters
    name: str = "mace_job",
    train_file: str = None,
    valid_file: Optional[str] = None,
    valid_fraction: Optional[float] = None,
    E0s: Any = None,
    max_num_epochs: Optional[int] = None,
    device: Optional[str] = "cuda",
    additional_args: Optional[Dict[str, Any]] = None,
    additional_flag_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Write a Slurm job script that runs mace_run_train into a new job bundle (job.sh). Every MACE option is
    checked against the option table of the MACE version the job will run (the table mace_cli_schema shows),
    and the script stops at the first failing command. Invalid input raises one error listing every problem:
    fix all of them in the next call. Result: {"script_path", "bundle", "script_content", "notes"}; notes say
    which values were taken from the cluster probe, which MACE version the options were checked against and which
    label keys the first frame of the training file has (energy_key, forces_key, stress_key must be among them).
    Next: preflight(script_path), then slurm_submit(script_path), which shows the script to the person and asks.

    Args:
        job_name: Slurm job name: letters, digits, '.', '_' and '-'.
        time: Slurm time limit in any sbatch form, e.g. "24:00:00" or "2-00:00:00"; within the partition's max_time.
        partition: Slurm partition, one of those slurm_config lists.
        nodes: Number of Slurm nodes.
        ntasks: Number of Slurm tasks.
        cpus_per_task: CPU cores for the training (torch threads and data loading), e.g. 8 to 16 for a CPU job, 4 to 8 next to a GPU. Left out: one core.
        gpus_per_node: GPUs per node, requested when device is "cuda". Use 0 for CPU jobs.
        mail_user: Optional e-mail address for Slurm notifications.
        mail_type: Slurm mail type such as ALL, FAIL, BEGIN or END (comma-separated).
        memory: Slurm memory, a number with an optional K, M, G or T suffix, e.g. "32G".
        additional_args_slurm: Other sbatch options as long option name -> single value, e.g. {"account": "proj1"}.
        conda_env: Conda environment with mace-torch on the cluster. Left out: taken from the cluster probe when exactly one environment there has MACE installed.
        conda_activation_path: Path of conda.sh on the cluster. Left out: taken from the cluster probe.
        name: MACE run name (--name).
        train_file: Training file (.xyz or .h5): an absolute path or one starting with "~/"; the job runs in its bundle directory.
        valid_file: Validation file, same rule as train_file. Give valid_file or valid_fraction, not both.
        valid_fraction: Fraction of the training set used for validation, between 0 and 1.
        E0s: Isolated-atom energies: "average", a dict of atomic number to energy such as {1: -13.6, 8: -2041.8}, the absolute path of a .json file, or "estimated" / "foundation" together with additional_args foundation_model.
        max_num_epochs: Maximum number of training epochs (an integer).
        device: MACE device: "cuda" (requests gpus_per_node GPUs), "cpu", "mps" or "xpu".
        additional_args: Other value options of mace_run_train as name -> value, e.g. {"model": "MACE", "r_max": 5.0}; look them up with mace_cli_schema. Values that name input files follow the train_file rule.
        additional_flag_args: Switches of mace_run_train without a value, e.g. ["swa"].
    """
    conda_activation_path, conda_env, notes = job_environment(conda_activation_path, conda_env)
    cluster = ClusterSpec(
        job_name=job_name,
        time=time,
        partition=partition,
        nodes=nodes,
        ntasks=ntasks,
        cpus_per_task=cpus_per_task,
        gpus_per_node=gpus_per_node,
        memory=memory,
        conda_env=conda_env,
        conda_activation_path=conda_activation_path,
        mail_user=mail_user,
        mail_type=mail_type,
        additional_slurm_args=additional_args_slurm or {},
    )
    spec = MACEJobSpec(
        cluster=cluster,
        name=name,
        train_file=train_file,
        valid_file=valid_file,
        valid_fraction=valid_fraction,
        e0s=repr(E0s) if isinstance(E0s, dict) else E0s,
        max_num_epochs=max_num_epochs,
        device=device,
        additional_args=additional_args or {},
        additional_flag_args=additional_flag_args or [],
    )
    problems = _conda_problems(spec.problems(), conda_activation_path, conda_env)
    key_notes: List[str] = []
    try:
        table = load_mace_cli_schema(conda_activation_path, conda_env)
    except ValueError as error:
        problems.append(f"the MACE option table could not be loaded, so no option was checked: {error}")
    else:
        problems += option_problems(spec, table)
        key_problems, key_notes = label_key_check(spec, table)
        problems += key_problems
    raise_problems("MACE_script", problems)

    script_content = render_mace_script(spec, table)
    session = get_session()
    bundle = session.new_bundle(spec.cluster.job_name)
    artifact = session.write_artifact(bundle, "job.sh", script_content, "mace_script")
    notes.append(f"MACE options checked against MACE {table.version} ({table.origin})")
    notes += key_notes
    return {"script_path": artifact.path, "bundle": artifact.bundle, "script_content": script_content, "notes": notes}


def script_options(text: str) -> Optional[List[Tuple[str, Optional[str]]]]:
    """(option, value; None for a switch) of the mace_run_train command of a job script; None without one."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("mace_run_train")), None)
    if start is None:
        return None
    command = []
    for line in lines[start:]:
        command.append(line.rstrip().removesuffix("\\"))
        if not line.rstrip().endswith("\\"):
            break
    options = []
    for token in shlex.split(" ".join(command))[1:]:
        name, equals, value = token.removeprefix("--").partition("=")
        options.append((name, value if equals else None))
    return options


def _setup_value(text: str, prefix: str) -> Optional[str]:
    return next((line[len(prefix):].strip() for line in text.splitlines() if line.startswith(prefix)), None)


def check_mace_script(artifact: Artifact, session: Session) -> List[Problem]:
    """Pre-flight: the options still validate against the table of the target version; input files exist."""
    text = Path(artifact.path).read_text(encoding="utf-8")
    options = script_options(text)
    if options is None:
        return []
    try:
        table = load_mace_cli_schema(_setup_value(text, "source "), _setup_value(text, "conda activate "))
    except ValueError as error:
        return [Problem("warning", f"the MACE options were not checked again: {error}")]
    on_cluster = bool(session.cluster and session.cluster.get("available"))
    problems = []
    for name, value in options:
        problem = table.problem(name, value, switch=value is None)
        if problem:
            problems.append(Problem("error", problem))
            continue
        if value is None or not is_input_path(name, value.replace("$HOME/", "~/", 1)):
            continue
        path = Path.home() / value[len("$HOME/"):] if value.startswith("$HOME/") else Path(value)
        if path.exists():
            continue
        if on_cluster:
            problems.append(Problem("error", f"--{name}: {value} does not exist"))
        else:
            problems.append(Problem("warning", f"--{name}: {value} was not found on this machine; Slurm is not "
                                               "available here, so it was not checked where the job runs"))
    return problems


register("mace_script", check_mace_script)
