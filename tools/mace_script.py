import json
import shlex
from typing import Any, Dict, List, Optional

from smolagents.tools import tool

from tools.mace_cli_schema import (
    MACECliSchema,
    load_mace_cli_schema,
    validate_mace_value,
)
from tools.session import get_session
from tools.specs import ClusterSpec, MACEJobSpec, raise_problems


def _format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _quote_cli(value: Any) -> str:
    return shlex.quote(_format_cli_value(value))


DEDICATED_MACE_OPTIONS = {
    "name",
    "train_file",
    "valid_file",
    "valid_fraction",
    "E0s",
    "max_num_epochs",
    "device",
}


def _canonical_option(schema: MACECliSchema, option: str) -> str:
    return schema.require_value(option).canonical


def validate_mace_cli_options(spec: MACEJobSpec, schema: MACECliSchema) -> None:
    for option in DEDICATED_MACE_OPTIONS:
        flag = schema.require_value(option)
        value = getattr(spec, "e0s" if option == "E0s" else option, None)
        if value is not None:
            validate_mace_value(flag, value)

    for key, value in spec.additional_args.items():
        flag = schema.require_value(key)
        if flag.canonical in DEDICATED_MACE_OPTIONS:
            raise ValueError(
                f"MACE option '--{flag.canonical}' has a dedicated argument and must not be repeated in additional_args."
            )
        validate_mace_value(flag, value)

    for flag_name in spec.additional_flag_args:
        flag = schema.require_switch(flag_name)
        if flag.canonical in DEDICATED_MACE_OPTIONS:
            raise ValueError(
                f"MACE option '--{flag.canonical}' has a dedicated argument and must not be repeated in additional_flag_args."
            )


def render_mace_script(spec: MACEJobSpec, schema: Optional[MACECliSchema] = None) -> str:
    spec.validate()
    schema = schema or load_mace_cli_schema()
    validate_mace_cli_options(spec, schema)

    cmd_lines = [
        "mace_run_train \\",
        f"    --{_canonical_option(schema, 'name')}={_quote_cli(spec.name)} \\",
        f"    --{_canonical_option(schema, 'train_file')}={_quote_cli(spec.train_file)} \\",
    ]
    if spec.valid_file:
        cmd_lines.append(
            f"    --{_canonical_option(schema, 'valid_file')}={_quote_cli(spec.valid_file)} \\"
        )
    else:
        cmd_lines.append(
            f"    --{_canonical_option(schema, 'valid_fraction')}={spec.valid_fraction} \\"
        )
    cmd_lines.extend(
        [
            f"    --{_canonical_option(schema, 'E0s')}={_quote_cli(spec.e0s)} \\",
            f"    --{_canonical_option(schema, 'max_num_epochs')}={spec.max_num_epochs} \\",
            f"    --{_canonical_option(schema, 'device')}={_quote_cli(spec.device)} \\",
        ]
    )

    for key, value in spec.additional_args.items():
        flag = schema.require_value(key)
        cmd_lines.append(f"    --{flag.canonical}={_quote_cli(value)} \\")
    for flag_name in spec.additional_flag_args:
        flag = schema.require_switch(flag_name)
        cmd_lines.append(f"    --{flag.canonical} \\")
    cmd_lines[-1] = cmd_lines[-1].rstrip(" \\")

    cluster = spec.cluster
    header = [
        "#!/bin/bash -l",
        f"#SBATCH --job-name={cluster.job_name}",
        f"#SBATCH --time={cluster.time}",
        f"#SBATCH --partition={cluster.partition}",
        f"#SBATCH --nodes={cluster.nodes}",
        f"#SBATCH --ntasks={cluster.ntasks}",
        f"#SBATCH --mem={cluster.memory}",
    ]
    if cluster.mail_user:
        header.append(f"#SBATCH --mail-user={cluster.mail_user}")
        if cluster.mail_type:
            header.append(f"#SBATCH --mail-type={cluster.mail_type}")
    if spec.device == "cuda" and cluster.gpus_per_node:
        header.append(f"#SBATCH --gres=gpu:{cluster.gpus_per_node}")
    for key, value in cluster.additional_slurm_args.items():
        if value is not None and value != "":
            header.append(f"#SBATCH --{key}={value}")

    setup = [
        "",
        f"source {cluster.conda_activation_path}",
        f"conda activate {cluster.conda_env}",
        "",
    ]

    return "\n".join(header + setup + cmd_lines)


@tool
def MACE_script(
    # Slurm parameters
    job_name: str = "mace_job",
    time: str = "24:00:00",
    partition: str = None,
    nodes: int = 1,
    ntasks: int = 1,
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
    E0s: str = None,
    max_num_epochs: Optional[int] = None,
    device: Optional[str] = "cuda",
    additional_args: Optional[Dict[str, Any]] = None,
    additional_flag_args: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Generate and save a validated Slurm .sh script for MACE training.

    The tool builds a MACEJobSpec first, fetches and parses the official MACE
    arg_parser.py schema, validates every MACE CLI option against that schema,
    then renders the bash script. Invalid input raises one error that lists every
    problem; fix all of them before calling again. Each call writes a new job
    bundle directory containing job.sh and returns {"script_path", "bundle",
    "script_content"}; submit it with slurm_submit(script_path).

    Args:
        job_name: Slurm job name.
        time: Slurm time limit, e.g. "24:00:00" or "2-00:00:00" (any sbatch form).
        partition: Slurm partition name.
        nodes: Number of Slurm nodes.
        ntasks: Number of Slurm tasks.
        gpus_per_node: Number of GPUs per node. Use 0 for CPU jobs.
        mail_user: Optional email address for Slurm notifications.
        mail_type: Optional Slurm mail type such as ALL, FAIL, BEGIN or END.
        memory: Slurm memory allocation, for example 32G.
        additional_args_slurm: Additional Slurm options as long option name -> single value, e.g. {"account": "proj1"}.
        conda_env: Conda environment to activate before running MACE.
        conda_activation_path: Path to conda.sh.
        name: MACE run name passed to --name.
        train_file: Absolute path of the MACE training XYZ file (the job runs in its bundle directory).
        valid_file: Optional absolute path of the MACE validation XYZ file.
        valid_fraction: Optional validation fraction. Mutually exclusive with valid_file.
        E0s: Isolated atom energy setting passed to --E0s.
        max_num_epochs: Maximum number of training epochs.
        device: MACE device. Valid choices are checked against the official MACE CLI schema.
        additional_args: Additional MACE key-value CLI flags.
        additional_flag_args: Additional MACE switch flags without values.

    Returns:
        Dictionary with script_path, bundle and script_content.
    """
    cluster = ClusterSpec(
        job_name=job_name,
        time=time,
        partition=partition,
        nodes=nodes,
        ntasks=ntasks,
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
        e0s=E0s,
        max_num_epochs=max_num_epochs,
        device=device,
        additional_args=additional_args or {},
        additional_flag_args=additional_flag_args or [],
    )
    problems = spec.problems()
    schema = load_mace_cli_schema()
    try:
        validate_mace_cli_options(spec, schema)
    except ValueError as error:
        problems.append(str(error))
    raise_problems("MACE_script", problems)

    script_content = render_mace_script(spec, schema)

    session = get_session()
    bundle = session.new_bundle(spec.cluster.job_name)
    artifact = session.write_artifact(bundle, "job.sh", script_content, "mace_script")
    return {"script_path": artifact.path, "bundle": artifact.bundle, "script_content": script_content}
