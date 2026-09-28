import os
from datetime import datetime
from typing import Any, Dict, Optional

from smolagents.tools import tool

from tools.specs import ClusterSpec, IPISubmitSpec, WorkflowArtifact


def render_ipi_submit_script(spec: IPISubmitSpec) -> str:
    spec.validate()
    cluster = spec.cluster

    header = [
        "#!/bin/bash -l",
        "set -euo pipefail",
        f"#SBATCH --job-name={cluster.job_name}",
        f"#SBATCH --time={cluster.time}",
        f"#SBATCH --partition={cluster.partition}",
        f"#SBATCH --nodes={cluster.nodes}",
        f"#SBATCH --ntasks={cluster.ntasks}",
        f"#SBATCH --mem={cluster.memory}",
    ]
    if cluster.gpus_per_node:
        header.append(f"#SBATCH --gres=gpu:{cluster.gpus_per_node}")
    for key, value in cluster.additional_slurm_args.items():
        if value is not None and value != "":
            header.append(f"#SBATCH --{key}={value}")

    script_lines = header + [
        "",
        f"source {spec.ipi_env_path}",
        f"source {cluster.conda_activation_path}",
        f"conda activate {cluster.conda_env}",
        "",
        f"i-pi {spec.ipi_input_path} &> log.i-pi &",
        "IPI_PID=$!",
        f"sleep {spec.sleep_time}",
        "",
    ]

    if spec.multi_run:
        script_lines.extend(
            [
                "CLIENT_PIDS=()",
                f"for x in {{1..{spec.num_force_providers}}}",
                "do",
                f"    python {spec.ase_path} &",
                "    CLIENT_PIDS+=(\"$!\")",
                "done",
                "",
                "for pid in \"${CLIENT_PIDS[@]}\"",
                "do",
                "    wait \"$pid\"",
                "done",
            ]
        )
    else:
        script_lines.append(f"python {spec.ase_path}")

    script_lines.extend(["", "wait \"$IPI_PID\""])
    return "\n".join(script_lines)


@tool
def ipi_submit_generator(
    ipi_input_path: str = "input.xml",
    ase_path: str = "run_ase.py",
    job_name: str = "I-PI",
    time: str = "24:00:00",
    partition: str = None,
    nodes: int = 1,
    ntasks: int = 1,
    gpus_per_node: int = 1,
    memory: str = "20G",
    conda_env: str = None,
    conda_path: str = None,
    ipi_env_path: str = "~/i-pi/env.sh",
    sleep_time: int = 30,
    multi_run: bool = False,
    num_force_providers: int = 1,
    additional_slurm_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate a validated Slurm script for running i-PI with one or more ASE clients.

    Args:
        ipi_input_path: Path to the generated i-PI input XML file.
        ase_path: Path to the generated ASE socket client script.
        job_name: Slurm job name.
        time: Slurm time limit in HH:MM:SS format.
        partition: Slurm partition name.
        nodes: Number of Slurm nodes.
        ntasks: Number of Slurm tasks.
        gpus_per_node: Number of GPUs per node. Use 0 for CPU jobs.
        memory: Slurm memory allocation, for example 20G.
        conda_env: Conda environment to activate before running the ASE client.
        conda_path: Path to conda.sh.
        ipi_env_path: Path to the i-PI environment setup script.
        sleep_time: Seconds to wait after starting i-PI before launching clients.
        multi_run: Whether to launch multiple ASE force-provider clients.
        num_force_providers: Number of ASE force-provider clients to launch.
        additional_slurm_args: Additional Slurm arguments as key-value pairs.

    Returns:
        Dictionary with script_path, script_content and num_force_providers.
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
        conda_activation_path=conda_path,
        additional_slurm_args=additional_slurm_args or {},
    )
    spec = IPISubmitSpec(
        cluster=cluster,
        ipi_input_path=ipi_input_path,
        ase_path=ase_path,
        ipi_env_path=ipi_env_path,
        sleep_time=sleep_time,
        multi_run=multi_run,
        num_force_providers=num_force_providers,
    ).validate()

    script_content = render_ipi_submit_script(spec)

    os.makedirs("ipi_scripts", exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    script_path = f"ipi_scripts/{spec.cluster.job_name}_{timestamp}.sh"
    artifact = WorkflowArtifact(
        kind="ipi_slurm_script",
        path=script_path,
        content=script_content,
        metadata={
            "job_name": spec.cluster.job_name,
            "num_force_providers": spec.num_force_providers,
        },
    ).validate()

    with open(artifact.path, "w", encoding="utf-8") as file:
        file.write(artifact.content)

    return {
        "script_path": artifact.path,
        "script_content": artifact.content,
        "num_force_providers": spec.num_force_providers,
    }
