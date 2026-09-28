import subprocess
import os
import re
from smolagents.tools import tool
from typing import Any, Dict, List, Optional
import shutil


def _find_conda_path() -> Optional[str]:
    """Tries to find the path to conda.sh for activation via shutil.which."""
    conda_exe = shutil.which("conda")
    if not conda_exe:
        return None

    conda_base_dir = os.path.dirname(os.path.dirname(conda_exe))
    conda_sh_path = os.path.join(conda_base_dir, "etc", "profile.d", "conda.sh")

    return conda_sh_path if os.path.exists(conda_sh_path) else None


def _get_conda_envs() -> List[str]:
    """Gets a list of available conda environments."""
    try:
        result = subprocess.run(
            ["conda", "info", "--envs"], capture_output=True, text=True, check=True
        )
        envs = []
        for line in result.stdout.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            match = re.match(r"^(\S+)", line)
            if match:
                envs.append(match.group(1))
        return envs
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _get_available_modules() -> List[str]:
    """Gets a list of available environment modules."""
    try:
        # The `module avail` command often prints to stderr.
        result = subprocess.run(
            ["module", "avail"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Filter out empty lines from the output.
        return [line.strip() for line in result.stderr.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


@tool
def slurm_config() -> Dict[str, Any]:
    """
    Query the Slurm scheduler and environment for current cluster configuration.

    Returns a dict with five keys:
      - "partitions": a list of partitions, each being a dict with
           name (str), state (str), avail_nodes (int), total_nodes (int), max_time (str)
      - "nodes": a list of nodes, each being a dict with
           name (str), state (str), cpus (int), gres (str), gpu_model (str|None)
      - "conda_activation_path": path to the conda.sh script, or None if not found.
      - "available_conda_envs": a list of available conda environment names.
      - "available_modules": a list of available environment module names.

    Uses `sinfo`, `command`, `conda`, and `module` under the hood. Requires them to be in PATH.
    """

    # Query partitions: sinfo with format PartitionName|State|AllocNodes|TotalNodes|Timelimit
    p = subprocess.run(
        ["sinfo", "--noheader", "-o", "%P|%T|%a|%D|%l"], capture_output=True, text=True
    )
    partitions: List[Dict[str, Any]] = []
    for line in p.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        name, state, alloc, total, timelimit = parts
        try:
            alloc_n = int(alloc) if alloc.isdigit() else 0
            total_n = int(total) if total.isdigit() else 0
        except ValueError:
            alloc_n = total_n = 0
        partitions.append(
            {
                "name": name,
                "state": state,
                "alloc_nodes": alloc_n,
                "total_nodes": total_n,
                "max_time": timelimit,
            }
        )

    # Query nodes: sinfo with format NodeList|State|CPUsState|Gres
    p2 = subprocess.run(
        ["sinfo", "--noheader", "-N", "-o", "%N|%T|%c|%G"],
        capture_output=True,
        text=True,
    )
    nodes: List[Dict[str, Any]] = []
    for line in p2.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        name, state, cpus, gres = parts
        gpu_model = None
        if gres.startswith("gpu:"):
            gres_parts = gres.split(":")
            # Format can be gpu:model:count or gpu:count
            if len(gres_parts) == 3:
                gpu_model = gres_parts[1]

        try:
            cpu_n = int(cpus)
        except ValueError:
            cpu_n = 0
        nodes.append(
            {
                "name": name,
                "state": state,
                "cpus": cpu_n,
                "gres": gres,
                "gpu_model": gpu_model,
            }
        )

    conda_path = _find_conda_path()
    conda_envs = _get_conda_envs()
    modules = _get_available_modules()

    return {
        "partitions": partitions,
        "nodes": nodes,
        "conda_activation_path": conda_path,
        "available_conda_envs": conda_envs,
        "available_modules": modules,
    }
