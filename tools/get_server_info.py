import copy
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from smolagents.tools import tool

from tools.session import get_session

COMMAND_TIMEOUT = 30


def run_command(argv: List[str], timeout: int = COMMAND_TIMEOUT) -> Tuple[Optional[str], str]:
    """(stdout, "") on success, else (None, a one-line reason)."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, f"{argv[0]} was not found on this machine"
    except subprocess.TimeoutExpired:
        return None, f"{argv[0]} did not answer in {timeout} s"
    if result.returncode != 0:
        detail = " ".join((result.stderr or result.stdout).split())[:300]
        return None, f"{argv[0]} exited with code {result.returncode}: {detail}"
    return result.stdout, ""


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
    stdout, _ = run_command(["conda", "info", "--envs"])
    envs = []
    for line in (stdout or "").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        match = re.match(r"^(\S+)", line)
        if match:
            envs.append(match.group(1))
    return envs


def _get_available_modules() -> List[str]:
    """Gets a list of available environment modules (POSIX only: `module` is a shell function).

    `module -t avail` (terse) prints one name per line, with "<directory>:" headers and "<family>/" lines.
    """
    if os.name != "posix":
        return []
    stdout, _ = run_command(["bash", "-lc", "module -t avail 2>&1"])
    lines = (line.strip().removesuffix("(default)") for line in (stdout or "").splitlines())
    return [line for line in lines if line and not line.endswith((":", "/"))]


def parse_partitions(text: str) -> List[Dict[str, Any]]:
    """Rows of `sinfo -o '%P|%a|%F|%l'`, merged by partition name (sinfo may print one per node state)."""
    merged: Dict[str, Dict[str, Any]] = {}
    for line in text.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        name, state, node_counts, timelimit = (part.strip() for part in parts)
        if not name:
            continue
        counts = node_counts.split("/")
        avail_n, total_n = (int(counts[1]), int(counts[3])) if len(counts) == 4 and all(
            c.isdigit() for c in counts) else (0, 0)
        entry = merged.setdefault(name.rstrip("*"), {
            "name": name.rstrip("*"), "state": state, "avail_nodes": 0, "total_nodes": 0, "max_time": timelimit,
        })
        entry["avail_nodes"] += avail_n
        entry["total_nodes"] += total_n
    return list(merged.values())


def parse_nodes(text: str) -> List[Dict[str, Any]]:
    """Rows of `sinfo -N -o '%N|%P|%T|%c|%G'`, one per node with the partitions it belongs to
    (sinfo prints a node once per partition)."""
    nodes: Dict[str, Dict[str, Any]] = {}
    for line in text.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        name, partition, state, cpus, gres = parts
        gres_parts = gres.split(":")
        # Format can be gpu:model:count or gpu:count
        gpu_model = gres_parts[1] if gres.startswith("gpu:") and len(gres_parts) == 3 else None
        node = nodes.setdefault(name, {
            "name": name,
            "partitions": [],
            "state": state,
            "cpus": int(cpus) if cpus.isdigit() else 0,
            "gres": gres,
            "gpu_model": gpu_model,
        })
        node["partitions"].append(partition.rstrip("*"))
    return list(nodes.values())


def probe_cluster() -> Dict[str, Any]:
    partitions_out, reason = run_command(["sinfo", "--noheader", "-o", "%P|%a|%F|%l"])
    nodes_out = None
    if partitions_out is not None:
        nodes_out, reason = run_command(["sinfo", "--noheader", "-N", "-o", "%N|%P|%T|%c|%G"])
    available = partitions_out is not None and nodes_out is not None
    return {
        "available": available,
        "reason": "" if available else f"Slurm facts unavailable: {reason}",
        "partitions": parse_partitions(partitions_out or ""),
        "nodes": parse_nodes(nodes_out or ""),
        "conda_activation_path": _find_conda_path(),
        "available_conda_envs": _get_conda_envs(),
        "available_modules": _get_available_modules(),
    }


@tool
def slurm_config(refresh: bool = False) -> Dict[str, Any]:
    """
    Query the Slurm scheduler and environment for current cluster configuration.

    The result is kept for the session (and used by the pre-flight checks of slurm_submit);
    pass refresh=True to ask the cluster again. Result: a dict with keys
      - "available": True when sinfo answered; False when Slurm is not usable here.
      - "reason": why Slurm facts are unavailable ("" when available).
      - "partitions": a list of partitions, each being a dict with
           name (str), state (str), avail_nodes (int), total_nodes (int), max_time (str)
      - "nodes": a list of nodes, each being a dict with
           name (str), partitions (list of str), state (str), cpus (int), gres (str), gpu_model (str|None)
      - "conda_activation_path": path to the conda.sh script, or None if not found.
      - "available_conda_envs": a list of available conda environment names.
      - "available_modules": a list of available environment module names.

    Args:
        refresh: Ask the cluster again instead of returning the result kept for this session.
    """
    session = get_session()
    if refresh or session.cluster is None:
        session.cluster = probe_cluster()
    return copy.deepcopy(session.cluster)  # the pre-flight checks read the kept dict; the caller gets a copy
