"""Checks run on a job script before it is submitted; also available to the agents as a read-only tool."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from smolagents.tools import tool

from tools.session import Artifact, Session, get_session
from tools.specs import parse_slurm_limit, parse_slurm_time

SCRIPT_KINDS = ("mace_script", "ipi_submit")
# Files a script of this kind runs; they must sit in its bundle, unchanged, and be named in the script.
NEEDS = {"ipi_submit": ("ipi_input", "ase_client")}
SBATCH_RE = re.compile(r"^#SBATCH\s+--([A-Za-z][A-Za-z-]*)(?:[=\s]+(\S+))?")


@dataclass(frozen=True)
class Problem:
    severity: str  # "error" | "warning"
    message: str


Checker = Callable[[Artifact, Session], list[Problem]]
_CHECKERS: dict[str, list[Checker]] = {}


def register(kind: str, checker: Checker) -> None:
    _CHECKERS.setdefault(kind, []).append(checker)


def run_preflight(artifact: Artifact, session: Session) -> list[Problem]:
    problems: list[Problem] = []
    for checker in _CHECKERS.get(artifact.kind, []):
        problems.extend(checker(artifact, session))
    return problems


def sbatch_options(text: str) -> dict[str, str]:
    """`#SBATCH --key=value` lines before the first command (sbatch ignores later ones)."""
    options: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#SBATCH"):
            match = SBATCH_RE.match(line)
            if match:
                options[match[1]] = match[2] or ""
        elif line.strip() and not line.startswith("#"):
            break
    return options


def _options(artifact: Artifact) -> dict[str, str]:
    return sbatch_options(Path(artifact.path).read_text(encoding="utf-8"))


def check_header(artifact: Artifact, session: Session) -> list[Problem]:
    options = _options(artifact)
    problems = []
    if not options.get("job-name"):
        problems.append(Problem("error", "the #SBATCH header has no --job-name"))
    try:
        parse_slurm_time(options.get("time"))
    except ValueError as error:
        problems.append(Problem("error", f"#SBATCH --time: {error}"))
    return problems


def check_cluster(artifact: Artifact, session: Session) -> list[Problem]:
    cluster = session.cluster
    if not cluster or not cluster.get("available"):
        return [Problem("warning", "cluster facts unknown (no slurm_config result with Slurm available); "
                                   "partition and time limit were not checked")]
    options = _options(artifact)
    partitions = {p["name"]: p for p in cluster.get("partitions", [])}
    name = options.get("partition", "")
    if name not in partitions:
        return [Problem("error", f"partition {name!r} does not exist on this cluster; available: {sorted(partitions)}")]
    limit = parse_slurm_limit(partitions[name].get("max_time", ""))
    try:
        seconds = parse_slurm_time(options.get("time"))
    except ValueError:
        return []  # reported by check_header
    if limit is not None and seconds > limit:
        return [Problem("error", f"time {options['time']} exceeds the limit {partitions[name]['max_time']} "
                                 f"of partition {name!r}")]
    return []


def check_bundle(artifact: Artifact, session: Session) -> list[Problem]:
    problems = []
    members = [a for a in session.artifacts if a.bundle == artifact.bundle]
    for member in members:
        if not session.is_unchanged(member):
            problems.append(Problem("error", f"{member.path} was changed or removed after it was generated"))
    words = Path(artifact.path).read_text(encoding="utf-8").split()
    for kind in NEEDS.get(artifact.kind, ()):
        if not any(a.kind == kind and Path(a.path).name in words for a in members):
            problems.append(Problem("error", f"the script does not run a {kind} file generated in its bundle "
                                             f"{artifact.bundle}"))
    return problems


for _kind in SCRIPT_KINDS:
    for _checker in (check_header, check_cluster, check_bundle):
        register(_kind, _checker)


@tool
def preflight(script_path: str) -> dict:
    """
    Run the checks slurm_submit runs before submitting, without submitting anything.
    Result: {"ok": bool, "errors": [...], "warnings": [...]}; ok is True when there is no error.

    Args:
        script_path: script_path returned by MACE_script or ipi_submit_generator.
    """
    session = get_session()
    artifact = session.find_artifact(script_path)
    if artifact is None or artifact.kind not in SCRIPT_KINDS:
        errors = [f"{script_path!r} is not a job script generated in this session"]
        return {"ok": False, "errors": errors, "warnings": []}
    if not session.is_unchanged(artifact):
        errors = [f"{artifact.path} was changed or removed after it was generated; generate it again"]
        return {"ok": False, "errors": errors, "warnings": []}
    problems = run_preflight(artifact, session)
    errors = [p.message for p in problems if p.severity == "error"]
    warnings = [p.message for p in problems if p.severity == "warning"]
    return {"ok": not errors, "errors": errors, "warnings": warnings}
