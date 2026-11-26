"""job_output: read what a job submitted in this session wrote into its bundle directory, and nothing else."""

from collections import deque
from pathlib import Path
from typing import Optional

from smolagents.tools import tool

from tools.session import get_session

MAX_LINES = 200


def _tail(path: Path, count: int) -> tuple[int, str]:
    """(number of lines, last `count` lines) of a text file, read once without holding it in memory."""
    with open(path, encoding="utf-8", errors="replace") as file:
        last, total = deque(maxlen=count), 0
        for line in file:
            last.append(line.rstrip("\n"))
            total += 1
    return total, "\n".join(last)


@tool
def job_output(job_id: str, name: Optional[str] = None, lines: int = 40) -> dict:
    """
    Read the files a job submitted in this session wrote into its bundle directory: MACE's training log
    (logs/*.log), i-PI's log.i-pi and simulation.out, slurm-<job_id>.out, ...
    Without name: {"bundle", "files": [{"name", "bytes"}]}. With name: {"name", "total_lines", "tail"}, the last
    lines of that file.

    Args:
        job_id: The job id returned by slurm_submit.
        name: A file name from the listing, relative to the bundle, e.g. "log.i-pi" or "logs/water_run-123.log".
        lines: How many lines from the end of the file to return (at most 200).
    """
    job = get_session().find_job(job_id)
    if job is None:
        raise ValueError(f"job {job_id!r} was not submitted in this session; job_output reads only those jobs' files")
    bundle = Path(job.bundle).resolve()
    if name is None:
        files = sorted(path for path in bundle.rglob("*") if path.is_file())
        return {"bundle": job.bundle,
                "files": [{"name": path.relative_to(bundle).as_posix(), "bytes": path.stat().st_size} for path in files]}
    path = (bundle / name).resolve()
    if not path.is_relative_to(bundle) or not path.is_file():
        raise ValueError(f"{name!r} is not a file of the bundle {job.bundle}; job_output({job_id!r}) lists them")
    with open(path, "rb") as file:
        if b"\0" in file.read(4096):
            raise ValueError(f"{name!r} is a binary file (a model or checkpoint), not a log")
    total, tail = _tail(path, min(max(int(lines), 1), MAX_LINES))
    return {"name": path.relative_to(bundle).as_posix(), "total_lines": total, "tail": tail}
