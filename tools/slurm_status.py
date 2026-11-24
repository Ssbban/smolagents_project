import subprocess
import time
from pathlib import Path
from typing import Optional

from smolagents.tools import tool

from tools.session import get_session

SQUEUE_FORMAT = "%i|%T|%r|%M"
SACCT_FIELDS = "JobID,State,ExitCode,Elapsed,Reason"
FINISHED = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
    "BOOT_FAIL", "DEADLINE",
}  # fmt: skip
POLL_SECONDS = 15
MAX_WAIT_SECONDS = 600
QUERY_TIMEOUT = 30
LOG_LINES = 40

_sleep = time.sleep
_clock = time.monotonic


class _NoSlurm(Exception):
    pass


def _reason(text: str) -> Optional[str]:
    text = text.strip()
    return None if text in {"", "None", "(null)"} else text


def parse_squeue(text: str, job_id: str) -> Optional[dict]:
    """The row of `squeue --noheader -o '%i|%T|%r|%M'` for this job, or None."""
    for line in text.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 4 and fields[0] == job_id:
            return {"state": fields[1].strip(), "reason": _reason(fields[2]), "elapsed": fields[3].strip(),
                    "exit_code": None}
    return None


def parse_sacct(text: str, job_id: str) -> Optional[dict]:
    """The allocation row of `sacct -X -P -n --format=JobID,State,ExitCode,Elapsed,Reason`, or None."""
    for line in text.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 4 and fields[0] == job_id:
            words = fields[1].split()
            state = words[0].rstrip("+") if words else "UNKNOWN"
            reason = _reason(fields[4]) if len(fields) > 4 else None
            if len(words) > 1:  # "CANCELLED by 1234"
                reason = " ".join(filter(None, [fields[1].strip(), reason]))
            return {"state": state, "reason": reason, "elapsed": fields[3].strip(),
                    "exit_code": fields[2].strip() or None}
    return None


def _run(argv: list[str]) -> Optional[str]:
    """stdout of a Slurm query; None when it failed or timed out; _NoSlurm when squeue is missing."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=QUERY_TIMEOUT)
    except FileNotFoundError:
        if argv[0] == "squeue":
            raise _NoSlurm from None
        return None
    except subprocess.TimeoutExpired:
        return None
    return result.stdout if result.returncode == 0 else None


def _query(job_id: str) -> dict:
    queued = parse_squeue(_run(["squeue", "--noheader", "-j", job_id, "-o", SQUEUE_FORMAT]) or "", job_id)
    if queued and queued["state"] not in FINISHED:
        return queued
    accounted = parse_sacct(_run(["sacct", "-X", "-P", "-n", "-j", job_id, f"--format={SACCT_FIELDS}"]) or "", job_id)
    return accounted or queued or {"state": "UNKNOWN", "reason": "neither squeue nor sacct knows this job (yet)",
                                   "elapsed": None, "exit_code": None}


def _log_tail(job_id: str) -> Optional[str]:
    job = get_session().find_job(job_id)
    if job is None:
        return None
    log = Path(job.bundle) / f"slurm-{job_id}.out"
    if not log.is_file():
        return None
    return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_LINES:])


@tool
def job_status(job_id: str, wait_seconds: int = 0) -> dict:
    """
    Report the state of a Slurm job: squeue while it is queued or running, sacct after it left the queue.

    Result: {"job_id", "state" (e.g. PENDING, RUNNING, COMPLETED, FAILED, UNKNOWN), "finished" (bool),
    "exit_code" ("code:signal" from sacct, or None), "elapsed", "reason", "log_tail" (last 40 lines of
    slurm-<id>.out of a job submitted in this session, or None)}. A job still running is not an error.
    UNKNOWN means Slurm is not installed here or does not know the job (yet); "reason" says which.

    Args:
        job_id: The job id returned by slurm_submit.
        wait_seconds: Wait up to this many seconds (at most 600) for the job to finish, checking every 15 s; 0 = ask once.
    """
    job_id = str(job_id).strip()
    if not job_id.isdigit():
        raise ValueError(f"job_id must be the number slurm_submit returned, e.g. '123456'; got {job_id!r}")
    deadline = _clock() + min(max(int(wait_seconds or 0), 0), MAX_WAIT_SECONDS)
    try:
        status = _query(job_id)
        while status["state"] not in FINISHED and _clock() < deadline:
            _sleep(min(POLL_SECONDS, deadline - _clock()))
            status = _query(job_id)
    except _NoSlurm:
        status = {"state": "UNKNOWN", "reason": "squeue was not found on this machine", "elapsed": None,
                  "exit_code": None}
    result = {"job_id": job_id, **status, "finished": status["state"] in FINISHED, "log_tail": _log_tail(job_id)}
    session = get_session()
    job = session.find_job(job_id)
    if job is not None and status["state"] != "UNKNOWN":
        job.state = status["state"]
    session.trace.emit("job_status", **{k: v for k, v in result.items() if k != "log_tail"})
    return result
