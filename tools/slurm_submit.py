import subprocess
from pathlib import Path

from smolagents.tools import tool

from tools import approval
from tools.preflight import SCRIPT_KINDS, run_preflight
from tools.session import Session, get_session

SBATCH_TIMEOUT = 60


def _submit(session: Session, script_path: str) -> dict:
    artifact = session.find_artifact(script_path)
    if artifact is None or artifact.kind not in SCRIPT_KINDS:
        return {"status": "refused", "reason": f"{script_path!r} is not a job script generated in this session "
                "by MACE_script or ipi_submit_generator; generate it with one of them and pass its script_path"}
    if not session.is_unchanged(artifact):
        return {"status": "refused", "reason": f"{artifact.path} was changed or removed after it was generated; "
                "generate the script again"}
    job = session.job_for(artifact)
    if job is not None:
        return {"status": "refused", "reason": f"{artifact.path} was already submitted as job {job.job_id}; "
                "check it with job_status, or generate a new script for another run"}
    problems = run_preflight(artifact, session)
    errors = [p.message for p in problems if p.severity == "error"]
    if errors:
        return {"status": "refused", "reason": "pre-flight checks failed", "errors": errors}

    text = Path(artifact.path).read_text(encoding="utf-8")
    notes = [p.message for p in problems if p.severity == "warning"]
    decision = approval.ask(f"Job script {artifact.path}", text, notes)
    session.trace.emit("approval", script_path=artifact.path, approved=decision.approved, feedback=decision.feedback)
    if not decision.approved:
        return {"status": "declined", "feedback": decision.feedback}

    try:
        result = subprocess.run(
            ["sbatch", "--parsable", Path(artifact.path).name],
            cwd=artifact.bundle,
            capture_output=True,
            text=True,
            timeout=SBATCH_TIMEOUT,
        )
    except FileNotFoundError:
        return {"status": "failed", "reason": "sbatch was not found on this machine; submit from a Slurm login node"}
    except subprocess.TimeoutExpired:
        return {"status": "unknown", "reason": f"sbatch did not answer in {SBATCH_TIMEOUT} s; a job may exist - "
                "check with job_status before submitting again"}
    if result.returncode != 0:
        reason = (result.stderr or result.stdout).strip() or f"sbatch exited with code {result.returncode}"
        return {"status": "failed", "reason": reason}
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        return {"status": "unknown", "reason": f"sbatch answered {result.stdout.strip()!r} without a job id; "
                "a job may exist - check squeue before submitting again"}
    job = session.record_job(artifact, job_id)
    session.trace.emit("job_submitted", job_id=job_id, script_path=job.script, bundle=job.bundle)
    return {"status": "submitted", "job_id": job_id, "script_path": job.script, "bundle": job.bundle}


@tool
def slurm_submit(script_path: str) -> dict:
    """
    Submit a job script generated in this session to Slurm with sbatch.

    The tool itself runs the pre-flight checks, shows the script to the person at the
    terminal and asks them; do not ask for confirmation yourself. Only scripts written by
    MACE_script or ipi_submit_generator, unchanged since, are accepted, and each one once.

    Result: a dict with "status":
    "submitted" (with job_id, script_path, bundle) - follow it with job_status(job_id);
    "refused" (with reason, maybe errors) - fix the reason, usually by generating the script again;
    "declined" (with feedback) - the person said no; their feedback says what to change;
    "failed" (with reason) - sbatch rejected the job or is not installed;
    "unknown" (with reason) - a job may exist; check before submitting again.

    Args:
        script_path: script_path returned by MACE_script or ipi_submit_generator.
    """
    session = get_session()
    outcome = _submit(session, script_path)
    session.trace.emit("slurm_submit", **{"script_path": script_path, **outcome})
    return outcome
