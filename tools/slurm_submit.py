import subprocess
from pathlib import Path

from smolagents.tools import tool

from tools import approval
from tools.preflight import NEEDS, SCRIPT_KINDS, run_preflight
from tools.session import Artifact, Session, get_session

SBATCH_TIMEOUT = 60


def _shown_to_the_person(session: Session, artifact: Artifact) -> tuple[str, list[str]]:
    """The script, then every file of its bundle that it runs, each under its own heading;
    one line (size, start of the sha256) for each other file of the bundle."""
    script = Path(artifact.path).read_text(encoding="utf-8")
    text, others = script, []
    for member in session.artifacts:
        if member.bundle != artifact.bundle or member.path == artifact.path:
            continue
        path = Path(member.path)
        if member.kind in NEEDS.get(artifact.kind, ()) and path.name in script.split():
            text += "" if text.endswith("\n") else "\n"
            text += f"{approval.RULE}\n{member.path} (used by this job)\n{approval.RULE}\n"
            text += path.read_text(encoding="utf-8")
        else:
            others.append(f"also in the bundle, not shown: {member.path} "
                          f"({path.stat().st_size} bytes, sha256 {member.sha256[:12]})")
    return text, others


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

    text, others = _shown_to_the_person(session, artifact)
    notes = others + [p.message for p in problems if p.severity == "warning"]
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
        return {"status": "unknown", "reason": f"sbatch did not answer in {SBATCH_TIMEOUT} s, so there is no job id; "
                "a job may exist all the same - ask the person to check their queue (squeue --me) before any new "
                "submission"}
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

    The tool itself runs the pre-flight checks, shows the script and the files it runs to the
    person at the terminal and asks them; do not ask for confirmation yourself. Only scripts written by
    MACE_script or ipi_submit_generator, unchanged since, are accepted, and each one once.

    Result: a dict with "status":
    "submitted" (with job_id, script_path, bundle) - follow it with job_status(job_id);
    "refused" (with reason, maybe errors) - fix the reason, usually by generating the script again;
    "declined" (with feedback) - the person said no; their feedback says what to change;
    "failed" (with reason) - sbatch rejected the job or is not installed;
    "unknown" (with reason) - a job may exist without a job id; the person must check the queue first.

    Args:
        script_path: script_path returned by MACE_script or ipi_submit_generator.
    """
    session = get_session()
    outcome = _submit(session, script_path)
    session.trace.emit("slurm_submit", **{"script_path": script_path, **outcome})
    return outcome
