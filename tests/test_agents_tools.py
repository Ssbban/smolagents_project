"""The plan review and the follow-ups in slurm_submit / job_status that the agent layer relies on."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from smolagents import LogLevel
from smolagents.memory import PlanningStep
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import Timing

import tools.slurm_status as status_module
from tools import approval
from tools.approval import Decision
from tools.customize_plan import UserCancelled, interrupt_after_plan
from tools.job_output import job_output
from tools.slurm_status import job_status
from tools.slurm_submit import slurm_submit

SUBMIT = "#!/bin/bash\n#SBATCH --job-name=md\n#SBATCH --time=1:00:00\ni-pi input.xml &\npython run_ase.py\n"
AGENT = SimpleNamespace(logger=SimpleNamespace(level=LogLevel.INFO))  # smolagents prints the plan at INFO


def plan_step(plan="1. Delegate."):
    return PlanningStep(model_input_messages=[], plan=plan, timing=Timing(start_time=0.0),
                        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan))


@pytest.fixture
def answers(monkeypatch):
    def install(*replies):
        replies = list(replies)

        def answer(prompt=""):
            if not replies:
                raise EOFError
            return replies.pop(0)

        monkeypatch.setattr("builtins.input", answer)

    return install


@pytest.mark.parametrize("replies", [("3",), ("x", "3"), ()])
def test_cancel_or_end_of_input_at_the_plan_review_raises_user_cancelled(answers, replies):
    answers(*replies)
    with pytest.raises(UserCancelled):
        interrupt_after_plan(plan_step(), agent=AGENT)
    assert not issubclass(UserCancelled, Exception)


def test_approve_and_modify(answers):
    answers("1")
    step = plan_step()
    interrupt_after_plan(step, agent=AGENT)
    assert step.plan == "1. Delegate."
    answers("2", "1. Ask the person first.", "2. Then delegate.", "", "")
    interrupt_after_plan(step, agent=AGENT)
    assert step.plan == "1. Ask the person first.\n2. Then delegate."
    answers("2", "1. Only this.")  # input ends while typing: the text so far is the plan
    interrupt_after_plan(step, agent=AGENT)
    assert step.plan == "1. Only this."


def test_the_plan_is_printed_once(answers, capsys):
    answers("1")
    interrupt_after_plan(plan_step("1. A plan."), agent=AGENT)
    assert "1. A plan." not in capsys.readouterr().out  # smolagents' own log has just shown it
    answers("1")
    interrupt_after_plan(plan_step("1. A plan."), agent=SimpleNamespace(logger=SimpleNamespace(level=LogLevel.ERROR)))
    assert "1. A plan." in capsys.readouterr().out


@pytest.fixture
def ipi_bundle(session, monkeypatch):
    """An i-PI bundle with the input, two clients (one unused) and the job script."""
    monkeypatch.setattr("tools.slurm_submit.run_preflight", lambda artifact, session: [])
    bundle = session.new_bundle("ipi")
    files = {
        "input": session.write_artifact(bundle, "input.xml", "<simulation>\n</simulation>\n", "ipi_input"),
        "client": session.write_artifact(bundle, "run_ase.py", "print('client')\n", "ase_client"),
        "unused": session.write_artifact(bundle, "run_ase-2.py", "print('old client')\n", "ase_client"),
        "script": session.write_artifact(bundle, "job.sh", SUBMIT, "ipi_submit"),
    }
    return files


def test_the_person_sees_every_file_the_job_runs(ipi_bundle, fake_run, monkeypatch):
    shown = []
    monkeypatch.setattr(approval, "ask", lambda title, text, notes: shown.append((title, text, notes)) or
                        Decision(False, ""))
    run = fake_run()
    assert slurm_submit(script_path=ipi_bundle["script"].path)["status"] == "declined"
    [(title, text, notes)] = shown
    rule = approval.RULE
    assert text.startswith(SUBMIT)
    assert f"{rule}\n{ipi_bundle['input'].path} (used by this job)\n{rule}\n<simulation>\n</simulation>\n" in text
    assert f"{rule}\n{ipi_bundle['client'].path} (used by this job)\n{rule}\nprint('client')\n" in text
    unused = ipi_bundle["unused"]
    assert "old client" not in text
    assert notes == [f"also in the bundle, not shown: {unused.path} "
                     f"({Path(unused.path).stat().st_size} bytes, sha256 {unused.sha256[:12]})"]
    assert run.calls == []


@pytest.fixture
def sleeps(monkeypatch):
    clock = {"now": 0.0, "sleeps": []}

    def sleep(seconds):
        clock["sleeps"].append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(status_module, "_sleep", sleep)
    monkeypatch.setattr(status_module, "_clock", lambda: clock["now"])
    return clock["sleeps"]


def test_an_id_nobody_knows_returns_at_once(session, fake_run, sleeps):
    fake_run({"squeue": (1, "", "Invalid job id specified"), "sacct": (0, "", "")})
    result = job_status(job_id="31", wait_seconds=600)
    assert result["state"] == "UNKNOWN" and "not submitted in this session" in result["reason"]
    assert sleeps == []


def test_a_job_of_this_session_unknown_to_slurm_yet_is_still_polled(session, fake_run, sleeps):
    artifact = session.write_artifact(session.new_bundle("w"), "job.sh", "#!/bin/bash\n", "mace_script")
    session.record_job(artifact, "32")
    fake_run({"squeue": (0, "", ""), "sacct": (0, "", "")})
    result = job_status(job_id="32", wait_seconds=30)
    assert result["state"] == "UNKNOWN" and sleeps == [15, 15]


def submitted_job(session, job_id="77"):
    bundle = session.new_bundle("md")
    session.record_job(session.write_artifact(bundle, "job.sh", SUBMIT, "ipi_submit"), job_id)
    return bundle


def test_job_output_lists_and_reads_the_files_of_its_job(session):
    bundle = submitted_job(session)
    (bundle / "log.i-pi").write_text("".join(f"step {i}\n" for i in range(300)))
    (bundle / "checkpoints").mkdir()
    (bundle / "checkpoints" / "run.pt").write_bytes(b"\0" * 10)
    names = {f["name"] for f in job_output("77")["files"]}
    assert names == {"job.sh", "log.i-pi", "checkpoints/run.pt"}
    assert job_output("77", "log.i-pi", lines=2) == {"name": "log.i-pi", "total_lines": 300, "tail": "step 298\nstep 299"}
    with pytest.raises(ValueError, match="binary file"):
        job_output("77", "checkpoints/run.pt")


def test_job_output_reads_nothing_outside_the_bundle_of_a_job_of_this_session(session):
    bundle = submitted_job(session)
    Path("secret.txt").write_text("key")
    with pytest.raises(ValueError, match="is not a file of the bundle"):
        job_output("77", "../../../secret.txt")
    if os.name != "nt":  # a link the job itself could have made
        (bundle / "link").symlink_to(Path("secret.txt").resolve())
        with pytest.raises(ValueError, match="is not a file of the bundle"):
            job_output("77", "link")
    with pytest.raises(ValueError, match="was not submitted in this session"):
        job_output("78")
