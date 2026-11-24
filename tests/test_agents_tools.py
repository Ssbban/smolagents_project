"""The plan review and the follow-ups in slurm_submit / job_status that the agent layer relies on."""

from pathlib import Path

import pytest
from smolagents.memory import PlanningStep
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import Timing

import tools.slurm_status as status_module
from tools import approval
from tools.approval import Decision
from tools.customize_plan import UserCancelled, interrupt_after_plan
from tools.slurm_status import job_status
from tools.slurm_submit import slurm_submit

SUBMIT = "#!/bin/bash\n#SBATCH --job-name=md\n#SBATCH --time=1:00:00\ni-pi input.xml &\npython run_ase.py\n"


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
        interrupt_after_plan(plan_step(), agent=None)
    assert not issubclass(UserCancelled, Exception)


def test_approve_and_modify(answers):
    answers("1")
    step = plan_step()
    interrupt_after_plan(step, agent=None)
    assert step.plan == "1. Delegate."
    answers("2", "1. Ask the person first.", "2. Then delegate.", "", "")
    interrupt_after_plan(step, agent=None)
    assert step.plan == "1. Ask the person first.\n2. Then delegate."
    answers("2", "1. Only this.")  # input ends while typing: the text so far is the plan
    interrupt_after_plan(step, agent=None)
    assert step.plan == "1. Only this."


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
