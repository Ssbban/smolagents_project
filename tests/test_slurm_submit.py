import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.approval as approval
from tools.approval import Decision
from tools.preflight import preflight
from tools.slurm_submit import slurm_submit

SCRIPT = "#!/bin/bash -l\n#SBATCH --job-name=water\n#SBATCH --time={time}\n#SBATCH --partition=gpu\n\necho run\n"
CLUSTER = {"available": True, "reason": "", "partitions": [{"name": "gpu", "max_time": "1-00:00:00"}]}


@pytest.fixture
def script(session):
    def make(time="12:00:00", text=None):
        bundle = session.new_bundle("water")
        return session.write_artifact(bundle, "job.sh", text or SCRIPT.format(time=time), "mace_script")

    return make


@pytest.fixture
def asked(monkeypatch):
    """Replace the person: answers in order, records what they were shown."""
    shown = []

    def install(*decisions):
        answers = list(decisions)

        def ask(title, text, notes):
            shown.append((title, text, notes))
            return answers.pop(0)

        monkeypatch.setattr(approval, "ask", ask)
        return shown

    return install


def test_not_an_artifact_is_refused(session, fake_run, asked):
    Path("job.sh").write_text(SCRIPT.format(time="1:00:00"))
    run, shown = fake_run(), asked()
    result = slurm_submit(script_path="job.sh")
    assert result["status"] == "refused" and "not a job script generated in this session" in result["reason"]
    assert run.calls == [] and shown == []


def test_wrong_kind_is_refused(session, fake_run, asked):
    other = session.write_artifact(session.new_bundle("ipi"), "input.xml", "<simulation/>", "ipi_input")
    run = fake_run()
    assert slurm_submit(script_path=other.path)["status"] == "refused" and run.calls == []


def test_changed_after_generation_is_refused(script, fake_run, asked):
    artifact = script()
    Path(artifact.path).write_text(SCRIPT.format(time="1:00:00") + "rm -rf ~\n")
    run, shown = fake_run(), asked()
    result = slurm_submit(script_path=artifact.path)
    assert result["status"] == "refused" and "changed" in result["reason"]
    assert run.calls == [] and shown == []


def test_preflight_errors_refuse_with_all_errors(session, script, fake_run, asked):
    session.cluster = CLUSTER
    artifact = script(text="#!/bin/bash\n#SBATCH --time=soon\n#SBATCH --partition=cpu\necho\n")
    run, shown = fake_run(), asked()
    result = slurm_submit(script_path=artifact.path)
    assert result["status"] == "refused" and result["reason"] == "pre-flight checks failed"
    assert len(result["errors"]) == 3
    assert any("--job-name" in e for e in result["errors"])
    assert any("time must use a Slurm format" in e for e in result["errors"])
    assert any("partition 'cpu' does not exist" in e for e in result["errors"])
    assert run.calls == [] and shown == []


def test_time_over_partition_limit_is_an_error(session, script):
    session.cluster = CLUSTER
    report = preflight(script_path=script(time="2-00:00:00").path)
    assert not report["ok"] and "exceeds the limit 1-00:00:00" in report["errors"][0]


def test_unprobed_cluster_is_a_warning_shown_to_the_person(script, fake_run, asked):
    artifact = script()
    fake_run({"sbatch": (0, "77\n", "")})
    shown = asked(Decision(True))
    assert preflight(script_path=artifact.path)["ok"] is True
    slurm_submit(script_path=artifact.path)
    title, text, notes = shown[0]
    assert text == Path(artifact.path).read_text() and "cluster facts unknown" in notes[0]


def test_declined_never_calls_sbatch(session, script, fake_run, asked):
    artifact = script()
    run = fake_run({"sbatch": (0, "1\n", "")})
    asked(Decision(False, "use 4 hours"))
    assert slurm_submit(script_path=artifact.path) == {"status": "declined", "feedback": "use 4 hours"}
    assert run.calls == [] and session.jobs == []


def test_approved_submits_once_from_the_bundle_and_records_the_job(session, script, fake_run, asked):
    artifact = script()
    run = fake_run({"sbatch": (0, "4242;cluster\n", "")})
    asked(Decision(True), Decision(True))
    result = slurm_submit(script_path=artifact.path)
    assert result == {"status": "submitted", "job_id": "4242", "script_path": artifact.path,
                      "bundle": artifact.bundle}
    [(argv, kwargs)] = run.calls
    assert argv == ["sbatch", "--parsable", "job.sh"] and kwargs["cwd"] == artifact.bundle
    assert kwargs["timeout"] == 60
    assert session.find_job("4242").script == artifact.path

    again = slurm_submit(script_path=artifact.path)
    assert again["status"] == "refused" and "already submitted as job 4242" in again["reason"]
    assert len(run.calls) == 1

    events = [json.loads(line)["event"] for line in (session.dir / "trace.jsonl").read_text().splitlines()]
    assert events.count("slurm_submit") == 2 and "approval" in events and "job_submitted" in events


def test_sbatch_missing_is_failed(script, fake_run, asked):
    fake_run({})
    asked(Decision(True))
    result = slurm_submit(script_path=script().path)
    assert result["status"] == "failed" and "sbatch was not found" in result["reason"]


def test_sbatch_error_is_failed_with_stderr(script, fake_run, asked):
    fake_run({"sbatch": (1, "", "sbatch: error: invalid partition specified\n")})
    asked(Decision(True))
    result = slurm_submit(script_path=script().path)
    assert result == {"status": "failed", "reason": "sbatch: error: invalid partition specified"}


def test_sbatch_timeout_is_unknown(session, script, fake_run, asked):
    fake_run({"sbatch": subprocess.TimeoutExpired("sbatch", 60)})
    asked(Decision(True))
    result = slurm_submit(script_path=script().path)
    # Without a job id job_status cannot help; the person has to look at the queue.
    assert result["status"] == "unknown" and "job_status" not in result["reason"]
    assert "no job id" in result["reason"] and "check their queue" in result["reason"]
    assert session.jobs == []


def test_ask_without_terminal_declines(monkeypatch):
    monkeypatch.setattr("sys.stdin", None)
    assert approval.ask("t", "text", []) == Decision(False, "no interactive terminal: nobody could confirm")


@pytest.mark.parametrize("answer, expected", [("y", Decision(True)), ("YES", Decision(True)),
                                              ("n", Decision(False)), ("", Decision(False)),
                                              ("use 2 GPUs", Decision(False, "use 2 GPUs"))])
def test_ask_answers(monkeypatch, capsys, answer, expected):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    assert approval.ask("Job script x", "#!/bin/bash\n", ["a note"]) == expected
    assert "#!/bin/bash" in capsys.readouterr().out


def test_ask_end_of_input_declines(monkeypatch):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))

    def eof(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert approval.ask("t", "x", []) == Decision(False, "")
