import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from smolagents.memory import ActionStep, PlanningStep
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import Timing, TokenUsage

import tools.session as session_module
from config import Settings
from tools.trace import step_callback


def test_get_session_without_session(monkeypatch):
    monkeypatch.setattr(session_module, "_current", None)
    with pytest.raises(RuntimeError, match="no session"):
        session_module.get_session()


def test_session_id_and_dir(session):
    assert session_module.get_session() is session
    assert session.dir == Path("runs") / session.id
    assert session.dir.is_dir()
    date, clock, tag = session.id.split("-")
    assert len(date) == 8 and len(clock) == 6 and len(tag) == 4


def test_bundles_are_unique(session):
    first, second = session.new_bundle("water"), session.new_bundle("water")
    assert first != second and first.is_dir() and second.is_dir()
    assert first.name == "01-water" and second.name == "02-water"


def test_write_artifact_lf_bytes_and_registration(session):
    bundle = session.new_bundle("job")
    artifact = session.write_artifact(bundle, "job.sh", "#!/bin/bash\r\necho a\necho b\r\n", "mace_script")
    data = Path(artifact.path).read_bytes()
    assert b"\r" not in data and data == b"#!/bin/bash\necho a\necho b\n"
    assert artifact.path == f"runs/{session.id}/01-job/job.sh"
    assert artifact.bundle == f"runs/{session.id}/01-job"
    assert len(artifact.sha256) == 64 and artifact.kind == "mace_script"
    assert session.artifacts == [artifact]


def test_write_artifact_refuses_overwrite(session):
    bundle = session.new_bundle("job")
    session.write_artifact(bundle, "job.sh", "one", "mace_script")
    with pytest.raises(FileExistsError, match="never overwritten"):
        session.write_artifact(bundle, "job.sh", "two", "mace_script")
    assert (bundle / "job.sh").read_text() == "one"


def test_find_artifact_by_several_spellings(session):
    artifact = session.write_artifact(session.new_bundle("job"), "job.sh", "x", "mace_script")
    spellings = [
        artifact.path,
        "./" + artifact.path,
        artifact.path.replace("/", "\\"),
        str(Path(artifact.path).resolve()),
        f" {artifact.path} ",
    ]
    for spelling in spellings:
        assert session.find_artifact(spelling) == artifact, spelling
    assert session.find_artifact("runs/other/job.sh") is None


def test_unchanged_and_changed_detection(session):
    artifact = session.write_artifact(session.new_bundle("job"), "job.sh", "x\n", "mace_script")
    assert session.is_unchanged(artifact)
    Path(artifact.path).write_bytes(b"y\n")
    assert not session.is_unchanged(artifact)
    Path(artifact.path).unlink()
    assert not session.is_unchanged(artifact)


def test_jobs_mark_since_and_snapshot(session):
    first = session.write_artifact(session.new_bundle("a"), "job.sh", "a", "mace_script")
    mark = session.mark()
    second = session.write_artifact(session.new_bundle("b"), "job.sh", "b", "mace_script")
    job = session.record_job(second, "4242")
    assert session.job_for(second) is job and session.job_for(first) is None
    assert session.find_job(" 4242 ") is job and session.find_job("1") is None
    later = session.since(mark)
    assert [a["path"] for a in later["artifacts"]] == [second.path]
    assert later["jobs"][0]["job_id"] == "4242" and later["jobs"][0]["state"] == "SUBMITTED"
    assert session.since(session.mark()) == {"artifacts": [], "jobs": []}
    snapshot = session.snapshot()
    json.dumps(snapshot)
    assert snapshot["bundles"][second.bundle] == [{"path": second.path, "kind": "mace_script"}]
    assert snapshot["jobs"] == [{"job_id": "4242", "script": second.path, "state": "SUBMITTED"}]


def _trace_lines(session):
    return [json.loads(line) for line in (session.dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]


def test_trace_lines_are_json_and_never_contain_the_key(session):
    secret = "sk-test-not-a-real-key-123"
    settings = Settings.from_env({"SMOLAGENTS_MODEL_ID": "m", "SMOLAGENTS_API_KEY": secret}, dotenv_path="none.env")
    session.trace.emit("session_start", settings=settings.public(), smolagents="1.26.0")
    session.write_artifact(session.new_bundle("job"), "job.sh", "x", "mace_script")
    agent = SimpleNamespace(name="mace_agent")
    step = ActionStep(
        step_number=1,
        timing=Timing(start_time=0.0, end_time=1.5),
        model_input_messages=[ChatMessage(role=MessageRole.USER, content=f"key {secret}")],
        code_action="print(1)",
        observations="1",
        token_usage=TokenUsage(input_tokens=10, output_tokens=2),
    )
    step_callback(step, agent=agent)
    plan = PlanningStep(
        model_input_messages=[ChatMessage(role=MessageRole.USER, content=secret)],
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="plan"),
        plan="1. delegate", timing=Timing(start_time=0.0, end_time=0.5),
    )
    step_callback(plan, agent=agent)

    text = (session.dir / "trace.jsonl").read_text(encoding="utf-8")
    assert secret not in text
    records = _trace_lines(session)
    assert [r["event"] for r in records] == ["session_start", "artifact_written", "step", "step"]
    assert all(r["session"] == session.id and r["ts"] for r in records)
    action = records[2]
    assert action["agent"] == "mace_agent" and action["step_type"] == "ActionStep" and action["step"] == 1
    assert action["code"] == "print(1)" and action["tokens"] == {"input": 10, "output": 2}
    assert action["duration_s"] == 1.5
    assert records[3]["plan"] == "1. delegate"
    assert records[0]["settings"]["model_id"] == "m" and "api_key" not in records[0]["settings"]


def test_step_callback_caps_observation(session):
    step = ActionStep(step_number=2, timing=Timing(start_time=0.0), observations="x" * 5000)
    step_callback(step, agent=SimpleNamespace(name="ipi_agent"))
    assert len(_trace_lines(session)[-1]["observation"]) == 2000
