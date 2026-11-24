import json
from pathlib import Path

import pytest

import tools.slurm_status as status_module
from tools.slurm_status import job_status, parse_sacct, parse_squeue


def test_parse_squeue():
    text = "123|PENDING|Priority|0:00\n124|RUNNING|None|1:02:03\n"
    assert parse_squeue(text, "123") == {"state": "PENDING", "reason": "Priority", "elapsed": "0:00",
                                         "exit_code": None}
    assert parse_squeue(text, "124")["reason"] is None
    assert parse_squeue(text, "12") is None and parse_squeue("", "123") is None


def test_parse_sacct():
    assert parse_sacct("55|COMPLETED|0:0|00:10:00|None\n", "55") == {
        "state": "COMPLETED", "reason": None, "elapsed": "00:10:00", "exit_code": "0:0"}
    failed = parse_sacct("56|FAILED|1:0|1-02:00:00|NonZeroExitCode\n", "56")
    assert failed["state"] == "FAILED" and failed["exit_code"] == "1:0" and failed["reason"] == "NonZeroExitCode"
    cancelled = parse_sacct("57|CANCELLED by 1000|0:15|00:01:00|None\n", "57")
    assert cancelled["state"] == "CANCELLED" and cancelled["reason"] == "CANCELLED by 1000"
    assert parse_sacct("58|RUNNING+|0:0|00:00:05\n", "58")["state"] == "RUNNING"
    assert parse_sacct("59.batch|COMPLETED|0:0|00:10:00|None\n", "59") is None


def squeue_and_sacct(squeue_rows, sacct_rows=()):
    """Answers that change per call: squeue_rows[i] / sacct_rows[i] for the i-th call (last one repeats)."""
    calls = {"squeue": 0, "sacct": 0}

    def answer(rows, name):
        def respond(argv):
            row = rows[min(calls[name], len(rows) - 1)] if rows else ""
            calls[name] += 1
            return (0, row, "")
        return respond

    return {"squeue": answer(list(squeue_rows), "squeue"), "sacct": answer(list(sacct_rows), "sacct")}


def test_running_job_once(session, fake_run):
    fake_run(squeue_and_sacct(["7|RUNNING|None|5:00\n"]))
    result = job_status(job_id="7")
    assert result == {"job_id": "7", "state": "RUNNING", "reason": None, "elapsed": "5:00", "exit_code": None,
                      "finished": False, "log_tail": None}


def test_finished_job_comes_from_sacct_with_log_tail(session, fake_run):
    artifact = session.write_artifact(session.new_bundle("w"), "job.sh", "#!/bin/bash\n", "mace_script")
    session.record_job(artifact, "8")
    lines = [f"epoch {i}" for i in range(100)]
    (Path(artifact.bundle) / "slurm-8.out").write_text("\n".join(lines) + "\n")
    fake_run(squeue_and_sacct([""], ["8|COMPLETED|0:0|01:00:00|None\n"]))
    result = job_status(job_id="8")
    assert result["state"] == "COMPLETED" and result["finished"] and result["exit_code"] == "0:0"
    assert result["log_tail"].splitlines() == lines[-40:]
    assert session.find_job("8").state == "COMPLETED"
    record = json.loads((session.dir / "trace.jsonl").read_text().splitlines()[-1])
    assert record["event"] == "job_status" and record["state"] == "COMPLETED"


def test_unknown_job(session, fake_run):
    fake_run({"squeue": (1, "", "slurm_load_jobs error: Invalid job id specified"), "sacct": (0, "", "")})
    result = job_status(job_id="9")
    assert result["state"] == "UNKNOWN" and not result["finished"]


def test_without_slurm(session, fake_run):
    fake_run({})
    result = job_status(job_id="10", wait_seconds=600)
    assert result["state"] == "UNKNOWN" and result["reason"] == "squeue was not found on this machine"


def test_bad_job_id(session):
    with pytest.raises(ValueError, match="job_id must be the number"):
        job_status(job_id="abc")


@pytest.fixture
def fake_clock(monkeypatch):
    clock = {"now": 0.0, "sleeps": []}

    def sleep(seconds):
        clock["sleeps"].append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(status_module, "_sleep", sleep)
    monkeypatch.setattr(status_module, "_clock", lambda: clock["now"])
    return clock


def test_wait_polls_until_finished(session, fake_run, fake_clock):
    running = "11|RUNNING|None|1:00\n"
    fake_run(squeue_and_sacct([running, running, ""], ["11|FAILED|2:0|00:00:40|None\n"]))
    result = job_status(job_id="11", wait_seconds=300)
    assert result["state"] == "FAILED" and result["finished"] and result["exit_code"] == "2:0"
    assert fake_clock["sleeps"] == [15, 15]


def test_wait_is_capped_and_returns_a_running_job(session, fake_run, fake_clock):
    fake_run(squeue_and_sacct(["12|PENDING|Resources|0:00\n"]))
    result = job_status(job_id="12", wait_seconds=10_000)
    assert result["state"] == "PENDING" and not result["finished"] and result["reason"] == "Resources"
    assert sum(fake_clock["sleeps"]) == 600 and len(fake_clock["sleeps"]) == 40


def test_short_wait_sleeps_only_the_remaining_time(session, fake_run, fake_clock):
    fake_run(squeue_and_sacct(["13|RUNNING|None|1:00\n"]))
    job_status(job_id="13", wait_seconds=20)
    assert fake_clock["sleeps"] == [15, 5]
