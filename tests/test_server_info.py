import os
import subprocess

from tools.get_server_info import parse_nodes, parse_partitions, slurm_config


def test_slurm_config_without_slurm(session, fake_run):
    run = fake_run({})
    result = slurm_config()
    assert result["available"] is False and "sinfo was not found" in result["reason"]
    assert result["partitions"] == [] and result["nodes"] == []
    assert set(result) >= {"partitions", "nodes", "conda_activation_path", "available_conda_envs",
                           "available_modules", "available", "reason"}
    assert session.cluster == result
    assert run.calls and all(kwargs.get("timeout") for _, kwargs in run.calls)
    if os.name != "posix":
        assert not any(argv[0] == "bash" for argv, _ in run.calls)


def test_slurm_config_is_cached_until_refresh(session, fake_run):
    run = fake_run({})
    first = slurm_config()
    calls = len(run.calls)
    assert slurm_config() == first and len(run.calls) == calls
    slurm_config(refresh=True)
    assert len(run.calls) > calls


def test_changing_the_result_does_not_change_the_kept_facts(session, fake_run):
    fake_run({"sinfo": (0, "gpu|up|1/1/0/2|1:00:00\n", "")})
    result = slurm_config()
    result["partitions"][0]["max_time"] = "infinite"
    assert session.cluster["partitions"][0]["max_time"] == "1:00:00"


def test_slurm_config_timeout_is_a_reason(session, fake_run):
    fake_run({"sinfo": subprocess.TimeoutExpired("sinfo", 30)})
    result = slurm_config()
    assert result["available"] is False and "did not answer in 30 s" in result["reason"]


def test_slurm_config_with_canned_sinfo(session, fake_run):
    def sinfo(argv):
        if "-N" in argv:
            return (0, "node1|idle|64|gpu:a100:4\nnode2|mixed|32|(null)\n", "")
        return (0, "gpu*|up|1/2/0/3|2-00:00:00\ngpu*|up|0/1/1/2|2-00:00:00\ncpu|up|0/4/0/4|infinite\n", "")

    fake_run({"sinfo": sinfo})
    result = slurm_config()
    assert result["available"] is True and result["reason"] == ""
    assert [p["name"] for p in result["partitions"]] == ["gpu", "cpu"]
    assert result["partitions"][0] == {"name": "gpu", "state": "up", "avail_nodes": 3, "total_nodes": 5,
                                       "max_time": "2-00:00:00"}
    assert result["nodes"][0]["gpu_model"] == "a100" and result["nodes"][1]["cpus"] == 32
    assert session.snapshot()["cluster"] == {"available": True, "partitions": ["gpu", "cpu"]}


def test_parsers_skip_malformed_lines():
    assert parse_partitions("garbage\n|||\n") == []
    assert parse_nodes("a|b\n") == []
