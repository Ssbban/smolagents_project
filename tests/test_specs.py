import pytest

from tools.specs import ClusterSpec, IPISubmitSpec, parse_slurm_limit, parse_slurm_time

CLUSTER = dict(job_name="job", time="1:00:00", partition="gpu", conda_env="mace", conda_activation_path="/opt/c.sh")


def problems_of(**changes):
    return ClusterSpec(**{**CLUSTER, **changes}).problems()


@pytest.mark.parametrize(
    "text, seconds",
    [
        ("90", 90 * 60),  # M
        ("30:15", 30 * 60 + 15),  # M:S
        ("24:00:00", 24 * 3600),  # H:M:S
        ("100:00:00", 100 * 3600),
        ("2-12", 2 * 86400 + 12 * 3600),  # D-H
        ("1-02:30", 86400 + 2 * 3600 + 30 * 60),  # D-H:M
        ("2-00:00:00", 2 * 86400),  # D-H:M:S
    ],
)
def test_slurm_time_forms(text, seconds):
    assert parse_slurm_time(text) == seconds


@pytest.mark.parametrize("text", ["2 days", "", "1:2:3:4", "-1", "0", "0:00:00", "1-", "12h", None, 24])
def test_slurm_time_rejects(text):
    with pytest.raises(ValueError):
        parse_slurm_time(text)


def test_slurm_limit():
    assert parse_slurm_limit("infinite") is None and parse_slurm_limit("UNLIMITED") is None
    assert parse_slurm_limit("2-00:00:00") == 2 * 86400


def test_one_error_lists_every_problem():
    spec = ClusterSpec(job_name="a b", time="2 days", partition=None, nodes=0, memory="lots",
                       conda_env=None, conda_activation_path="x y")
    with pytest.raises(ValueError) as caught:
        spec.validate()
    message = str(caught.value)
    assert "7 problem(s)" in message
    for fragment in ["job_name may only", "time must use a Slurm format", "partition is required",
                     "nodes must be greater than zero", "memory must be a number", "conda_env is required",
                     "conda_activation_path may only"]:
        assert f"- {fragment}" in message


def test_valid_header_values_pass():
    assert problems_of(memory="500M", mail_user="ada.lovelace@uni.ac.uk", mail_type="BEGIN,END,FAIL",
                       additional_slurm_args={"account": "proj-12", "qos": "normal", "output": "slurm-%j.out",
                                              "constraint": ""}) == []


@pytest.mark.parametrize(
    "changes, fragment",
    [
        ({"memory": "32G; rm -rf ~"}, "memory must be"),  # baseline M12
        ({"memory": "32GB"}, "memory must be"),
        ({"mail_user": "a@b.c; rm -rf ~"}, "mail_user must be one e-mail address"),  # baseline M13
        ({"mail_user": "a b c"}, "mail_user must be"),
        ({"mail_type": "ALWAYS"}, "mail_type must be"),
        ({"additional_slurm_args": {"account": "proj; rm -rf ~"}}, "must be one token"),  # baseline M14
        ({"additional_slurm_args": {"account": ["a", "b"]}}, "single text or number"),
        ({"additional_slurm_args": {"time": "1:00:00"}}, "dedicated argument"),
        ({"additional_slurm_args": {"chdir": "/tmp"}}, "dedicated argument"),
        ({"additional_slurm_args": {"cpus_per_task": 4}}, "'-' instead of '_'"),
        ({"additional_slurm_args": {"cpus-per-task": 8}}, "dedicated argument"),
        ({"additional_slurm_args": {"export": "ALL"}}, "dedicated argument"),
        ({"cpus_per_task": 0}, "cpus_per_task must be greater than zero"),
        ({"additional_slurm_args": {"A": "proj"}}, "long option name"),
        ({"memory": "32G\nrm -rf ~"}, "control characters"),  # baseline M16
    ],
)
def test_rejected_header_values(changes, fragment):
    found = problems_of(**changes)
    assert any(fragment in p for p in found), found


def test_header_requests_cpus_and_exports_nothing():
    header = ClusterSpec(**CLUSTER, cpus_per_task=8).sbatch_header(gpus=0)
    assert "#SBATCH --cpus-per-task=8" in header and "#SBATCH --export=NONE" in header
    assert not any("cpus-per-task" in line for line in ClusterSpec(**CLUSTER).sbatch_header(gpus=0))


def test_ipi_submit_spec_collects_cluster_and_own_problems():
    spec = IPISubmitSpec(cluster=ClusterSpec(**{**CLUSTER, "memory": "x"}), ipi_input_path="in put.xml",
                         ase_path="run_ase.py", num_force_providers=3)
    found = spec.problems()
    assert len(found) == 3
    assert any("memory" in p for p in found) and any("ipi_input_path" in p for p in found)
    assert any("num_force_providers must be 1" in p for p in found)

def test_client_keyword_arguments_must_be_python_names():
    from tools.specs import IPIASEClientSpec

    calculators = {"mace_mp": "mace_mp"}
    bad = IPIASEClientSpec(calculator="mace_mp", calculator_kwargs={"a-b": 1}).problems(supported_calculators=calculators)
    good = IPIASEClientSpec(calculator="mace_mp", calculator_kwargs={"model": "medium"}).problems(supported_calculators=calculators)
    assert any("'a-b'" in problem for problem in bad)
    assert good == []
