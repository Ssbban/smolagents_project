"""MACE_script: the 23 cases of the v1 baseline (U4 3.2) with their outcomes now, and the MACE pre-flight check."""

from pathlib import Path

import pytest
from conftest import MACE_PARAMS

import tools.mace_cli_schema as schema_module
from tools.mace_cli_schema import parse_mace_arg_parser
from tools.mace_script import MACE_script, script_options
from tools.preflight import preflight

DATA = Path(__file__).parent / "data"
PARAMS = {**MACE_PARAMS, "train_file": "/data/train.xyz"}  # the job runs in its bundle: input paths are absolute


def table(version):
    text = (DATA / f"mace_arg_parser_v{version}.py.txt").read_text(encoding="utf-8")
    return parse_mace_arg_parser(text, source_url="release tag", version=version)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the network was used")

    monkeypatch.setattr(schema_module.requests, "get", refuse)


@pytest.fixture
def mace17(monkeypatch):
    """MACE_script and the pre-flight check use the (trimmed) option table of MACE 0.3.17."""
    loads = []

    def load(*args, **kwargs):
        loads.append(args)
        return table("0.3.17")

    monkeypatch.setattr("tools.mace_script.load_mace_cli_schema", load)
    return loads


def run(**changes):
    return MACE_script(**{**PARAMS, **changes})


def test_m00_valid_script(session, mace17):
    result = run()
    script = result["script_content"]
    assert result["script_path"] == f"{result['bundle']}/job.sh"
    assert Path(result["script_path"]).read_bytes() == script.encode()
    assert session.find_artifact(result["script_path"]).kind == "mace_script"
    assert result["notes"] == ["MACE options checked against MACE 0.3.17 (release tag)"]
    assert script_options(script) == [
        ("name", "water_mace"), ("train_file", "/data/train.xyz"), ("valid_fraction", "0.1"), ("E0s", "average"),
        ("max_num_epochs", "100"), ("device", "cuda"), ("model", "MACE"), ("r_max", "5.0"), ("batch_size", "10"),
        ("swa", None)]  # fmt: skip
    assert mace17 == [("/opt/conda/etc/profile.d/conda.sh", "mace")]  # the table of the job's environment


def test_n3_the_script_stops_at_the_first_failing_command(session, mace17):
    lines = run()["script_content"].splitlines()
    last_sbatch = max(i for i, line in enumerate(lines) if line.startswith("#SBATCH"))
    assert lines[last_sbatch + 1] == "# MACE options checked against MACE 0.3.17 (release tag)"
    assert lines[last_sbatch + 2] == "set -euo pipefail"
    setup = lines[lines.index("set +u  # environment and conda activation scripts may read unset variables"):]
    assert setup[1:4] == ["source /opt/conda/etc/profile.d/conda.sh", "conda activate mace", "set -u"]


@pytest.mark.parametrize(
    "case, changes, fragment",
    [
        ("M01", {"additional_args": {"model": "MACE", "learning_rat": 0.01}}, "unknown MACE option 'learning_rat'"),
        ("M02", {"max_num_epochs": "100"}, "max_num_epochs must be an integer"),
        ("M03", {"additional_args": {"batch_size": "ten"}}, "--batch_size: invalid int value 'ten'"),
        ("M05", {"device": "tpu"}, "--device must be one of ['cpu', 'cuda', 'mps', 'xpu']"),
        ("M06", {"additional_args": {"swa": True}}, "--swa is a switch without a value"),
        ("M09", {"train_file": "/data/my train.xyz"}, "train_file may only contain"),
        ("M10", {"name": "water; rm -rf ~"}, "name may only contain"),
        ("M11", {"E0s": "average; rm -rf ~"}, "E0s must be 'average', a dict of atomic number"),
        ("M12", {"memory": "32G; rm -rf ~"}, "memory must be a number"),
        ("M13", {"mail_user": "a@b.c; rm -rf ~"}, "mail_user must be one e-mail address"),
        ("M14", {"additional_args_slurm": {"account": "proj; rm -rf ~"}}, "must be one token"),
        ("M15", {"partition": "$(whoami)"}, "partition may only contain"),
        ("M16", {"memory": "32G\nrm -rf ~"}, "memory must not contain control characters"),
        ("M18", {"valid_file": "/data/valid.xyz"}, "mutually exclusive"),
        ("M19", {"additional_args": {"max_num_epochs": 5}}, "--max_num_epochs has a dedicated argument"),
        ("M20", {"partition": None}, "partition is required"),
        ("M22", {"time": "2 days"}, "time must use a Slurm format"),
        ("R6", {"train_file": "data/train.xyz"}, "train_file must be an absolute path or start with '~/'"),
        ("R6b", {"additional_args": {"test_file": "test.xyz"}}, "test_file must be an absolute path"),
        ("E0s", {"E0s": {"H": -13.6}}, "E0s must be 'average'"),
        ("E0s2", {"E0s": "estimated"}, "E0s='estimated' needs a foundation_model"),
        ("E0s3", {"E0s": "e0s.json"}, "E0s must be an absolute path or start with '~/'"),
        ("model", {"additional_args": {"foundation_model": "models/my.model"}}, "foundation_model must be an absolute"),
        ("switch", {"additional_flag_args": ["r_max"]}, "--r_max needs a value"),
    ],
)
def test_rejected_cases_write_nothing(session, mace17, case, changes, fragment):
    with pytest.raises(ValueError, match="MACE_script") as caught:
        run(**changes)
    assert fragment in str(caught.value), case
    assert session.artifacts == [] and not any(session.dir.iterdir())


@pytest.mark.parametrize(
    "case, changes, expected",
    [
        ("M04", {"additional_args": {"r_max": "5.0"}}, "--r_max=5.0 \\"),
        ("M07", {"train_file": "~/data/train.xyz"}, '--train_file="$HOME/data/train.xyz" \\'),
        ("M08", {"conda_activation_path": "~/miniconda3/etc/profile.d/conda.sh"},
         "source ~/miniconda3/etc/profile.d/conda.sh"),
        ("M17", {"additional_args": {"log_dir": "logs; rm -rf ~"}}, "--log_dir='logs; rm -rf ~' \\"),
        ("M22b", {"time": "2-00:00:00"}, "#SBATCH --time=2-00:00:00"),
        ("R6c", {"additional_args": {"test_file": "~/data/test.xyz"}}, '--test_file="$HOME/data/test.xyz" \\'),
        ("E0s", {"E0s": {1: -13.6, 8: -2041.8}}, "--E0s='{1: -13.6, 8: -2041.8}' \\"),
        ("E0s2", {"E0s": "Estimated", "additional_args": {"foundation_model": "medium"}}, "--E0s=Estimated \\"),
        ("alias", {"additional_flag_args": ["stage_two"]}, "--swa\n"),
        ("bool", {"additional_args": {"plot": False}}, "--plot=false"),
    ],
)
def test_accepted_cases(session, mace17, case, changes, expected):
    assert expected in run(**changes)["script_content"], case


def test_m21_cpu_device_requests_no_gpu(session, mace17):
    assert "--gres" not in run(device="cpu")["script_content"]


def test_all_problems_come_together(session, mace17):
    with pytest.raises(ValueError) as caught:
        run(memory="lots", partition=None, train_file="train.xyz", E0s="banana",
            additional_args={"learning_rat": 1, "batch_size": "ten"})
    message = str(caught.value)
    assert "6 problem(s)" in message
    for fragment in ["memory", "partition is required", "train_file must be an absolute path", "E0s must be",
                     "unknown MACE option 'learning_rat'", "invalid int value"]:
        assert fragment in message


def test_two_calls_give_two_bundles(session, mace17):
    first, second = run(), run()
    assert first["bundle"] != second["bundle"] and len(session.artifacts) == 2


def test_a_table_that_cannot_be_loaded_is_one_of_the_problems(session, monkeypatch):
    def offline(*args):
        raise ValueError("MACE 0.3.17's option table could not be fetched")

    monkeypatch.setattr("tools.mace_script.load_mace_cli_schema", offline)
    with pytest.raises(ValueError) as caught:
        run(partition=None)
    assert "2 problem(s)" in str(caught.value) and "could not be fetched" in str(caught.value)


# --- cluster facts from the probe ---------------------------------------------------------------


@pytest.fixture
def probed_conda(session, tmp_path):
    """The session's probe found conda.sh and three environments, of which only 'mace' has MACE installed."""
    base = tmp_path / "conda"
    (base / "etc" / "profile.d").mkdir(parents=True)
    (base / "etc" / "profile.d" / "conda.sh").write_text("")
    parser = base / "envs" / "mace" / "lib" / "python3.11" / "site-packages" / "mace" / "tools" / "arg_parser.py"
    parser.parent.mkdir(parents=True)
    parser.write_text("")
    conda_sh = (base / "etc" / "profile.d" / "conda.sh").as_posix()
    session.cluster = {"available": True, "conda_activation_path": conda_sh,
                       "available_conda_envs": ["base", "mace", "other"], "partitions": []}
    return conda_sh


def test_conda_facts_left_out_are_taken_from_the_probe(probed_conda, mace17):
    result = run(conda_env=None, conda_activation_path=None)
    assert f"source {probed_conda}\nconda activate mace\n" in result["script_content"]
    assert result["notes"][:2] == [
        f"conda_activation_path {probed_conda} was taken from the cluster probe",
        "conda_env mace was taken from the cluster probe: the only environment with MACE",
    ]


def test_the_probe_runs_when_it_has_not_and_a_missing_value_is_listed(session, mace17, monkeypatch):
    probes = []

    def probe():
        probes.append(1)
        return {"available": False, "conda_activation_path": None, "available_conda_envs": ["base", "ml"]}

    monkeypatch.setattr(schema_module, "probe_cluster", probe)
    with pytest.raises(ValueError) as caught:
        run(conda_env=None, conda_activation_path=None)
    message = str(caught.value)
    assert probes == [1] and "2 problem(s)" in message
    assert "conda_env is required: the conda environment with mace-torch on the cluster; " \
           "environments on this machine: ['base', 'ml']" in message
    assert "conda_activation_path is required: the path of conda.sh on the cluster" in message


def test_an_environment_the_cluster_does_not_have_is_a_problem(probed_conda, mace17):
    with pytest.raises(ValueError, match="conda_env 'maec' is not a conda environment of this cluster"):
        run(conda_env="maec", conda_activation_path=probed_conda)


# --- pre-flight ---------------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "data").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return home


def test_preflight_rechecks_options_against_the_target_version(session, mace17, home, monkeypatch):
    (home / "data" / "train.xyz").write_text("")
    script = run(train_file="~/data/train.xyz", additional_args={"launcher": "slurm"})["script_path"]
    assert preflight(script_path=script)["errors"] == []
    monkeypatch.setattr("tools.mace_script.load_mace_cli_schema", lambda *a: table("0.3.13"))
    errors = preflight(script_path=script)["errors"]
    assert errors == ["unknown MACE option 'launcher' (MACE 0.3.13); find it with mace_cli_schema(search=...)"]


def test_preflight_missing_input_file_is_an_error_on_the_cluster(session, mace17, home):
    script = run(train_file="~/data/train.xyz", valid_fraction=None, valid_file="/no/such/valid.xyz")["script_path"]
    report = preflight(script_path=script)
    assert report["errors"] == [] and len(report["warnings"]) == 3  # cluster unknown; two files not seen here
    assert any("--valid_file: /no/such/valid.xyz was not found on this machine" in w for w in report["warnings"])
    session.cluster = {"available": True, "partitions": [{"name": "gpu", "max_time": "1-00:00:00"}]}
    assert preflight(script_path=script)["errors"] == [
        "--train_file: $HOME/data/train.xyz does not exist", "--valid_file: /no/such/valid.xyz does not exist"]
    (home / "data" / "train.xyz").write_text("")
    assert preflight(script_path=script)["errors"] == ["--valid_file: /no/such/valid.xyz does not exist"]
