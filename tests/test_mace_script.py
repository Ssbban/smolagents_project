"""The 23 MACE_script cases of the v1 baseline, as far as wave 0a changes or keeps their outcome."""

from pathlib import Path

import pytest
from conftest import MACE_PARAMS

from tools.mace_script import MACE_script


def run(**changes):
    return MACE_script(**{**MACE_PARAMS, **changes})


def test_m00_valid_writes_job_sh_in_a_bundle(session, mace_schema):
    result = run()
    assert set(result) == {"script_path", "bundle", "script_content"}
    assert result["script_path"] == f"{result['bundle']}/job.sh"
    assert Path(result["script_path"]).read_bytes() == result["script_content"].encode()
    assert "--model=MACE" in result["script_content"] and result["script_content"].endswith("--swa")
    assert session.find_artifact(result["script_path"]).kind == "mace_script"


@pytest.mark.parametrize(
    "case, changes, fragment",
    [
        ("M01", {"additional_args": {"model": "MACE", "learning_rat": 0.01}}, "Unknown MACE CLI option"),
        ("M02", {"max_num_epochs": "100"}, "max_num_epochs must be an integer"),
        ("M03", {"additional_args": {"batch_size": "ten"}}, "requires an integer value"),
        ("M04", {"additional_args": {"r_max": "5.0"}}, "requires a numeric value"),
        ("M05", {"device": "tpu"}, "must be one of"),
        ("M06", {"additional_args": {"swa": True}}, "is a switch flag"),
        ("M09", {"train_file": "data/my train.xyz"}, "train_file may only contain"),
        ("M10", {"name": "water; rm -rf ~"}, "name may only contain"),
        ("M12", {"memory": "32G; rm -rf ~"}, "memory must be a number"),
        ("M13", {"mail_user": "a@b.c; rm -rf ~"}, "mail_user must be one e-mail address"),
        ("M14", {"additional_args_slurm": {"account": "proj; rm -rf ~"}}, "must be one token"),
        ("M15", {"partition": "$(whoami)"}, "partition may only contain"),
        ("M16", {"memory": "32G\nrm -rf ~"}, "memory must not contain control characters"),
        ("M18", {"valid_file": "data/valid.xyz"}, "mutually exclusive"),
        ("M19", {"additional_args": {"max_num_epochs": 5}}, "has a dedicated argument"),
        ("M20", {"partition": None}, "partition is required"),
        ("M22", {"time": "2 days"}, "time must use a Slurm format"),
    ],
)
def test_rejected_cases_write_nothing(session, mace_schema, case, changes, fragment):
    with pytest.raises(ValueError, match="MACE_script") as caught:
        run(**changes)
    assert fragment in str(caught.value), case
    assert session.artifacts == [] and not any(session.dir.iterdir())


@pytest.mark.parametrize(
    "case, changes, expected",
    [
        ("M07", {"train_file": "~/data/train.xyz"}, "--train_file='~/data/train.xyz'"),  # tilde: wave 1c
        ("M08", {"conda_activation_path": "~/miniconda3/etc/profile.d/conda.sh"},
         "source ~/miniconda3/etc/profile.d/conda.sh"),
        ("M11", {"E0s": "average; rm -rf ~"}, "--E0s='average; rm -rf ~'"),
        ("M17", {"additional_args": {"log_dir": "logs; rm -rf ~"}}, "--log_dir='logs; rm -rf ~'"),
        ("M22b", {"time": "2-00:00:00"}, "#SBATCH --time=2-00:00:00"),
    ],
)
def test_accepted_cases_keep_values_quoted(session, mace_schema, case, changes, expected):
    assert expected in run(**changes)["script_content"], case


def test_m21_cpu_device_requests_no_gpu(session, mace_schema):
    assert "--gres" not in run(device="cpu")["script_content"]


def test_spec_and_option_problems_come_together(session, mace_schema):
    with pytest.raises(ValueError) as caught:
        run(memory="lots", partition=None, additional_args={"learning_rat": 1})
    message = str(caught.value)
    assert "3 problem(s)" in message
    assert "memory" in message and "partition is required" in message and "Unknown MACE CLI option" in message


def test_two_calls_in_the_same_second_give_two_bundles(session, mace_schema):
    first, second = run(), run()
    assert first["bundle"] != second["bundle"]
    assert Path(first["script_path"]).is_file() and Path(second["script_path"]).is_file()
    assert len(session.artifacts) == 2
