import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tools.session as session_module
from tools.mace_cli_schema import parse_mace_arg_parser

# A small stand-in for MACE's arg_parser.py, so no test fetches the real option table.
MACE_ARG_PARSER = '''
def build_default_arg_parser():
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--valid_fraction", type=float)
    parser.add_argument("--E0s", type=str)
    parser.add_argument("--max_num_epochs", type=int)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "xpu"])
    parser.add_argument("--model", type=str, choices=["MACE", "ScaleShiftMACE"])
    parser.add_argument("--r_max", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", "--learning_rate", type=float)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--swa", action="store_true")
'''

MACE_PARAMS = {
    "job_name": "mace_baseline", "time": "12:00:00", "partition": "gpu", "nodes": 1, "ntasks": 1,
    "gpus_per_node": 1, "memory": "32G", "conda_env": "mace",
    "conda_activation_path": "/opt/conda/etc/profile.d/conda.sh", "name": "water_mace",
    "train_file": "~/data/train.xyz", "valid_fraction": 0.1, "E0s": "average", "max_num_epochs": 100,
    "device": "cuda", "additional_args": {"model": "MACE", "r_max": 5.0, "batch_size": 10},
    "additional_flag_args": ["swa"],
}  # fmt: skip


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session_module, "_current", None)
    return session_module.start_session("runs")


def _find_bash():
    """A real bash for syntax checks. On Windows only the one shipped with Git: a bare `bash` may start WSL."""
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    if git is None:
        return None
    try:
        exec_path = subprocess.run([git, "--exec-path"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    root = Path(exec_path).parents[2]  # <root>/mingw64/libexec/git-core
    return next((str(p) for p in (root / "bin" / "bash.exe", root / "usr" / "bin" / "bash.exe") if p.is_file()), None)


BASH = _find_bash()


@pytest.fixture
def bash_syntax_ok():
    """Returns a function: True when `bash -n` accepts the script. Skips the test when there is no bash."""
    if BASH is None:
        pytest.skip("no bash available for a syntax check")

    def check(script_path):
        return subprocess.run([BASH, "-n", str(script_path)], capture_output=True, timeout=30).returncode == 0

    return check


@pytest.fixture
def mace_schema(monkeypatch):
    schema = parse_mace_arg_parser(MACE_ARG_PARSER, source_url="test")
    monkeypatch.setattr("tools.mace_script.load_mace_cli_schema", lambda *a, **k: schema)
    return schema


REAL_RUN = subprocess.run


class FakeRun:
    """Stands in for subprocess.run: answers by program name and records every call.
    This Python itself (the i-PI input check runs it as a child) still runs for real."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []

    def __call__(self, argv, **kwargs):
        if argv[0] == sys.executable:
            return REAL_RUN(argv, **kwargs)
        self.calls.append((list(argv), kwargs))
        answer = self.answers.get(argv[0], FileNotFoundError(2, "not found", argv[0]))
        if callable(answer) and not isinstance(answer, BaseException):
            answer = answer(argv)
        if isinstance(answer, BaseException):
            raise answer
        returncode, stdout, stderr = answer
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


@pytest.fixture
def fake_run(monkeypatch):
    def install(answers=None):
        fake = FakeRun(answers)
        monkeypatch.setattr(subprocess, "run", fake)
        return fake

    return install
