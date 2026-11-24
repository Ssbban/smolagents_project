"""Generated code against the submission gate, run by smolagents' interpreter as a CodeAgent runs it.

The person declines unless a test says otherwise; sbatch is a fake that records its calls.
"""

from pathlib import Path

import pytest
from conftest import MACE_PARAMS
from smolagents.default_tools import FinalAnswerTool
from smolagents.local_python_executor import InterpreterError, LocalPythonExecutor

import tools.approval as approval
from tools.approval import Decision
from tools.get_server_info import slurm_config
from tools.ipi_InputFile_generator import ipi_InputFile_generator
from tools.ipi_runase_generator import ipi_runase_generator
from tools.ipi_submit_generator import ipi_submit_generator
from tools.mace_script import MACE_script
from tools.preflight import preflight
from tools.show_content import show_content
from tools.slurm_status import job_status
from tools.slurm_submit import slurm_submit

TOOLS = [MACE_script, ipi_InputFile_generator, ipi_runase_generator, ipi_submit_generator, show_content, preflight,
         slurm_submit, job_status, slurm_config, FinalAnswerTool()]
TEMPLATE = '<simulation><ffsocket name="driver" mode="unix"><address>driver</address></ffsocket></simulation>'


@pytest.fixture
def gate(session, mace_schema, fake_run, monkeypatch):
    """(run generated code, sbatch calls, scripts shown to the person)."""
    monkeypatch.setattr("tools.ipi_InputFile_generator._select_template", lambda template_type: TEMPLATE)
    fake = fake_run({"sbatch": (0, "999\n", "")})
    shown = []

    def decline(title, text, notes):
        shown.append(text)
        return Decision(False, "no")

    monkeypatch.setattr(approval, "ask", decline)
    executor = LocalPythonExecutor(additional_authorized_imports=["json"])
    executor.send_variables({"params": dict(MACE_PARAMS)})
    executor.send_tools({tool.name: tool for tool in TOOLS})
    sbatch_calls = [argv for argv, _ in fake.calls if argv[0] == "sbatch"]

    def run(code):
        output = executor(code).output
        sbatch_calls[:] = [argv for argv, _ in fake.calls if argv[0] == "sbatch"]
        return output

    return run, sbatch_calls, shown


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "import subprocess",
        "import sys",
        "import importlib",
        "import tools.session",
        "import tools.approval",
        "open('job.sh', 'w')",
        "__import__('os')",
        "eval('1')",
        "exec('1')",
        "globals()",
        "vars(slurm_submit)",
        "slurm_submit.__dict__",
        "slurm_submit.forward",
        "slurm_submit.forward.__globals__",
        "getattr(preflight, '__globals__')",
        "slurm_submit.session",
        "job_status.__wrapped__",
        "type(slurm_submit).__call__",
        "import json\njson.codecs",
        "import json\ngetattr(json, 'decoder')",
    ],
)
def test_no_way_to_the_session_the_approver_or_a_shell(gate, code):
    run, _, _ = gate
    with pytest.raises(InterpreterError):
        run(code)


def test_tool_names_carry_no_attributes(gate):
    run, _, _ = gate
    names = "slurm_submit, preflight, job_status, slurm_config, MACE_script"
    code = f"[getattr(t, a) for t in [{names}] for a in ['forward', 'session', 'approval', 'name']]"
    assert set(run(code)) == {None}


def test_a_file_no_tool_wrote_is_refused(gate, session):
    run, sbatch_calls, shown = gate
    text = "#!/bin/bash\n#SBATCH --job-name=x\n#SBATCH --time=10\nsleep 1\n"
    Path("job.sh").write_text(text)
    (session.dir / "99-x").mkdir()
    (session.dir / "99-x" / "job.sh").write_text(text)
    for path in ["job.sh", f"{session.dir.as_posix()}/99-x/job.sh"]:
        assert run(f"slurm_submit(script_path={path!r})")["status"] == "refused"
    assert run("slurm_submit(script_path={'script_path': 'job.sh'})")["status"] == "refused"
    assert shown == [] and sbatch_calls == [] and session.jobs == []


def test_a_generated_script_changed_afterwards_is_refused(gate, session):
    run, sbatch_calls, shown = gate
    path = run("r = MACE_script(**params)\nr['script_path']")
    with open(path, "a", encoding="utf-8") as file:
        file.write("rm -rf ~\n")
    assert run("slurm_submit(script_path=r['script_path'])")["status"] == "refused"
    assert shown == [] and sbatch_calls == [] and session.jobs == []


def test_a_declined_script_stays_unsubmitted_however_often_it_is_tried(gate, session):
    run, sbatch_calls, shown = gate
    code = """
r = MACE_script(**params)
results = []
for path in [r['script_path'], './' + r['script_path'], r['script_path'].replace('/', '\\\\')]:
    results.append(slurm_submit(script_path=path)['status'])
results
"""
    assert run(code) == ["declined", "declined", "declined"]
    assert len(shown) == 3 and sbatch_calls == [] and session.jobs == []


def test_a_submitted_script_is_not_submitted_twice(gate, monkeypatch, session):
    run, sbatch_calls, _ = gate
    run("r = MACE_script(**params)")
    monkeypatch.setattr(approval, "ask", lambda title, text, notes: Decision(True))
    assert run("slurm_submit(script_path=r['script_path'])")["status"] == "submitted"
    again = run("[slurm_submit(script_path=p)['status'] for p in [r['script_path'], './' + r['script_path']]]")
    assert again == ["refused", "refused"]
    assert len(sbatch_calls) == 1 and len(session.jobs) == 1


def test_replacing_attributes_of_a_tool_name_changes_nothing(gate, session):
    run, sbatch_calls, shown = gate
    code = """
setattr(slurm_submit, 'forward', lambda script_path: {'status': 'submitted'})
slurm_submit.ask = lambda *a: True
r = MACE_script(**params)
slurm_submit(script_path=r['script_path'])['status']
"""
    assert run(code) == "declined"
    assert len(shown) == 1 and sbatch_calls == [] and session.jobs == []


def test_changing_returned_objects_does_not_change_the_ledger(gate, session):
    run, _, _ = gate
    run("""
cfg = slurm_config()
cfg['available'] = True
cfg['partitions'].append({'name': 'gpu', 'max_time': 'infinite'})
r = MACE_script(**params)
r['script_path'] = 'elsewhere/job.sh'
p = preflight(script_path=r['bundle'] + '/job.sh')
p['ok'] = True
""")
    assert session.cluster["available"] is False and session.cluster["partitions"] == []
    [artifact] = session.artifacts
    assert artifact.path == f"{artifact.bundle}/job.sh"


def test_an_object_with_its_own_repr_cannot_write_code_into_the_client(gate, session):
    run, _, _ = gate
    code = """
class Payload:
    def __repr__(self):
        return "__import__('os').system('id')"
x = ipi_InputFile_generator(template_type='00-NVE')
ipi_runase_generator(ipi_input_path=x['script_path'], calculator='mace_mp', calculator_kwargs={'model': Payload()})
"""
    with pytest.raises(InterpreterError, match="calculator_kwargs values must be plain data"):
        run(code)
    assert [a.kind for a in session.artifacts] == ["ipi_input"]
