"""Generated code against the submission gate, run in the interpreter of the agents build_agents builds,
with the tools each agent has, as a CodeAgent runs it.

The person declines unless a test says otherwise; sbatch is a fake that records its calls.
"""

from pathlib import Path

import pytest
from conftest import MACE_PARAMS
from fakemodel import ScriptedModel
from smolagents.local_python_executor import InterpreterError

import tools.approval as approval
from agents import build_agents
from config import Settings
from tools.approval import Decision

WATER = "3\nwater\nO 0 0 0\nH 0.757 0.586 0\nH -0.757 0.586 0\n"
AGENTS = ["manager", "mace_agent", "ipi_agent"]


@pytest.fixture
def gate(request, session, mace_schema, fake_run, monkeypatch):
    """(run generated code in one built agent's interpreter, sbatch calls, scripts shown to the person).
    The agent is mace_agent unless the test parametrizes `gate` indirectly with another name."""
    name = getattr(request, "param", "mace_agent")
    fake = fake_run({"sbatch": (0, "999\n", "")})
    shown = []

    def decline(title, text, notes):
        shown.append(text)
        return Decision(False, "no")

    monkeypatch.setattr(approval, "ask", decline)
    manager = build_agents(ScriptedModel(), Settings(model_id="scripted", api_key="unused"))
    agent = manager if name == "manager" else manager.managed_agents[name]
    executor = agent.python_executor
    executor.send_variables({"params": dict(MACE_PARAMS)})
    executor.send_tools({**agent.tools, **agent.managed_agents})  # as agent.run() does
    sbatch_calls = [argv for argv, _ in fake.calls if argv[0] == "sbatch"]

    def run(code):
        output = executor(code).output
        sbatch_calls[:] = [argv for argv, _ in fake.calls if argv[0] == "sbatch"]
        return output

    return run, sbatch_calls, shown


@pytest.mark.parametrize("gate", AGENTS, indirect=True)
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
        "open('job.sh').read()",
        "__import__('os')",
        "eval('1')",
        "exec('1')",
        "globals()",
        "compile('1', 'x', 'eval')",
        "import json\njson.codecs",
        "import json\ngetattr(json, 'decoder')",
    ],
)
def test_no_shell_no_files_no_modules_in_any_agent(gate, code):
    run, sbatch_calls, _ = gate
    with pytest.raises(InterpreterError):
        run(code)
    assert sbatch_calls == []


@pytest.mark.parametrize("gate", ["mace_agent", "ipi_agent"], indirect=True)
@pytest.mark.parametrize(
    "code",
    [
        "vars(slurm_submit)",
        "slurm_submit.__dict__",
        "slurm_submit.forward",
        "slurm_submit.forward.__globals__",
        "getattr(preflight, '__globals__')",
        "slurm_submit.session",
        "job_status.__wrapped__",
        "type(slurm_submit).__call__",
    ],
)
def test_no_way_through_a_tool_to_the_session_or_the_approver(gate, code):
    run, _, _ = gate
    with pytest.raises(InterpreterError):
        run(code)


@pytest.mark.parametrize(
    "gate, names",
    [("mace_agent", "slurm_submit, preflight, job_status, slurm_config, MACE_script, mace_cli_schema"),
     ("ipi_agent", "slurm_submit, preflight, ipi_InputFile_generator, ipi_submit_generator, ipi_tag_help")],
    indirect=["gate"],
)
def test_tool_names_carry_no_attributes(gate, names):
    run, _, _ = gate
    code = f"[getattr(t, a) for t in [{names}] for a in ['forward', 'session', 'approval', 'name']]"
    assert set(run(code)) == {None}


@pytest.mark.parametrize("gate", ["mace_agent", "ipi_agent"], indirect=True)
def test_no_tool_runs_sbatch_on_a_file_it_did_not_generate(gate, session):
    run, sbatch_calls, shown = gate
    Path("job.sh").write_text("#!/bin/bash\n#SBATCH --job-name=x\n#SBATCH --time=10\nsbatch other.sh\n")
    result = run("""
statuses = [slurm_submit(script_path=p)['status'] for p in ['job.sh', './job.sh', 'job.sh ', '../job.sh']]
statuses.append(preflight(script_path='job.sh')['ok'])
statuses.append(job_status(job_id='1')['state'])
show_content(script='job.sh')
statuses
""")
    assert result == ["refused"] * 4 + [False, "UNKNOWN"]
    assert sbatch_calls == [] and shown == [] and session.jobs == []
    with pytest.raises(InterpreterError, match="job_id must be the number"):
        run("job_status(job_id='1; sbatch job.sh')")


def test_changing_a_looked_up_option_does_not_change_what_mace_script_accepts(gate, mace_schema, monkeypatch, session):
    run, _, _ = gate
    monkeypatch.setattr("tools.mace_cli_schema.load_mace_cli_schema", lambda *a, **k: mace_schema)
    session.cluster = {"available": False, "partitions": []}
    run("d = mace_cli_schema(names=['device'])\nd['options']['device']['choices'].append('tpu')")
    with pytest.raises(InterpreterError, match="--device must be one of"):
        run("MACE_script(**dict(params, device='tpu'))")
    assert session.artifacts == []


@pytest.mark.xfail(strict=True, reason="documented limitation: the manager's code can reach the sub-agents' "
                   "interpreters (no handle hides them); this probe fails while that is so")
@pytest.mark.parametrize("gate", ["manager"], indirect=True)
def test_the_manager_cannot_widen_a_sub_agents_imports(gate):
    run, _, _ = gate
    run("mace_agent.python_executor.authorized_imports.append('subprocess')")
    with pytest.raises(InterpreterError):
        run("mace_agent.python_executor('import subprocess')")


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


@pytest.mark.parametrize("gate", ["ipi_agent"], indirect=True)
def test_an_object_with_its_own_repr_cannot_write_code_into_the_client(gate, session):
    run, _, _ = gate
    Path("water.xyz").write_text(WATER)
    code = """
class Payload:
    def __repr__(self):
        return "__import__('os').system('id')"
x = ipi_InputFile_generator(template_type='00-NVE', structure_path='water.xyz')
ipi_runase_generator(ipi_input_path=x['script_path'], calculator='mace_mp', calculator_kwargs={'model': Payload()})
"""
    with pytest.raises(InterpreterError, match="calculator_kwargs values must be plain data"):
        run(code)
    assert [a.kind for a in session.artifacts] == ["structure", "ipi_input"]
