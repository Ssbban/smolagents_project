"""The agents built by build_agents, driven by a scripted model, a scripted person and a fake sbatch.

The scenarios are those of the v1 baseline (cancel at plan review, submit without review, shell from generated
code, sub-agent provider error, sub-agent step limit, executor time-out, Ctrl+C), which now end differently.
"""

import json
import sys
from types import SimpleNamespace

import pytest
from fakemodel import ScriptedModel, code, text_of
from smolagents.local_python_executor import InterpreterError

import agents
import main
from agents import build_agents
from config import Settings
from tools import approval
from tools.session import get_session

SCRIPT = "#!/bin/bash -l\n#SBATCH --job-name=water\n#SBATCH --time=1:00:00\n#SBATCH --partition=gpu\n\necho train\n"
DICT_LINE = "def mace_agent(task: str, additional_args: dict[str, Any]) -> dict:"


def build(model, **settings):
    return build_agents(model, Settings(model_id="scripted", api_key="unused", **settings))


def every_agent(manager):
    return {"manager": manager, **manager.managed_agents}


def events(name=None):
    lines = (get_session().dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    return [r for r in records if name is None or r["event"] == name]


@pytest.fixture
def person(monkeypatch):
    """The person at the terminal: answers input() prompts in order (an exception is raised), records the prompts.
    End of input when the answers run out."""
    prompts, answers = [], []

    def answer(prompt=""):
        prompts.append(prompt)
        if not answers:
            raise EOFError
        reply = answers.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))

    def say(*replies):
        answers.extend(replies)
        return prompts

    return say


@pytest.fixture
def repl(session, person, monkeypatch, capsys):
    """Run the real command-line loop with a scripted model; returns (printed output, input prompts)."""
    for name, value in {"SMOLAGENTS_MODEL_ID": "scripted", "SMOLAGENTS_API_KEY": "unused",
                        "SMOLAGENTS_MAX_STEPS": "3", "SMOLAGENTS_PLANNING_INTERVAL": "10"}.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(main, "start_session", lambda runs_dir: session)

    def run(model, *inputs):
        prompts = person(*inputs)
        monkeypatch.setattr(main, "build_model", lambda settings: model)
        main.main()
        return capsys.readouterr().out, prompts

    return run


@pytest.fixture
def no_preflight(monkeypatch):
    """These tests are about the person's decision; the pre-flight checks have their own tests."""
    monkeypatch.setattr("tools.slurm_submit.run_preflight", lambda artifact, session: [])


def outcome_lines(out):
    return [line for line in out.splitlines() if line.startswith("Outcome:")]


# --- (b) Cancel at the plan review: nothing runs ------------------------------------------------------------

def test_b_cancel_at_plan_review_runs_nothing_and_the_session_goes_on(repl, fake_run):
    sbatch = fake_run({"sbatch": (0, "4242\n", "")})
    model = ScriptedModel(manager=[code('r = mace_agent(task="Submit a MACE training job")')],
                          mace_agent=[code('slurm_submit(script_path="job.sh")')])
    out, prompts = repl(model, "Submit a MACE training job", "3", "exit")
    assert [who for who, _ in model.calls] == ["plan"]
    assert sbatch.calls == [] and get_session().jobs == []
    assert outcome_lines(out) == ["Outcome: cancelled at the plan review"]
    assert prompts.count("User> ") == 2 and "Goodbye" in out
    assert [e["outcome"] for e in events("turn_end")] == ["cancelled at the plan review"]


def test_end_of_input_at_the_plan_menu_cancels(repl, fake_run):
    model = ScriptedModel(manager=[code('final_answer("should not run")')])
    out, _ = repl(model, "Explain r_max")
    assert [who for who, _ in model.calls] == ["plan"]
    assert outcome_lines(out) == ["Outcome: cancelled at the plan review"]


# --- (e) a sub-agent that goes straight to slurm_submit -----------------------------------------------------

def test_e_submit_without_review_asks_the_person_and_a_no_stops_it(session, person, fake_run, no_preflight):
    sbatch = fake_run({"sbatch": (0, "4242\n", "")})
    script = session.write_artifact(session.new_bundle("water"), "job.sh", SCRIPT, "mace_script")
    model = ScriptedModel(
        manager=[code(f'r = mace_agent(task="Submit the job", additional_args={{"script": "{script.path}"}})\n'
                      'final_answer(r)')],
        mace_agent=[code("result = slurm_submit(script_path=script)\nprint(result)"),
                    code('final_answer("submission " + result["status"])')],
    )
    prompts = person("1", "n")
    result = build(model).run("Submit my water job", return_full_result=True)
    assert approval.QUESTION in prompts
    assert sbatch.calls == [] and session.jobs == []
    assert "'status': 'declined'" in model.seen_by("mace_agent")[1]
    assert result.output["status"] == "success" and result.output["summary"] == "submission declined"


# --- (d) no shell and no file access from generated code, in all three agents ------------------------------

@pytest.mark.parametrize("agent_name", ["manager", "mace_agent", "ipi_agent"])
@pytest.mark.parametrize("source", ["import os", "import subprocess", "open('x.txt', 'w')"])
def test_d_generated_code_cannot_reach_a_shell_or_files(session, agent_name, source, tmp_path):
    agent = every_agent(build(ScriptedModel()))[agent_name]
    assert agent.additional_authorized_imports == ["json"]
    agent.python_executor.send_tools({**agent.tools, **agent.managed_agents})
    with pytest.raises(InterpreterError):
        agent.python_executor(source)
    assert not (tmp_path / "x.txt").exists()
    assert agent.python_executor("import json\njson.dumps([1])").output == "[1]"


# --- (f2) provider error inside a sub-agent ----------------------------------------------------------------

def test_f2_provider_error_in_a_sub_agent_reaches_the_manager_as_status_error(repl):
    model = ScriptedModel(
        manager=[code('r = ipi_agent(task="Explain the NVT ensemble")\nprint(r["status"], r["summary"])'),
                 code('final_answer("the specialist could not answer: " + r["status"])')],
        ipi_agent=[RuntimeError("Error code: 503 - upstream overloaded")],
    )
    out, _ = repl(model, "Explain the NVT ensemble", "1", "exit")
    observation = model.seen_by("manager")[1]
    assert "error AgentGenerationError" in observation and "503" in observation
    assert outcome_lines(out) == ["Outcome: answered"]
    assert "Agent> the specialist could not answer: error" in out
    assert "successfully" not in out.lower()
    [end] = events("delegation_end")
    assert end["agent"] == "ipi_agent" and end["status"] == "error"


# --- (f4) provider error in the manager's planning call, which smolagents does not wrap in an AgentError ------

def test_f4_provider_error_at_planning_fails_the_turn_and_the_session_goes_on(repl):
    model = ScriptedModel(manager=[code('final_answer("second turn")')],
                          plan=[ConnectionError("provider unreachable"), "1. Answer."])
    out, _ = repl(model, "Explain r_max", "Explain r_max again", "1", "exit")
    assert outcome_lines(out) == ["Outcome: failed: ConnectionError: provider unreachable", "Outcome: answered"]
    assert "Agent> second turn" in out


def test_a_manager_out_of_steps_is_reported_as_such(repl):
    model = ScriptedModel(manager=[code(f"x = undefined_{i}") for i in range(3)] + ["fallback answer"])
    out, _ = repl(model, "Explain r_max", "1", "exit")
    assert outcome_lines(out) == ["Outcome: stopped at the step limit (fallback answer)"]
    assert "Agent> fallback answer" in out


# --- (f3) a sub-agent that exhausts its steps --------------------------------------------------------------

def test_f3_sub_agent_out_of_steps_is_incomplete(session, person):
    model = ScriptedModel(
        manager=[code('r = ipi_agent(task="Generate an i-PI input for water")\nfinal_answer(r)')],
        ipi_agent=[code("x = undefined_1"), code("x = undefined_2"), "I could not generate the input."],
    )
    person("1")
    result = build(model, max_steps=2).run("Generate an i-PI input for water", return_full_result=True)
    assert result.output["status"] == "incomplete"
    assert result.output["summary"] == "I could not generate the input."


# --- (a) no executor time-out ------------------------------------------------------------------------------

def test_a_no_agent_has_an_executor_time_out(session):
    for agent in every_agent(build(ScriptedModel())).values():
        assert agent.python_executor.timeout_seconds is None


# --- (c) Ctrl+C inside a turn ------------------------------------------------------------------------------

def test_c_ctrl_c_inside_a_turn_leaves_a_usable_session(repl):
    model = ScriptedModel(
        manager=[code('r = mace_agent(task="Train MACE")\nprint(r)'),
                 code('final_answer("second turn; history " + str(history))')],
        mace_agent=[code('p = user_input(question="Which partition?")')],
    )
    out, prompts = repl(model, "Train MACE", "1", KeyboardInterrupt(), "Explain r_max", "1", "exit")
    assert outcome_lines(out) == ["Outcome: interrupted (Ctrl+C)", "Outcome: answered"]
    assert "Agent> second turn; history [{'request': 'Train MACE', 'outcome': 'interrupted (Ctrl+C)'}]" in out
    assert [e["event"] for e in events() if e["event"].startswith("turn")] == ["turn_start", "turn_end"] * 2


def test_c1_ctrl_c_at_the_prompt_ends_the_session_cleanly(repl):
    out, prompts = repl(ScriptedModel(), KeyboardInterrupt())
    assert prompts == ["User> "] and "Goodbye" in out


def test_c3_ctrl_c_during_a_model_call_ends_only_that_turn(repl):
    model = ScriptedModel(manager=[KeyboardInterrupt(), code('final_answer("second turn")')])
    out, _ = repl(model, "Explain r_max", "1", "Explain r_max again", "1", "exit")
    assert outcome_lines(out) == ["Outcome: interrupted (Ctrl+C)", "Outcome: answered"]


# --- the result dict ---------------------------------------------------------------------------------------

def test_manager_code_gets_a_dict_and_branches_on_it(session, person):
    model = ScriptedModel(
        manager=[code('r = mace_agent(task="What is r_max?")\n'
                      'if r["status"] == "success":\n'
                      '    final_answer({"said": r["summary"], "keys": sorted(r)})')],
        mace_agent=[code('final_answer("r_max is the radial cutoff")')],
    )
    person("1")
    result = build(model).run("What is r_max?", return_full_result=True)
    assert result.output == {"said": "r_max is the radial cutoff",
                             "keys": ["agent", "artifacts", "jobs", "status", "summary"]}
    [start], [end] = events("delegation_start"), events("delegation_end")
    assert start["agent"] == end["agent"] == "mace_agent" and start["task"] == "What is r_max?"
    assert end["status"] == "success" and end["tokens"] == {"input": 100, "output": 10}
    [plan_messages] = [messages for who, messages in model.calls if who == "plan"]
    assert DICT_LINE in text_of(plan_messages[0])


def test_artifacts_and_jobs_of_the_delegation_come_from_the_ledger(session, person):
    model = ScriptedModel(manager=[code('r = mace_agent(task="x")\nfinal_answer(r)')],
                          mace_agent=[code('final_answer("I wrote /tmp/invented.sh and submitted job 1")')])
    session.write_artifact(session.new_bundle("before"), "job.sh", SCRIPT, "mace_script")
    person("1")
    result = build(model).run("x", return_full_result=True)
    assert result.output["artifacts"] == [] and result.output["jobs"] == []


def test_keyword_arguments_from_generated_code_do_not_reach_run(session, person, monkeypatch):
    model = ScriptedModel(
        manager=[code('r = mace_agent(task="x", max_steps=999, reset=False, return_full_result=False)\n'
                      'final_answer(r["status"])')],
        mace_agent=[code('final_answer("ok")')],
    )
    manager = build(model)
    mace = manager.managed_agents["mace_agent"]
    seen, real_run = [], mace.run

    def spy(task, **kwargs):
        seen.append(kwargs)
        return real_run(task, **kwargs)

    monkeypatch.setattr(mace, "run", spy)
    person("1")
    assert manager.run("x") == "success"
    assert seen == [{"additional_args": None, "return_full_result": True}]


def test_additional_args_reach_the_sub_agent_as_variables(session, person):
    model = ScriptedModel(manager=[code('r = mace_agent(task="x", additional_args={"epochs": 7})\n'
                                        'final_answer(r["summary"])')],
                          mace_agent=[code("final_answer(str(epochs * 2))")])
    person("1")
    assert build(model).run("x") == "14"


# --- prompt templates --------------------------------------------------------------------------------------

def test_the_stock_template_still_has_the_three_signatures_that_are_replaced():
    stock = agents.stock_templates()
    places = [stock["system_prompt"], stock["planning"]["initial_plan"],
              stock["planning"]["update_plan_post_messages"]]
    assert [text.count(agents.STR_SIGNATURE) for text in places] == [1, 1, 1]
    assert json.dumps(stock).count(agents.STR_SIGNATURE) == 3
    assert "{{name}}" in stock["managed_agent"]["task"] and "{{task}}" in stock["managed_agent"]["task"]
    replaced = json.dumps(agents.manager_templates())
    assert agents.STR_SIGNATURE not in replaced and replaced.count(agents.DICT_SIGNATURE) == 3


def test_the_manager_prompt_declares_dicts_and_documents_the_keys(session):
    manager = build(ScriptedModel())
    prompt = manager.system_prompt
    assert DICT_LINE in prompt and "ipi_agent(task: str, additional_args: dict[str, Any]) -> dict:" in prompt
    assert "-> str:" not in prompt
    for agent in manager.managed_agents.values():
        assert all(f'"{key}"' in agent.description for key in ("status", "summary", "artifacts", "jobs"))


def test_the_sub_agents_task_asks_for_a_short_summary_and_no_invented_ids(session):
    for agent in build(ScriptedModel()).managed_agents.values():
        task = agent.prompt_templates["managed_agent"]["task"]
        assert task == agents.SUBAGENT_TASK and "### 1. Task outcome" not in task
        assert "never invent a file path or a job id" in task


# --- model client and the command line ---------------------------------------------------------------------

def test_build_model_sets_a_request_time_out_and_retries():
    model = agents.build_model(Settings(model_id="m", api_key="unused", api_base="http://127.0.0.1:9"))
    assert model.client.timeout == 120 and model.client.max_retries == 2


def test_empty_lines_are_ignored_and_end_of_input_ends_the_session(repl):
    model = ScriptedModel()
    out, prompts = repl(model, "", "   ")
    assert model.calls == [] and prompts == ["User> "] * 3 and "Goodbye" in out


def test_an_approved_submission_is_printed_from_the_ledger(repl, session, fake_run, no_preflight):
    sbatch = fake_run({"sbatch": (0, "4242\n", "")})
    script = session.write_artifact(session.new_bundle("water"), "job.sh", SCRIPT, "mace_script")
    model = ScriptedModel(
        manager=[code(f'r = mace_agent(task="Submit", additional_args={{"script": "{script.path}"}})\n'
                      'final_answer(r["summary"])')],
        mace_agent=[code('result = slurm_submit(script_path=script)\nfinal_answer("job " + result["job_id"])')],
    )
    out, _ = repl(model, "Submit my water job", "1", "y", "exit")
    assert len([argv for argv, _ in sbatch.calls if argv[0] == "sbatch"]) == 1
    assert "Agent> job 4242" in out and f"  job submitted: 4242 from {script.path}" in out
    assert "file written" not in out  # the script was written before this turn


def test_building_the_agents_needs_no_retrieval_packages(session):
    for name in ("langchain", "faiss", "datasets", "torch"):
        assert name not in sys.modules
    manager = build(ScriptedModel())
    for agent in manager.managed_agents.values():
        assert {"preflight", "slurm_submit", "job_status", "slurm_config"} <= set(agent.tools)
    for name in ("langchain", "faiss", "datasets", "torch"):
        assert name not in sys.modules
