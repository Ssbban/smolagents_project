# Design

This note is for someone who will change the code. It gives the reasoning behind each mechanism, the smolagents
behaviour it relies on, and the alternative that was rejected. smolagents references are to version 1.26.0
(`requirements.txt` pins it); file names are inside the installed `smolagents` package.

## What stays from v1, and the one idea of the upgrade

Kept from v1:

- Code as action. The agents act by writing Python that calls tools, so the manager can hold a result in a
  variable, branch on it and chain two delegations in one step.
- Delegation to domain sub-agents. A manager without tools; `mace_agent` and `ipi_agent` each own one tool set.
- Typed tools that validate and render. Specs in `tools/specs.py`, pure renderers, restricted alphabets for
  everything written into a shell script.
- Grounding in external sources: the cluster probe, MACE's own option parser, the i-PI templates. v1 fetched the
  last two from moving branches; they are now pinned to versions (see below).
- A person in the loop: plan review in the manager, the person's answers to the agents' questions.

The idea of this upgrade: the agents keep deciding their steps, and every side effect is guarded by the tool
that performs it, with state kept by code in a session ledger (`tools/session.py`). v1 kept the order of steps,
the preview and the "type yes" rule in prompt text; a scripted model that skipped them submitted a job. Now the
prompts give guidance, and a tool refuses, or asks the person, whatever order the model chooses.

## Mechanisms

### Delegation contract

What: a call `mace_agent(task=..., additional_args=...)` in the manager's code returns a dict:
`agent`, `status` (`success`, `incomplete`, `error`), `summary` (the model's short report), `artifacts` and
`jobs` (taken from the ledger, never from the model's text).
Where: `agents.py` `ScienceAgent.__call__`; `manager_templates()` and `subagent_templates()`; `RESULT_DOC`.
smolagents behaviour: `MultiStepAgent.__call__` (`agents.py:868-890`) is the managed-agent entry point and a plain
method, so a subclass can override it. The manager's interpreter calls it and passes the return value through
unchanged (`local_python_executor.py:941-951` resolves the name, `safer_func` at `211-237` only checks the result
for modules and dangerous builtins). `run(..., return_full_result=True)` gives `RunResult.state`
(`max_steps_error` or `success`, `agents.py:523-526`). `_setup_managed_agents` forces `output_type = "string"`
(`agents.py:387`) and `prompts/code_agent.yaml` hard-codes `-> str:` (lines 146, 215, 276), so the manager's
templates are edited to say `-> dict:`; a test fails if the stock text changes. The stock `__call__` forwards any
keyword argument from generated code into `run()` (`agents.py:876`), so `max_steps=999` or `reset=False` from the
manager would reach the sub-agent; the override drops them.
Rejected: a Tool wrapper around the sub-agent (it gives the same dict but loses the managed-agent section of the
prompt, and the wrapper's attributes expose the child the same way); prose reports (the stock template; a failed
sub-agent's fallback text looked like a success, measured on v1).

### Submission gate

What: `slurm_submit(script_path)` accepts only a job script that a generator tool wrote in this session, whose
bytes still hash to the recorded sha256, and that was not submitted before. It runs the pre-flight checks, shows
the script and every file of the bundle it runs, asks the person, and then calls `sbatch --parsable` once, in the
bundle directory, with a 60 s time-out. Refusals and declines are returned as a `status`, not raised.
Where: `tools/slurm_submit.py` `_submit`; `tools/approval.py` `ask`; `tools/preflight.py`; `tools/session.py`.
smolagents behaviour: a tool name in generated code resolves to a `functools.wraps` wrapper
(`local_python_executor.py:231`), which copies the tool object's `__dict__`. Instance attributes are therefore
readable from generated code and the objects they reference are shared; dunder reads are blocked
(`nodunder_getattr`, line 68), attribute writes land on the wrapper. So no approval state lives on a tool or
agent instance: the current session is a module global (`tools/session.py` `_current`), and the decision is taken
by `approval.ask` at call time. Tools that return cached data return deep copies: a probe showed that appending
to a list returned by `mace_cli_schema` changed what `MACE_script` accepted until the copy was added.
Rejected: the v1 protocol (the model asks "type yes" with `user_input` and reads the answer itself, so the model
decides); an approval token the model passes back; approval state on the tool instance; an approval tool the
model calls before submitting.

### Cancel that cancels

What: Cancel (or end of input) at the plan review raises `UserCancelled`, a `BaseException`. `main.py` catches it
and ends the turn; the session goes on.
Where: `tools/customize_plan.py` `interrupt_after_plan`; `main.py` `run_turn`.
smolagents behaviour: `_run_stream` runs the PlanningStep callbacks in `_finalize_step` (`agents.py:566`) before
the action step (`569-604`) and outside its `try`. `agent.interrupt()` (`754-756`) only sets a flag that is read at
the top of the next loop (`546`), after one more action step. Raising in the callback is what stops the turn:
the callback runs outside the step's `try`, so the exception leaves `run()` before any action step. A
`BaseException` also passes the interpreter's `except Exception` (`local_python_executor.py:1655`), the `try` of
generated code (`evaluate_try`, `1159-1169`) and the agent's step handler (`agents.py:1734`); smolagents itself
uses this for `FinalAnswerException` (`local_python_executor.py:1572`). That matters only if a plan review ever
runs inside a sub-agent, whose run happens inside the manager's code action; today only the manager has one. With
`UserCancelled` changed to an `Exception` subclass,
`tests/test_agents.py::test_b_cancel_at_plan_review_runs_nothing_and_the_session_goes_on` still passes; with v1's `agent.interrupt()` it fails. Callbacks run in MRO order (`memory.py:314`), so the plan
review runs before the tracer and the trace holds the approved or modified plan.
Rejected: `agent.interrupt()` as in the smolagents plan-customisation example (measured on v1: the one extra step
was a whole sub-agent run including `sbatch`); an `Exception` subclass (in a sub-agent with a plan review it would
become a manager code error that the model retries; no sub-agent has one now, so no test shows this).

### No executor time-out

What: all three agents run with `executor_kwargs={"timeout_seconds": None}`. Time limits sit where the blocking
happens: `sbatch` 60 s, `squeue` / `sacct` / `sinfo` / `conda` 30 s, the i-PI check child 60 s, GitHub and the
MACE release fetch 30 s, the model client 120 s with 2 retries. `user_input` has none: the person may think.
Where: `agents.py` `EXECUTOR`, `build_model`; the tools named.
smolagents behaviour: the default is 30 s (`local_python_executor.py:60`), applied to one whole code action
(`1664-1665`) through a thread pool whose `with` block waits for the worker on exit (`308`). A delegation that
takes longer is not stopped; it finishes, and then the step reports a time-out error, which the model may retry.
Measured on v1: the person answered after 35 s and both agents reported a time-out after the sub-agent succeeded.
Rejected: a larger limit (it still wraps the person's think time and still cannot stop the work).

### One turn is one run, with the ledger passed as data

What: each request is `manager.run(request, additional_args={"session": snapshot, "history": last 3 turns},
return_full_result=True)` with a fresh memory. `snapshot()` lists bundles, files, jobs with their last state and
the cluster partitions; `history` holds each earlier request with its outcome line and the first 1,000 characters
of the answer. The outcome printed to the person comes from the run state and the ledger, never from the model.
Where: `main.py` `run_turn`; `tools/session.py` `Session.snapshot`, `mark`, `since`.
smolagents behaviour: `additional_args` become variables in the interpreter and are also printed into the task
text (`agents.py:471-475`), so the snapshot is kept small. With `reset=False`, memory grows every
turn and later turns start with the update-plan templates (`is_first_step = len(self.memory.steps) == 1`,
`agents.py:556`), so earlier, unrelated requests are shown to the model as attempts at the current one (v1 did
this). Sub-agents already start fresh on every delegation; the dict carries what they produced.
Rejected: an ever-growing transcript; a model-written summary of earlier turns as the only state.

### All problems at once

What: validators collect messages (`check`) and a tool raises one `ValueError` listing every problem
(`raise_problems`): spec fields, MACE options, i-PI's own messages. The model fixes all of them in the next call.
Where: `tools/specs.py`; every generator; `tools/mace_script.py` `option_problems`.
smolagents behaviour: an exception in a tool becomes the next observation: the interpreter wraps it ("Code
execution failed at line ... due to", `local_python_executor.py:1659-1661`), `CodeAgent` raises
`AgentExecutionError` (`agents.py:1750`) and the loop stores it on the step (`597-599`).
Rejected: v1's first-problem-only validation (one model round per missing field).

### Option tables and templates pinned to versions

What: the MACE option table is read with `ast` from `build_default_arg_parser` of the MACE that will run the job:
in the job's conda environment if visible, else importable here, else release tag 0.3.17 fetched once and cached.
MACE is never imported (it pulls in torch). The script records the version, and pre-flight re-validates the
options. The ten i-PI templates are vendored at one upstream commit with sha256 checksums
(`tools/ipi_templates/SOURCE.json`), with one catalogue (`catalog.json`). `ipi_tag_help` reads the installed i-PI's
input classes. The retriever's corpus and embedding model are pinned to revisions.
Where: `tools/mace_cli_schema.py` `load_mace_cli_schema`; `tools/ipi_InputFile_generator.py`; `tools/ipi_help.py`;
`tools/RetrieverTool.py`.
Rejected: fetching from GitHub `main` on every call (v1: four preprocess-only MACE options accepted, a dump of
about 39,000 characters in the context, templates from a moving third-party branch); fetching four i-PI source
files into the context (about 88,000 characters; 87,886 in i-PI 3.3.1) to look up a tag. v1's code gives 39,414
characters for MACE 0.3.17's parser; the summary of `mace_cli_schema` now gives about 3,500.

### Validation by i-PI's own loader, in a child process

What: before an input is written, `check_input` builds a validation copy (every force field replaced by i-PI's
dummy potential, every structure file by a three-atom placeholder) and loads it with `python -I
tools/ipi_check.py input.xml` in a temporary directory, with a 60 s time-out. The child prints one JSON line: i-PI's
error if any, tags written twice, and the values i-PI understood (temperature in K, time step in fs, pressure in
bar per replica, beads, replicas, steps). Values outside plain ranges are reported with the value i-PI read: "300"
without units is about 9.5e7 K. Answers are cached by the text of the validation copy.
Where: `tools/ipi_check.py`; called by `ipi_InputFile_generator`, `ipi_submit_generator` and the i-PI pre-flight
check.
Why a child: i-PI builds every force field while loading, may leave threads behind and prints to stdout; `-I`
keeps the child from importing the working directory. One check took 0.44-0.84 s per template (measured once).
Rejected: value rules written again in this project (they would drift from the i-PI that reads the file); loading
in the agent's process.

## Rejected for this project

- A fixed pipeline or state machine instead of agents. The open decisions (what to ask, which template, which
  options, how to react to an error) are what the agents are for. The guarantees that matter attach to side
  effects, and the tools check them whatever the order of steps.
- JSON tool-calling agents instead of CodeAgents. The manager would lose the variable that holds a sub-agent's
  dict, and with it branching on `status` and passing paths to the next delegation in code.
- Remote or container executors. smolagents refuses them with managed agents: "Managed agents are not yet
  supported with remote code execution." (`agents.py:1608-1609`). The tools must also run where `sbatch` is.
- A handle object that hides a sub-agent's internals from the manager's code. It would put a non-agent object
  into `managed_agents`; in a one-off feasibility check (not a committed test) smolagents' `visualize()` and
  `to_dict()` / `save()` failed on it.
  The risk addressed here is an over-helpful model, not a hostile one; the reach that remains is documented and
  kept as an expected-failure test.
- A simulated scheduler. Its likeness to a real Slurm would itself need checking. Fakes of `subprocess.run` test
  the tools' logic; behaviour on a real scheduler can only come from a real cluster.

## What comparable public projects do

Only what informed a decision here. None of the four asks a person before a side effect.

- [gromacs_smolagent](https://github.com/virtualramblas/gromacs_smolagent): a smolagents manager CodeAgent with
  four ToolCallingAgent sub-agents, one per GROMACS stage, each with typed `@tool` functions that run `gmx` through
  `subprocess` (energy minimisation runs `gmx mdrun` directly from a tool). A second script checks generated `gmx`
  command lines with a parser and feeds the errors back to the model. It supports the manager plus domain
  sub-agent shape on smolagents and the "report problems to the model" loop.
- [MatClaw](https://github.com/cz2014/MatClaw): one CodeAgent on a vendored copy of smolagents 1.24.0 with code
  run in Jupyter kernels and every import allowed; a container is the only boundary; the model submits HPC jobs
  itself. It removed the executor time-out and bounds waits inside tools, tests the real agent loop with a scripted
  model, and checks prompt text against code constants. The first two are used here, the third as a test that
  prompts name only registered tools; its safety model is not used.
- [Biomni](https://github.com/snap-stanford/Biomni): a LangGraph agent that runs generated code with `exec` in its
  own process, no allow-list, a time-out per execution block; validator tools the agent may call; the
  human-feedback tool is commented out. The validator idea is used here, with the check moved into the tool that
  has the side effect.
- [Science-Star](https://github.com/ustc-ai4science/Science-Star): a smolagents manager CodeAgent with a
  ToolCallingAgent worker for benchmarks; the worker returns a string report plus a run summary; no side effects to
  gate. Its manager prompt carries a list of tools the manager does not have, because the manager kept calling
  them. Here the same drift between prompt and tools is caught by a test (`tests/test_prompts.py`), and the string
  boundary is replaced by the dict.

## Known limits and what would lift each

| Limit | What would lift it |
|---|---|
| No live-model run of this version | A set of requests run against a real model, read from the trace: outcome, steps, tokens |
| No real cluster run | Small MACE and i-PI jobs submitted from a login node, then `job_status` and the logs |
| Manager code can reach a sub-agent object: import list (expected-failure test), API key and tools (one-off probe, no test) | A handle in `managed_agents`, giving up `visualize()` and `save()`; or sub-agents in another process |
| The local interpreter is not a sandbox | Operating-system isolation around the whole process |
| Executor time-out: only the setting is tested, the 35 s wait was not re-run | A scripted-model test whose person answers after more than 30 s |
| The 512-token chunk test is skipped without the retrieval packages | A CI job with those packages and the cached model |
| Retriever checked by a keyword heuristic on eight queries | A labelled query set with expected pages |
| Timings from one Windows machine | The same measurements on the login node |
| `max_num_epochs="100"` refused | Convert numeric strings for the integer fields of `MACEJobSpec`, as argparse would |
| Template comments dropped | `ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))` when reading templates |
| `conda_path` / `conda_activation_path` and `additional_slurm_args` / `additional_args_slurm` differ | Accept both names in both generators |
