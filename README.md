# MACE & I-PI Agent (Public Version)

A scientific agent for people who train MACE machine-learning potentials and run i-PI molecular dynamics on a
Slurm cluster. A person describes a job in plain language at a terminal. A manager agent hands the request to a
MACE agent or an i-PI agent. The sub-agents are prompted to ask for what is missing and write the job files
through typed tools. The submit tool itself shows the job to the person and runs `sbatch` only after the person
types `y` (or `yes`). It is built on Hugging Face [smolagents](https://github.com/huggingface/smolagents) 1.26.0, with agents
that act by writing Python code.

Status. On 2026-11-26 it ran on a real Slurm cluster (Slurm 19.05, CentOS 7) with a live model (`claude-sonnet-5-5`
through Anthropic's OpenAI-compatible endpoint): two requests in one session, a MACE training job and then an i-PI
simulation with the model just trained, each submitted after the person's `y`, both completed; after the changes
that run led to, one request for both in a new session also completed, and the agent checked the result. The tools were also
driven without a model: MACE training on CPU and GPU, i-PI runs on CPU, two at once on one node. What that day found
and what changed is in [First run on a real cluster](#first-run-on-a-real-cluster-2026-11-26). The agent-level tests
still drive the real agents with a scripted model, and Slurm is a fake in the tests.

## Architecture

```mermaid
flowchart LR
    P([person at the terminal]) -- request, plan review --> M[manager_agent]
    M -- "mace_agent(task, additional_args) -> dict" --> A[mace_agent]
    M -- "ipi_agent(task, additional_args) -> dict" --> B[ipi_agent]
    A --> T1[MACE tools]
    B --> T2[i-PI tools]
    A & B --> S[slurm_submit]
    S -- "shows the job, asks y / n" --> P
    T1 & T2 & S --> L[("session ledger (files and trace in runs/)")]
    L -- "snapshot, passed in by main.py with the last turns" --> M
```

| Part | What it is | Where |
|---|---|---|
| `manager_agent` | CodeAgent with no tools; delegates; the person reviews its plan | `agents.py` `build_agents` |
| `mace_agent` | CodeAgent: MACE option look-up, job script, pre-flight, submit, job status and output | `agents.py`, `tools/mace_*.py` |
| `ipi_agent` | CodeAgent: starting structure, i-PI templates, tag help, input, ASE client, job script, submit | `agents.py`, `tools/structure_writer.py`, `tools/ipi_*.py` |
| Shared tools | `slurm_config`, `preflight`, `slurm_submit`, `job_status`, `job_output`, `show_content`, `user_input`, `web_search` | `tools/` |
| Session ledger | Files the tools wrote (path, sha256, bundle), jobs submitted, the cluster probe | `tools/session.py` |
| Trace | One JSON line per event of a session | `tools/trace.py` |
| REPL | One request = one run of the manager | `main.py` |

## Who decides what

| The agents decide | The tools enforce in code | The person decides |
|---|---|---|
| Which sub-agent gets a request, in what order | Only a job script written by this session's generator tools, unchanged since (sha256), is submitted, and only once (`tools/slurm_submit.py`, `tools/session.py`) | Approve, modify or cancel each plan of the manager |
| What to ask the person, and when | Pre-flight before every submission: `#SBATCH` header, partition and time limit against the cluster probe, every file of the bundle unchanged (`tools/preflight.py`) | Each submission: `y`, `n`, or what to change |
| Which template, options and values to try | The question to the person is asked by the submit tool at call time; without a terminal, or at end of input, it declines (`tools/approval.py`) | Answers to the agents' questions (paths, partition, model) |
| Which tools to call, and when to stop | Generated code may import `json` besides smolagents' base modules (`math`, `re`, ...); `os`, `subprocess`, `sys`, `open` are blocked (`agents.py`) | Whether a failed job is resubmitted |
| What to report back | A sub-agent's status comes from the run state or the exception; its files and jobs from the ledger (`agents.py` `ScienceAgent`) | |
| | Cancel at plan review stops the turn before any action (`tools/customize_plan.py`) | |
| | Every input problem in one error; names and paths in a restricted alphabet (`tools/specs.py`) | |
| | MACE options checked against the option table of the job's MACE when it is visible, else a local MACE or release 0.3.17; the script records which (`tools/mace_cli_schema.py`, `tools/mace_script.py`) | |
| | i-PI input loaded by i-PI itself; only listed paths editable; values without units reported (`tools/ipi_check.py`, `tools/ipi_InputFile_generator.py`) | |
| | Socket, structure file and client count agree across input, client and job script (`tools/ipi_runase_generator.py`, `tools/ipi_submit_generator.py`); on the cluster a structure path must exist (`tools/ipi_InputFile_generator.py`) | |
| | A structure the agent works out reaches the disk only through `write_structure`: known elements, one position per atom, a right-handed cell, no atoms closer than 0.5 Å (`tools/structure_writer.py`) | |
| | Generated files are never overwritten and have LF line endings (`tools/session.py`) | |
| | Job scripts run with `--export=NONE`: nothing comes from the shell the agent was started in; the i-PI job's conda environment has `i-pi` and MACE when it is visible (`tools/specs.py`, `tools/ipi_submit_generator.py`) | |
| | Label keys given for training are in the first frame of the training file (`tools/mace_script.py`) | |
| | `job_output` reads only the bundle of a job submitted in this session (`tools/job_output.py`) | |
| | `sbatch` without an answer in 60 s gives status `unknown`, never an invented job id (`tools/slurm_submit.py`) | |

## What changed since v1

v1 is commit `3fc8270`. It was measured with a scripted model driving the real v1 code, with `sbatch` replaced by
a recorder; the tool cases were run by calling the v1 tools directly. Each row gives the v1 code that causes the
behaviour, so it can be checked by reading. [docs/V1_DEFECTS.md](docs/V1_DEFECTS.md) has every defect, including
the open ones.

| v1 behaviour (measured) | v1 code at 3fc8270 | Now | Test |
|---|---|---|---|
| Person answers after 35 s: time-out errors after the sub-agent succeeded | `main.py:71-115`, no `executor_kwargs` (default 30 s) | No executor time-out; time-outs inside blocking tools | `test_agents.py::test_a_no_agent_has_an_executor_time_out` (configuration only) |
| Cancel at plan review: a sub-agent still ran and submitted | `tools/customize_plan.py:75` `agent.interrupt()` | Only the planning call runs; session goes on | `test_agents.py::test_b_cancel_at_plan_review_runs_nothing_and_the_session_goes_on` |
| Ctrl+C at `User>`, in `user_input` or in a model call ends the CLI | `main.py:121`, `main.py:130` | Ends the turn; at the prompt ends cleanly | `test_agents.py::test_c1_ctrl_c_at_the_prompt_ends_the_session_cleanly`, `test_c_ctrl_c_inside_a_turn_leaves_a_usable_session`, `test_c3_ctrl_c_during_a_model_call_ends_only_that_turn` |
| Generated code ran shell commands in all three agents | `main.py:77-82, 92-97, 109-113` | `import os`, `open`, `eval` ... blocked | `test_gate_probes.py::test_no_shell_no_files_no_modules_in_any_agent` |
| A sub-agent submitted without any question | `tools/slurm_submit.py:13` | The tool asks; `n` means no `sbatch` | `test_agents.py::test_e_submit_without_review_asks_the_person_and_a_no_stops_it` |
| `slurm_submit` accepted any file | `tools/slurm_submit.py:13` | Refused: 4 spellings of a hand-written file | `test_gate_probes.py::test_no_tool_runs_sbatch_on_a_file_it_did_not_generate` |
| Sub-agent 503 or step limit: "Task completed successfully!" | `main.py:127-129` | Dict with `status` `error` / `incomplete` | `test_agents.py::test_f2_provider_error_in_a_sub_agent_reaches_the_manager_as_status_error`, `test_f3_sub_agent_out_of_steps_is_incomplete` |
| Manager provider error: printed, loop went on (no defect) | `main.py:130-134` | Kept: turn `failed: <error>`, next turn answered | `test_agents.py::test_f4_provider_error_at_planning_fails_the_turn_and_the_session_goes_on` |
| Slurm tools on a machine without Slurm: raw `[WinError 2]` | `tools/get_server_info.py:73` | `available: False` with a reason; `failed`; `UNKNOWN` | `test_server_info.py::test_slurm_config_without_slurm`, `test_slurm_submit.py::test_sbatch_missing_is_failed`, `test_slurm_status.py::test_without_slurm` |
| MACE cases: junk in `#SBATCH`, `r_max "5.0"` refused, `~` never expanded | `tools/specs.py:154-157`, `tools/mace_cli_schema.py:194-197`, `tools/mace_script.py:25-26` | Junk refused, `"5.0"` accepted, `"$HOME/..."` | `test_mace_script.py::test_rejected_cases_write_nothing`, `test_accepted_cases` |
| Two files in one second: second overwrote the first | `tools/mace_script.py:220-221` | One bundle per job; never overwritten | `test_session.py::test_write_artifact_refuses_overwrite` |
| i-PI: socket pointed at another host, `total_steps "-5 abc"`, two `<prng>` accepted | `tools/ipi_InputFile_generator.py:95-136` | Refused or reported by i-PI's loader | `test_ipi_generators.py::test_baseline_input_cases` |
| A temperature without units is read in atomic units | no check | Reported: "300" is 9.47e7 K | `test_ipi_check.py::test_the_unit_trap_is_reported_with_the_value_ipi_understood` |
| i-PI job always requested a GPU | `tools/ipi_submit_generator.py:23-24` | GPU only for a `cuda` client | `test_ipi_generators.py::test_gpu_request_follows_the_client_device` |
| `github_fetch`: HTML for blob URLs, no time-out, any URL | `tools/github_fetch.py:15` | Raw URL, GitHub only, 30 s, 1 MB | `test_fetch_retriever_main.py::test_github_fetch_uses_timeout_and_raw_url` |
| Retriever: chunks up to 600 tokens (model reads 512), no sources | `tools/RetrieverTool.py:36-43, 83-89` | At most 512 tokens; a source line per passage | `test_retriever.py::test_each_passage_is_followed_by_its_source` (the 512 check is a test that is skipped without the retrieval packages) |
| About 52 s to the first prompt (index built at import) | `main.py:6`, `tools/RetrieverTool.py:13-62` | 1.72 s median of 5, index built on first query | `test_agents.py::test_building_the_agents_needs_no_retrieval_packages` (imports only, not the time) |

Test files are under `tests/`. Both start-up times come from one Windows machine: v1 about 52 s (with network
access); now the median of 5 runs with placeholder settings and no model call.

## Quick start

Python 3.11 is the only version used so far.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
cp .env.example .env                             # then set SMOLAGENTS_MODEL_ID and SMOLAGENTS_API_KEY
.venv/bin/python main.py                         # run from the repository root
```

On a cluster, `requirements.txt` pins `faiss-cpu` 1.9.0.post1, the newest wheel for glibc 2.17 (CentOS 7). The
retriever needs only CPU torch: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu` before
the requirements leaves out about 2.5 GB of CUDA libraries.

Settings (`config.py`, environment first, then `.env`): `SMOLAGENTS_MODEL_ID` and `SMOLAGENTS_API_KEY` are
required; `SMOLAGENTS_API_BASE` (any OpenAI-compatible endpoint, default `https://api.deepseek.com/v1`),
`SMOLAGENTS_TEMPERATURE`, `SMOLAGENTS_MAX_STEPS` (20), `SMOLAGENTS_PLANNING_INTERVAL` (10) and
`SMOLAGENTS_RUNS_DIR` (`runs`) are optional. Requests, tool results and cluster facts go to that endpoint. A
variable exported in the shell (for example in `~/.bashrc`) wins over `.env`; the CLI names each one that does.

The tools are written for a Slurm login node, where they call `sbatch`, `squeue`, `sacct` and `sinfo`; they ran on
one (see below). Without Slurm, every file is still generated and checked; `slurm_config`
answers `available: False` with a reason, pre-flight warns that partition and time limit were not checked, and
after the person's `y` `slurm_submit` returns `failed: sbatch was not found`. The MACE option table comes from the
MACE in the job's conda environment if visible, else from a MACE importable locally, else from the release tag
0.3.17 fetched once from GitHub into `.cache/mace/`. The i-PI checks need the `ipi` package of
`requirements.txt`. On the first `retriever` call the i-PI documentation pages and the embedding model are
downloaded at pinned revisions and the index is built; it is kept in `.cache/retriever/`. The build took 38.6 s
once, with both already downloaded, on a machine that was busier than usual.

Each session writes to `runs/<session>/`: one directory per job (a bundle) and `trace.jsonl`. The listing and
trace lines below were produced without a model, by calling the i-PI tools from a short script:
`ipi_InputFile_generator` (template `03-NVT-SVR`, 350 K, a water molecule), `ipi_runase_generator`,
`ipi_submit_generator`, `preflight`, then `slurm_submit`, which declined because standard input was not a terminal.
The trace lines shown are the last three of six; the first three record the other written files.

```text
runs/20261124-210358-3e97/01-ipi/input.xml
runs/20261124-210358-3e97/01-ipi/job.sh
runs/20261124-210358-3e97/01-ipi/run_ase.py
runs/20261124-210358-3e97/01-ipi/water.xyz
runs/20261124-210358-3e97/trace.jsonl
```

```json
{"ts": "2026-11-24T21:03:59+00:00", "session": "20261124-210358-3e97", "event": "artifact_written", "kind": "ipi_submit", "path": "runs/20261124-210358-3e97/01-ipi/job.sh", "sha256": "e3757711cabb6e67d8a807f6634222f05467b08527efcaf5f3cf11e0cd52884e", "bundle": "runs/20261124-210358-3e97/01-ipi", "created_at": "2026-11-24T21:03:59+00:00"}
{"ts": "2026-11-24T21:03:59+00:00", "session": "20261124-210358-3e97", "event": "approval", "script_path": "runs/20261124-210358-3e97/01-ipi/job.sh", "approved": false, "feedback": "no interactive terminal: nobody could confirm"}
{"ts": "2026-11-24T21:03:59+00:00", "session": "20261124-210358-3e97", "event": "slurm_submit", "script_path": "runs/20261124-210358-3e97/01-ipi/job.sh", "status": "declined", "feedback": "no interactive terminal: nobody could confirm"}
```

The job script removes a stale socket file, waits up to 300 s for i-PI's socket (`/tmp/ipi_<session>-<bundle>`),
then starts the client, and stops on the first failing command. A session run through `main.py` (in the tests,
with the scripted model) also writes `session_start`, `turn_start`, `turn_end`, `delegation_start`,
`delegation_end` (with the sub-agent's token counts), one `step` line per agent step, `job_submitted` and
`job_status`. The trace does not record the model's input messages, and the settings line leaves out the API key
(`tests/test_session.py` checks both). Generated code that printed the key would still put it into a `step`
line: see the limits below.

## Tests

```bash
pip install -r requirements-dev.txt     # no retrieval packages needed
python -m pytest                        # pyproject.toml already adds -q; a second -q hides the summary line
python -m ruff check .
```

Last run (the cluster's login node, CentOS 7, Python 3.11.13, all of `requirements.txt` installed): 398 passed,
1 expected failure, in about 70 s; ruff: all checks passed. The test that builds a real retrieval index needs the
packages of `requirements.txt` and the downloaded model and corpus, and is skipped without them (it was skipped on
Windows, 363 passed then). The `bash -n` tests use `bash` on POSIX and the bash shipped with Git on Windows, and
skip when neither exists.

| Kind | Tests | What they cover |
|---|---|---|
| Deterministic tools | 293, in 16 files | Specs, session ledger, structure writer, pre-flight, submit and status with a fake `subprocess.run`, the MACE option table on two trimmed MACE versions, i-PI checks with the real i-PI 3.3.1 loader, `bash -n` on generated job scripts, retriever parts, settings, prompts naming only registered tools |
| Scripted-model scenarios | 32, `tests/test_agents.py` | `tests/fakemodel.py` drives the agents `build_agents` builds; the person is scripted through `input()`; `sbatch` is a fake. The v1 scenarios (cancel, submit without review, shell access, sub-agent failures, Ctrl+C) and the delegation dict |
| Gate probes | 74, `tests/test_gate_probes.py` | Generated code run in each built agent's interpreter, trying to get a shell, a file, a module, the approver, or `sbatch` on a file no tool wrote; 73 are blocked, 1 is the known gap kept as an expected failure |

## First run on a real cluster (2026-11-26)

One cluster: a Slurm 19.05.2 login node on CentOS 7 (glibc 2.17), a CPU partition and a partition of A100 MIG
slices, MACE 0.3.10 to 0.3.13 in three conda environments, i-PI 3.1.5 in one of them.

| What ran | Outcome |
|---|---|
| Without a model: cluster probe, `MACE_script`, `preflight`, `slurm_submit` (answered `y` at a terminal), `job_status` | MACE training jobs completed on CPU (one node pinned) and on a GPU; `job_status` read `sacct` |
| Without a model: two i-PI runs (`03-NVT-SVR`, 192-atom water box, MACE-MP client on CPU) on one node at once | Both completed, each on its own socket |
| Live model, request 1: "train a quick test MACE model on my bulk ice data ... on CPU on the compute-0-0 node" | Plan approved, script generated on the first `MACE_script` call, submitted after `y`, job completed, followed with `job_status` |
| Live model, request 2: "run a short i-PI NVT simulation ... using the model we just trained ... on a GPU" | Two delegations (the MACE agent located the model, the i-PI agent built the run), submitted after `y`, 200 steps completed |
| Both live submissions: the person answered the submit question about 50 s after it was asked, inside a sub-agent's code action | The turn went on; no time-out (v1 failed at 30 s) |
| Live model, after the changes below, one request: "train a small MACE model ... then use that model to run a 100-step i-PI NVT simulation ... tell me whether the simulation ran all its steps and what temperature it ended at" | The MACE agent asked which of the three MACE environments to use (answered after about 400 s, no time-out); `ipi_submit_generator` refused an environment without `i-pi` and the agent took the one with it; the person declined a client whose model path was wrong, with the right path as feedback, and the agent regenerated it; both jobs completed; read with `job_output`, the agent reported step 100 of 100 and 370.97 K, as in `simulation.out` |
| Live model, a person who has no structure file: "give me an input file for ipi that using any random mace model to run a nve simulation, submit on any lightgpu" | Before `write_structure`: both agents built 64-atom silicon in memory and could not save it, and the i-PI agent's probes of guessed paths were all accepted, a made-up one too; no job. After: the agent wrote `si64.xyz` itself, chose MACE-MP-0 small (downloaded by the job) and the one environment with i-PI, and the job completed on the GPU partition (5000 steps NVE); read with `job_output`, its report of the conserved energy (-341.4815 eV throughout) matches `simulation.out` |

What the run showed, and what changed:

| Seen | Cause | Now |
|---|---|---|
| `faiss-cpu==1.16.0` does not install | No wheel for glibc 2.17 | Pinned to 1.9.0.post1, the newest that has one |
| First model call: 401 | An old `SMOLAGENTS_API_KEY` in `~/.bashrc` beat the key in `.env`, next to the API base from `.env` | The CLI names every setting the environment overrides (`config.shadowed_settings`) |
| The agent could not tell which partition a node belongs to | `sinfo -N` prints a node once per partition and the probe had no partition column (38 rows, 16 nodes) | One row per node with its partitions; modules from `module -t avail` |
| `mace_cli_schema` showed the MACE 0.3.17 options, labelled "environment not visible" | Three visible environments have MACE, so none was chosen and the release table was used | It asks for `conda_env` and lists the environments with their MACE versions |
| The MACE agent could not check the label keys of the training file | No tool reads it; MACE trains on with weight zero when a key is missing | `MACE_script` lists the first frame's keys in its notes and refuses a given key that is not there |
| The i-PI job ran in an environment without `i-pi` | `sbatch` exported the submitting shell's `PATH`, which had another environment's `i-pi` | `--export=NONE` in every job script; `ipi_submit_generator` checks the environment for `i-pi` and MACE |
| Both agents ended with "please check the output files yourself" | No tool read `log.i-pi`, `simulation.out` or MACE's log; `show_content` returned a path it did not know as if it were the text | `job_output` reads the files in the bundle of a job of this session; `show_content` refuses other paths |
| A wasted step on `modifications` given as a list; a model path relative to the repository; after the change, one made absolute from the home directory (the person caught it) | The job runs in its bundle directory; the paths the tools return are relative to the repository | Lists accepted; a model file must be an absolute path whose directory exists, and the message names the directory the tools' paths are relative to |
| 4 of 35 actions rejected: written as `<invoke name="python_interpreter">` or `<invoke name="final_answer">` | Claude's own tool-call markup instead of a `<code>` block | `agents.CodeActionModel` turns both into the `<code>` block; all 4 then parse |
| Each plan printed twice | smolagents logs plans at INFO, and the plan review printed it again | Printed by the review only below INFO |
| CPU jobs got one core | `--ntasks=1` and no CPU count | `cpus_per_task` in both generators |
| A person without a structure file got no job; `/nonexistent/definitely/fake.xyz` was accepted as a structure | Generated code cannot write files and no tool wrote a structure; an absolute structure path was never checked | `write_structure`; on the cluster a structure path must exist |
| `log.i-pi` read during a run was empty | Python buffers output redirected to a file until it exits | `PYTHONUNBUFFERED=1` in the i-PI job script: the log showed MD step 80 of a running job |

Not changed: one MIG slice of the GPU node (`SLURM_JOB_GPUS=12`) failed CUDA initialisation with "out of memory" in
two short test jobs while `nvidia-smi` showed it idle; jobs on the other slices ran. That is for the cluster's
administrators.

## Not done / limits

- Two live-model sessions: three requests, one model. Whether a model follows the prompts and finishes a task in
  general is not measured. There are no success rates or costs. In the last session 4 of 32 actions still failed
  and were retried by the agent: the intended "which conda environment?" error, an environment without `i-pi`
  refused by the tool, a final answer written as plain text and a stray "Thought:" line read as code.
- One cluster only. The tests still use fake Slurm commands; `sinfo`, `squeue` and `sacct` parsing was also checked
  on that cluster's output. Generated scripts ran there for MACE (CPU and GPU) and for i-PI with templates
  `03-NVT-SVR` and `00-NVE` (CPU and GPU clients); the other eight templates are only syntax-checked and loaded by i-PI.
- smolagents' local Python interpreter is not a sandbox. Code written by the manager can reach a sub-agent object:
  it can widen the sub-agent's import list (kept as the expected failure
  `tests/test_gate_probes.py::test_the_manager_cannot_widen_a_sub_agents_imports`), read the model's API key
  through `mace_agent.model.client.api_key` and call a sub-agent's tools directly. A direct call to `slurm_submit`
  that way still asked the person. The last two were seen in a one-off probe with a dummy key; no test covers them.
- The 512-token chunk limit is checked by a test that is skipped without the retrieval packages; it passed once in
  an environment that had them. The retriever was checked with a keyword heuristic on eight queries only: the top
  passage holds every expected keyword for 6 of 8 (v1: 7 of 8). No labelled relevance set exists.
- The retriever's corpus is a dataset on a personal Hugging Face account (`DATASET` in `tools/RetrieverTool.py`),
  pinned to a revision.
- Timings in "What changed since v1" come from one Windows 11 machine. The test suite ran on Windows and on the
  cluster's Linux login node, with Python 3.11 only; the CI workflow (Ubuntu) is configured but no run of it is
  recorded.
- Open v1 defects (string epochs refused, template comments dropped, two parameter names that differ between the
  Slurm generators) are listed in [docs/V1_DEFECTS.md](docs/V1_DEFECTS.md).
- No licence file yet.

Design reasons and rejected alternatives: [docs/DESIGN.md](docs/DESIGN.md).

## Credits

- [smolagents](https://github.com/huggingface/smolagents) (Hugging Face): agents, code execution, managed agents.
- [MLIPs-with-iPI](https://github.com/venkatkapil24/MLIPs-with-iPI): the ten i-PI input templates, vendored at
  commit `55d6ef7` with checksums; licence and source in `tools/ipi_templates/LICENSE` and `SOURCE.json`.
- [MACE](https://github.com/ACEsuit/mace): `tests/data/` holds options trimmed from `mace/tools/arg_parser.py`
  of v0.3.13 and v0.3.17, with MACE's MIT header.
- [i-PI](https://github.com/i-pi/i-pi): its loader checks generated inputs and its input classes answer
  `ipi_tag_help`.
- [gte-small](https://huggingface.co/thenlper/gte-small) embeds the i-PI documentation pages for `retriever`.
