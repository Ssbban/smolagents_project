# v1 defect register

One row per defect found in v1 (commit `3fc8270`, tag `baseline-v1`). Locations are `file:line` at that commit
and can be checked with `git show 3fc8270:<file>`.

How a defect was observed:

- scripted: a scripted model drove the real v1 `main.py` and agents; `builtins.input` gave the person's answers;
  `sbatch` was replaced by a recorder. Ctrl+C was simulated with `_thread.interrupt_main()`.
- tool case: the v1 tool was called directly with the listed input.
- timing: measured on one Windows machine, once.
- fresh install: `pip install -r requirements.txt` into an empty environment.
- probe: generated code run in a built agent's interpreter.
- reading: found in the code; no run needed to show it.

Status is "fixed" with the place in the code (file and function) that gives the new behaviour, or "open" with the
reason.


## Summary

The behaviours measured on v1, the v1 code that causes each, and where the new behaviour is in the code.

| v1 behaviour (measured) | v1 code at 3fc8270 | Now | Where fixed |
|---|---|---|---|
| Person answers after 35 s: time-out errors after the sub-agent succeeded | `main.py:71-115`, no `executor_kwargs` (default 30 s) | No executor time-out; time-outs inside blocking tools | `agents.py` `EXECUTOR`, given to all three agents in `build_agents` |
| Cancel at plan review: a sub-agent still ran and submitted | `tools/customize_plan.py:75` `agent.interrupt()` | Only the planning call runs; session goes on | `tools/plan_review.py` `interrupt_after_plan` raises `UserCancelled`; `main.py` `run_turn` ends the turn |
| Ctrl+C at `User>`, in `user_input` or in a model call ends the CLI | `main.py:121`, `main.py:130` | Ends the turn; at the prompt ends cleanly | `main.py` `run_turn` (`except KeyboardInterrupt`) and `main` (the `User>` loop) |
| Generated code ran shell commands in all three agents | `main.py:77-82, 92-97, 109-113` | `import os`, `open`, `eval` ... blocked | `agents.py` `IMPORTS`: only `json` beyond smolagents' base modules |
| A sub-agent submitted without any question | `tools/slurm_submit.py:13` | The tool asks; `n` means no `sbatch` | `tools/slurm_submit.py` `_submit`, `tools/approval.py` `ask` |
| `slurm_submit` accepted any file | `tools/slurm_submit.py:13` | Refused: 4 spellings of a hand-written file | `tools/slurm_submit.py` `_submit` (`Session.find_artifact`, `SCRIPT_KINDS`) |
| Sub-agent 503 or step limit: "Task completed successfully!" | `main.py:127-129` | Dict with `status` `error` / `incomplete` | `agents.py` `ScienceAgent.__call__` |
| Manager provider error: printed, loop went on (no defect) | `main.py:130-134` | Kept: turn `failed: <error>`, next turn answered | `main.py` `run_turn` (`except Exception`) |
| Slurm tools on a machine without Slurm: raw `[WinError 2]` | `tools/get_server_info.py:73` | `available: False` with a reason; `failed`; `UNKNOWN` | `tools/slurm_config.py` `run_command`; `tools/slurm_submit.py` `_submit`; `tools/job_status.py` `job_status` |
| MACE cases: junk in `#SBATCH`, `r_max "5.0"` refused, `~` never expanded | `tools/specs.py:154-157`, `tools/mace_cli_schema.py:194-197`, `tools/mace_script.py:25-26` | Junk refused, `"5.0"` accepted, `"$HOME/..."` | `tools/specs.py` `ClusterSpec.problems`; `tools/mace_cli_schema.py` `convert`; `tools/mace_script.py` `cli_argument` |
| Two files in one second: second overwrote the first | `tools/mace_script.py:220-221` | One bundle per job; never overwritten | `tools/session.py` `Session.new_bundle`, `Session.write_artifact` |
| i-PI: socket pointed at another host, `total_steps "-5 abc"`, two `<prng>` accepted | `tools/ipi_InputFile_generator.py:95-136` | Refused or reported by i-PI's loader | `tools/ipi_InputFile_generator.py` `_apply`; `tools/ipi_check.py` `check_input` |
| A temperature without units is read in atomic units | no check | Reported: "300" is 9.47e7 K | `tools/ipi_check.py` `_range_problems` |
| i-PI job always requested a GPU | `tools/ipi_submit_generator.py:23-24` | GPU only for a `cuda` client | `tools/ipi_submit_generator.py` `ipi_submit_generator` |
| `github_fetch`: HTML for blob URLs, no time-out, any URL | `tools/github_fetch.py:15` | Raw URL, GitHub only, 30 s, 1 MB | `tools/github_fetch.py` `raw_url`, `github_fetch` |
| Retriever: chunks up to 600 tokens (model reads 512), no sources | `tools/RetrieverTool.py:36-43, 83-89` | At most 512 tokens; a source line per passage | `tools/retriever.py` `build_chunks` (`CHUNK_TOKENS`), `format_results` |
| About 52 s to the first prompt (index built at import) | `main.py:6`, `tools/RetrieverTool.py:13-62` | 1.72 s median of 5, index built on first query | `tools/retriever.py` `get_vectordb`, called from `RetrieverTool.forward` |

Both start-up times come from one Windows machine: v1 about 52 s (with network access); now the median of 5 runs
with placeholder settings and no model call.

## Agent layer

| Id | What v1 did | Where (3fc8270) | Observed | Status |
|---|---|---|---|---|
| AG-1 | The 30 s executor time-out wrapped a whole delegation, the person's answers included: after a 35 s answer both agents reported a time-out although the sub-agent had succeeded | `main.py:71-115` (no `executor_kwargs`) | scripted | fixed: `agents.py` `EXECUTOR` (`timeout_seconds: None`) for all three agents; time limits sit in the blocking tools and the model client. In the cluster run the person answered after about 50 s and about 400 s without a time-out (`docs/CLUSTER_RUN.md`) |
| AG-2 | Cancel at the plan review let one more action step run: a whole sub-agent, which generated and submitted a job | `tools/customize_plan.py:75` | scripted | fixed: `tools/plan_review.py` `interrupt_after_plan` raises `UserCancelled` (a `BaseException`) on Cancel, and `get_user_choice` treats end of input as Cancel; `main.py` `run_turn` ends the turn |
| AG-3 | Ctrl+C at the prompt, inside `user_input` or during a model call ended the CLI with a traceback; end of input crashed it | `main.py:121` (outside `try`), `main.py:130` (`except Exception`) | scripted | fixed: `main.py` `main` ends the session cleanly on Ctrl+C or end of input at `User>` and skips empty lines; `run_turn` ends only the current turn on Ctrl+C ("interrupted (Ctrl+C)") |
| AG-4 | `os`, `subprocess` and `tempfile` were authorised in all three agents; generated code ran shell commands and wrote a file | `main.py:77-82, 92-97, 109-113` | scripted | fixed: `agents.py` `IMPORTS = ["json"]` is the only import added to smolagents' base modules, in all three agents (`build_agents`) |
| AG-5 | A sub-agent's provider error (503) and an exhausted step budget both ended in "Task completed successfully!" | `main.py:127-129` | scripted | fixed: `agents.py` `ScienceAgent.__call__` returns a dict whose `status` comes from the run state (`incomplete` at the step limit) or the exception (`error`); the manager's code branches on it |
| AG-6 | Prompts named tools that are not registered (`mace_script` for `MACE_script`, `UserInputTool` for `user_input`, `DuckDuckGoSearchTool`, `RetrieverTool`) | `prompts/mace_instruction.txt:7, 9, 11`; `prompts/ipi_instruction.txt:11-13` | reading | fixed: the prompts in `prompts/` name only registered tools (`MACE_script`, `user_input`, `web_search`, `retriever`) |
| AG-7 | The manager kept its memory across requests (`reset=False`); later requests were planned as new attempts at the old task | `main.py:127` | reading | fixed: one run per request with the session and history as data: `main.py` `run_turn` calls `manager.run` without `reset=False`, and `main` passes the last three turns as `history` |
| AG-8 | No log or trace: console output only | `main.py:120-134` | reading | fixed: a JSONL trace per session (`tools/trace.py` `Trace`, `step_callback`; in the session directory of `tools/session.py` `Session`); model input messages are not recorded and the settings are traced without the key (`config.py` `Settings.public`) |
| AG-9 | Importing `main` built the retrieval index: about 52 s and network access before the first prompt and before the key check | `main.py:6`, `tools/RetrieverTool.py:13-62` | timing | fixed: the index is built on the first query (`tools/retriever.py` `get_vectordb`, called from `RetrieverTool.forward`), and the retrieval packages are imported only then; 1.72 s to the prompt (median of 5, once) |
| AG-10 | `.env.example` existed but nothing read `.env` | `main.py:19-26` | reading | fixed: the file was removed; settings come only from environment variables (`config.py` `Settings.from_env`), documented in the README |
| AG-11 | Generated code could read the API key (`os.environ`, since `os` was authorised) | `main.py:22`, `main.py:109-113` | reading | open: `os` is blocked now, but code in the manager can read `mace_agent.model.client.api_key` (seen once with a dummy key). See AG-12 |
| AG-12 | Code in the manager can reach a sub-agent object and widen its import list (found while probing the new gate; in v1 `os` was open anyway) | `main.py:114` (the real agent objects are the managed agents, as now) | probe | open, decided: a known limit; reasons in `docs/DESIGN.md` |

## Submission

| Id | What v1 did | Where (3fc8270) | Observed | Status |
|---|---|---|---|---|
| SUB-1 | `slurm_submit` ran `sbatch` on any path: no check that a tool wrote it, that it was shown, or that the person agreed; no time-out; job id not parsed | `tools/slurm_submit.py:13` | scripted | fixed: `tools/slurm_submit.py` `_submit` accepts only a job script generated in this session (`Session.find_artifact`), unchanged (`Session.is_unchanged`) and not submitted before (`Session.job_for`), shows it and asks the person, then runs `sbatch --parsable` with a 60 s time-out and parses the job id |
| SUB-2 | Review and "type yes" existed only as prompt text | `prompts/mace_instruction.txt:38-45` | reading | fixed: the tool asks (`tools/approval.py` `ask`, called by `tools/slurm_submit.py` `_submit`); no prompt asks the person to type yes |
| SUB-3 | `ipi_agent` had no submit tool; its "Submit" step only generated a script | `main.py:58-69` | reading | fixed: `slurm_submit`, `preflight`, `job_status` on `ipi_agent` (`agents.py` `build_agents`) |
| SUB-4 | Without Slurm, `slurm_config` and `slurm_submit` raised a bare `FileNotFoundError: [WinError 2]` naming no program | `tools/get_server_info.py:73`, `tools/slurm_submit.py:13` | scripted | fixed: `tools/slurm_config.py` `run_command` (`available: False` with a reason), `tools/slurm_submit.py` `_submit` (`failed`, naming `sbatch`), `tools/job_status.py` `job_status` (`UNKNOWN`, naming `squeue`) |
| SUB-5 | Partition and time limit were checked against the cluster only by a prompt sentence | `prompts/mace_instruction.txt:15-17` | reading | fixed: `tools/preflight.py` `check_cluster` checks partition and time limit against the cluster probe; an unprobed cluster is a warning shown to the person by `tools/slurm_submit.py` `_submit` |
| SUB-6 | File names had second resolution: 9 MACE scripts gave 7 files, 4 i-PI job scripts 1, 5 clients 1 | `tools/mace_script.py:220-221`, `tools/ipi_runase_generator.py:89-90`, `tools/ipi_submit_generator.py:132-133`, `tools/ipi_InputFile_generator.py:252-253` | tool case | fixed: one bundle per job (`tools/session.py` `Session.new_bundle`); `Session.write_artifact` never overwrites; a second client or job script in a bundle gets a new name (`unused_name`) |
| SUB-7 | Scripts written on Windows had CRLF line endings | `tools/mace_script.py:229`, `tools/ipi_submit_generator.py:144` | tool case | fixed: `tools/session.py` `Session.write_artifact` writes LF line endings |
| SUB-8 | Validation stopped at the first problem | `tools/specs.py:146-197` | reading | fixed: `tools/specs.py` `check` and `raise_problems` collect every problem into one error; `tools/mace_script.py` `MACE_script` adds every option problem (`option_problems`) |
| SUB-9 | `memory`, `mail_user` and extra Slurm values were free text written into `#SBATCH` lines (`32G; rm -rf ~` accepted) | `tools/specs.py:154-157`, `tools/mace_script.py:117-119` | tool case | fixed: `tools/specs.py` `ClusterSpec.problems` (`_memory`, `_mail_user`, `_mail_type`, `_slurm_arg_problems`), used by both generators |
| SUB-10 | Only `HH:MM:SS` was accepted; Slurm's `D-HH:MM:SS`, the form `sinfo` prints, was refused | `tools/specs.py:32, 148` | tool case | fixed: `tools/specs.py` `parse_slurm_time` accepts every sbatch form |

## MACE

| Id | What v1 did | Where (3fc8270) | Observed | Status |
|---|---|---|---|---|
| MACE-1 | The option table was fetched from MACE's `main` branch on every call, not from the MACE the job runs | `tools/mace_cli_schema.py:11-13, 179-182`, `tools/mace_script.py:69` | tool case | fixed: `tools/mace_cli_schema.py` `load_mace_cli_schema`: the job's conda environment first, then a MACE importable here, then release tag 0.3.17 fetched once and cached (`_release_table`) |
| MACE-2 | The whole module was parsed, so four options of `mace_prepare_data` were accepted for `mace_run_train` | `tools/mace_cli_schema.py:129` | tool case | fixed: `tools/mace_cli_schema.py` `parse_mace_arg_parser` reads only `build_default_arg_parser` |
| MACE-3 | `mace_cli_schema()` returned about 39,000 characters into the context (39,414 for MACE 0.3.17's parser) | `tools/mace_cli_schema.py:215` | tool case | fixed: `tools/mace_cli_schema.py` `mace_cli_schema` returns option names only unless names or a search are given. On MACE 0.3.17's full `arg_parser.py` the summary is about 3,500 characters; v1's code gives 39,414 on the same file |
| MACE-4 | A numeric string argparse accepts (`r_max="5.0"`) was refused | `tools/mace_cli_schema.py:194-197` | tool case | fixed: `tools/mace_cli_schema.py` `convert` and `MACECliSchema.problem` judge values as argparse would |
| MACE-5 | `~/data/train.xyz` was single-quoted, so it never expanded and MACE would get a literal `~` | `tools/mace_script.py:25-26, 75` | tool case | fixed: `tools/mace_script.py` `cli_argument` writes `"$HOME/..."`; relative input paths are refused (`path_problem`) |
| MACE-6 | Any non-empty `E0s` was accepted (`average; rm -rf ~`) | `tools/specs.py:192` | tool case | fixed: `tools/mace_script.py` `e0s_problem` accepts only the forms MACE reads |
| MACE-7 | The job script had no `set -e`: a failed `conda activate` did not stop the job | `tools/mace_script.py:121-128` | reading | fixed: `set -euo pipefail` in `tools/mace_script.py` `render_mace_script` |
| MACE-8 | The conda path from the cluster probe was used only if the prompt was followed | `prompts/mace_instruction.txt:17` | reading | fixed: `tools/mace_cli_schema.py` `job_environment`, called by `MACE_script`, takes conda values left out from the cluster probe |
| MACE-9 | `max_num_epochs="100"` (a string) is refused although argparse would accept it | `tools/specs.py:193` | tool case | open: still refused (`tools/specs.py` `MACEJobSpec.problems`); the message says an integer is needed and the model can retry |

## i-PI

| Id | What v1 did | Where (3fc8270) | Observed | Status |
|---|---|---|---|---|
| IPI-1 | Templates were fetched from a third-party branch on every call | `tools/ipi_InputFile_generator.py:13-15, 27-30` | tool case | fixed: the templates are vendored in `tools/ipi_templates/` at one upstream commit, with sha256 checksums in `SOURCE.json`; `tools/ipi_InputFile_generator.py` `_select_template` reads them from there |
| IPI-2 | Any XPath was accepted: the socket could be pointed at `attacker.example.com` over inet | `tools/ipi_InputFile_generator.py:95-119` | tool case | fixed: `tools/ipi_InputFile_generator.py` `_apply` edits only listed paths; the socket is not among them (`_tool_owned`) and cannot be added (`_add_child_element`) |
| IPI-3 | Values were not checked: `total_steps` `-5 abc` was accepted | none | tool case | fixed: i-PI's own loader reports it (`tools/ipi_check.py` `check_input`) |
| IPI-4 | An append block added a second `<prng>`; i-PI reads only the last | `tools/ipi_InputFile_generator.py:122-136` | tool case | fixed: `tools/ipi_check.py` `_repeated` reports tags written twice |
| IPI-5 | A value without a `units` attribute is read in atomic units; nothing checked it | none | reading | fixed: `tools/ipi_check.py` `_range_problems` reports temperature, time step and pressure with the value i-PI read |
| IPI-6 | `sockets_prefix` was accepted although the ASE client always uses `/tmp/ipi_`; the client's socket name had to match by prompt | `tools/specs.py:223`, `prompts/ipi_instruction.txt:102` | reading | fixed: `sockets_prefix` refused (`tools/specs.py` `IPIInputSpec.problems`); the client takes its socket and structure from the input (`tools/ipi_runase_generator.py`, `tools/ipi_check.py` `links`), and pre-flight checks that the three files agree (`tools/ipi_submit_generator.py` `check_ipi_job`) |
| IPI-7 | The i-PI job always requested a GPU (default `gpus_per_node=1`) | `tools/ipi_submit_generator.py:23-24, 73` | tool case | fixed: `tools/ipi_submit_generator.py` `ipi_submit_generator`: `gpus_per_node` defaults to 1 for a cuda client and 0 for a cpu client |
| IPI-8 | The job script slept 30 s instead of waiting for the socket, sourced `~/i-pi/env.sh`, had no error handling and used the socket `driver` for every job | `tools/ipi_submit_generator.py:31-37, 77-78`; socket default `tools/specs.py:231` | reading | fixed: `tools/ipi_submit_generator.py` `render_ipi_submit_script` waits for the socket, runs under `set -euo pipefail` and sources nothing unless `ipi_env_path` is given; one socket per bundle (`tools/ipi_InputFile_generator.py` `socket_address`) |
| IPI-9 | Paths in the job were relative to the agent's working directory and the structure file was never staged | `tools/ipi_submit_generator.py:35, 47, 58` | reading | fixed: `ipi_InputFile_generator` copies the structure file into the bundle, `tools/slurm_submit.py` `_submit` runs `sbatch` in the bundle, and pre-flight needs the structure there (`tools/ipi_submit_generator.py` `check_ipi_job`) |
| IPI-10 | The number of clients was not tied to beads and replicas | `tools/specs.py:269-270` | reading | fixed: `tools/specs.py` `IPISubmitSpec.problems` (at most beads x replicas clients), checked again by `check_ipi_job` |
| IPI-11 | A calculator keyword `a-b` was accepted (a `TypeError` at run time) | `tools/specs.py:246` | tool case | fixed: `tools/specs.py` `IPIASEClientSpec.problems` requires Python names as keys |
| IPI-12 | The template catalogue was kept twice, in code and in the prompt | `tools/specs.py:10-21`, `prompts/ipi_instruction.txt:26-65` | reading | fixed: one `tools/ipi_templates/catalog.json`, read by `tools/ipi_InputFile_generator.py` and listed by `ipi_template_schema` |
| IPI-13 | The prompt fetched four i-PI source files from GitHub `main` (about 88,000 characters; 87,886 in i-PI 3.3.1) to look up tags | `prompts/ipi_instruction.txt:67-72` | reading | fixed: `ipi_tag_help` (`tools/ipi_help.py`) reads the installed i-PI's input classes |
| IPI-14 | XML comments of the template are dropped when the input is rewritten | `tools/ipi_InputFile_generator.py:140, 159` | tool case | open: ElementTree still drops them; i-PI ignores comments |

## Retriever and other

| Id | What v1 did | Where (3fc8270) | Observed | Status |
|---|---|---|---|---|
| RET-1 | Chunks of up to 600 tokens for an embedding model that reads 512 (median 533) | `tools/RetrieverTool.py:36-43` | tool case | fixed: at most 512 tokens with special tokens (`tools/retriever.py` `build_chunks`, `CHUNK_TOKENS`) |
| RET-2 | Passages were returned without their source | `tools/RetrieverTool.py:83-89` | tool case | fixed: `tools/retriever.py` `format_results` puts a source line after each passage |
| RET-3 | The index was rebuilt at every start, with the embedding model in float16 on the CPU | `tools/RetrieverTool.py:13-62` | timing | fixed: kept on disk under `.cache/retriever/` and embedded in float32 (`tools/retriever.py` `build_vectordb`) |
| RET-4 | Corpus and model had no pinned revision | `tools/RetrieverTool.py:13, 37, 56` | reading | fixed: `tools/retriever.py` `DATASET_REVISION`, `MODEL_REVISION`; the cache key changes with them and with the chunking (`index_params`) |
| OPS-1 | `github_fetch` returned GitHub's HTML for blob URLs, had no time-out and fetched any URL | `tools/github_fetch.py:15` | tool case | fixed: `tools/github_fetch.py` `raw_url` (GitHub only, raw URL) and `github_fetch` (30 s, 1 MB) |
| OPS-2 | 11 of 12 requirements unpinned; `mace-torch` and `pandas` installed but never imported | `requirements.txt:1-12` | fresh install | fixed: pinned `requirements.txt` |
| OPS-3 | Prompts were read relative to the working directory | `main.py:35-45` | reading | fixed: read relative to `agents.py` (`PROMPTS`, `load_prompt`) |
| OPS-4 | The two Slurm generators name the same things differently (`conda_activation_path` / `conda_path`, `additional_args_slurm` / `additional_slurm_args`) | `tools/mace_script.py:143-145`, `tools/ipi_submit_generator.py:76, 81` | reading | open: v1 names kept so existing calls and notes stay valid |
