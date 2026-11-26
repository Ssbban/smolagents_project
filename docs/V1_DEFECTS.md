# v1 defect register

One row per defect found in v1 (commit `3fc8270`, tag `baseline-v1`). Locations are `file:line` at that commit
and can be checked with `git show 3fc8270:<file>`. Test ids are relative to `tests/`.

How a defect was observed:

- scripted: a scripted model drove the real v1 `main.py` and agents; `builtins.input` gave the person's answers;
  `sbatch` was replaced by a recorder. Ctrl+C was simulated with `_thread.interrupt_main()`.
- tool case: the v1 tool was called directly with the listed input.
- timing: measured on one Windows machine, once.
- fresh install: `pip install -r requirements.txt` into an empty environment.
- probe: generated code run in a built agent's interpreter (now part of `test_gate_probes.py`).
- reading: found in the code; no run needed to show it.

Status is "fixed" with the test that shows the new behaviour, or "open" with the reason.


## Summary

The behaviours measured on v1, the v1 code that causes each, and the test that shows the new behaviour.

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
| Slurm tools on a machine without Slurm: raw `[WinError 2]` | `tools/get_server_info.py:73` | `available: False` with a reason; `failed`; `UNKNOWN` | `test_slurm_config.py::test_slurm_config_without_slurm`, `test_slurm_submit.py::test_sbatch_missing_is_failed`, `test_job_status.py::test_without_slurm` |
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

## Agent layer

| Id | What v1 did | Where (3fc8270) | Observed | Status |
|---|---|---|---|---|
| AG-1 | The 30 s executor time-out wrapped a whole delegation, the person's answers included: after a 35 s answer both agents reported a time-out although the sub-agent had succeeded | `main.py:71-115` (no `executor_kwargs`) | scripted | fixed: `test_agents.py::test_a_no_agent_has_an_executor_time_out` (checks the setting; the 35 s wait was not re-run) |
| AG-2 | Cancel at the plan review let one more action step run: a whole sub-agent, which generated and submitted a job | `tools/customize_plan.py:75` | scripted | fixed: `test_agents.py::test_b_cancel_at_plan_review_runs_nothing_and_the_session_goes_on`, `test_end_of_input_at_the_plan_menu_cancels` |
| AG-3 | Ctrl+C at the prompt, inside `user_input` or during a model call ended the CLI with a traceback; end of input crashed it | `main.py:121` (outside `try`), `main.py:130` (`except Exception`) | scripted | fixed: `test_agents.py::test_c1_ctrl_c_at_the_prompt_ends_the_session_cleanly`, `test_c_ctrl_c_inside_a_turn_leaves_a_usable_session`, `test_c3_ctrl_c_during_a_model_call_ends_only_that_turn`, `test_empty_lines_are_ignored_and_end_of_input_ends_the_session` |
| AG-4 | `os`, `subprocess` and `tempfile` were authorised in all three agents; generated code ran shell commands and wrote a file | `main.py:77-82, 92-97, 109-113` | scripted | fixed: `test_agents.py::test_d_generated_code_cannot_reach_a_shell_or_files`, `test_gate_probes.py::test_no_shell_no_files_no_modules_in_any_agent` |
| AG-5 | A sub-agent's provider error (503) and an exhausted step budget both ended in "Task completed successfully!" | `main.py:127-129` | scripted | fixed: `test_agents.py::test_f2_provider_error_in_a_sub_agent_reaches_the_manager_as_status_error`, `test_f3_sub_agent_out_of_steps_is_incomplete`, `test_manager_code_gets_a_dict_and_branches_on_it` |
| AG-6 | Prompts named tools that are not registered (`mace_script` for `MACE_script`, `UserInputTool` for `user_input`, `DuckDuckGoSearchTool`, `RetrieverTool`) | `prompts/mace_instruction.txt:7, 9, 11`; `prompts/ipi_instruction.txt:11-13` | reading | fixed: `test_prompts.py::test_names_in_a_prompt_are_registered_on_its_agent` |
| AG-7 | The manager kept its memory across requests (`reset=False`); later requests were planned as new attempts at the old task | `main.py:127` | reading | fixed: one run per request with the session and history as data; `test_agents.py::test_c_ctrl_c_inside_a_turn_leaves_a_usable_session` (next turn sees the history). The fresh memory itself is by reading `main.py` `run_turn` (no `reset=False`); no test asserts it |
| AG-8 | No log or trace: console output only | `main.py:120-134` | reading | fixed: `test_session.py::test_trace_lines_are_json_and_never_contain_the_key` |
| AG-9 | Importing `main` built the retrieval index: about 52 s and network access before the first prompt and before the key check | `main.py:6`, `tools/RetrieverTool.py:13-62` | timing | fixed: `test_agents.py::test_building_the_agents_needs_no_retrieval_packages`; 1.72 s to the prompt (median of 5, once) |
| AG-10 | `.env.example` existed but nothing read `.env` | `main.py:19-26` | reading | fixed: `test_config.py::test_dotenv_file_is_read_and_environment_wins` |
| AG-11 | Generated code could read the API key (`os.environ`, since `os` was authorised) | `main.py:22`, `main.py:109-113` | reading | open: `os` is blocked now, but code in the manager can read `mace_agent.model.client.api_key` (seen once with a dummy key; no test). See AG-12 |
| AG-12 | Code in the manager can reach a sub-agent object and widen its import list (found while testing the new gate; in v1 `os` was open anyway) | `main.py:114` (the real agent objects are the managed agents, as now) | probe | open, decided: kept as the strict expected failure `test_gate_probes.py::test_the_manager_cannot_widen_a_sub_agents_imports`; reasons in `docs/DESIGN.md` |

## Submission

| Id | What v1 did | Where (3fc8270) | Observed | Status |
|---|---|---|---|---|
| SUB-1 | `slurm_submit` ran `sbatch` on any path: no check that a tool wrote it, that it was shown, or that the person agreed; no time-out; job id not parsed | `tools/slurm_submit.py:13` | scripted | fixed: `test_agents.py::test_e_submit_without_review_asks_the_person_and_a_no_stops_it`, `test_gate_probes.py::test_no_tool_runs_sbatch_on_a_file_it_did_not_generate`, `test_slurm_submit.py::test_changed_after_generation_is_refused`, `test_gate_probes.py::test_a_submitted_script_is_not_submitted_twice` |
| SUB-2 | Review and "type yes" existed only as prompt text | `prompts/mace_instruction.txt:38-45` | reading | fixed: the tool asks (`test_slurm_submit.py::test_ask_answers`, `test_declined_never_calls_sbatch`); `test_prompts.py::test_no_prompt_asks_the_person_to_type_yes` |
| SUB-3 | `ipi_agent` had no submit tool; its "Submit" step only generated a script | `main.py:58-69` | reading | fixed: `slurm_submit`, `preflight`, `job_status` on `ipi_agent`; exercised by `test_gate_probes.py::test_no_tool_runs_sbatch_on_a_file_it_did_not_generate[ipi_agent]` |
| SUB-4 | Without Slurm, `slurm_config` and `slurm_submit` raised a bare `FileNotFoundError: [WinError 2]` naming no program | `tools/get_server_info.py:73`, `tools/slurm_submit.py:13` | scripted | fixed: `test_slurm_config.py::test_slurm_config_without_slurm`, `test_slurm_submit.py::test_sbatch_missing_is_failed`, `test_job_status.py::test_without_slurm` |
| SUB-5 | Partition and time limit were checked against the cluster only by a prompt sentence | `prompts/mace_instruction.txt:15-17` | reading | fixed: `test_slurm_submit.py::test_time_over_partition_limit_is_an_error`, `test_unprobed_cluster_is_a_warning_shown_to_the_person` |
| SUB-6 | File names had second resolution: 9 MACE scripts gave 7 files, 4 i-PI job scripts 1, 5 clients 1 | `tools/mace_script.py:220-221`, `tools/ipi_runase_generator.py:89-90`, `tools/ipi_submit_generator.py:132-133`, `tools/ipi_InputFile_generator.py:252-253` | tool case | fixed: `test_session.py::test_write_artifact_refuses_overwrite`, `test_mace_script.py::test_two_calls_give_two_bundles`, `test_ipi_generators.py::test_second_client_and_second_job_get_new_names` |
| SUB-7 | Scripts written on Windows had CRLF line endings | `tools/mace_script.py:229`, `tools/ipi_submit_generator.py:144` | tool case | fixed: `test_session.py::test_write_artifact_lf_bytes_and_registration` |
| SUB-8 | Validation stopped at the first problem | `tools/specs.py:146-197` | reading | fixed: `test_specs.py::test_one_error_lists_every_problem`, `test_mace_script.py::test_all_problems_come_together` |
| SUB-9 | `memory`, `mail_user` and extra Slurm values were free text written into `#SBATCH` lines (`32G; rm -rf ~` accepted) | `tools/specs.py:154-157`, `tools/mace_script.py:117-119` | tool case | fixed: `test_mace_script.py::test_rejected_cases_write_nothing[M12]`, `[M13]`, `[M14]`; i-PI: `test_ipi_generators.py::test_submit_reports_missing_client_and_spec_problems_together` |
| SUB-10 | Only `HH:MM:SS` was accepted; Slurm's `D-HH:MM:SS`, the form `sinfo` prints, was refused | `tools/specs.py:32, 148` | tool case | fixed: `test_mace_script.py::test_accepted_cases[M22b]`, `test_specs.py::test_slurm_time_forms` |

## MACE

| Id | What v1 did | Where (3fc8270) | Observed | Status |
|---|---|---|---|---|
| MACE-1 | The option table was fetched from MACE's `main` branch on every call, not from the MACE the job runs | `tools/mace_cli_schema.py:11-13, 179-182`, `tools/mace_script.py:69` | tool case | fixed: `test_mace_cli_schema.py::test_the_jobs_conda_environment_comes_first`, `test_then_the_pinned_release_tag_fetched_once_and_cached`, `test_two_versions_give_two_tables` |
| MACE-2 | The whole module was parsed, so four options of `mace_prepare_data` were accepted for `mace_run_train` | `tools/mace_cli_schema.py:129` | tool case | fixed: `test_mace_cli_schema.py::test_only_the_training_parser_is_read` |
| MACE-3 | `mace_cli_schema()` returned about 39,000 characters into the context (39,414 for MACE 0.3.17's parser) | `tools/mace_cli_schema.py:215` | tool case | fixed: `test_mace_cli_schema.py::test_summary_lists_names_only`. On MACE 0.3.17's full `arg_parser.py` the summary is about 3,500 characters; v1's code gives 39,414 on the same file |
| MACE-4 | A numeric string argparse accepts (`r_max="5.0"`) was refused | `tools/mace_cli_schema.py:194-197` | tool case | fixed: `test_mace_script.py::test_accepted_cases[M04]`, `test_mace_cli_schema.py::test_values_are_judged_as_argparse_would` |
| MACE-5 | `~/data/train.xyz` was single-quoted, so it never expanded and MACE would get a literal `~` | `tools/mace_script.py:25-26, 75` | tool case | fixed: `test_mace_script.py::test_accepted_cases[M07]`; relative input paths refused: `test_rejected_cases_write_nothing[R6]` |
| MACE-6 | Any non-empty `E0s` was accepted (`average; rm -rf ~`) | `tools/specs.py:192` | tool case | fixed: `test_mace_script.py::test_rejected_cases_write_nothing[M11]` |
| MACE-7 | The job script had no `set -e`: a failed `conda activate` did not stop the job | `tools/mace_script.py:121-128` | reading | fixed: `test_mace_script.py::test_n3_the_script_stops_at_the_first_failing_command` |
| MACE-8 | The conda path from the cluster probe was used only if the prompt was followed | `prompts/mace_instruction.txt:17` | reading | fixed: `test_mace_script.py::test_conda_facts_left_out_are_taken_from_the_probe` |
| MACE-9 | `max_num_epochs="100"` (a string) is refused although argparse would accept it | `tools/specs.py:193` | tool case | open: still refused (`test_mace_script.py::test_rejected_cases_write_nothing[M02]`); the message says an integer is needed and the model can retry |

## i-PI

| Id | What v1 did | Where (3fc8270) | Observed | Status |
|---|---|---|---|---|
| IPI-1 | Templates were fetched from a third-party branch on every call | `tools/ipi_InputFile_generator.py:13-15, 27-30` | tool case | fixed: `test_ipi_check.py::test_every_vendored_file_matches_its_recorded_checksum`, `test_every_template_loads_unmodified` |
| IPI-2 | Any XPath was accepted: the socket could be pointed at `attacker.example.com` over inet | `tools/ipi_InputFile_generator.py:95-119` | tool case | fixed: `test_ipi_generators.py::test_baseline_input_cases[I04_unrestricted_ffsocket_to_inet]`, `test_redirecting_the_socket_lists_both_refused_paths` |
| IPI-3 | Values were not checked: `total_steps` `-5 abc` was accepted | none | tool case | fixed: `test_ipi_generators.py::test_baseline_input_cases[I03_malformed_value_steps]` (i-PI's own message) |
| IPI-4 | An append block added a second `<prng>`; i-PI reads only the last | `tools/ipi_InputFile_generator.py:122-136` | tool case | fixed: `test_ipi_generators.py::test_baseline_input_cases[I07_append_block]` |
| IPI-5 | A value without a `units` attribute is read in atomic units; nothing checked it | none | reading | fixed: `test_ipi_check.py::test_the_unit_trap_is_reported_with_the_value_ipi_understood`, `test_a_pressure_without_units_is_reported` |
| IPI-6 | `sockets_prefix` was accepted although the ASE client always uses `/tmp/ipi_`; the client's socket name had to match by prompt | `tools/specs.py:223`, `prompts/ipi_instruction.txt:102` | reading | fixed: `test_ipi_generators.py::test_sockets_prefix_is_refused`, `test_socket_and_structure_agree_across_the_three_files` |
| IPI-7 | The i-PI job always requested a GPU (default `gpus_per_node=1`) | `tools/ipi_submit_generator.py:23-24, 73` | tool case | fixed: `test_ipi_generators.py::test_gpu_request_follows_the_client_device` |
| IPI-8 | The job script slept 30 s instead of waiting for the socket, sourced `~/i-pi/env.sh`, had no error handling and used the socket `driver` for every job | `tools/ipi_submit_generator.py:31-37, 77-78`; socket default `tools/specs.py:231` | reading | fixed: `test_ipi_generators.py::test_job_script_fails_on_errors_waits_for_the_socket_and_needs_no_source_checkout`, `test_two_inputs_get_two_sockets` |
| IPI-9 | Paths in the job were relative to the agent's working directory and the structure file was never staged | `tools/ipi_submit_generator.py:35, 47, 58` | reading | fixed: `test_ipi_generators.py::test_only_structure_files_are_staged`, `test_preflight_needs_the_structure_in_the_bundle` |
| IPI-10 | The number of clients was not tied to beads and replicas | `tools/specs.py:269-270` | reading | fixed: `test_ipi_generators.py::test_client_count_is_bounded_by_beads_times_replicas` |
| IPI-11 | A calculator keyword `a-b` was accepted (a `TypeError` at run time) | `tools/specs.py:246` | tool case | fixed: `test_specs.py::test_client_keyword_arguments_must_be_python_names` |
| IPI-12 | The template catalogue was kept twice, in code and in the prompt | `tools/specs.py:10-21`, `prompts/ipi_instruction.txt:26-65` | reading | fixed: one `tools/ipi_templates/catalog.json`; `test_ipi_generators.py::test_schema_lists_the_catalogue_and_only_editable_paths` |
| IPI-13 | The prompt fetched four i-PI source files from GitHub `main` (about 88,000 characters; 87,886 in i-PI 3.3.1) to look up tags | `prompts/ipi_instruction.txt:67-72` | reading | fixed: `ipi_tag_help`; `test_ipi_help.py::test_thermostat_lists_its_modes_and_children` |
| IPI-14 | XML comments of the template are dropped when the input is rewritten | `tools/ipi_InputFile_generator.py:140, 159` | tool case | open: ElementTree still drops them; i-PI ignores comments |

## Retriever and other

| Id | What v1 did | Where (3fc8270) | Observed | Status |
|---|---|---|---|---|
| RET-1 | Chunks of up to 600 tokens for an embedding model that reads 512 (median 533) | `tools/RetrieverTool.py:36-43` | tool case | fixed: at most 512 tokens with special tokens; checked by `test_retriever.py::test_index_is_built_once_then_loaded_from_the_cache`, which is skipped without the retrieval packages (passed once where they were installed) |
| RET-2 | Passages were returned without their source | `tools/RetrieverTool.py:83-89` | tool case | fixed: `test_retriever.py::test_each_passage_is_followed_by_its_source` |
| RET-3 | The index was rebuilt at every start, with the embedding model in float16 on the CPU | `tools/RetrieverTool.py:13-62` | timing | fixed: kept on disk; `test_retriever.py::test_index_is_built_once_then_loaded_from_the_cache` (skipped as above) |
| RET-4 | Corpus and model had no pinned revision | `tools/RetrieverTool.py:13, 37, 56` | reading | fixed: `test_retriever.py::test_cache_key_changes_with_corpus_revision_and_chunking` |
| OPS-1 | `github_fetch` returned GitHub's HTML for blob URLs, had no time-out and fetched any URL | `tools/github_fetch.py:15` | tool case | fixed: `test_fetch_retriever_main.py::test_raw_url_refuses_other_urls`, `test_github_fetch_uses_timeout_and_raw_url`, `test_github_fetch_size_cap` |
| OPS-2 | 11 of 12 requirements unpinned; `mace-torch` and `pandas` installed but never imported | `requirements.txt:1-12` | fresh install | fixed: pinned `requirements.txt`; no test |
| OPS-3 | Prompts were read relative to the working directory | `main.py:35-45` | reading | fixed: read relative to `agents.py`; the tests build the agents from a temporary working directory |
| OPS-4 | The two Slurm generators name the same things differently (`conda_activation_path` / `conda_path`, `additional_args_slurm` / `additional_slurm_args`) | `tools/mace_script.py:143-145`, `tools/ipi_submit_generator.py:76, 81` | reading | open: v1 names kept so existing calls and notes stay valid |
