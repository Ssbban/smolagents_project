"""The three agents: a manager CodeAgent that delegates to two science CodeAgents."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import yaml
from smolagents import CodeAgent, DuckDuckGoSearchTool, PlanningStep, UserInputTool
from smolagents.agents import populate_template
from smolagents.memory import MemoryStep
from smolagents.models import OpenAIServerModel
from smolagents.utils import AgentError

from config import Settings
from tools.RetrieverTool import RetrieverTool
from tools.customize_plan import interrupt_after_plan
from tools.get_server_info import slurm_config
from tools.github_fetch import github_fetch
from tools.ipi_help import ipi_tag_help
from tools.ipi_InputFile_generator import ipi_InputFile_generator, ipi_template_schema
from tools.ipi_runase_generator import ipi_runase_generator
from tools.ipi_submit_generator import ipi_submit_generator
from tools.mace_cli_schema import mace_cli_schema
from tools.mace_script import MACE_script
from tools.preflight import preflight
from tools.session import get_session
from tools.show_content import show_content
from tools.slurm_status import job_status
from tools.slurm_submit import slurm_submit
from tools.trace import step_callback

PROMPTS = Path(__file__).parent / "prompts"
IMPORTS = ["json"]
# No executor time limit: one code action may hold a whole delegation and the person's answers.
# The tools that block (sbatch, squeue, HTTP) and the model client have their own time-outs.
EXECUTOR = {"timeout_seconds": None}

# The stock template declares managed agents as returning str; ScienceAgent returns a dict.
STR_SIGNATURE = "additional_args: dict[str, Any]) -> str:"
DICT_SIGNATURE = "additional_args: dict[str, Any]) -> dict:"

SUBAGENT_TASK = """You're a helpful agent named '{{name}}'. Your manager gives you this task:
---
{{task}}
---
When the task is done, or you cannot go further, call final_answer with a short plain-text summary for your manager:
what was done, what is still pending, and what the person must still decide (ask it there instead of guessing).
The files your tools wrote and the jobs they submitted are recorded by the tools and handed to your manager with
your summary, so do not list them from memory and never invent a file path or a job id."""

RESULT_DOC = """Returns a dict: "status" is "success", "incomplete" (it ran out of steps) or "error" (it failed;
"summary" says why); "summary" is its short report: what was done, what is pending, what the person must decide;
"artifacts" lists the files its tools wrote (kind, path, sha256, bundle, created_at); "jobs" lists the jobs it
submitted (job_id, script, bundle, submitted_at, state); "agent" is its name."""


def build_model(settings: Settings) -> OpenAIServerModel:
    extra = {} if settings.temperature is None else {"temperature": settings.temperature}
    return OpenAIServerModel(
        model_id=settings.model_id,
        api_base=settings.api_base,
        api_key=settings.api_key,
        flatten_messages_as_text=True,
        client_kwargs={"timeout": 120, "max_retries": 2},
        **extra,
    )


def stock_templates() -> dict:
    text = importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def manager_templates() -> dict:
    """The stock templates with the three managed-agent signatures returning dict instead of str."""
    templates = stock_templates()
    planning = templates["planning"]
    templates["system_prompt"] = templates["system_prompt"].replace(STR_SIGNATURE, DICT_SIGNATURE)
    for key in ("initial_plan", "update_plan_post_messages"):
        planning[key] = planning[key].replace(STR_SIGNATURE, DICT_SIGNATURE)
    return templates


def subagent_templates() -> dict:
    templates = stock_templates()
    templates["managed_agent"]["task"] = SUBAGENT_TASK
    return templates


class ScienceAgent(CodeAgent):
    """A managed CodeAgent whose call from the manager's code returns a dict instead of a report string."""

    def __call__(self, task: str, additional_args: dict | None = None, **ignored) -> dict:
        # `ignored`: the stock entry point forwards max_steps, reset, ... from generated code into run().
        session = get_session()
        mark = session.mark()
        session.trace.emit("delegation_start", agent=self.name, task=task)
        full_task = populate_template(self.prompt_templates["managed_agent"]["task"], {"name": self.name, "task": task})
        try:
            result = self.run(full_task, additional_args=additional_args, return_full_result=True)
        except AgentError as error:
            status, summary = "error", f"{type(error).__name__}: {error}"
        else:
            status = "success" if result.state == "success" else "incomplete"
            summary = str(result.output)
        tokens = self.monitor.get_total_token_counts()
        session.trace.emit("delegation_end", agent=self.name, status=status,
                           tokens={"input": tokens.input_tokens, "output": tokens.output_tokens})
        return {"agent": self.name, "status": status, "summary": summary, **session.since(mark)}


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def build_agents(model, settings: Settings) -> CodeAgent:
    """The manager, with mace_agent and ipi_agent as its managed agents."""
    mace_agent = ScienceAgent(
        name="mace_agent",
        description="Specialist for MACE potentials: writes, checks and submits MACE training jobs on Slurm, follows "
        "them, and answers questions about MACE and its options.\n" + RESULT_DOC,
        tools=[github_fetch, mace_cli_schema, MACE_script, show_content, preflight, slurm_submit, job_status,
               slurm_config, DuckDuckGoSearchTool(), UserInputTool()],
        instructions=load_prompt("mace_instruction.txt"),
        prompt_templates=subagent_templates(),
        model=model,
        max_steps=settings.max_steps,
        additional_authorized_imports=IMPORTS,
        executor_kwargs=EXECUTOR,
        step_callbacks={MemoryStep: step_callback},
    )
    ipi_agent = ScienceAgent(
        name="ipi_agent",
        description="Specialist for molecular dynamics with i-PI: writes the i-PI input, the ASE client and the job "
        "script, checks and submits the job on Slurm, follows it, and answers questions about i-PI.\n" + RESULT_DOC,
        tools=[DuckDuckGoSearchTool(), UserInputTool(), RetrieverTool(), ipi_template_schema, ipi_tag_help,
               ipi_InputFile_generator, ipi_runase_generator, ipi_submit_generator, show_content,
               slurm_config, preflight, slurm_submit, job_status],
        instructions=load_prompt("ipi_instruction.txt"),
        prompt_templates=subagent_templates(),
        model=model,
        max_steps=settings.max_steps,
        additional_authorized_imports=IMPORTS,
        executor_kwargs=EXECUTOR,
        step_callbacks={MemoryStep: step_callback},
    )
    return CodeAgent(
        name="manager_agent",
        tools=[],
        managed_agents=[mace_agent, ipi_agent],
        instructions=load_prompt("manager_instruction.txt"),
        prompt_templates=manager_templates(),
        model=model,
        max_steps=settings.max_steps,
        planning_interval=settings.planning_interval,
        additional_authorized_imports=IMPORTS,
        executor_kwargs=EXECUTOR,
        step_callbacks={MemoryStep: step_callback, PlanningStep: interrupt_after_plan},
    )
