import smolagents
from smolagents import CodeAgent, DuckDuckGoSearchTool, PlanningStep, UserInputTool
from smolagents.models import OpenAIServerModel

from config import Settings, SettingsError
from tools.RetrieverTool import RetrieverTool
from tools.customize_plan import interrupt_after_plan
from tools.get_server_info import slurm_config
from tools.github_fetch import github_fetch
from tools.ipi_InputFile_generator import ipi_InputFile_generator, ipi_template_schema
from tools.ipi_runase_generator import ipi_runase_generator
from tools.ipi_submit_generator import ipi_submit_generator
from tools.mace_cli_schema import mace_cli_schema
from tools.mace_script import MACE_script
from tools.preflight import preflight
from tools.session import start_session
from tools.show_content import show_content
from tools.slurm_status import job_status
from tools.slurm_submit import slurm_submit


def build_model(settings: Settings) -> OpenAIServerModel:
    extra = {} if settings.temperature is None else {"temperature": settings.temperature}
    return OpenAIServerModel(
        model_id=settings.model_id,
        api_base=settings.api_base,
        api_key=settings.api_key,
        flatten_messages_as_text=True,
        **extra,
    )


def load_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as file:
        return file.read()


def main():
    try:
        settings = Settings.from_env()
    except SettingsError as error:
        raise SystemExit(f"Error: {error}") from None
    session = start_session(settings.runs_dir)
    session.trace.emit("session_start", settings=settings.public(), smolagents=smolagents.__version__)
    print(f"Session {session.id}: generated files and the trace are in {session.dir.as_posix()}/")
    model = build_model(settings)

    mace_instruction = load_prompt("prompts/mace_instruction.txt")
    ipi_instruction = load_prompt("prompts/ipi_instruction.txt")
    manager_instruction = load_prompt("prompts/manager_instruction.txt")

    mace_tools = [
        github_fetch,
        mace_cli_schema,
        MACE_script,
        show_content,
        preflight,
        slurm_submit,
        job_status,
        slurm_config,
        DuckDuckGoSearchTool(),
        UserInputTool(),
    ]

    ipi_tools = [
        DuckDuckGoSearchTool(),
        UserInputTool(),
        RetrieverTool(),
        ipi_template_schema,
        ipi_InputFile_generator,
        ipi_runase_generator,
        ipi_submit_generator,
        show_content,
        github_fetch,
        slurm_config,
        preflight,
        slurm_submit,
        job_status,
    ]

    ipi_agent = CodeAgent(
        name="ipi_agent",
        tools=ipi_tools,
        model=model,
        max_steps=settings.max_steps,
        instructions=ipi_instruction,
        additional_authorized_imports=[
            "tempfile",
            "subprocess",
            "os",
            "typing",
        ],
        description="i-PI input file generator agent, it can help users to generate i-PI input file.",
    )

    mace_agent = CodeAgent(
        name="mace_agent",
        tools=mace_tools,
        model=model,
        max_steps=settings.max_steps,
        instructions=mace_instruction,
        additional_authorized_imports=[
            "tempfile",
            "subprocess",
            "os",
            "json",
        ],
        description="MACE training agent, it can help users to train MACE model on their dataset.",
    )

    manager_agent = CodeAgent(
        name="manager_agent",
        tools=[],
        model=model,
        max_steps=settings.max_steps,
        instructions=manager_instruction,
        planning_interval=settings.planning_interval,
        step_callbacks={PlanningStep: interrupt_after_plan},
        additional_authorized_imports=[
            "tempfile",
            "subprocess",
            "os",
        ],
        managed_agents=[ipi_agent, mace_agent],
    )
    print(
        "Agent: Hello! I'm here to guide you through MACE training and I-PI. Type 'exit' or `q` to quit."
    )

    while True:
        user_query = input("User> ").strip()
        if user_query.lower() in {"exit", "quit", "q"}:
            print("Agent: Goodbye!")
            break
        try:
            print("\nAgent starting execution...")
            response = manager_agent.run(user_query, reset=False)
            print("\nTask completed successfully!")
            print("\nAgent>", response)
        except Exception as error:
            if "interrupted" in str(error).lower():
                print("\nAgent execution was cancelled by user.")
            else:
                print(f"\nError: {error}")


if __name__ == "__main__":
    main()
