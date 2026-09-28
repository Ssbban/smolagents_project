import os

from smolagents import CodeAgent, DuckDuckGoSearchTool, PlanningStep, UserInputTool
from smolagents.models import OpenAIServerModel

from tools.RetrieverTool import RetrieverTool, vectordb
from tools.customize_plan import interrupt_after_plan
from tools.get_server_info import slurm_config
from tools.github_fetch import github_fetch
from tools.ipi_InputFile_generator import ipi_InputFile_generator, ipi_template_schema
from tools.ipi_runase_generator import ipi_runase_generator
from tools.ipi_submit_generator import ipi_submit_generator
from tools.mace_cli_schema import mace_cli_schema
from tools.mace_script import MACE_script
from tools.show_content import show_content
from tools.slurm_submit import slurm_submit


def build_model() -> OpenAIServerModel:
    model_id = os.environ.get("SMOLAGENTS_MODEL_ID", "deepseek-chat")
    api_base = os.environ.get("SMOLAGENTS_API_BASE", "https://api.deepseek.com/v1")
    api_key = os.environ.get("SMOLAGENTS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SMOLAGENTS_API_KEY is required. Set it in your shell before running the agent."
        )
    return OpenAIServerModel(
        model_id=model_id,
        api_base=api_base,
        api_key=api_key,
        flatten_messages_as_text=True,
    )


def load_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as file:
        return file.read()


def main():
    model = build_model()

    mace_instruction = load_prompt("prompts/mace_instruction.txt")
    ipi_instruction = load_prompt("prompts/ipi_instruction.txt")
    manager_instruction = load_prompt("prompts/manager_instruction.txt")

    mace_tools = [
        github_fetch,
        mace_cli_schema,
        MACE_script,
        show_content,
        slurm_submit,
        slurm_config,
        DuckDuckGoSearchTool(),
        UserInputTool(),
    ]

    ipi_tools = [
        DuckDuckGoSearchTool(),
        UserInputTool(),
        RetrieverTool(vectordb=vectordb),
        ipi_template_schema,
        ipi_InputFile_generator,
        ipi_runase_generator,
        ipi_submit_generator,
        show_content,
        github_fetch,
        slurm_config,
    ]

    ipi_agent = CodeAgent(
        name="ipi_agent",
        tools=ipi_tools,
        model=model,
        max_steps=20,
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
        max_steps=20,
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
        max_steps=20,
        instructions=manager_instruction,
        planning_interval=10,
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
