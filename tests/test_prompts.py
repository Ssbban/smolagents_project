"""The instruction files name only tools and agents registered on their agent, and leave tool lists and
confirmation protocols to the framework and the tools."""

import builtins
import re

import pytest
from fakemodel import ScriptedModel

from agents import PROMPTS, build_agents
from config import Settings

FILES = {"manager_instruction.txt": "manager", "mace_instruction.txt": "mace_agent", "ipi_instruction.txt": "ipi_agent"}
CALL = re.compile(r"\b([A-Za-z_]\w*)\(")
BACKTICKED = re.compile(r"`([A-Za-z_]\w*)(?:\(\))?`")
WORD = re.compile(r"\b[A-Za-z_]\w*\b")  # bare words count when they cannot be English: snake_case or ...Tool
LIST_ENTRY = re.compile(r"^\s*-\s*\**`?([A-Za-z_]\w*)`?\**\s*:", re.MULTILINE)
TOOL_LIST = re.compile(r"^\s*tools?\s*:|following tools", re.IGNORECASE | re.MULTILINE)
TYPE_YES = re.compile(r"type\s+['\"`]?yes", re.IGNORECASE)


@pytest.fixture(scope="module")
def team():
    manager = build_agents(ScriptedModel(), Settings(model_id="scripted", api_key="unused"))
    return {"manager": manager, **manager.managed_agents}


def registered(agent) -> set[str]:
    return set(agent.tools) | set(agent.managed_agents)


def tool_like(team) -> dict[str, str]:
    """Lower-case spelling -> name, for every tool and agent name and tool class name in the project."""
    names = {}
    for agent in team.values():
        for name in registered(agent) | {"manager_agent"}:
            names[name.lower()] = name
        for tool in agent.tools.values():
            names[type(tool).__name__.lower()] = type(tool).__name__
    return names


@pytest.mark.parametrize("filename", FILES)
def test_names_in_a_prompt_are_registered_on_its_agent(team, filename):
    text = (PROMPTS / filename).read_text(encoding="utf-8")
    known = registered(team[FILES[filename]])
    calls = {name for name in CALL.findall(text) if not hasattr(builtins, name)}
    distinctive = {name for name in WORD.findall(text) if "_" in name or name.endswith("Tool")}
    mentioned = {name for name in set(BACKTICKED.findall(text)) | distinctive if name.lower() in tool_like(team)}
    assert sorted((calls | mentioned) - known) == []


@pytest.mark.parametrize("agent", ["mace_agent", "ipi_agent"])
def test_tool_descriptions_name_tools_of_their_agent(team, agent):
    """preflight and slurm_submit serve both agents and name both script generators; nothing else crosses over."""
    known = registered(team[agent]) | {"MACE_script", "ipi_submit_generator"}
    for tool in team[agent].tools.values():
        text = tool.description + " ".join(spec.get("description", "") for spec in tool.inputs.values())
        mentioned = {word for word in WORD.findall(text) if word.lower() in tool_like(team)}
        assert sorted(mentioned - known) == [], tool.name


@pytest.mark.parametrize("filename", FILES)
def test_no_prompt_lists_the_tools(team, filename):
    text = (PROMPTS / filename).read_text(encoding="utf-8")
    entries = [name for name in LIST_ENTRY.findall(text) if name.lower() in tool_like(team)]
    assert TOOL_LIST.findall(text) == [] and entries == []


@pytest.mark.parametrize("filename", FILES)
def test_no_prompt_asks_the_person_to_type_yes(filename):
    assert TYPE_YES.findall((PROMPTS / filename).read_text(encoding="utf-8")) == []
