"""A scripted stand-in for the LLM, shared by the three agents like the real model.

Each agent has its own queue of replies. A reply is the text the model would write (use `code(...)` for a code
action) or an exception to raise, as a provider error would. Planning calls get `plan` (a list: one per call).
"""

from __future__ import annotations

from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage


def code(source: str) -> str:
    return f"Thought: scripted step.\n<code>\n{source}\n</code>"


def text_of(message) -> str:
    content = message.content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def agent_of(messages) -> str:
    """Which agent is calling. A sub-agent's task (first message after the system prompt, also in the
    step-limit fallback call) starts with its name, from the managed_agent task template."""
    head = " ".join(text_of(m) for m in messages[:2])
    for name in ("mace_agent", "ipi_agent"):
        if f"agent named '{name}'" in head:
            return name
    return "manager"


class ScriptedModel(Model):
    def __init__(self, manager=(), mace_agent=(), ipi_agent=(), plan="1. Delegate to the right specialist."):
        super().__init__(model_id="scripted")
        self.queues = {"manager": list(manager), "mace_agent": list(mace_agent), "ipi_agent": list(ipi_agent)}
        self.plan = plan
        self.calls: list[tuple[str, list]] = []  # (agent or "plan", messages)

    def seen_by(self, agent: str) -> list[str]:
        """The last message of each call this agent made: the observation it was reacting to."""
        return [text_of(messages[-1]) for who, messages in self.calls if who == agent]

    def generate(self, messages, stop_sequences=None, **kwargs):
        if stop_sequences and "<end_plan>" in stop_sequences:
            self.calls.append(("plan", messages))
            reply = self.plan.pop(0) if isinstance(self.plan, list) else self.plan
        else:
            agent = agent_of(messages)
            self.calls.append((agent, messages))
            if not self.queues[agent]:
                raise AssertionError(f"no scripted reply left for {agent}")
            reply = self.queues[agent].pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return ChatMessage(role=MessageRole.ASSISTANT, content=reply, token_usage=TokenUsage(100, 10))
