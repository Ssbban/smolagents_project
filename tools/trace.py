"""JSONL trace of one session: one JSON object per line."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep

OBSERVATION_CHARS = 2000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Trace:
    def __init__(self, path: str | Path, session_id: str = ""):
        self.path = Path(path)
        self.session_id = session_id

    def emit(self, event: str, **data: Any) -> None:
        record = {"ts": utc_now(), "session": self.session_id, "event": event, **data}
        line = json.dumps(record, default=str, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8", newline="\n") as file:
            file.write(line + "\n")


def step_callback(step: Any, agent: Any = None) -> None:
    """Step callback for `step_callbacks={MemoryStep: step_callback}`; never records model input messages."""
    from tools.session import get_session

    try:
        trace = get_session().trace
    except RuntimeError:
        return
    data: dict[str, Any] = {"agent": getattr(agent, "name", None), "step_type": type(step).__name__}
    if isinstance(step, ActionStep):
        data["step"] = step.step_number
        data["code"] = step.code_action
        data["observation"] = (step.observations or "")[:OBSERVATION_CHARS]
        data["error"] = str(step.error) if step.error else None
    elif isinstance(step, PlanningStep):
        data["plan"] = step.plan
    elif isinstance(step, FinalAnswerStep):
        data["output"] = str(step.output)[:OBSERVATION_CHARS]
    usage = getattr(step, "token_usage", None)
    if usage is not None:
        data["tokens"] = {"input": usage.input_tokens, "output": usage.output_tokens}
    timing = getattr(step, "timing", None)
    if timing is not None and timing.duration is not None:
        data["duration_s"] = round(timing.duration, 3)
    trace.emit("step", **data)
