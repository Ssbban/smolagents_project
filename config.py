"""Settings, read from SMOLAGENTS_* environment variables."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional

DEFAULT_API_BASE = "https://api.anthropic.com/v1"  # OpenAI-compatible; the provider the agent was tested with


class SettingsError(ValueError):
    pass


def _required(env: Mapping[str, str], name: str, example: str) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise SettingsError(f"{name} is not set. Set it in your shell before starting: export {name}={example}")
    return value


def _number(env: Mapping[str, str], name: str, default, kind):
    text = (env.get(name) or "").strip()
    if not text:
        return default
    try:
        value = kind(text)
    except ValueError:
        raise SettingsError(f"{name} must be a {'whole ' if kind is int else ''}number; got {text!r}.") from None
    if kind is int and value <= 0:
        raise SettingsError(f"{name} must be greater than zero; got {text!r}.")
    return value


@dataclass(frozen=True)
class Settings:
    model_id: str
    api_key: str = field(repr=False)
    api_base: str = DEFAULT_API_BASE
    temperature: Optional[float] = None
    max_steps: int = 20
    planning_interval: int = 10
    runs_dir: str = "runs"

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if environ is None else environ
        return cls(
            model_id=_required(env, "SMOLAGENTS_MODEL_ID", "<a model your provider serves>"),
            api_key=_required(env, "SMOLAGENTS_API_KEY", "<your API key>"),
            api_base=(env.get("SMOLAGENTS_API_BASE") or "").strip() or DEFAULT_API_BASE,
            temperature=_number(env, "SMOLAGENTS_TEMPERATURE", None, float),
            max_steps=_number(env, "SMOLAGENTS_MAX_STEPS", 20, int),
            planning_interval=_number(env, "SMOLAGENTS_PLANNING_INTERVAL", 10, int),
            runs_dir=(env.get("SMOLAGENTS_RUNS_DIR") or "").strip() or "runs",
        )

    def public(self) -> dict:
        """Everything except the API key, for the trace."""
        values = asdict(self)
        del values["api_key"]
        return values
