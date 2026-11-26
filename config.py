"""Settings from the environment, with a .env file in the working directory as fallback."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from dotenv import dotenv_values

DEFAULT_API_BASE = "https://api.deepseek.com/v1"


class SettingsError(ValueError):
    pass


def _required(env: Mapping[str, str], name: str, hint: str) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise SettingsError(f"{name} is not set: {hint} (in the environment or in .env, see .env.example).")
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
    def from_env(cls, environ: Optional[Mapping[str, str]] = None, dotenv_path: str | Path = ".env") -> "Settings":
        """Read SMOLAGENTS_* variables; values in the environment win over the .env file."""
        file_values = {k: v for k, v in dotenv_values(dotenv_path).items() if v is not None}
        env = {**file_values, **(os.environ if environ is None else environ)}
        return cls(
            model_id=_required(env, "SMOLAGENTS_MODEL_ID", "set it to a model name your provider serves now"),
            api_key=_required(env, "SMOLAGENTS_API_KEY", "set it to your provider's API key"),
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


def shadowed_settings(environ: Optional[Mapping[str, str]] = None, dotenv_path: str | Path = ".env") -> list[str]:
    """SMOLAGENTS_* names that .env sets but the environment overrides with another value.

    A key exported in ~/.bashrc for another provider, next to an API base taken from .env, gives a 401.
    """
    environ = os.environ if environ is None else environ
    return sorted(name for name, value in dotenv_values(dotenv_path).items()
                  if name.startswith("SMOLAGENTS_") and value and environ.get(name, value) != value)
