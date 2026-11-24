"""The person at the terminal decides; the model cannot answer for them."""

from __future__ import annotations

import sys
from dataclasses import dataclass

QUESTION = "Submit this job? [y = submit, n = do not submit, or type what should change]: "
RULE = "=" * 60


@dataclass(frozen=True)
class Decision:
    approved: bool
    feedback: str = ""


def ask(title: str, text: str, notes: list[str]) -> Decision:
    """Show `text` exactly as it will be submitted, then the notes, and ask the person."""
    if sys.stdin is None or not sys.stdin.isatty():
        return Decision(False, "no interactive terminal: nobody could confirm")
    print(f"\n{RULE}\n{title}\n{RULE}\n{text}\n{RULE}")
    for note in notes:
        print(f"note: {note}")
    try:
        answer = input(QUESTION).strip()
    except EOFError:
        return Decision(False, "")
    if answer.lower() in {"y", "yes"}:
        return Decision(True, "")
    if answer.lower() in {"", "n", "no"}:
        return Decision(False, "")
    return Decision(False, answer)
