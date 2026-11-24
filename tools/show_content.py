from pathlib import Path
from typing import Any

from smolagents.tools import tool

from tools.session import get_session


@tool
def show_content(script: Any) -> str:
    """
    Print a generated file for the person at the terminal and return its text.

    This only displays; it is not a confirmation. slurm_submit shows the script and asks the person itself.

    Args:
        script (any): A generator result dict, the script_path of a generated file, or the text itself.
    Returns:
        str: The text that was shown.
    """
    if isinstance(script, dict):
        script_content = script.get("script_content", "")
    else:
        artifact = get_session().find_artifact(str(script)) if "\n" not in str(script) else None
        script_content = Path(artifact.path).read_text(encoding="utf-8") if artifact else str(script)

    print("\n" + "=" * 60)
    print("Generated script:")
    print("=" * 60)
    print(script_content)
    return script_content
