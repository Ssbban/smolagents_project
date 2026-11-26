from pathlib import Path
from typing import Any

from smolagents.tools import tool

from tools.session import get_session


@tool
def show_content(script: Any) -> str:
    """
    Print a generated file for the person at the terminal and return its text.

    This only displays; it is not a confirmation. slurm_submit shows the script and asks the person itself.
    It does not read other files: job_output reads what a submitted job wrote (logs, i-PI output).

    Args:
        script (any): A generator result dict, the script_path of a generated file, or a text of several lines.
    Returns:
        str: The text that was shown.
    """
    if isinstance(script, dict):
        script_content = script.get("script_content", "")
    elif "\n" in str(script):
        script_content = str(script)
    else:
        artifact = get_session().find_artifact(str(script))
        if artifact is None:
            raise ValueError(f"{script!r} is not a file generated in this session; read the files a submitted job "
                             "wrote with job_output(job_id, name)")
        script_content = Path(artifact.path).read_text(encoding="utf-8")

    print("\n" + "=" * 60)
    print("Generated script:")
    print("=" * 60)
    print(script_content)
    return script_content
