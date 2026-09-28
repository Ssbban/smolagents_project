import requests
from smolagents.tools import tool


@tool
def github_fetch(url_path: str) -> str:
    """
    Fetch the raw text of a file from GitHub.

    Args:
        url_path (str): Path to the github file want to fetch, e.g. "https://github.com/ACEsuit/mace/blob/main/mace/cli/run_train.py"
    Returns:
        str: The full source code of the requested file.
    """
    response = requests.get(url_path)
    response.raise_for_status()
    return response.text
