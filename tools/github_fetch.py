from urllib.parse import urlparse

import requests
from smolagents.tools import tool

TIMEOUT_SECONDS = 30
MAX_BYTES = 1_000_000
RAW_HOST = "raw.githubusercontent.com"


def raw_url(url: str) -> str:
    """The raw.githubusercontent.com URL of a GitHub file URL; ValueError for anything else."""
    parsed = urlparse(url.strip())
    if parsed.scheme != "https":
        raise ValueError(f"only https GitHub file URLs can be fetched; got {url!r}")
    if parsed.netloc == RAW_HOST:
        return parsed.geturl()
    parts = parsed.path.strip("/").split("/")
    if parsed.netloc in {"github.com", "www.github.com"} and len(parts) >= 5 and parts[2] in {"blob", "raw"}:
        owner, repo, _, ref, *path = parts
        return f"https://{RAW_HOST}/{owner}/{repo}/{ref}/{'/'.join(path)}"
    raise ValueError(
        "github_fetch only reads files from GitHub: give https://github.com/<owner>/<repo>/blob/<ref>/<path> "
        f"or a https://{RAW_HOST}/... URL; got {url!r}"
    )


@tool
def github_fetch(url_path: str) -> str:
    """
    Fetch the raw text of a file from GitHub (files up to 1 MB; 30 s time-out).

    Args:
        url_path (str): GitHub file URL, e.g. "https://github.com/ACEsuit/mace/blob/main/mace/cli/run_train.py"; blob URLs are read as raw files.
    Returns:
        str: The full text of the requested file.
    """
    url = raw_url(url_path)
    with requests.get(url, timeout=TIMEOUT_SECONDS, stream=True) as response:
        response.raise_for_status()
        data = b""
        for chunk in response.iter_content(chunk_size=65536):
            data += chunk
            if len(data) > MAX_BYTES:
                raise ValueError(f"{url} is larger than {MAX_BYTES // 1_000_000} MB; fetch a smaller file")
    return data.decode("utf-8", errors="replace")
