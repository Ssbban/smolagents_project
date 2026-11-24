import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.github_fetch as fetch_module
from tools.github_fetch import github_fetch, raw_url
from tools.RetrieverTool import RetrieverTool
from tools.show_content import show_content


def test_raw_url():
    blob = "https://github.com/ACEsuit/mace/blob/main/mace/cli/run_train.py"
    assert raw_url(blob) == "https://raw.githubusercontent.com/ACEsuit/mace/main/mace/cli/run_train.py"
    raw = "https://raw.githubusercontent.com/ACEsuit/mace/main/README.md"
    assert raw_url(raw) == raw


@pytest.mark.parametrize("url", ["https://example.com/a.py", "http://raw.githubusercontent.com/a/b/c/d",
                                 "https://github.com/ACEsuit/mace", "file:///etc/passwd"])
def test_raw_url_refuses_other_urls(url):
    with pytest.raises(ValueError):
        raw_url(url)


class FakeResponse:
    def __init__(self, chunks):
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        return iter(self.chunks)


def test_github_fetch_uses_timeout_and_raw_url(monkeypatch):
    seen = {}

    def get(url, **kwargs):
        seen.update(url=url, **kwargs)
        return FakeResponse([b"print(", b"'hi')\n"])

    monkeypatch.setattr(fetch_module.requests, "get", get)
    assert github_fetch(url_path="https://github.com/a/b/blob/v1/x.py") == "print('hi')\n"
    assert seen["url"] == "https://raw.githubusercontent.com/a/b/v1/x.py" and seen["timeout"] == 30


def test_github_fetch_size_cap(monkeypatch):
    monkeypatch.setattr(fetch_module.requests, "get", lambda url, **k: FakeResponse([b"x" * 600_000] * 2))
    with pytest.raises(ValueError, match="larger than 1 MB"):
        github_fetch(url_path="https://raw.githubusercontent.com/a/b/main/big.txt")


def test_retriever_uses_injected_index_without_building():
    docs = [SimpleNamespace(page_content="thermostat"), SimpleNamespace(page_content="barostat")]
    index = SimpleNamespace(similarity_search=lambda query, k: docs[:k])
    text = RetrieverTool(vectordb=index)(query="thermostat")
    assert "===== Document 0 =====\nthermostat" in text and "barostat" in text


def test_main_imports_without_retrieval_packages():
    for name in ("langchain", "faiss", "datasets", "torch"):
        assert name not in sys.modules
    main = importlib.import_module("main")
    names = {tool.name for tool in main.__dict__.values() if hasattr(tool, "forward") and hasattr(tool, "name")}
    assert {"preflight", "slurm_submit", "job_status", "slurm_config"} <= names
    for name in ("langchain", "faiss", "datasets", "torch"):
        assert name not in sys.modules


def test_show_content_reads_an_artifact_path(session, capsys):
    artifact = session.write_artifact(session.new_bundle("x"), "job.sh", "#!/bin/bash\necho hi\n", "mace_script")
    assert show_content(script=artifact.path) == Path(artifact.path).read_text()
    assert show_content(script={"script_content": "abc"}) == "abc"
    assert "echo hi" in capsys.readouterr().out
