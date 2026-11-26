import json
from types import SimpleNamespace

import pytest

import tools.retriever as rt
from tools.retriever import RetrieverTool

PAGES = [
    {"text": "alpha|beta|alpha", "title": "Input file tags", "source": "https://docs.ipi-code.org/input-tags.html"},
    {"text": "beta| gamma ||", "title": "Output files", "source": "https://docs.ipi-code.org/output-files.html"},
]


def doc(text, title="", source=""):
    return SimpleNamespace(page_content=text, metadata={"title": title, "source": source})


def test_make_chunks_keeps_page_source_and_drops_empty_and_repeated_chunks():
    chunks = rt.make_chunks(PAGES, lambda text: text.split("|"))
    assert [c["text"] for c in chunks] == ["alpha", "beta", "gamma"]
    assert chunks[0]["title"] == "Input file tags" and chunks[2]["source"].endswith("output-files.html")


def test_read_pages_skips_rows_without_text(tmp_path):
    rows = [{"text": " page one ", "metadata": {"title": "T", "source": "https://x/1"}}, {"text": ""}, {"text": "two"}]
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    assert rt.read_pages(path) == [
        {"text": "page one", "title": "T", "source": "https://x/1"},
        {"text": "two", "title": "", "source": ""},
    ]


def test_each_passage_is_followed_by_its_source():
    docs = [doc("thermostat text", "Input file tags", "https://docs.ipi-code.org/input-tags.html"), doc("no metadata")]
    text = rt.format_results(docs)
    assert "===== Document 0 =====\nthermostat text\nSource: Input file tags - https://docs.ipi-code.org/input-tags.html\n" in text
    assert "===== Document 1 =====\nno metadata\nSource: unknown\n" in text


def test_tool_keeps_its_name_and_argument_and_uses_top_k():
    seen = {}

    def search(query, k):
        seen.update(query=query, k=k)
        return [doc("barostat", "Input file tags", "https://docs.ipi-code.org/input-tags.html")]

    tool = RetrieverTool(vectordb=SimpleNamespace(similarity_search=search))
    assert tool.name == "retriever" and list(tool.inputs) == ["query"] and "Source:" in tool.description
    assert "Source: Input file tags" in tool(query="barostat")
    assert seen == {"query": "barostat", "k": rt.TOP_K}


def test_cache_key_changes_with_corpus_revision_and_chunking():
    params = rt.index_params()
    assert rt.cache_key(params) == rt.cache_key(dict(reversed(list(params.items()))))
    for name, value in [("dataset_revision", "0" * 40), ("chunk_tokens", 256), ("chunk_overlap", 0),
                        ("model_revision", "1" * 40), ("separators", ["\n"])]:
        assert rt.cache_key({**params, name: value}) != rt.cache_key(params), name


def test_index_file_round_trip(tmp_path):
    path = tmp_path / "retriever" / "abc.json"
    assert rt.load_index(path) is None
    chunks = [{"text": "a", "title": "T", "source": "S"}]
    rt.save_index(path, chunks, [[0.25, -1.0]])
    assert rt.load_index(path) == (chunks, [[0.25, -1.0]])
    assert [p.name for p in path.parent.iterdir()] == ["abc.json"]


def test_index_is_built_once_then_loaded_from_the_cache(tmp_path, monkeypatch):
    for module in ("faiss", "langchain_community", "langchain_huggingface", "langchain_text_splitters", "transformers"):
        pytest.importorskip(module, reason="retrieval packages are in requirements.txt, not in requirements-dev.txt")
    from huggingface_hub import try_to_load_from_cache

    model = try_to_load_from_cache(rt.MODEL, "config.json", revision=rt.MODEL_REVISION)
    corpus = try_to_load_from_cache(rt.DATASET, rt.CORPUS_FILE, revision=rt.DATASET_REVISION, repo_type="dataset")
    if not (isinstance(model, str) and isinstance(corpus, str)):
        pytest.skip("embedding model or corpus not in the local Hugging Face cache (the test does not download)")

    db = rt.build_vectordb(tmp_path)
    tokenizer = db.embedding_function._client.tokenizer
    texts = [d.page_content for d in db.docstore._dict.values()]
    assert max(len(tokenizer.encode(t)) for t in texts) <= rt.MODEL_MAX_TOKENS

    monkeypatch.setattr(rt, "build_chunks", lambda: pytest.fail("index rebuilt although it is cached"))
    cached = rt.build_vectordb(tmp_path)
    assert cached.index.ntotal == len(texts)
    text = RetrieverTool(vectordb=cached)(query="Restart a simulation from a checkpoint file")
    assert text.count("Source: ") == rt.TOP_K and "https://docs.ipi-code.org/" in text
