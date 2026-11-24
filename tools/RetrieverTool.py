"""Semantic search over the i-PI documentation: dense retrieval with gte-small over a FAISS index.

The index is built on first use and kept under .cache/retriever/, keyed by the corpus revision and the chunking
parameters, so later runs load it instead of embedding the corpus again.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from smolagents import Tool

DATASET = "Sslmj/I-PI"  # ten pages of docs.ipi-code.org
DATASET_REVISION = "ccc6a123c250fa7e9eedaa2ef739f6671a1633a7"
CORPUS_FILE = "ipi_docs.jsonl"
MODEL = "thenlper/gte-small"
MODEL_REVISION = "17e1f347d17fe144873b1201da91788898c639cd"
MODEL_MAX_TOKENS = 512  # gte-small reads at most 512 tokens, [CLS] and [SEP] included
CHUNK_TOKENS = MODEL_MAX_TOKENS - 2  # counted without [CLS] and [SEP], so a whole chunk reaches the embedding
CHUNK_OVERLAP = 51  # 10 %, as before
SEPARATORS = ["\n\n", "\n", ".", " ", ""]
TOP_K = 7
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "retriever"

_vectordb: Any = None


def index_params() -> dict:
    """Everything that decides the content of the index."""
    return {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "corpus_file": CORPUS_FILE,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP,
        "separators": SEPARATORS,
    }


def cache_key(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def page_from_row(row: dict) -> Optional[dict]:
    """One dataset row -> {"text", "title", "source"}; None for a row without text."""
    text = (row.get("text") or "").strip()
    if not text:
        return None
    md = row.get("metadata") or {}
    return {"text": text, "title": md.get("title", ""), "source": md.get("source", "")}


def make_chunks(pages: Iterable[dict], split: Callable[[str], list[str]]) -> list[dict]:
    """Split every page, keep the page's title and source on each chunk, drop empty and repeated chunks."""
    chunks, seen = [], set()
    for page in pages:
        for piece in split(page["text"]):
            text = piece.strip()
            if text and text not in seen:
                seen.add(text)
                chunks.append({"text": text, "title": page["title"], "source": page["source"]})
    return chunks


def format_results(docs: list) -> str:
    """Documents found -> the text the agent reads, each passage followed by where it comes from."""
    parts = []
    for i, doc in enumerate(docs):
        md = getattr(doc, "metadata", None) or {}
        source = " - ".join(s for s in (md.get("title"), md.get("source")) if s) or "unknown"
        parts.append(f"===== Document {i} =====\n{doc.page_content}\nSource: {source}\n")
    return "Retrieved documents (most similar first):\n" + "\n".join(parts)


def save_index(path: Path, chunks: list[dict], vectors: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    data = {"params": index_params(), "chunks": chunks, "vectors": vectors}
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def load_index(path: Path) -> Optional[tuple[list[dict], list[list[float]]]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["chunks"], data["vectors"]


def read_pages(path: Path) -> list[dict]:
    """The corpus file (one JSON row per line) -> pages with text."""
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return [p for p in (page_from_row(row) for row in rows) if p is not None]


def build_chunks() -> list[dict]:
    """Read the corpus at the pinned revision and split it into chunks of at most CHUNK_TOKENS tokens."""
    from huggingface_hub import hf_hub_download
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer

    # hf_hub_download honours the revision also offline (the datasets loader then takes any cached version).
    corpus = hf_hub_download(DATASET, CORPUS_FILE, repo_type="dataset", revision=DATASET_REVISION)
    pages = read_pages(Path(corpus))
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION),
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        strip_whitespace=True,
        separators=SEPARATORS,
    )
    return make_chunks(pages, splitter.split_text)


def build_vectordb(cache_dir: Path = CACHE_DIR):
    """Load the index from the cache, or build it (about a minute) and store it there."""
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_huggingface import HuggingFaceEmbeddings

    # float32: transformers loads gte-small in float16 by default, which embeds several times slower on a CPU.
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL, model_kwargs={"revision": MODEL_REVISION, "model_kwargs": {"dtype": "float32"}}
    )
    path = cache_dir / f"{cache_key(index_params())}.json"
    cached = load_index(path)
    if cached is None:
        print("Building the i-PI documentation index (once; kept in .cache/retriever)...")
        chunks = build_chunks()
        vectors = embeddings.embed_documents([c["text"] for c in chunks])
        save_index(path, chunks, vectors)
    else:
        chunks, vectors = cached
    return FAISS.from_embeddings(
        text_embeddings=[(c["text"], v) for c, v in zip(chunks, vectors)],
        embedding=embeddings,
        metadatas=[{"title": c["title"], "source": c["source"]} for c in chunks],
        distance_strategy=DistanceStrategy.COSINE,
    )


def get_vectordb():
    global _vectordb
    if _vectordb is None:
        _vectordb = build_vectordb()
    return _vectordb


class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Searches the i-PI documentation (docs.ipi-code.org) by semantic similarity to the query. Returns the "
        f"{TOP_K} most similar passages, most similar first; each passage is followed by a line "
        "'Source: <page title> - <page URL>' naming where it comes from, so an answer can cite its page."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vectordb: Optional[Any] = None, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        vectordb = self.vectordb if self.vectordb is not None else get_vectordb()
        return format_results(vectordb.similarity_search(query, k=TOP_K))
