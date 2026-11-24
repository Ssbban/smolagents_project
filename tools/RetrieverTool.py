from typing import Any, Optional

from smolagents import Tool

_vectordb: Any = None


def row_to_document(row):
    from langchain.docstore.document import Document

    text = (row.get("text") or "").strip()
    if not text:
        return None
    md = row.get("metadata") or {}
    return Document(
        page_content=text,
        metadata={
            "source": md.get("source", ""),
            "title": md.get("title", ""),
        },
    )


def build_vectordb():
    """Download the i-PI documentation corpus and build the FAISS index (slow; done on first use)."""
    from datasets import load_dataset
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_huggingface import HuggingFaceEmbeddings
    from tqdm import tqdm
    from transformers import AutoTokenizer

    knowledge_base = load_dataset("Sslmj/I-PI", split="train")
    source_docs = [d for d in (row_to_document(row) for row in knowledge_base) if d is not None]

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained("thenlper/gte-small"),
        chunk_size=600,
        chunk_overlap=60,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    print("Splitting documents...")
    docs_processed, seen = [], set()
    for doc in tqdm(source_docs):
        for d in text_splitter.split_documents([doc]):
            t = d.page_content.strip()
            if t and t not in seen:
                seen.add(t)
                docs_processed.append(d)

    print("Embedding documents...")
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    return FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )


def get_vectordb():
    global _vectordb
    if _vectordb is None:
        _vectordb = build_vectordb()
    return _vectordb


class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves documentation from the knowledge base that have the closest embeddings to the input query."
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
        assert isinstance(query, str), "Your search query must be a string"

        vectordb = self.vectordb if self.vectordb is not None else get_vectordb()
        docs = vectordb.similarity_search(query, k=7)
        return "\nRetrieved documents:\n" + "".join(
            [
                f"===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )
