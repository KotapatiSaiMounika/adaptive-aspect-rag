"""The ONE entry point Members 3, 4 and 5 should use."""
from pathlib import Path

from data.preprocessing import get_clean_corpus

# Anchored to the project root, so indexes are found no matter where you run from.
INDEX_ROOT = Path(__file__).resolve().parent.parent / "indexes"
METHODS = ("bm25", "dense")

_RETRIEVERS = {}  # (task, method) -> retriever; built/loaded once, then reused


def _get_retriever(task: str, method: str):
    if method not in METHODS:                       # validate BEFORE loading any data
        raise ValueError(f"Invalid retrieval method '{method}': use one of {METHODS}")
    key = (task, method)
    if key not in _RETRIEVERS:
        corpus = get_clean_corpus(task)
        if method == "bm25":
            from retrieval.bm25 import BM25Retriever
            _RETRIEVERS[key] = BM25Retriever(corpus, INDEX_ROOT / "bm25_index" / task)
        else:
            from retrieval.dense import DenseRetriever
            _RETRIEVERS[key] = DenseRetriever(corpus, INDEX_ROOT / "dense_index" / task)
    return _RETRIEVERS[key]


def retrieve(query: str, task: str, method: str = "bm25", top_k: int = 10) -> list:
    """Returns [{'doc_id','text','score','retriever'}, ...] sorted by score."""
    return _get_retriever(task, method).search(query, top_k)