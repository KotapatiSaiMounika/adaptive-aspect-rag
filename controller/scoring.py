"""Document-aspect support scoring. Scores are always in [0, 1].

Two interchangeable backends (same interface):
  * LexicalScorer - no model, no network, works anywhere (good for tests / fallback).
  * DenseScorer   - bge-small embeddings (same model as Member 1's dense retriever).
"""
import math
import re
from typing import Dict, List, Optional, Protocol

_STOP = set("""a an the and or of to in on for with by from at as is are was were be been it its this
that these those which who whom what why how do does did can could should would may might not no
than then so such into about over under between while also more most very""".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def normalize_aspects(aspects) -> List[Dict[str, str]]:
    """Accept whatever Member 2 / the dataset gives us -> [{'id','text'}].
    Handles: list[str], list[dict], dict{id: text|dict}, or None."""
    if not aspects:
        return []
    if isinstance(aspects, dict):
        aspects = [{"id": k, **(v if isinstance(v, dict) else {"text": v})}
                   for k, v in aspects.items()]
    out = []
    for i, a in enumerate(aspects):
        if isinstance(a, str):
            out.append({"id": f"a{i}", "text": a})
            continue
        aid = next((a[k] for k in ("id", "aspect_id", "name") if a.get(k) not in (None, "")), f"a{i}")
        text = next((a[k] for k in ("description", "text", "aspect", "content", "name")
                     if isinstance(a.get(k), str) and a[k].strip()), "")
        out.append({"id": str(aid), "text": text})
    return out


class AspectScorer(Protocol):
    def score(self, aspect_text: str, doc_text: str) -> float: ...


class LexicalScorer:
    """Fraction of (idf-weighted) aspect terms that appear in the document."""

    def __init__(self, idf: Optional[Dict[str, float]] = None):
        self.idf = idf or {}

    def fit(self, corpus_texts: List[str]) -> "LexicalScorer":
        n = len(corpus_texts)
        df: Dict[str, int] = {}
        for t in corpus_texts:
            for w in set(tokenize(t)):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}
        return self

    def score(self, aspect_text: str, doc_text: str) -> float:
        a_terms = set(tokenize(aspect_text))
        if not a_terms:
            return 0.0
        d_terms = set(tokenize(doc_text))
        default = max(self.idf.values(), default=1.0)  # unseen term = rare = important
        w = lambda t: self.idf.get(t, default)
        total = sum(w(t) for t in a_terms)
        return sum(w(t) for t in a_terms & d_terms) / total if total else 0.0


class DenseScorer:
    """Cosine similarity, linearly rescaled from [lo, hi] to [0, 1].
    bge cosines for related text sit around 0.5-0.85, so raw cosine is not a probability."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 lo: float = 0.45, hi: float = 0.85, max_seq_length: int = 256):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        self.lo, self.hi = lo, hi

    def score(self, aspect_text: str, doc_text: str) -> float:
        e = self.model.encode([aspect_text, doc_text], normalize_embeddings=True)
        cos = float(e[0] @ e[1])
        return min(1.0, max(0.0, (cos - self.lo) / (self.hi - self.lo)))
