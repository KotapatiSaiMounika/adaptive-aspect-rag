"""Member 2 - Document-aspect relevance ("reflection").

Produces ``{aspect_id: {doc_id: score in [0, 1]}}`` - exactly the ``reflections`` argument of
``AdaptiveController.decide(...)`` / ``score_matrix(...)``.

What it adds over Member 4's document-level scorers:
  * Passage-level evidence: a long document is split into sentence windows and the best
    window is scored, so an aspect that is actually *discussed* in one passage beats one whose
    words are merely scattered across a 1000-word document.
  * Optional ``judge_fn`` (LLM / cross-encoder) blended in for higher precision.
  * Incremental cache: only new (aspect, doc) pairs are scored in later retrieval rounds.

Backend-agnostic: ``base_scorer`` is anything with ``.score(aspect_text, doc_text) -> [0,1]``
(Member 4's LexicalScorer or DenseScorer).  The class itself also satisfies that protocol, so it
can alternatively be passed to the controller as its ``scorer``.
"""
import re
from typing import Callable, Dict, List, Optional, Tuple

from controller.scoring import AspectScorer, LexicalScorer, normalize_aspects

Reflections = Dict[str, Dict[str, float]]
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _coerce_aspects(aspects):
    """Accept Aspect objects (anything with .to_dict()) as well as whatever normalize_aspects takes."""
    if isinstance(aspects, (list, tuple)):
        aspects = [a.to_dict() if hasattr(a, "to_dict") else a for a in aspects]
    return normalize_aspects(aspects)


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


class AspectRelevanceScorer:
    def __init__(self, base_scorer: Optional[AspectScorer] = None, window: int = 3, stride: int = 2,
                 max_chunks: int = 40, passage_weight: float = 0.5,
                 judge_fn: Optional[Callable[[str, str], float]] = None, judge_weight: float = 0.3):
        """passage_weight: 0 = whole-document score only (same as Member 4's scorer), 1 = best passage only.
        judge_fn(aspect_text, passage_text) -> [0,1]; run on the best passage only (cheap)."""
        self.base = base_scorer or LexicalScorer()
        self.window, self.stride, self.max_chunks = max(1, window), max(1, stride), max(1, max_chunks)
        self.passage_weight = _clamp(passage_weight)
        self.judge_fn, self.judge_weight = judge_fn, _clamp(judge_weight)

    def fit(self, corpus_texts: List[str]) -> "AspectRelevanceScorer":
        """Fit idf on the task corpus (only relevant for the lexical backend)."""
        if hasattr(self.base, "fit"):
            self.base.fit(corpus_texts)
        return self

    # ---------------------------------------------------------------- passages
    def split_passages(self, text: str) -> List[str]:
        sents = [s.strip() for s in _SENT.split(text or "") if s.strip()]
        if len(sents) <= self.window:
            return [" ".join(sents)] if sents else []
        chunks = [" ".join(sents[i:i + self.window])
                  for i in range(0, len(sents) - self.window + self.stride, self.stride)]
        if len(chunks) > self.max_chunks:         # evenly subsample long documents
            n = self.max_chunks
            chunks = [chunks[round(i * (len(chunks) - 1) / (n - 1))] for i in range(n)] if n > 1 else chunks[:1]
        return chunks

    # ---------------------------------------------------------------- scoring
    def explain(self, aspect_text: str, doc_text: str) -> Dict:
        """Score plus the evidence behind it: {'score','doc_score','passage_score','snippet'}."""
        if not (aspect_text or "").strip() or not (doc_text or "").strip():
            return {"score": 0.0, "doc_score": 0.0, "passage_score": 0.0, "snippet": ""}
        doc_s = _clamp(self.base.score(aspect_text, doc_text))
        best_s, best_p = doc_s, doc_text           # passage stage skipped when it cannot matter
        if self.passage_weight > 0 or self.judge_fn is not None:
            best_s, best_p = 0.0, ""
            for p in self.split_passages(doc_text):
                s = _clamp(self.base.score(aspect_text, p))
                if s > best_s or not best_p:
                    best_s, best_p = s, p
        score = (1 - self.passage_weight) * doc_s + self.passage_weight * best_s
        if self.judge_fn is not None and best_p and self.judge_weight > 0:
            score = (1 - self.judge_weight) * score + self.judge_weight * _clamp(self.judge_fn(aspect_text, best_p))
        return {"score": round(_clamp(score), 4), "doc_score": round(doc_s, 4),
                "passage_score": round(best_s, 4), "snippet": best_p[:300]}

    def score(self, aspect_text: str, doc_text: str) -> float:
        """AspectScorer protocol -> usable directly as ``AdaptiveController(..., scorer=this)``."""
        return self.explain(aspect_text, doc_text)["score"]


class ReflectionEngine:
    """Stateful per-question helper: call ``reflect(evidence)`` after every retrieval round."""

    def __init__(self, aspects, scorer: Optional[AspectRelevanceScorer] = None):
        self.aspects = _coerce_aspects(aspects)
        if not self.aspects:
            raise ValueError("ReflectionEngine needs at least one aspect.")
        self.scorer = scorer or AspectRelevanceScorer()
        self._cache: Dict[Tuple[str, str], Dict] = {}

    def reflect(self, evidence: List[Dict]) -> Reflections:
        """evidence = [{'doc_id','text'}, ...] (retrieve() output works as-is).
        Returns {aspect_id: {doc_id: score}} covering every aspect x every evidence doc."""
        out: Reflections = {}
        for a in self.aspects:
            row = {}
            for d in evidence:
                did = str(d["doc_id"])
                key = (a["id"], did)
                if key not in self._cache:
                    self._cache[key] = self.scorer.explain(a["text"], d.get("text", ""))
                row[did] = self._cache[key]["score"]
            out[a["id"]] = row
        return out

    def evidence_snippets(self, aspect_id: str, top_n: int = 3) -> List[Dict]:
        """Best supporting passages seen so far for an aspect (for answer generation / Member 5)."""
        hits = [{"doc_id": did, **info} for (aid, did), info in self._cache.items() if aid == aspect_id]
        return sorted(hits, key=lambda h: -h["score"])[:top_n]


def reflect(aspects, evidence: List[Dict], scorer: Optional[AspectRelevanceScorer] = None) -> Reflections:
    """One-shot convenience (no caching across calls)."""
    return ReflectionEngine(aspects, scorer).reflect(evidence)


def summarize_support(reflections: Reflections, threshold: float = 0.38) -> Dict[str, Dict]:
    """Per aspect: best doc, best score, docs above ``threshold`` (default = ControllerConfig.support_threshold)."""
    out = {}
    for aid, row in reflections.items():
        ranked = sorted(row.items(), key=lambda kv: -kv[1])
        out[aid] = {"best_doc": ranked[0][0] if ranked else None,
                    "best_score": ranked[0][1] if ranked else 0.0,
                    "supporting_docs": [d for d, s in ranked if s >= threshold],
                    "supported": bool(ranked and ranked[0][1] >= threshold)}
    return out