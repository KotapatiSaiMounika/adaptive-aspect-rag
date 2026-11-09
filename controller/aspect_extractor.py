"""Member 2 - Aspect extraction.

Turns a complex question into a small list of independent, retrievable aspects.

Design (avoids duplicating existing work):
  1. Bright-Pro already ships gold aspects (ids like ``biology-0-a1``) and Member 1's
     ``load_questions()`` attaches them as ``q["aspects"]``.  If they exist we REUSE them
     (via Member 4's ``normalize_aspects``) instead of re-extracting.
  2. If a question has no aspects (new / user-typed questions), we extract them:
       - ``llm_fn`` given  -> LLM extraction (any callable ``prompt:str -> str``), validated,
                              with automatic fallback to (3) on any failure.
       - otherwise         -> deterministic heuristic extractor (no model, no network).

Output is a list of ``Aspect``.  ``Aspect.to_dict()`` is directly accepted by
``AdaptiveController(aspects, ...)`` and ``normalize_aspects``  ->  {"id", "description", "text", ...}.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from controller.scoring import normalize_aspects, tokenize

LLMFn = Callable[[str], str]

# --------------------------------------------------------------------------- data model


@dataclass
class Aspect:
    id: str                                   # unique per question, e.g. "biology-0-a1"
    description: str                          # self-contained sub-question / information need
    keywords: List[str] = field(default_factory=list)
    source: str = "heuristic"                 # dataset | heuristic | llm

    def to_dict(self) -> Dict:
        # "text" duplicates "description" so every consumer (controller, scoring, Member 3/5) works.
        return {"id": self.id, "description": self.description, "text": self.description,
                "keywords": list(self.keywords), "source": self.source}

    def search_query(self, extra_terms: Iterable[str] = ()) -> str:
        """Query string for Member 3: the aspect description + optional controller hint terms
        (e.g. ``decision.hints["missing_terms"]``)."""
        extra = [t for t in extra_terms if t and t.lower() not in self.description.lower()]
        return " ".join([self.description] + extra).strip()


def to_dicts(aspects: Iterable[Aspect]) -> List[Dict]:
    """Controller-ready format: ``AdaptiveController(to_dicts(aspects))``."""
    return [a.to_dict() for a in aspects]


def make_aspect_id(n: int, qid: Optional[str] = None, task: Optional[str] = None) -> str:
    """1-based, mirrors Bright-Pro: ``<task>-<qid>-a<n>`` (parts omitted if unknown)."""
    parts = [p for p in (task, qid) if p not in (None, "")]
    return "-".join(map(str, parts + [f"a{n}"]))


def top_keywords(text: str, n: int = 6) -> List[str]:
    """Up to n distinctive content words (longer words first), kept in reading order."""
    uniq = list(dict.fromkeys(tokenize(text)))
    keep = set(sorted(uniq, key=lambda w: -len(w))[:n])
    return [w for w in uniq if w in keep]


# --------------------------------------------------------------------------- helpers

_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_URL = re.compile(r"https?://\S+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_INTERROG = ("how", "why", "what", "whether", "which", "when", "where", "who", "is", "are",
             "does", "do", "can", "could", "should", "would", "will", "any")
_CLAUSE_SPLIT = re.compile(
    r";|\b(?:and also|as well as|in addition to|along with)\b|"
    r",\s*(?:and|or)\s+(?=(?:" + "|".join(_INTERROG) + r")\b)", re.I)


def _clean(text: str) -> str:
    text = _CODE_BLOCK.sub(" ", text or "")
    text = _URL.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


_LEAD = re.compile(r"^(?:(?:also|and|but|plus|so|additionally|finally|then|secondly|lastly)\b[ ,]*)+", re.I)


def _is_question(clause: str) -> bool:
    c = clause.strip().lower()
    return c.endswith("?") or c.split(" ", 1)[0].strip(",") in _INTERROG


def _truncate_words(text: str, max_words: int) -> str:
    w = text.split()
    return text if len(w) <= max_words else " ".join(w[:max_words])


def _build(descs: List[str], source: str, qid, task, max_aspects: int) -> List[Aspect]:
    out, seen = [], []
    for d in descs:
        d = re.sub(r"\s+", " ", str(d)).strip()
        if not d or any(_jaccard(d, s) >= 0.7 for s in seen):   # drop empties / near-duplicates
            continue
        seen.append(d)
        out.append(Aspect(make_aspect_id(len(out) + 1, qid, task), d, top_keywords(d), source))
        if len(out) >= max_aspects:
            break
    return out


# --------------------------------------------------------------------------- extractors

def heuristic_extract(query: str, qid: Optional[str] = None, task: Optional[str] = None,
                      max_aspects: int = 5, min_tokens: int = 3, max_words: int = 40) -> List[Aspect]:
    """Rule-based: split into sentences/clauses, prefer question-like clauses, rank by
    information content, drop near-duplicates.  Always returns >= 1 aspect for non-empty text."""
    text = _clean(query)
    if not text:
        return []
    cands = []                                   # (position, clause, score)
    for s in _SENT_SPLIT.split(text):
        for c in _CLAUSE_SPLIT.split(s):
            c = _LEAD.sub("", (c or "").strip(" ,.-")).strip(" ,.-")
            n = len(tokenize(c))
            if n < min_tokens:
                continue
            cands.append((len(cands), c, 2.0 * _is_question(c) + min(n, 12) / 12))
    if not cands:                                # nothing passed the filter -> whole question
        return _build([_truncate_words(text, max_words)], "heuristic", qid, task, 1)

    questions = [c for c in cands if _is_question(c[1])]
    pool = questions if len(questions) >= 2 else cands   # one lone '?' is not enough to cover a post
    chosen = []
    for c in sorted(pool, key=lambda c: -c[2]):
        if len(chosen) < max_aspects and all(_jaccard(c[1], x[1]) < 0.7 for x in chosen):
            chosen.append(c)
    chosen.sort(key=lambda c: c[0])              # back to reading order
    return _build([_truncate_words(c[1], max_words) for c in chosen],
                  "heuristic", qid, task, max_aspects)


ASPECT_PROMPT = """Break the question below into {k} or fewer INDEPENDENT aspects: distinct pieces \
of information a complete answer needs. Each aspect must be a short, self-contained \
sub-question or topic (max 25 words) that can be searched on its own, and aspects must not overlap.
Return ONLY JSON: {{"aspects": [{{"description": "...", "keywords": ["..."]}}]}}

Question:
{query}
"""


def _parse_llm(raw: str) -> List[Dict]:
    m = re.search(r"\{.*\}|\[.*\]", raw or "", re.S)   # tolerate prose / ```json fences
    if not m:
        raise ValueError("no JSON found")
    data = json.loads(m.group(0))
    items = data.get("aspects", []) if isinstance(data, dict) else data
    out = []
    for it in items:
        if isinstance(it, str):
            out.append({"description": it})
        elif isinstance(it, dict):
            d = next((it[k] for k in ("description", "text", "aspect") if isinstance(it.get(k), str)), "")
            out.append({"description": d, "keywords": it.get("keywords")})
    return out


def llm_extract(query: str, llm_fn: LLMFn, qid: Optional[str] = None, task: Optional[str] = None,
                max_aspects: int = 5) -> List[Aspect]:
    """LLM extraction; any failure (exception, bad JSON, empty list) falls back to heuristics."""
    try:
        items = _parse_llm(llm_fn(ASPECT_PROMPT.format(k=max_aspects, query=_clean(query)[:6000])))
        aspects = _build([i["description"] for i in items], "llm", qid, task, max_aspects)
        if not aspects:
            raise ValueError("empty extraction")
        by_desc = {i["description"].strip(): i.get("keywords") for i in items}
        for a in aspects:                         # keep LLM keywords when they are a clean list of str
            kw = by_desc.get(a.description)
            if isinstance(kw, list) and kw and all(isinstance(k, str) for k in kw):
                a.keywords = [k.lower() for k in kw][:8]
        return aspects
    except Exception:
        return heuristic_extract(query, qid, task, max_aspects)


def from_dataset(raw_aspects, source: str = "dataset") -> List[Aspect]:
    """Wrap Bright-Pro / user-supplied aspects (any shape normalize_aspects understands)."""
    out, used = [], set()
    for a in normalize_aspects(raw_aspects):
        if not a["text"].strip():
            continue
        aid, k = a["id"], 2
        while aid in used:                        # ids must be unique
            aid, k = f"{a['id']}_{k}", k + 1
        used.add(aid)
        out.append(Aspect(aid, a["text"].strip(), top_keywords(a["text"]), source))
    return out


# --------------------------------------------------------------------------- public entry point

def get_aspects(question: Dict, task: Optional[str] = None, llm_fn: Optional[LLMFn] = None,
                use_dataset: bool = True, max_aspects: int = 5) -> List[Aspect]:
    """Main entry point.  ``question`` is a record from ``data.dataset_loader.load_questions``
    ({'qid','query','aspects',...}) or any dict with 'query' (+ optional 'qid', 'aspects').

    Priority: dataset aspects (if present and use_dataset) -> LLM -> heuristic.
    """
    if use_dataset and question.get("aspects"):
        found = from_dataset(question["aspects"])
        if found:
            return found
    qid = None if question.get("qid") is None else str(question["qid"])
    query = question.get("query") or question.get("question") or ""
    if llm_fn is not None:
        return llm_extract(query, llm_fn, qid, task, max_aspects)
    return heuristic_extract(query, qid, task, max_aspects)


def extract_aspects(query: str, **kw) -> List[Aspect]:
    """Convenience for a bare query string: ``extract_aspects("...", llm_fn=my_llm)``."""
    return get_aspects({"query": query}, **kw)