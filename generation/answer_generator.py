import re
from generation import llm
from generation.prompts import ANSWER_SYSTEM, REVISE_SYSTEM, MAX_DOC_CHARS

CITE = re.compile(r"\[([^\[\]]+)\]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\-\*])")
NO_INFO = re.compile(
    r"(do|does) not (cover|discuss|address|mention|contain)|none of the (provided|retrieved)"
    r"|not enough information|no information|no evidence|untested", re.I)
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")

def _norm_id(x):
    x = x.translate(_DASHES).replace("\u00a0", " ")
    x = re.sub(r"\.(txt|md|json)$", "", x.strip(), flags=re.I)
    return re.sub(r"[^A-Za-z0-9]", "", x).lower()


def _cited_ids(s, valid_ids):
    lookup = {_norm_id(v): v for v in valid_ids}
    out = []
    for grp in CITE.findall(s):
        for part in re.split(r"[;,]\s*", grp):
            key = _norm_id(part)
            if key in lookup:
                out.append(lookup[key])
    return list(dict.fromkeys(out))

def norm_aspects(aspects):
    """Accepts None, list[str], or list[{'id','text'}] -> list[{'id','text'}]."""
    out = []
    for i, a in enumerate(aspects or []):
        if isinstance(a, dict):
            out.append({"id": str(a.get("id", i)),
                        "text": a.get("text") or a.get("content") or a.get("description") or a.get("aspect") or str(a)})
        else:
            out.append({"id": str(i), "text": str(a)})
    return out

def extract_claims(text, valid_ids):
    claims = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*\u2022])\s+", "", line.strip())
        if not line or line.startswith("#") or re.fullmatch(r"\*\*[^*]+\*\*:?", line):
            continue  # blank lines and headings are not claims
        line_claims = []
        for sent in SENT_SPLIT.split(line):
            sent = sent.strip(" -*#\u2022\t")
            cited = _cited_ids(sent, valid_ids)
            body = CITE.sub("", sent).strip(" -*#\u2022\t")
            if not body:  # citation-only fragment: belongs to the previous claim
                if cited and line_claims:
                    line_claims[-1]["cited"] = list(dict.fromkeys(line_claims[-1]["cited"] + cited))
                continue
            if len(body.split()) < 5 or NO_INFO.search(body):
                continue
            line_claims.append({"text": body, "cited": cited})
        line_ids = list(dict.fromkeys(i for c in line_claims for i in c["cited"]))
        for c in line_claims:
            if not c["cited"]:
                c["cited"] = list(line_ids)  # a citation at the end of a bullet covers the bullet
        claims.extend(line_claims)
    return claims


def _dedupe(evidence, max_docs):
    seen, out = set(), []
    for d in evidence:
        if d["doc_id"] not in seen:
            seen.add(d["doc_id"])
            out.append(d)
    return out[:max_docs]

def _context(docs):
    return "\n\n".join(f"[{d['doc_id']}]\n{d['text'][:MAX_DOC_CHARS]}" for d in docs)

def _pack(text, docs):
    ids = {d["doc_id"] for d in docs}
    return {"text": text, "claims": extract_claims(text, ids), "doc_ids": list(ids)}

def answer(question, evidence, aspects=None, max_docs=12):
    docs = _dedupe(evidence, max_docs)
    asp = norm_aspects(aspects)
    block = ("\n\nMake sure the answer addresses these aspects:\n"
             + "\n".join(f"- {a['text']}" for a in asp)) if asp else ""
    user = f"DOCUMENTS:\n{_context(docs)}\n\nQUESTION:\n{question}{block}"
    text = llm.chat(ANSWER_SYSTEM, user, max_tokens=4000)
    if not text.strip():
        raise RuntimeError("LLM returned an empty answer; check LLM_MODEL / max_tokens / prompt size")
    return _pack(text, docs)

def revise(question, ans, report, evidence, max_docs=12):
    docs = _dedupe(evidence, max_docs)
    bad = "\n".join(f"- {c}" for c in report["unsupported_claims"])
    user = (f"DOCUMENTS:\n{_context(docs)}\n\nQUESTION:\n{question}\n\n"
            f"DRAFT ANSWER:\n{ans['text']}\n\nUNSUPPORTED CLAIMS:\n{bad}")
    return _pack(llm.chat(REVISE_SYSTEM, user, max_tokens=4000), docs)
