from generation import llm
from generation.prompts import SUPPORT_SYSTEM, UTILITY_SYSTEM, MAX_DOC_CHARS
from generation.answer_generator import norm_aspects

SCORE = {"fully": 1.0, "partial": 0.5, "none": 0.0}

def check_support(claims, evidence, batch=8):
    """Returns a label per claim. Uncited claims are 'none' without an LLM call."""
    by_id = {d["doc_id"]: d["text"] for d in evidence}
    labels = ["none"] * len(claims)
    todo = [i for i, c in enumerate(claims) if c["cited"]]
    for s in range(0, len(todo), batch):
        chunk, blocks = todo[s:s + batch], []
        for i in chunk:
            ctx = "\n".join(f"[{d}] {by_id[d][:MAX_DOC_CHARS]}" for d in claims[i]["cited"] if d in by_id)
            blocks.append(f"CLAIM {i}: {claims[i]['text']}\nCITED EVIDENCE:\n{ctx}")
        res = llm.chat_json(SUPPORT_SYSTEM, "\n\n".join(blocks))
        for item in (res or {}).get("labels", []):
            try:
                if item["label"] in SCORE:
                    labels[int(item["id"])] = item["label"]
            except (KeyError, ValueError, IndexError):
                pass
    return labels

def judge_answer(question, answer_text, aspects, model=None):
    """Returns {'utility': 1-5, 'covered': {aspect_id: bool}, 'covered_frac': float}."""
    asp = norm_aspects(aspects)
    alist = "\n".join(f"{i}: {a['text']}" for i, a in enumerate(asp)) or "(none given)"
    res = llm.chat_json(UTILITY_SYSTEM,
                        f"QUESTION:\n{question}\n\nASPECTS:\n{alist}\n\nANSWER:\n{answer_text}",
                        model=model) or {}
    try:
        utility = max(1, min(5, int(res.get("utility", 1))))
    except (TypeError, ValueError):
        utility = 1
    covered = {a["id"]: False for a in asp}
    for item in res.get("aspects", []):
        try:
            covered[asp[int(item["id"])]["id"]] = bool(item["covered"])
        except (KeyError, ValueError, IndexError):
            pass
    frac = sum(covered.values()) / len(covered) if covered else None
    return {"utility": utility, "covered": covered, "covered_frac": frac}