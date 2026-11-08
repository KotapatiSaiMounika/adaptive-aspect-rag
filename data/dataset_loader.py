"""Load Bright-Pro questions, documents and aspects."""
import json
import os
from pathlib import Path

HF_REPO = "yale-nlp/Bright-Pro"
TASKS = ["biology", "earth_science", "economics", "psychology",
         "robotics", "stackoverflow", "sustainable_living"]

ID_KEYS = ["id", "doc_id", "qid", "query_id"]
TEXT_KEYS = ["content", "text", "document"]
QUERY_KEYS = ["query", "question"]
GOLD_KEYS = ["gold_ids", "gold_doc_ids", "positive_ids", "gold_passages"]


def _first(rec, keys, default=None):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def _read(config: str, task: str) -> list:
    """config is one of: examples, documents, aspects."""
    root = os.environ.get("BRIGHT_PRO_DATA_ROOT")
    if root:  # offline mirror: <root>/<config>/<task>.json
        with open(Path(root) / config / f"{task}.json", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "value": v}
                    for k, v in data.items()]
        return data
    from datasets import load_dataset
    return [dict(r) for r in load_dataset(HF_REPO, config)[task]]


def load_documents(task: str) -> list:
    """Returns [{'doc_id': str, 'text': str}] with original ids/text kept."""
    docs = []
    for r in _read("documents", task):
        docs.append({"doc_id": str(_first(r, ID_KEYS)),
                     "text": _first(r, TEXT_KEYS, "")})
    return docs


def load_questions(task: str) -> list:
    """Returns [{'qid','query','gold_ids','aspects','raw'}]."""
    aspects_by_id = {}

    try:
        for r in _read("aspects", task):
            aspect_id = str(_first(r, ID_KEYS))
            prefix = f"{task}-"

            if aspect_id.startswith(prefix):
                qid = aspect_id[len(prefix):].split("-a")[0]
                aspects_by_id.setdefault(qid, []).append(r)

    except Exception as e:
        print(f"[warn] could not load aspects for {task}: {e}")

    questions = []
    for r in _read("examples", task):
        qid = str(_first(r, ID_KEYS))
        gold = _first(r, GOLD_KEYS, [])
        if isinstance(gold, str):
            gold = [gold]
        questions.append({
            "qid": qid,
            "query": _first(r, QUERY_KEYS, ""),
            "gold_ids": [str(g) for g in gold],
            "aspects": r.get("aspects") or aspects_by_id.get(qid),
            "raw": r,
        })
    return questions