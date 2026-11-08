"""Clean corpus: drop empty docs and duplicate ids, never alter text/ids."""
from data.dataset_loader import load_documents, load_questions

_CORPUS_CACHE = {}


def clean_corpus(docs: list):
    seen, clean = set(), []
    report = {"total": len(docs), "empty_removed": 0, "duplicate_ids_removed": 0}
    for d in docs:
        did, text = str(d["doc_id"]), d["text"]
        if not isinstance(text, str) or not text.strip():
            report["empty_removed"] += 1
            continue
        if did in seen:
            report["duplicate_ids_removed"] += 1
            continue
        seen.add(did)
        clean.append({"doc_id": did, "text": text})  # original text preserved
    report["final"] = len(clean)
    return clean, report


def get_clean_corpus(task: str) -> list:
    if task not in _CORPUS_CACHE:
        _CORPUS_CACHE[task], _ = clean_corpus(load_documents(task))
    return _CORPUS_CACHE[task]


def gold_coverage(task: str):
    """Check that gold ids exist in the cleaned corpus."""
    ids = {d["doc_id"] for d in get_clean_corpus(task)}
    gold = {g for q in load_questions(task) for g in q["gold_ids"]}
    return {"gold_total": len(gold), "gold_missing": len(gold - ids)}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="biology")
    a = p.parse_args()
    _, rep = clean_corpus(load_documents(a.task))
    print(a.task, rep, gold_coverage(a.task))