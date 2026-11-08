"""Print dataset structure so you can verify field names."""
import argparse
import json
from collections import Counter

from data.dataset_loader import TASKS, _read, load_documents, load_questions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="biology", choices=TASKS)
    a = p.parse_args()

    for config in ["examples", "documents", "aspects"]:
        try:
            rows = _read(config, a.task)
            print(f"\n=== {config}: {len(rows)} rows ===")
            print("fields:", list(rows[0].keys()))
            print(json.dumps(rows[0], indent=2, default=str)[:800])
        except Exception as e:
            print(f"\n=== {config}: FAILED -> {e}")

    qs, docs = load_questions(a.task), load_documents(a.task)
    ids = [d["doc_id"] for d in docs]
    dups = [i for i, c in Counter(ids).items() if c > 1]
    empty = sum(1 for d in docs if not str(d["text"]).strip())
    print(f"\nQuestions: {len(qs)} | Documents: {len(docs)}")
    print(f"Duplicate doc ids: {len(dups)} | Empty docs: {empty}")
    print(f"Questions with gold ids: {sum(1 for q in qs if q['gold_ids'])}")
    print(f"Questions with aspects:  {sum(1 for q in qs if q['aspects'])}")
    print("Sample question:", qs[0]["query"][:200])


if __name__ == "__main__":
    main()