import argparse
import json
import time
from pathlib import Path

from data.dataset_loader import TASKS, load_questions
from retrieval.retrieval_interface import retrieve

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["bm25", "dense"])
    p.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS,
                   help="default: all 7 domains")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="only first N questions per task")
    p.add_argument("--out", default=None,
                   help="output file (default: experiments/baselines/<method>_results.jsonl)")
    a = p.parse_args()

    out = Path(a.out) if a.out else ROOT / "experiments" / "baselines" / f"{a.method}_results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    hits = total = 0

    with out.open("w", encoding="utf-8") as f:
        for task in a.tasks:
            questions = load_questions(task)[: a.limit]
            retrieve("warm up", task, a.method, 1)  # load index before timing
            t_hits = t_total = 0
            for q in questions:
                t0 = time.perf_counter()
                res = retrieve(q["query"], task, a.method, a.top_k)
                elapsed = time.perf_counter() - t0
                ids = [r["doc_id"] for r in res]
                gold = set(q["gold_ids"])
                if gold:
                    t_hits += len(gold & set(ids))
                    t_total += len(gold)
                f.write(json.dumps({
                    "task": task, "qid": q["qid"], "method": a.method,
                    "top_k": a.top_k,
                    "retrieved_doc_ids": ids,
                    "scores": [r["score"] for r in res],
                    "num_retrieved": len(res),
                    "retrieval_time_sec": round(elapsed, 4),
                    "gold_doc_ids": q["gold_ids"],
                }) + "\n")
            hits += t_hits
            total += t_total
            print(f"  {task:20s} questions={len(questions):4d} "
                  f"recall@{a.top_k}={t_hits / max(t_total, 1):.3f}")

    print(f"Saved {out} | overall gold recall@{a.top_k} = {hits / max(total, 1):.3f}")


if __name__ == "__main__":
    main()