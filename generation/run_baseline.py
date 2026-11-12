import argparse, json, os, time
from data.dataset_loader import load_questions
from retrieval.retrieval_interface import retrieve
from generation.pipeline import generate_and_verify


def is_daily_limit(e):
    m = str(e)
    return "per day" in m or "TPD" in m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="biology")
    ap.add_argument("--method", default="dense", choices=["bm25", "dense"])
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--pause", type=float, default=8.0, help="seconds between questions")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"experiments/generation/baseline_{a.method}_{a.task}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["qid"] for l in open(out) if l.strip()}
    failed = 0
    with open(out, "a") as f:
        for q in load_questions(a.task)[:a.limit]:
            if q["qid"] in done:
                continue
            res = None
            for attempt in range(3):
                try:
                    docs = retrieve(q["query"], task=a.task, method=a.method, top_k=a.top_k)
                    ev = [{"doc_id": d["doc_id"], "text": d["text"]} for d in docs]
                    res = generate_and_verify(q["query"], ev, aspects=None, max_revisions=0)
                    break
                except Exception as e:
                    print("FAILED", q["qid"], type(e).__name__, str(e)[:160])
                    if is_daily_limit(e):
                        print("Daily limit reached: stopping. Re-run tomorrow or switch model.")
                        return
                    time.sleep(65)  # per-minute limit: wait for the window to reset
            if res is None:
                failed += 1
                continue
            f.write(json.dumps({
                "system": f"baseline_{a.method}", "task": a.task, "qid": q["qid"],
                "query": q["query"], "answer": res["answer"]["text"],
                "claims": res["answer"]["claims"], "report": res["report"],
                "evidence_ids": [d["doc_id"] for d in ev], "n_retrieval_calls": 1,
            }) + "\n")
            f.flush()
            print("done", q["qid"])
            time.sleep(a.pause)
    print(f"finished; {failed} failed. Re-run the same command to retry the failed ones.")


if __name__ == "__main__":
    main()
