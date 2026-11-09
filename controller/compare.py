"""One table: baselines vs controller settings (same questions, per-question macro recall).
   python -m controller.compare
"""
import json
from pathlib import Path

def load(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8")]

mean = lambda v: sum(v) / max(len(v), 1)

ctrl_files = {
    "controller t=0.38":          "experiments/controller/t038.jsonl",
    "controller t=0.5 (default)": "experiments/controller/controller_results.jsonl",
    "controller t=0.6":           "experiments/controller/t060.jsonl",
    "controller t=0.5 patience=2":"experiments/controller/p2.jsonl",
}
base_files = {
    "BM25 top-10": "experiments/baselines/bm25_results.jsonl",
    "BM25 top-13": "experiments/baselines/bm25_k13.jsonl",
}

default = load(ctrl_files["controller t=0.5 (default)"])
keys = {(r["task"], r["qid"]) for r in default}   # same questions for everyone

print(f"{'method':30s} {'n':>4s} {'docs':>6s} {'rounds':>6s} {'recall':>7s}")
for name, p in base_files.items():
    rs = [r for r in load(p) if (r["task"], r["qid"]) in keys and r["gold_doc_ids"]]
    rec = [len(set(r["gold_doc_ids"]) & set(r["retrieved_doc_ids"])) / len(r["gold_doc_ids"]) for r in rs]
    print(f"{name:30s} {len(rs):4d} {mean([r['num_retrieved'] for r in rs]):6.1f} {'-':>6s} {mean(rec):7.3f}")
for name, p in ctrl_files.items():
    rs = [r for r in load(p) if (r["task"], r["qid"]) in keys]
    print(f"{name:30s} {len(rs):4d} {mean([r['n_docs'] for r in rs]):6.1f} "
          f"{mean([r['rounds_used'] for r in rs]):6.2f} {mean([r['gold_recall_final'] for r in rs]):7.3f}")