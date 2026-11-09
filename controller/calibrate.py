"""Find a score threshold that separates gold docs from non-gold retrieved docs.
   python -m controller.calibrate --task biology --n 15 --scorer dense
"""
import argparse

from controller import DenseScorer, LexicalScorer
from controller.scoring import normalize_aspects
from data.dataset_loader import load_questions
from data.preprocessing import get_clean_corpus
from retrieval.retrieval_interface import retrieve

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="biology")
ap.add_argument("--n", type=int, default=15)
ap.add_argument("--scorer", choices=["lexical", "dense"], default="dense")
a = ap.parse_args()

corpus = get_clean_corpus(a.task)
by_id = {d["doc_id"]: d for d in corpus}
scorer = DenseScorer() if a.scorer == "dense" else LexicalScorer().fit([d["text"] for d in corpus])

pos, neg = [], []
for q in [q for q in load_questions(a.task) if q["aspects"] and q["gold_ids"]][: a.n]:
    aspects = normalize_aspects(q["aspects"])
    gold = [by_id[g] for g in q["gold_ids"] if g in by_id]
    retrieved = retrieve(q["query"], a.task, "bm25", 10)
    non_gold = [r for r in retrieved if r["doc_id"] not in set(q["gold_ids"])]
    # a doc's score = its best score over the question's aspects
    best = lambda d: max(scorer.score(x["text"], d["text"]) for x in aspects)
    pos += [best(d) for d in gold]
    neg += [best(d) for d in non_gold]

mean = lambda v: sum(v) / max(len(v), 1)
print(f"gold docs: n={len(pos)} mean={mean(pos):.3f} | non-gold: n={len(neg)} mean={mean(neg):.3f}")

best_t, best_bal = 0, 0
for i in range(0, 101):
    t = i / 100
    bal = (sum(p >= t for p in pos) / max(len(pos), 1) + sum(n < t for n in neg) / max(len(neg), 1)) / 2
    if bal > best_bal:
        best_t, best_bal = t, bal
print(f"best threshold = {best_t:.2f} (balanced accuracy {best_bal:.3f})")