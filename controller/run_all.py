"""Run the controller on all questions and save results.
   python -m controller.run_all --limit 10          # quick test
   python -m controller.run_all                     # all domains, all questions
"""
import argparse
import json
from pathlib import Path

from controller import AdaptiveController, ControllerConfig, LexicalScorer
from controller.demo import stub_member3_retrieve   # replace with Member 3's retriever later
from data.dataset_loader import TASKS, load_questions
from data.preprocessing import get_clean_corpus
from retrieval.retrieval_interface import retrieve

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
ap.add_argument("--limit", type=int, default=None, help="questions per task")
ap.add_argument("--out", default="experiments/controller/controller_results.jsonl")
ap.add_argument("--target", type=float, default=0.5)
ap.add_argument("--patience", type=int, default=1)
a = ap.parse_args()

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
cfg = ControllerConfig(coverage_target=a.target, support_threshold=0.38,
                       top_m_docs=1, patience=a.patience)

with out.open("w", encoding="utf-8") as f:
    for task in a.tasks:
        corpus = get_clean_corpus(task)
        scorer = LexicalScorer().fit([d["text"] for d in corpus])
        qs = [q for q in load_questions(task) if q["aspects"]][: a.limit]
        n_rounds = 0
        for q in qs:
            gold = set(q["gold_ids"])
            ev = [{"doc_id": r["doc_id"], "text": r["text"]}
                  for r in retrieve(q["query"], task, "bm25", 10)]
            init_ids = [d["doc_id"] for d in ev]
            ctrl = AdaptiveController(q["aspects"], cfg, scorer)
            d = ctrl.decide(ev)
            targets = []
            while d.action == "retrieve":
                targets.append(d.weakest_aspect)
                new = stub_member3_retrieve(corpus, d.weakest_aspect_text, d.hints)
                have = {e["doc_id"] for e in ev}
                ev = ev + [x for x in new if x["doc_id"] not in have]
                d = ctrl.decide(ev)
            ids = [e["doc_id"] for e in ev]
            n_rounds += ctrl.rounds_used
            f.write(json.dumps({
                "task": task, "qid": q["qid"], **ctrl.summary(),
                "targeted_aspects": targets,
                "initial_doc_ids": init_ids, "final_doc_ids": ids,
                "gold_recall_initial": len(gold & set(init_ids)) / max(len(gold), 1),
                "gold_recall_final": len(gold & set(ids)) / max(len(gold), 1),
                "target": a.target, "patience": a.patience,
            }) + "\n")
        print(f"{task:20s} questions={len(qs):4d} avg_rounds={n_rounds / max(len(qs), 1):.2f}")
print("Saved", out)