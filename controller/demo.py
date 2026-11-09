"""Run Member 4's controller ALONE: no other member's code needed.

  python -m controller.demo                                   # toy data
  python -m controller.demo --task biology --n 5 --scorer lexical --target 0.38
"""
import argparse
import json

from controller import AdaptiveController, ControllerConfig, LexicalScorer, DenseScorer

TOY_CORPUS = [
    {"doc_id": "d1", "text": "Birds migrate south in winter to find food and warmth."},
    {"doc_id": "d2", "text": "Magnetoreception lets birds sense the Earth magnetic field for navigation."},
    {"doc_id": "d3", "text": "Fat storage fuels long migration flights; birds gain weight before departure."},
    {"doc_id": "d4", "text": "The stock market fell after the interest rate decision."},
]
TOY_ASPECTS = ["why birds migrate in winter", "how birds navigate using magnetic field",
               "energy fat storage for long flights"]


def stub_member3_retrieve(corpus, aspect_text, hints):
    """Stand-in for Member 3: rank unseen docs by word overlap with the aspect."""
    q = set(aspect_text.lower().split())
    seen = set(hints["exclude_doc_ids"])
    cand = [d for d in corpus if d["doc_id"] not in seen]
    cand.sort(key=lambda d: -len(q & set(d["text"].lower().split())))
    return cand[:2]


def run(aspects, corpus, evidence, cfg, scorer=None):
    scorer = scorer or LexicalScorer().fit([d["text"] for d in corpus])
    ctrl = AdaptiveController(aspects, cfg, scorer)
    d = ctrl.decide(evidence)
    while d.action == "retrieve":
        print(f"[round {ctrl.rounds_used}] {d.reason}")
        new = stub_member3_retrieve(corpus, d.weakest_aspect_text, d.hints)
        have = {e["doc_id"] for e in evidence}
        evidence = evidence + [x for x in new if x["doc_id"] not in have]
        d = ctrl.decide(evidence)
    print(f"STOP: {d.stop_reason} - {d.reason}")
    print(json.dumps(ctrl.summary(), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--scorer", choices=["lexical", "dense"], default="lexical")
    ap.add_argument("--target", type=float, default=0.38)
    a = ap.parse_args()
    cfg = ControllerConfig(coverage_target=a.target, support_threshold=a.target, top_m_docs=1)
    scorer = DenseScorer() if a.scorer == "dense" else None
    if not a.task:
        run(TOY_ASPECTS, TOY_CORPUS, TOY_CORPUS[:1], cfg, scorer)
    else:
        from data.dataset_loader import load_questions
        from data.preprocessing import get_clean_corpus
        from retrieval.retrieval_interface import retrieve
        corpus = get_clean_corpus(a.task)
        for q in [q for q in load_questions(a.task) if q["aspects"]][: a.n]:
            print("\nQ:", q["query"][:100])
            init = [{"doc_id": r["doc_id"], "text": r["text"]}
                    for r in retrieve(q["query"], a.task, "bm25", 10)]
            run(q["aspects"], corpus, init, cfg, scorer)