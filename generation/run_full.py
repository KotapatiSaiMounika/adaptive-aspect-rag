"""Full system: M3+M4 adaptive retrieval loop -> M5 cited answer + verification (+ feedback round)."""
import argparse, json, os, time

from controller import AdaptiveController, ControllerConfig, LexicalScorer
from query_reformulation.controller_adapter import retrieve_for_controller
from data.dataset_loader import load_questions
from data.preprocessing import get_clean_corpus
from retrieval.retrieval_interface import retrieve
from generation.pipeline import generate_and_verify
from generation.answer_generator import norm_aspects


def is_daily_limit(e):
    m = str(e)
    return "per day" in m or "TPD" in m


def run_question(q, a, scorer, cfg):
    gold = set(q["gold_ids"])
    init = retrieve(q["query"], a.task, method="bm25", top_k=a.init_k)
    ev = [{"doc_id": r["doc_id"], "text": r["text"]} for r in init]
    init_ids = [e["doc_id"] for e in ev]
    ctrl = AdaptiveController(q["aspects"], cfg, scorer)
    d = ctrl.decide(ev)
    targets, history = [], []
    calls = [1]  # initial retrieval

    def expand(aspect_id, aspect_text, hints):
        res = retrieve_for_controller(
            aspect_id=aspect_id, aspect_text=aspect_text, task=a.task,
            original_question=q["query"], previous_evidence=ev,
            previous_queries=history, hints=hints, method="bm25")
        history.extend(res["queries"])
        if res["queries"]:
            calls[0] += 1
        have = {str(e["doc_id"]) for e in ev}
        new = [x for x in res["documents"] if str(x["doc_id"]) not in have]
        ev.extend(new)
        return len(new)

    # M3 + M4 adaptive retrieval loop (same as controller/run_all.py)
    while d.action == "retrieve":
        targets.append(d.weakest_aspect)
        expand(d.weakest_aspect, d.weakest_aspect_text, d.hints)
        d = ctrl.decide(ev)
    summary = ctrl.summary()

    def gen():
        sel = ev[:a.init_k] + ev[a.init_k:][:6]  # initial docs + up to 6 newly retrieved
        sel = [{"doc_id": e["doc_id"], "text": e["text"]} for e in sel]
        return generate_and_verify(q["query"], sel, aspects=q["aspects"],
                                   max_revisions=a.max_revisions), sel

    res, sel = gen()

    # M5 -> M4 feedback: search again for aspects the answer left uncovered
    for _ in range(a.feedback_rounds):
        rep = res["report"]
        if not rep["needs_retrieval"] or not rep["uncovered_aspects"]:
            break
        texts = {x["id"]: x["text"] for x in norm_aspects(q["aspects"])}
        added = 0
        for aid in rep["uncovered_aspects"][:2]:
            if texts.get(aid):
                added += expand(aid, texts[aid],
                                {"top_k": 5, "exclude_doc_ids": [e["doc_id"] for e in ev]})
        if not added:
            break
        new_res, new_sel = gen()
        if new_res["report"]["confidence"] >= res["report"]["confidence"]:
            res, sel = new_res, new_sel

    final_ids = [e["doc_id"] for e in ev]
    return {
        "system": "full", "task": a.task, "qid": q["qid"], "query": q["query"],
        "answer": res["answer"]["text"], "claims": res["answer"]["claims"],
        "report": res["report"], "evidence_ids": [e["doc_id"] for e in sel],
        "n_retrieval_calls": calls[0], "targeted_aspects": targets,
        "gold_recall_initial": len(gold & set(init_ids)) / max(len(gold), 1),
        "gold_recall_final": len(gold & set(final_ids)) / max(len(gold), 1),
        "controller": summary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="biology")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--init_k", type=int, default=6)
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--feedback_rounds", type=int, default=1)
    ap.add_argument("--max_revisions", type=int, default=0)
    ap.add_argument("--pause", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"experiments/generation/full_{a.task}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    cfg = ControllerConfig(coverage_target=a.target, support_threshold=0.38,
                           top_m_docs=1, patience=a.patience)
    scorer = LexicalScorer().fit([d["text"] for d in get_clean_corpus(a.task)])
    done = set()
    if os.path.exists(out):
        done = {str(json.loads(l)["qid"]) for l in open(out) if l.strip()}

    failed = 0
    with open(out, "a") as f:
        for q in load_questions(a.task)[:a.limit]:
            if str(q["qid"]) in done:
                continue
            if not q["aspects"]:
                print("skip (no aspects)", q["qid"])
                continue
            rec = None
            for attempt in range(3):
                try:
                    rec = run_question(q, a, scorer, cfg)
                    break
                except Exception as e:
                    print("FAILED", q["qid"], type(e).__name__, str(e)[:160])
                    if is_daily_limit(e):
                        print("Daily limit reached: stopping. Re-run later or switch model.")
                        return
                    time.sleep(65)
            if rec is None:
                failed += 1
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print("done", q["qid"], "calls:", rec["n_retrieval_calls"])
            time.sleep(a.pause)
    print(f"finished; {failed} failed. Re-run the same command to retry the failed ones.")


if __name__ == "__main__":
    main()
