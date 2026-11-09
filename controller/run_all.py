
"""Run the adaptive controller on all questions and save results.

Usage:
    python -m controller.run_all --limit 10
    python -m controller.run_all --tasks biology --limit 2
    python -m controller.run_all
"""

import argparse
import json
from pathlib import Path

from controller import (
    AdaptiveController,
    ControllerConfig,
    LexicalScorer,
)

from query_reformulation.controller_adapter import (
    retrieve_for_controller,
)

from data.dataset_loader import TASKS, load_questions
from data.preprocessing import get_clean_corpus
from retrieval.retrieval_interface import retrieve


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

ap = argparse.ArgumentParser(
    description="Run adaptive aspect retrieval experiments."
)

ap.add_argument(
    "--tasks",
    nargs="+",
    default=TASKS,
    choices=TASKS,
)

ap.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Maximum questions per task.",
)

ap.add_argument(
    "--out",
    default="experiments/controller/controller_results.jsonl",
)

ap.add_argument(
    "--target",
    type=float,
    default=0.5,
)

ap.add_argument(
    "--patience",
    type=int,
    default=1,
)

a = ap.parse_args()


# ---------------------------------------------------------
# Output and controller configuration
# ---------------------------------------------------------

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)

cfg = ControllerConfig(
    coverage_target=a.target,
    support_threshold=0.38,
    top_m_docs=1,
    patience=a.patience,
)


# ---------------------------------------------------------
# Run experiments
# ---------------------------------------------------------

with out.open("w", encoding="utf-8") as f:

    for task in a.tasks:

        corpus = get_clean_corpus(task)

        scorer = LexicalScorer().fit(
            [doc["text"] for doc in corpus]
        )

        questions = [
            q for q in load_questions(task)
            if q["aspects"]
        ]

        if a.limit is not None:
            questions = questions[:a.limit]

        n_rounds = 0

        for q in questions:

            # Gold relevant document IDs
            gold = set(q["gold_ids"])

            # Initial retrieval
            initial_results = retrieve(
                q["query"],
                task,
                method="bm25",
                top_k=10,
            )

            ev = [
                {
                    "doc_id": r["doc_id"],
                    "text": r["text"],
                }
                for r in initial_results
            ]

            init_ids = [
                d["doc_id"] for d in ev
            ]

            # Initialize adaptive controller
            ctrl = AdaptiveController(
                q["aspects"],
                cfg,
                scorer,
            )

            d = ctrl.decide(ev)

            targets = []
            query_history = []

            # -------------------------------------------------
            # Adaptive retrieval loop - Member 3 + Member 4
            # -------------------------------------------------

            while d.action == "retrieve":

                targets.append(d.weakest_aspect)

                # Member 3 retrieves evidence for weakest aspect
                result = retrieve_for_controller(
                    aspect_id=d.weakest_aspect,
                    aspect_text=d.weakest_aspect_text,
                    task=task,
                    original_question=q["query"],
                    previous_evidence=ev,
                    previous_queries=query_history,
                    hints=d.hints,
                    method="bm25",
                )

                # Maintain query history
                query_history.extend(
                    result["queries"]
                )

                new = result["documents"]

                # Avoid adding documents already in evidence
                have = {
                    str(e["doc_id"])
                    for e in ev
                }

                new_documents = [
                    x for x in new
                    if str(x["doc_id"]) not in have
                ]

                ev.extend(new_documents)

                # Member 4 recalculates aspect coverage
                d = ctrl.decide(ev)

            # -------------------------------------------------
            # Save results for this question
            # -------------------------------------------------

            ids = [
                e["doc_id"] for e in ev
            ]

            n_rounds += ctrl.rounds_used

            result_record = {
                "task": task,
                "qid": q["qid"],
                **ctrl.summary(),
                "targeted_aspects": targets,
                "initial_doc_ids": init_ids,
                "final_doc_ids": ids,
                "gold_recall_initial": (
                    len(gold & set(init_ids))
                    / max(len(gold), 1)
                ),
                "gold_recall_final": (
                    len(gold & set(ids))
                    / max(len(gold), 1)
                ),
                "target": a.target,
                "patience": a.patience,
            }

            f.write(
                json.dumps(result_record) + "\n"
            )

        # Summary for each task
        print(
            f"{task:20s} "
            f"questions={len(questions):4d} "
            f"avg_rounds="
            f"{n_rounds / max(len(questions), 1):.2f}"
        )


# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------

print("Saved", out)
