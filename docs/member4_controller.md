# Aspect Scoring and the Adaptive Retrieval Controller

## Role and interface

Member 4 decides, after every retrieval round, whether the current evidence covers all aspects of the question. If not, it
names the weakest aspect for targeted retrieval; otherwise it stops and says why. Code: `controller/controller.py`,
`controller/scoring.py`, `controller/config.py`.

```python
from controller import AdaptiveController, ControllerConfig, LexicalScorer

ctrl = AdaptiveController(aspects, ControllerConfig(), LexicalScorer())
d = ctrl.decide(evidence)                    # evidence = [{"doc_id", "text"}, ...]
while d.action == "retrieve":
    new_docs = retrieve_for_aspect(d.weakest_aspect_text, d.hints)   # Member 3
    evidence += new_docs
    d = ctrl.decide(evidence)
print(d.stop_reason, ctrl.summary())
```

`python3 -m controller.demo` runs this loop on toy data with no dataset or API key.

## Scoring (`controller/scoring.py`)

All scores are in [0, 1]. Two interchangeable backends share the `score(aspect_text, doc_text)` interface:

- `LexicalScorer`: idf-weighted fraction of aspect terms found in the document. No model or network. Call `.fit(corpus)` to learn idf.
- `DenseScorer`: bge-small cosine similarity, linearly rescaled from [0.45, 0.85] to [0, 1].

`normalize_aspects` accepts a list of strings, a list of dicts, a dict of id to text, or `None`.

## Coverage and decision logic

1. **Score matrix.** Every aspect is scored against every evidence document (cached). Optional reflection scores from Member 2 are blended in with weight `reflection_weight` (0.4).
2. **Coverage.** Per aspect, the noisy-OR over the best `top_m_docs` scores (default 1, so the best document).
3. **Stop checks, in order:** `sufficient` (all aspects reach `coverage_target`), `max_rounds`, `doc_budget`, `token_budget`, `stagnation` (mean coverage gain below `min_gain` for `patience` consecutive rounds).
4. **Target selection.** Among uncovered aspects that have not failed `max_attempts_per_aspect` targeted rounds, pick the lowest coverage (ties: fewest supporting documents). If none remain, stop with `all_aspects_exhausted`.
5. **Hints for Member 3.** `missing_terms` (aspect words absent from all evidence), `top_k`, `exclude_doc_ids`, `verification_triggered`.
6. **Verification feedback.** If Member 5 reports an unsupported answer, the named aspects (or the weakest one) are forced to count as uncovered.

`Decision` fields: `action`, `weakest_aspect`, `weakest_aspect_text`, `stop_reason`, `reason`, `coverage`, `mean_coverage`, `uncovered`, `hints`, `budget`.

## Configuration (`ControllerConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `support_threshold` | 0.38 | Score at which a document counts as supporting an aspect |
| `coverage_target` | 0.5 | Coverage at which an aspect counts as covered |
| `top_m_docs` | 1 | Documents combined in the coverage score |
| `reflection_weight` | 0.4 | Blend of Member 2's reflection (0 = ignore) |
| `max_rounds` | 3 | Maximum additional retrieval rounds |
| `max_docs` | 50 | Maximum documents in the evidence set |
| `max_tokens` | 20000 | Approximate token budget of the evidence |
| `min_gain` | 0.02 | Minimum mean-coverage gain per round |
| `patience` | 1 | Low-gain rounds tolerated before stopping |
| `max_attempts_per_aspect` | 2 | Failed targeted rounds before giving up on an aspect |
| `top_k_per_query` | 10 | Documents requested per targeted query |

## Results (all 739 questions, per-question average gold recall)

| Method | Docs | Extra rounds | Recall |
|---|---|---|---|
| BM25 top-10 | 10.0 | - | 0.331 |
| BM25 top-13 (matched budget) | 13.0 | - | 0.375 |
| Controller, target 0.38 | 11.5 | 0.73 | 0.410 |
| **Controller, default (target 0.5)** | **12.6** | **1.29** | **0.454** |
| Controller, target 0.6 | 13.3 | 1.64 | 0.483 |
| Controller, patience 2 | 13.1 | 1.55 | 0.461 |

Recall never decreased on any question, and 23.8% of questions needed no extra search. Reproduce with
`python3 -m controller.analyze` and `python3 -m controller.compare` (saved results, no API key). To re-run the controller:

```bash
python3 -m controller.run_all --out experiments/controller/new.jsonl
```

Always pass `--out`, otherwise the committed `controller_results.jsonl` is overwritten.

Tests: `tests/test_controller.py`.