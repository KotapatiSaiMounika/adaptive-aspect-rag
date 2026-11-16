# Member 2: Aspect Extraction and Document-Aspect Relevance

## Role and interface

Member 2 turns a complex question into a short list of independent **aspects** (the distinct pieces of information a complete
answer needs) and measures how well each retrieved document supports each aspect. The aspects tell the controller *what* must
be covered, and the relevance scores (the "reflection") tell it *how well* the current evidence covers each one. Member 2 does
not decide whether to retrieve again (Member 4) or write the follow-up queries (Member 3). Code: `controller/aspect_extractor.py`
and `controller/relevance.py`.

| You need | Call |
|---|---|
| Aspects for a question (dataset, LLM or heuristic) | `controller.aspect_extractor.get_aspects(question, task=..., llm_fn=None)` |
| Aspects for a bare query string | `controller.aspect_extractor.extract_aspects(query, llm_fn=None)` |
| Controller-ready aspect list | `controller.aspect_extractor.to_dicts(aspects)` |
| Query string for one aspect (Member 3) | `aspect.search_query(extra_terms=decision.hints["missing_terms"])` |
| Document-aspect scores for the controller | `controller.relevance.ReflectionEngine(aspects).reflect(evidence)` |
| Which documents support which aspect | `controller.relevance.summarize_support(reflections)` |
| Best supporting passages per aspect (Member 5) | `ReflectionEngine.evidence_snippets(aspect_id, top_n=3)` |

## Where Member 2 sits in the pipeline

1. The question goes to `get_aspects()` (Member 2), which returns the aspects.
2. Member 1 runs the initial retrieval and returns the evidence documents.
3. `ReflectionEngine.reflect()` (Member 2) scores every aspect against every evidence document.
4. `AdaptiveController.decide()` (Member 4) receives the aspects and the reflections. If an aspect is weak, it returns the weakest aspect and hints to Member 3, who retrieves new documents; these are added to the evidence and the loop repeats from step 3.
5. When the controller stops (sufficient coverage, budget or stagnation), the evidence goes to cited answer generation (Member 5).

## Part 1: Aspect extraction (`controller/aspect_extractor.py`)

### Data model

Each aspect is an `Aspect` dataclass:

| Field | Meaning |
|---|---|
| `id` | Unique per question, in the Bright-Pro style `<task>-<qid>-a<n>`, e.g. `biology-0-a1` |
| `description` | A self-contained sub-question or information need that can be searched on its own |
| `keywords` | Up to 6 distinctive content words (LLM extraction keeps up to 8 model-supplied keywords) |
| `source` | Where it came from: `dataset`, `llm` or `heuristic` |

`Aspect.to_dict()` returns `{"id", "description", "text", "keywords", "source"}`. The `text` field duplicates `description` so
the controller, scorers and Members 3 and 5 can all read the same object without conversion.

### Three extraction routes

`get_aspects()` picks the route in this order, so existing work is never duplicated and the function never returns nothing
for a non-empty question: dataset aspects first, then the LLM (if `llm_fn` is given), then the heuristic. The LLM route falls back to the heuristic on any failure.

| Route | When it is used | Needs model or network |
|---|---|---|
| Dataset (`from_dataset`) | The question already carries Bright-Pro gold aspects (`q["aspects"]`). Aspect texts are cleaned, empty ones dropped, and duplicate ids made unique (`id`, `id_2`, ...) | No |
| LLM (`llm_extract`) | A new question with no aspects and an `llm_fn` (any callable `prompt: str -> str`) is supplied | Yes (caller-supplied) |
| Heuristic (`heuristic_extract`) | No dataset aspects and no `llm_fn`, or the LLM route failed | No |

### Heuristic extractor (deterministic, no model)

Steps:

1. Clean the text: remove code blocks and URLs.
2. Split into sentences, then into clauses on `;`, "and also", "as well as", "in addition to", "along with", and ", and/or" followed by a question word.
3. Strip lead-ins (also, and, but, finally, ...) and drop clauses with fewer than 3 content tokens.
4. Score each clause: +2 if it is question-like, plus min(tokens, 12) / 12.
5. Prefer question clauses (only if there are at least two).
6. Take the highest scores, skipping near-duplicates (Jaccard ≥ 0.7), up to 5.
7. Restore reading order and trim each aspect to 40 words.

Design points:

- A single question mark is not enough to throw away the rest of a long forum post, so question clauses are preferred only when at least two exist.
- If nothing passes the length filter (for example, the question is one word), the whole question becomes the single aspect.
- Code blocks and URLs are removed first, so pasted code never becomes an aspect.

Example (a sustainable-living post, produced by running `heuristic_extract`):

| Id | Aspect description | Keywords |
|---|---|---|
| `sustainable_living-7-a1` | Why do my compost piles smell like ammonia? | my, compost, piles, smell, like, ammonia |
| `sustainable_living-7-a2` | How do I balance green and brown materials? | balance, green, brown, materials |
| `sustainable_living-7-a3` | is it safe to add citrus peels and cooked food | safe, add, citrus, peels, cooked, food |
| `sustainable_living-7-a4` | how long does the process usually take? | long, process, usually, take |

### LLM extractor

The prompt asks for at most *k* independent aspects (each a self-contained sub-question of up to 25 words) and requires JSON only:
`{"aspects": [{"description": "...", "keywords": ["..."]}]}`. The question is cleaned and truncated to 6,000 characters first.
The parser tolerates ` ```json ` fences and surrounding prose, and accepts either a list of strings or a list of objects
(`description`, `text` or `aspect` keys). Model keywords are kept only when they form a clean, non-empty list of strings.
**Any failure** (exception, no JSON found, empty list) falls back to the heuristic extractor, so a broken API call never
stops the pipeline.

### Shared safeguards

| Safeguard | Behaviour |
|---|---|
| Maximum aspects | 5 by default (`max_aspects`), matching the dataset (1 to 5 per question) |
| Near-duplicate removal | Aspects with token Jaccard similarity ≥ 0.7 to an earlier one are dropped |
| Unique ids | `make_aspect_id` builds `<task>-<qid>-a<n>`, 1-based, omitting unknown parts |
| Controller compatibility | `to_dicts()` output is accepted directly by `AdaptiveController` and `normalize_aspects` |

## Part 2: Document-aspect relevance, the "reflection" (`controller/relevance.py`)

Member 4's scorers rate a *whole document* against an aspect. A long document that mentions the right words scattered across
1,000 words can look as good as a document that actually discusses the aspect. Member 2 adds a **passage-level** score on top
of any Member 4 scorer, plus an optional judge and a cache.

### Scoring one (aspect, document) pair

Steps (`pw` = `passage_weight`, `jw` = `judge_weight`):

1. `doc_score` = the base scorer on the whole document.
2. Split the document into sentence windows (window 3, stride 2, at most 40 windows); `passage_score` = the best window's score.
3. `score = (1 − pw) · doc_score + pw · passage_score`
4. If a `judge_fn` is given: `score = (1 − jw) · score + jw · judge(aspect, best passage)`
5. The final score is clamped to 0 to 1.

`AspectRelevanceScorer.explain()` returns `{"score", "doc_score", "passage_score", "snippet"}`; the snippet is the best passage
(first 300 characters), which gives Member 5 something to cite. `score()` returns only the number, so the class also satisfies
Member 4's `AspectScorer` protocol and can be passed to the controller directly as its `scorer`.

| Parameter | Default | Meaning |
|---|---|---|
| `base_scorer` | `LexicalScorer()` | Any object with `.score(aspect_text, doc_text) -> [0, 1]` (Member 4's `LexicalScorer` or `DenseScorer`) |
| `window` | 3 | Sentences per passage |
| `stride` | 2 | Step between passages |
| `max_chunks` | 40 | Cap per document; longer documents are evenly subsampled |
| `passage_weight` | 0.5 | 0 = whole-document score only (identical to the base scorer), 1 = best passage only |
| `judge_fn` | `None` | Optional LLM or cross-encoder `(aspect, passage) -> [0, 1]`, run only on the best passage to keep cost low |
| `judge_weight` | 0.3 | Blend weight of the judge when one is supplied |

Judge outputs are clamped to [0, 1]. Empty aspect or document text scores 0.

### Reflection engine

`ReflectionEngine(aspects).reflect(evidence)` is called after every retrieval round. It returns
`{aspect_id: {doc_id: score}}` for every aspect and every evidence document, which is exactly the `reflections` argument of
`AdaptiveController.decide(...)`.

- **Incremental cache:** each (aspect, document) pair is scored once; later rounds only score the newly retrieved documents.
- **Accepts both** `Aspect` objects and plain dicts or strings.
- `summarize_support(reflections, threshold=0.38)` returns, per aspect, the best document, best score, all supporting documents and a `supported` flag. The default threshold equals the controller's `support_threshold`.
- `evidence_snippets(aspect_id, top_n)` returns the best supporting passages seen so far for an aspect.

Example: the four composting aspects above scored against two toy documents (lexical backend):

| Aspect | `d1` (ammonia / green-brown balance) | `d2` (citrus and cooked food) | Supported (≥ 0.38) |
|---|---|---|---|
| a1 ammonia smell | 0.333 | 0.000 | no |
| a2 green and brown balance | **0.750** | 0.000 | yes (`d1`) |
| a3 citrus peels and cooked food | 0.167 | **0.833** | yes (`d2`) |
| a4 how long it takes | 0.000 | 0.000 | no, a gap the controller would target |

### How the controller uses it

In `AdaptiveController.score_matrix`, the final score for each pair is a blend of the controller's own scorer and the
reflection: `(1 − w) · scorer + w · reflection` with `w = reflection_weight` (default 0.4; 0 ignores Member 2's scores).

```python
from controller import AdaptiveController, ControllerConfig, LexicalScorer
from controller.aspect_extractor import get_aspects, to_dicts
from controller.relevance import ReflectionEngine

aspects = get_aspects(question, task="biology")          # dataset, LLM or heuristic
engine = ReflectionEngine(aspects)
ctrl = AdaptiveController(to_dicts(aspects), ControllerConfig(), LexicalScorer())

d = ctrl.decide(evidence, reflections=engine.reflect(evidence))
while d.action == "retrieve":
    query = next(a for a in aspects if a.id == d.weakest_aspect).search_query(d.hints["missing_terms"])
    evidence += retrieve_for_aspect(query)               # Member 3
    d = ctrl.decide(evidence, reflections=engine.reflect(evidence))
```

## Data used in the saved experiments

In all 739 Bright-Pro questions the saved controller runs (`experiments/controller/controller_results.jsonl`) used the
**dataset aspects** (ids such as `biology-0-a1`), averaging 3.74 aspects per question:

| Aspects per question | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Questions | 20 | 28 | 142 | 484 | 65 |

The saved controller runs used the controller's own `LexicalScorer` and did not pass Member 2's reflection scores
(`controller/run_all.py` calls `decide` without `reflections`). The heuristic and LLM extractors and the reflection engine are
implemented, tested and interface-compatible, and serve new, user-typed questions that have no gold aspects. The controller
results in `docs/member4_controller.md` therefore measure Member 4's logic on gold aspects, not Member 2's extraction.

## Limitations

- The heuristic extractor is sentence and clause based. It cannot split a single long sentence with several implicit needs, and its aspects keep the user's wording (for example "my" and "like" appear as keywords).
- It was checked on unit-test examples and sample posts, not scored against the Bright-Pro gold aspects, so no extraction accuracy is reported.
- The lexical backend rewards word overlap and does not understand paraphrase; use `DenseScorer` as `base_scorer` for semantic matching.
- The LLM extraction and the optional `judge_fn` need a caller-supplied model; tests use stub functions and never call an LLM.

## Tests

`tests/test_aspect_extractor.py` (10 tests) and `tests/test_relevance.py` (14 tests); all 24 pass offline in under a second.

| Area | What is checked |
|---|---|
| Ids | `make_aspect_id` follows the `biology-0-a1` convention |
| Heuristic | Multi-part posts split into 2 to 5 unique aspects; maximum and near-duplicate handling; empty text, one-word input and code blocks |
| Dataset route | Gold aspects reused, duplicate ids made unique, empty aspects dropped |
| Priority | Dataset beats LLM beats heuristic, and `use_dataset=False` is respected |
| LLM route | Fenced JSON parsed; exception, non-JSON and empty list all fall back to heuristic |
| Controller compatibility | Extractor output works with `normalize_aspects` and `AdaptiveController` |
| Query building | `search_query` adds only terms not already in the description |
| Reflection format | Scores are floats in [0, 1]; relevant documents outscore irrelevant ones |
| Passage scoring | A focused passage beats scattered mentions; `passage_weight=0` equals the base scorer; long documents capped at `max_chunks` |
| Judge | `judge_fn` is blended and clamped |
| Cache | Only new documents are scored in a later round |
| Integration | Reflections make the controller choose the correct weakest aspect and stop with `sufficient`; the scorer also works as the controller's `scorer` |

```bash
pytest -q tests/test_aspect_extractor.py tests/test_relevance.py
```
