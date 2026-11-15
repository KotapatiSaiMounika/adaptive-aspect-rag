# Adaptive Aspect-Aware Retrieval for Reasoning-Intensive QA

An adaptive RAG system on the Bright-Pro benchmark. It splits a question into aspects, retrieves for each,
checks whether the evidence covers every aspect, retrieves again only where it is missing, then generates a
cited, verified answer. It is compared with standard single-pass RAG.

## Structure
```text
data/                 Member 1  dataset loader, preprocessing, inspection
retrieval/            Member 1  BM25, dense retrieval, common interface, baselines
controller/           Member 2 + 4  aspect extractor, relevance scoring, adaptive controller, analysis
query_reformulation/  Member 3  follow-up query generation + iterative retrieval
generation/           Member 5  cited answers, claim checking, quality report
evaluation/           Member 5  answer evaluation
experiments/          saved results (.jsonl) and summaries
tests/  docs/
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Run everything from the project root. Indexes are built on first use and stored in `indexes/` (git-ignored).
Dense index building is slow (about 6 hours for all 7 domains on CPU): ask Member 1 for `indexes.zip` and unzip it in the project root.
Copy `.env.example` to `.env` only if you run the generation code (needs an LLM key). Leave `BRIGHT_PRO_DATA_ROOT` unset.

## Quick verification (no dataset, no API key, under 1 minute)
```bash
pytest -q tests/test_controller.py tests/test_generation.py tests/test_member3.py \
          tests/test_aspect_extractor.py tests/test_relevance.py tests/test_retrieval.py
python -m controller.analyze
python -m controller.compare
python -m controller.demo
```
`pytest -q` (all tests) also needs the dataset download and the biology BM25 index.

## Retrieval (Member 1)
```python
from retrieval.retrieval_interface import retrieve
retrieve("why do birds migrate", task="biology", method="bm25", top_k=10)
# [{"doc_id", "text", "score", "retriever"}, ...]   method: "bm25" | "dense"
```
Domains: biology, earth_science, economics, psychology, robotics, stackoverflow, sustainable_living.
Baselines (top-10, 739 questions): `experiments/baselines/bm25_results.jsonl` (pooled gold recall 0.308),
`dense_results.jsonl` (0.208), `bm25_k13.jsonl` (top-13, for a matched-budget comparison).

## Adaptive controller results (all 739 questions)
Per-question average gold recall (`python -m controller.compare`):

| method | docs | rounds | recall |
|---|---|---|---|
| BM25 top-10 | 10.0 | - | 0.331 |
| BM25 top-13 | 13.0 | - | 0.375 |
| controller t=0.38 | 11.5 | 0.73 | 0.410 |
| controller t=0.5 (default) | 12.6 | 1.29 | 0.454 |
| controller t=0.6 | 13.3 | 1.64 | 0.483 |
| controller t=0.5, patience=2 | 13.1 | 1.55 | 0.461 |

The experiments use Bright-Pro's annotated (gold) aspects with the lexical scorer. `controller/aspect_extractor.py`
and `controller/relevance.py` (Member 2) are implemented and tested for new questions but are not used in these runs.
Regenerate (about a few minutes per domain once indexes exist): `python -m controller.run_all --out experiments/controller/new.jsonl`.
Without `--out` it overwrites the committed results file.

## Generation and verification (Member 5)
Details: `docs/member5_generation.md`. Results on 20 biology questions (`experiments/generation/summary.md`):

| system | support rate | utility (1-5) | gold-aspect coverage | retrieval calls |
|---|---|---|---|---|
| baseline_bm25 | 0.56 | 2.75 | 0.345 | 1.0 |
| baseline_dense | 0.545 | 3.05 | 0.458 | 1.0 |
| full | 0.554 | 4.25 | 0.747 | 3.1 |

Needs `.env` and an LLM key:
```bash
python3 -m generation.run_baseline --task biology --method bm25  --top_k 6 --limit 20
python3 -m generation.run_baseline --task biology --method dense --top_k 6 --limit 20
python3 -m generation.run_full --task biology --limit 20 --pause 30
python3 -m evaluation.evaluate_answers experiments/generation/*_biology.jsonl
```
Runs are resumable: re-running skips questions already saved. If the provider reports a rate limit, wait and re-run.

## Limitations
- The answer-quality comparison covers 20 questions in one domain, so it is indicative only.
- Support rate is about the same for all systems; the full system mainly improves completeness and aspect coverage.
- Gold-aspect coverage favours the adaptive system by design (it searches for the same aspects it is scored on).
- LLM judgements are noisy; support rate is judged by the generator model itself, while utility and coverage use a separate judge model.