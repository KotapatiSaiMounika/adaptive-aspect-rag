# Adaptive Aspect-Aware RAG

An experimental project exploring **Retrieval-Augmented Generation (RAG)** with adaptive, aspect-aware retrieval for complex question answering.

## Overview

Traditional RAG systems retrieve relevant document chunks to provide context for a language model. However, complex questions may involve multiple aspects, relationships, and pieces of evidence that cannot always be captured through a single retrieval step.

This project builds and evaluates a basic RAG pipeline and an advanced aspect-aware retrieval approach to investigate whether adaptive retrieval can improve answer quality and evidence relevance.

## Objectives

- Build a baseline RAG pipeline from scratch.
- Explore aspect-aware retrieval for complex questions.
- Investigate adaptive and iterative retrieval strategies.
- Evaluate and compare the baseline and advanced approaches.
- Document experiments, findings, and limitations.

## Project Structure

```text
adaptive-aspect-rag/
├── data/            # Dataset loader, preprocessing, inspection
├── retrieval/       # BM25, dense retrieval, common interface, baselines
├── indexes/         # Saved BM25 / dense indexes (git-ignored)
├── experiments/     # Baseline results (.jsonl)
├── tests/           # Retrieval tests
├── requirements.txt
└── README.md
```

## Retrieval Infrastructure (Member 1)

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Run all commands from the project root.

### Load the dataset
```python
from data.dataset_loader import load_questions, load_documents, TASKS
questions = load_questions("biology")   # [{qid, query, gold_ids, aspects, raw}]
docs = load_documents("biology")        # [{doc_id, text}]
```
Domains (`task`): biology, earth_science, economics, psychology, robotics, stackoverflow, sustainable_living.
Inspect the data: `python -m data.inspect_dataset --task biology`

### Retrieve documents
```python
from retrieval.retrieval_interface import retrieve
results = retrieve("why do birds migrate", task="biology", method="dense", top_k=10)
# [{"doc_id": ..., "text": ..., "score": ..., "retriever": "dense"}, ...]
```
`method` is `"bm25"` or `"dense"`. Each domain has its own corpus, so `task` is required.
Indexes are built on first use, saved under `indexes/`, then reused. BM25 may return
fewer than `top_k` results if fewer documents match.

### Baselines
```bash
python -m retrieval.baseline --method bm25     # all 7 domains, top_k=10
python -m retrieval.baseline --method dense
python -m retrieval.baseline --method bm25 --tasks biology --top_k 5 --out experiments/baselines/bm25_k5.jsonl
```
Output: `experiments/baselines/<method>_results.jsonl` (one JSON line per question).

### Tests
```bash
pytest -v
```
## Generation & Verification (Member 5)
Cited answer generation, claim-level support checking (Self-RAG IsSup/IsUse style) and a quality
report that tells the controller whether to search again. Details and results: `docs/member5_generation.md`.

Setup: copy `.env.example` to `.env` and fill in `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
(optionally `JUDGE_MODEL`).

    python -m generation.run_baseline --task biology --method dense --top_k 6 --limit 20
    python -m generation.run_full --task biology --limit 10
    python -m evaluation.evaluate_answers experiments/generation/*.jsonl
    python -m pytest tests/test_generation.py

Controller entry point: `generation.pipeline.generate_and_verify(question, evidence, aspects)` returns
`{"answer", "report"}`; `report["needs_retrieval"]` and `report["uncovered_aspects"]` drive the re-search decision.
