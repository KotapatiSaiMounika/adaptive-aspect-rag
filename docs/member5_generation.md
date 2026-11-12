# Member 5: Answer Generation and Verification

## Role and interface
Member 5 turns the evidence collected by the retrieval loop into a final cited answer and checks its quality.
Entry point: `generation.pipeline.generate_and_verify(question, evidence, aspects=None, max_revisions=1)`
returns `{"answer": {...}, "report": {...}}`.

`report` fields (consumed by the controller):
- `support_rate`: fraction of claims supported by their cited documents (fully = 1, partial = 0.5, none = 0)
- `utility`: 1-5 completeness score for the answer
- `unsupported_claims`, `uncovered_aspects`: what the controller should search for next
- `confidence`: 0.6 * support_rate + 0.4 * (utility - 1) / 4 (a heuristic, not a calibrated probability)
- `needs_retrieval`: true if support_rate < 0.8, utility < 4, or any aspect is uncovered

## Design
1. **Cited generation** (`generation/answer_generator.py`): the LLM answers only from the provided documents and cites each claim with `[doc_id]`. Citation IDs are normalised (dash variants, missing file extensions) before matching.
2. **Claim extraction**: the answer is split into sentences; a citation at the end of a bullet covers the whole bullet; abstention sentences ("the documents do not cover...") are not counted as claims.
3. **Support check, Self-RAG IsSup style** (`generation/claim_checker.py`): each cited claim is labelled fully / partial / none against its cited documents. Uncited claims count as unsupported.
4. **Utility and aspect coverage, Self-RAG IsUse style**: one LLM call scores completeness (1-5) and marks which aspects the answer covers.
5. **Revision pass** (`revise`): optional rewrite that removes unsupported claims.
6. **Feedback to the controller**: in `generation/run_full.py`, if the report says an aspect is uncovered, the controller's retrieval adapter searches again for it and the answer is regenerated once. This is the FLARE-style "search only when needed" decision, driven by verified answer quality instead of token probabilities.

## Experimental setup
- Dataset: BRIGHT-Pro, biology split, first 8 questions that all systems completed.
- Generator: one LLM for all systems. Judge: a different model (`JUDGE_MODEL`).
- Baselines: single-shot RAG with BM25 and with dense retrieval (top-6 documents).
- Full system: BM25 initial retrieval (top-6) + Member 3/4 adaptive loop + Member 5 generation and one feedback round.

## Results (8 questions, biology)
| system | support rate | utility (1-5) | gold-aspect coverage | retrieval calls | abstained |
|---|---|---|---|---|---|
| baseline_bm25 | 0.558 | 2.625 | 0.344 | 1.0 | 1/8 |
| baseline_dense | 0.667 | 3.5 | 0.469 | 1.0 | 0/8 |
| full | 0.609 | 4.5 | 0.844 | 2.125 | 0/8 |

Retrieval side (full system): gold-document recall rose from 0.368 to 0.741; 2 of 8 questions needed no extra search.
Support rate is averaged over answers that made at least one claim.

## Findings
- The full system gives the most complete answers (utility 4.5 vs 3.5 for dense and 2.6 for BM25) and covers far more of the question's aspects (0.84 vs 0.47 and 0.34).
- The adaptive loop roughly doubles gold-document recall at about 2.1 retrieval calls per question, and skips extra searches when the first retrieval is already enough.
- Support rate did not improve over dense single-shot RAG (0.61 vs 0.67). Citation faithfulness is therefore not what the full system gains; it gains completeness. A possible reason is that longer, more complete answers contain more claims that can be unsupported, but this was not tested.

## Limitations
- Only 8 questions from one domain (free-tier LLM token limits), so differences are indicative, not statistically significant.
- Gold-aspect coverage favours the full system by design: the controller searches for the same aspects that coverage is scored against, and the full-system prompt lists them. Support rate and utility are the fairer comparison.
- Support, utility and coverage are LLM judgements and can be noisy. The judge is a different model from the generator, which reduces but does not remove bias.
- The confidence score is a heuristic, and claim extraction is rule-based (sentence splitting plus citation matching).
- Documents are truncated to 800 characters in prompts to fit token limits.

## Reproduce
    python -m generation.run_baseline --task biology --method bm25  --top_k 6 --limit 20
    python -m generation.run_baseline --task biology --method dense --top_k 6 --limit 20
    python -m generation.run_full --task biology --limit 10
    python -m evaluation.evaluate_answers experiments/generation/baseline_bm25_biology.jsonl \
        experiments/generation/baseline_dense_biology.jsonl experiments/generation/full_biology.jsonl
    python -m pytest tests/test_generation.py
