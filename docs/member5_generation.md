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
- Dataset: BRIGHT-Pro, biology split, first 20 questions (all three systems completed all 20).
- Generator: one LLM (`LLM_MODEL` in `.env`) for all systems. The claim-level support check (support rate) was done by the same LLM. Utility and gold-aspect coverage in the summary table were re-scored by a separate judge model (`JUDGE_MODEL` = `qwen/qwen3.8-27b`) in `evaluation/evaluate_answers.py`. Judgements are cached in `experiments/generation/judge_cache.jsonl`, so the evaluation can be resumed.
- Baselines: single-shot RAG with BM25 and with dense retrieval (top-6 documents).
- Full system: BM25 initial retrieval (top-6) + Member 3/4 adaptive loop + Member 5 generation and one feedback round.

## Results (20 questions, biology)
| system | support rate | utility (1-5) | gold-aspect coverage | retrieval calls | abstained |
|---|---|---|---|---|---|
| baseline_bm25 | 0.56 | 2.75 | 0.345 | 1.0 | 1/20 |
| baseline_dense | 0.545 | 3.05 | 0.458 | 1.0 | 1/20 |
| full | 0.554 | 4.25 | 0.747 | 3.1 | 0/20 |

Retrieval side (full system): gold-document recall rose from 0.295 to 0.738; 2 of 20 questions needed no extra search.
Support rate is averaged over answers that made at least one claim.

## Findings
- The full system gives the most complete answers (utility 4.25 vs 3.05 for dense and 2.75 for BM25) and covers far more of the question's aspects (0.75 vs 0.46 and 0.35).
- The adaptive loop raises gold-document recall about 2.5 times (0.295 to 0.738) at about 3.1 retrieval calls per question, and skips extra searches when the first retrieval is already enough (2 of 20 questions).
- Support rate is about the same for all three systems (0.545 to 0.56), so the differences are within noise. Citation faithfulness is therefore not what the full system gains; it gains completeness. A possible reason is that longer, more complete answers contain more claims that can be unsupported, but this was not tested.

## Limitations
- Only 20 questions from one domain (free-tier LLM limits), so differences are indicative, not statistically significant.
- Gold-aspect coverage favours the full system by design: the controller searches for the same aspects that coverage is scored against, and the full-system prompt lists them. Support rate and utility are the fairer comparison.
- Support rate is self-evaluated by the generator model, so it can be biased. Only utility and coverage use a separate judge.
- The confidence score is a heuristic, and claim extraction is rule-based (sentence splitting plus citation matching).
- Claim extraction only recognises `[doc_id]` citations. In one answer (q4) the model used 【doc_id】 brackets, so none of its claims counted as cited and its support rate is 0.0; another answer (q9) was cut off by the token limit. Both lower the full system's support rate for reasons unrelated to answer quality.
- Utility measures completeness, not correctness: for q18 the generator's own report gave utility 5 but a support rate of 0.14.
- The full system is more expensive: about 3.1 retrieval calls plus a verification and feedback round per question, versus one retrieval call for the baselines.
- Documents are truncated to 800 characters in prompts to fit token limits.

## Reproduce
Needs `.env` with `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` and, for evaluation, `JUDGE_MODEL`.

    python -m generation.run_baseline --task biology --method bm25  --top_k 6 --limit 20
    python -m generation.run_baseline --task biology --method dense --top_k 6 --limit 20
    python -m generation.run_full --task biology --limit 20 --pause 30
    python -m evaluation.evaluate_answers experiments/generation/baseline_bm25_biology.jsonl \
        experiments/generation/baseline_dense_biology.jsonl experiments/generation/full_biology.jsonl
    python -m pytest tests/test_generation.py

All runs are resumable: re-running the same command skips questions already saved. If the provider reports a rate limit, wait and re-run.