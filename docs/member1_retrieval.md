# Member 1: Dataset, Preprocessing and Retrieval Infrastructure

## Role and interface

Member 1 provides the data and search layer that every other component builds on: a reliable dataset loader, a clean
searchable corpus per domain, BM25 and dense retrievers, one common retrieval interface, and baseline results for
comparison with the adaptive system. It does not decide which aspect needs more evidence or when to retrieve again;
that belongs to the controller (Members 3 and 4).

| You need | Call |
|---|---|
| Questions, gold document IDs, aspects | `data.dataset_loader.load_questions(task)` |
| Raw documents | `data.dataset_loader.load_documents(task)` |
| Clean, deduplicated corpus | `data.preprocessing.get_clean_corpus(task)` |
| Search | `retrieval.retrieval_interface.retrieve(query, task, method="bm25", top_k=10)` |
| Baseline results | `experiments/baselines/*.jsonl` |

Every retriever returns the same format, sorted by score:

```python
[{"doc_id": "biology-0/extraction_0.txt", "text": "Document content...", "score": 12.7, "retriever": "bm25"}]
```

## Dataset

The benchmark is **Bright-Pro**, loaded from Hugging Face (`yale-nlp/Bright-Pro`). It has three configs, `examples`,
`documents` and `aspects`, and each domain is a split inside each config. For an offline copy, set
`BRIGHT_PRO_DATA_ROOT` to a folder laid out as `<root>/<config>/<task>.json`. If that path is wrong or a file is
missing, the loader prints a warning and falls back to Hugging Face.

Statistics computed from our baseline result files:

| Domain | Questions | Gold documents |
|---|---|---|
| biology | 103 | 804 |
| earth_science | 115 | 856 |
| economics | 99 | 773 |
| psychology | 100 | 707 |
| robotics | 101 | 623 |
| stackoverflow | 115 | 529 |
| sustainable_living | 106 | 980 |
| **Total** | **739** | **5,272** |

On average a question has 7.13 gold documents and 3.74 annotated aspects (between 1 and 5). Aspect IDs follow
`<task>-<qid>-a<n>`, for example `biology-0-a1`. The loader groups them by question.

`load_questions(task)` returns a list of dictionaries:

| Field | Meaning |
|---|---|
| `qid` | Question ID (string) |
| `query` | Question text |
| `gold_ids` | IDs of the gold (relevant) documents |
| `aspects` | List of aspect records for the question, or `None` if unavailable |
| `raw` | The original dataset record |

`load_documents(task)` returns `[{"doc_id": str, "text": str}]` with the original IDs and text. Field names are matched
against several common spellings (`id`/`doc_id`, `content`/`text`, `query`/`question`, `gold_ids`/...), so the loader
tolerates small schema differences. Run `python3 -m data.inspect_dataset --task biology` to print the structure, sizes
and sample records of any domain.

## Preprocessing

`data/preprocessing.py` prepares one consistent corpus for both retrievers:

- Documents with missing or whitespace-only text are removed.
- Documents with a repeated ID are removed, keeping the first.
- Original IDs and text are **never modified**, so gold IDs stay valid and BM25 and dense see identical text.
- `get_clean_corpus(task)` caches the result in memory.
- `gold_coverage(task)` reports how many gold IDs are missing from the cleaned corpus (it should be 0).

```bash
python3 -m data.preprocessing --task biology     # prints removal counts and gold coverage
```

## Retrieval

Each domain has its own corpus, so every retriever instance covers exactly one domain.

### BM25 (`retrieval/bm25.py`)

| Setting | Value |
|---|---|
| Library | `bm25s` (pure Python, no Java needed, unlike Pyserini) |
| Parameters | `k1 = 0.9`, `b = 0.4` (conventional values, not tuned) |
| Text processing | English stop words removed, English stemming (PyStemmer) |
| Index files | `indexes/bm25_index/<task>/` (sparse matrix `.npy` files, `params.index.json`, `vocab.index.json`) |

Behaviours worth knowing:

- The index is built once and loaded on later runs.
- A saved index whose size does not match the corpus raises a clear `RuntimeError` instead of failing later.
- Results with a score of zero are dropped, so a query with few matching documents returns **fewer than `top_k`** results rather than padding with irrelevant ones.

### Dense (`retrieval/dense.py`)

| Setting | Value |
|---|---|
| Model | `BAAI/bge-small-en-v1.5` (Sentence Transformers) |
| Query prefix | `Represent this sentence for searching relevant passages: ` |
| Max sequence length | 256 tokens (longer text is truncated) |
| Batch size | 64 |
| Similarity | Cosine, via normalised embeddings and an exact FAISS `IndexFlatIP` |
| Stored files | `indexes/dense_index/<task>/index.faiss`, `embeddings.npy`, `meta.json` |

`meta.json` records the model name and document count. Loading an index built with a different model, or for a corpus
of a different size, raises an error.

## Common interface (`retrieval/retrieval_interface.py`)

```python
from retrieval.retrieval_interface import retrieve

retrieve(query, task, method="bm25", top_k=10)
```

- `method` must be `"bm25"` or `"dense"`; anything else raises `ValueError` before any data is loaded.
- `task` is required because each domain has its own corpus. This is the one deliberate difference from the original
  specification, which had no `task` argument.
- Retrievers are created lazily and cached per `(task, method)`, so indexes and embeddings are loaded once per
  process and reused for every later query. BM25-only users never import PyTorch.
- Index locations are anchored to the project root, so it works from any working directory.

## Baselines (`retrieval/baseline.py`)

A baseline is one retrieval pass with a fixed `top_k`.

```bash
python3 -m retrieval.baseline --method bm25                  # all 7 domains, top_k = 10
python3 -m retrieval.baseline --method dense
python3 -m retrieval.baseline --method bm25 --top_k 13 --out experiments/baselines/bm25_k13.jsonl
python3 -m retrieval.baseline --method bm25 --tasks biology --limit 5
```

Each output line (one per question) contains `task`, `qid`, `method`, `top_k`, `retrieved_doc_ids`, `scores`,
`num_retrieved`, `retrieval_time_sec` and `gold_doc_ids`.

| File | Contents |
|---|---|
| `experiments/baselines/bm25_results.jsonl` | BM25, top-10, 739 questions |
| `experiments/baselines/dense_results.jsonl` | Dense, top-10, 739 questions |
| `experiments/baselines/bm25_k13.jsonl` | BM25, top-13, for a matched-budget comparison with the controller |

### Results (top-10, pooled gold recall)

Recall is the number of gold documents retrieved divided by the total number of gold documents, pooled over all
questions of the domain.

| Domain | BM25 | Dense | BM25 time | Dense time |
|---|---|---|---|---|
| biology | 0.320 | 0.210 | 0.7 ms | 57.2 ms |
| earth_science | 0.376 | 0.275 | 2.3 ms | 86.1 ms |
| economics | 0.299 | 0.176 | 0.5 ms | 89.5 ms |
| psychology | 0.256 | 0.184 | 0.5 ms | 85.6 ms |
| robotics | 0.252 | 0.207 | 1.0 ms | 163.9 ms |
| stackoverflow | 0.382 | 0.259 | 1.2 ms | 113.7 ms |
| sustainable_living | 0.279 | 0.164 | 0.5 ms | 80.1 ms |
| **All** | **0.308** | **0.208** | **1.0 ms** | **96.5 ms** |

BM25 beats dense retrieval in every domain, and is about 100 times faster per query on our CPU. At least one gold
document appears in the top-10 for 571 of 739 questions with BM25 and 498 of 739 with dense. These recall values are
low for both methods, which is expected for reasoning-intensive questions and is the gap the adaptive controller aims
to close. We did not investigate why dense retrieval is lower. The small embedding model and the 256-token cut-off on
long questions and documents are plausible factors, but this was not tested.

The controller report (`python3 -m controller.compare`) uses per-question average recall, so its top-10 BM25 figure
(0.331) differs from the pooled 0.308 above.

## Indexes, storage and sharing

```text
indexes/
├── bm25_index/<task>/     params.index.json, vocab.index.json, data/indices/indptr .npy files
└── dense_index/<task>/    index.faiss, embeddings.npy, meta.json
```

Indexes are built on first use and are git-ignored. BM25 indexes build quickly. Dense indexes are slow: building all
seven domains took about 6 hours on a CPU (stackoverflow alone took about 1 hour 50 minutes). To share finished
indexes, zip them without the redundant embedding copies and unzip in the project root:

```bash
zip -r indexes.zip indexes -x "*embeddings.npy"
```

## Testing (`tests/test_retrieval.py`)

11 test functions (14 cases) run on a tiny in-memory corpus and need no dataset:

| Area | What is checked |
|---|---|
| Output format | Exactly the four keys, correct retriever name, known IDs, float scores, descending order |
| Relevance | Four different queries each return the expected document first |
| Index reuse | A second construction loads the saved BM25 and dense indexes instead of rebuilding |
| Stale index | A BM25 index built for a different corpus size is rejected |
| Edge cases | No zero-score padding, empty result for unknown terms, `top_k` larger than the corpus |
| Interface | One retriever instance per `(task, method)`, BM25 and dense paths work, invalid method raises `ValueError` without loading data |

```bash
pytest -q tests/test_retrieval.py
```

## Integration with other members

| Member | What they use |
|---|---|
| 2 | `load_questions` for questions and annotated aspects (`controller/aspect_extractor.py` reuses them) |
| 3 | `retrieve` through `query_reformulation/iterative_retriever.py` and `controller_adapter.py` |
| 4 | Retrieved documents with IDs and scores; `get_clean_corpus` to fit the lexical scorer |
| 5 | `retrieve` for the generation baselines, and `experiments/baselines/*.jsonl` |

## Design decisions and limitations

- **One corpus per domain.** Questions are only searched against their own domain, which requires the `task` argument.
- **`bm25s` instead of Pyserini**, to avoid a Java dependency and keep setup simple.
- **Exact dense search.** `IndexFlatIP` gives exact results and is fast enough at this corpus size, so no approximate index is used.
- **Dense model is small and untuned**, with 256-token truncation. No reranker, query rewriting or hybrid fusion is applied at this layer.
- **BM25 hyperparameters are not tuned.**
- **In-process cache only.** Retrievers are cached per process and are not designed for concurrent use from multiple threads.
- **Results depend on the Hugging Face dataset version** downloaded at run time.

## Reproduce

```bash
python3 -m data.inspect_dataset --task biology
python3 -m data.preprocessing --task biology
python3 -m retrieval.baseline --method bm25
python3 -m retrieval.baseline --method dense          # slow on first run: builds dense indexes
pytest -q tests/test_retrieval.py
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named ...` | The virtual environment is not active. Run `source .venv/bin/activate`. |
| `python: command not found` | Use `python3`, or activate the virtual environment first. |
| `[warn] BRIGHT_PRO_DATA_ROOT=... has no ...json` | The offline path is wrong. Remove that variable from `.env` to use Hugging Face. |
| `RuntimeError: Saved BM25 index has N docs but corpus has M` | The index is stale. Delete `indexes/bm25_index/<task>` and re-run. |
| Dense search takes hours the first time | It is building the embedding index. Unzip a shared `indexes.zip` instead. |
| "unauthenticated requests to the HF Hub" | Harmless. Set `HF_TOKEN` for faster downloads if you wish. |

## File reference

| File | Purpose |
|---|---|
| `data/dataset_loader.py` | Loads questions, documents and aspects; Hugging Face or offline |
| `data/preprocessing.py` | Cleans the corpus and checks gold-ID coverage |
| `data/inspect_dataset.py` | Prints dataset structure and statistics |
| `retrieval/bm25.py` | BM25 retriever with persistent index |
| `retrieval/dense.py` | Dense retriever (Sentence Transformers + FAISS) |
| `retrieval/retrieval_interface.py` | Single `retrieve()` entry point with caching |
| `retrieval/baseline.py` | Fixed top-k baseline runner |
| `tests/test_retrieval.py` | Retrieval unit tests |