from pathlib import Path

import bm25s
import Stemmer


class BM25Retriever:
    name = "bm25"

    def __init__(self, corpus: list, index_dir: str, k1: float = 0.9, b: float = 0.4):
        self.doc_ids = [d["doc_id"] for d in corpus]
        self.texts = [d["text"] for d in corpus]
        self.stemmer = Stemmer.Stemmer("english")
        index_dir = Path(index_dir)

        if (index_dir / "params.index.json").exists():      # reuse saved index
            self.index = bm25s.BM25.load(str(index_dir), load_corpus=False)
            n_indexed = self.index.scores["num_docs"]
            if n_indexed != len(self.doc_ids):
                raise RuntimeError(
                    f"Saved BM25 index has {n_indexed} docs but corpus has "
                    f"{len(self.doc_ids)}. Delete {index_dir} and rebuild.")
        else:                                                # build once, save
            tokens = bm25s.tokenize(self.texts, stopwords="en",
                                    stemmer=self.stemmer, show_progress=False)
            self.index = bm25s.BM25(k1=k1, b=b)
            self.index.index(tokens, show_progress=False)
            index_dir.mkdir(parents=True, exist_ok=True)
            self.index.save(str(index_dir))

    def search(self, query: str, top_k: int = 10) -> list:
        q = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer,
                           return_ids=False, show_progress=False)
        k = min(top_k, len(self.doc_ids))
        idx, scores = self.index.retrieve(q, k=k, show_progress=False)
        return [{"doc_id": self.doc_ids[i], "text": self.texts[i],
                 "score": float(s), "retriever": self.name}
                for i, s in zip(idx[0], scores[0]) if s > 0]  # drop zero-score padding