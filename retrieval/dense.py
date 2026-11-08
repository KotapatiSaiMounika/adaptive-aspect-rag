import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    name = "dense"

    def __init__(self, corpus: list, index_dir: str, model_name: str = DEFAULT_MODEL,
                 max_seq_length: int = 256, batch_size: int = 64,
                 query_prefix: str = QUERY_PREFIX):
        self.doc_ids = [d["doc_id"] for d in corpus]
        self.texts = [d["text"] for d in corpus]
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length

        index_dir = Path(index_dir)
        faiss_path, meta_path = index_dir / "index.faiss", index_dir / "meta.json"

        if faiss_path.exists() and meta_path.exists():       # reuse saved index
            meta = json.loads(meta_path.read_text())
            assert meta["model"] == model_name, "Index built with a different model"
            self.index = faiss.read_index(str(faiss_path))
            assert self.index.ntotal == len(self.doc_ids), "Corpus/index size mismatch"
        else:                                                 # build once, save
            emb = self.model.encode(self.texts, batch_size=batch_size,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=True).astype("float32")
            index_dir.mkdir(parents=True, exist_ok=True)
            np.save(index_dir / "embeddings.npy", emb)        # stored for reuse
            self.index = faiss.IndexFlatIP(emb.shape[1])      # cosine (normalized)
            self.index.add(emb)
            faiss.write_index(self.index, str(faiss_path))
            meta_path.write_text(json.dumps({"model": model_name, "n_docs": len(emb)}))

    def search(self, query: str, top_k: int = 10) -> list:
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True,
                              convert_to_numpy=True).astype("float32")
        k = min(top_k, len(self.doc_ids))
        scores, idx = self.index.search(q, k)
        return [{"doc_id": self.doc_ids[i], "text": self.texts[i],
                 "score": float(s), "retriever": self.name}
                for i, s in zip(idx[0], scores[0])]