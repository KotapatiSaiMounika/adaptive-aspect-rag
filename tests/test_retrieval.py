import pytest

from retrieval.bm25 import BM25Retriever

TOY = [
    {"doc_id": "D1", "text": "Birds migrate south in winter to find food and warmth."},
    {"doc_id": "D2", "text": "Photosynthesis converts sunlight into chemical energy in plants."},
    {"doc_id": "D3", "text": "The stock market fell sharply after the interest rate decision."},
    {"doc_id": "D4", "text": "Mitochondria produce ATP through cellular respiration."},
]
KEYS = {"doc_id", "text", "score", "retriever"}
TOY_IDS = {d["doc_id"] for d in TOY}


def check_format(res, retriever_name):
    assert all(set(x) == KEYS for x in res)
    assert all(x["retriever"] == retriever_name for x in res)
    assert all(x["doc_id"] in TOY_IDS for x in res)
    assert all(isinstance(x["score"], float) for x in res)
    scores = [x["score"] for x in res]
    assert scores == sorted(scores, reverse=True)


# ---------------- BM25 ----------------
def test_bm25_format_and_relevance(tmp_path):
    r = BM25Retriever(TOY, tmp_path / "bm25")
    res = r.search("why do birds migrate in winter", top_k=2)
    assert len(res) <= 2 and res[0]["doc_id"] == "D1"
    check_format(res, "bm25")


@pytest.mark.parametrize("query,expected", [
    ("why do birds migrate in winter", "D1"),
    ("how do plants use sunlight", "D2"),
    ("interest rate stock market", "D3"),
    ("ATP cellular respiration", "D4"),
])
def test_bm25_multiple_queries(tmp_path, query, expected):
    r = BM25Retriever(TOY, tmp_path / "bm25")
    assert r.search(query, top_k=3)[0]["doc_id"] == expected


def test_bm25_index_is_reused(tmp_path):
    BM25Retriever(TOY, tmp_path / "bm25")
    assert (tmp_path / "bm25" / "params.index.json").exists()
    r2 = BM25Retriever(TOY, tmp_path / "bm25")
    assert r2.search("ATP cellular respiration", 1)[0]["doc_id"] == "D4"


def test_bm25_stale_index_is_rejected(tmp_path):
    BM25Retriever(TOY, tmp_path / "bm25")
    with pytest.raises(RuntimeError):
        BM25Retriever(TOY[:2], tmp_path / "bm25")


def test_bm25_no_zero_score_padding(tmp_path):
    r = BM25Retriever(TOY, tmp_path / "bm25")
    res = r.search("birds", top_k=4)
    assert all(x["score"] > 0 for x in res)
    assert r.search("zzzunknownword", top_k=4) == []


def test_top_k_larger_than_corpus(tmp_path):
    r = BM25Retriever(TOY, tmp_path / "bm25")
    assert len(r.search("market birds plants", top_k=100)) <= len(TOY)


# ---------------- Dense ----------------
def test_dense_format_and_relevance(tmp_path):
    pytest.importorskip("faiss")
    from retrieval.dense import DenseRetriever
    r = DenseRetriever(TOY, tmp_path / "dense")
    res = r.search("how do plants turn light into energy", top_k=2)
    assert len(res) == 2 and res[0]["doc_id"] == "D2"
    check_format(res, "dense")


def test_dense_index_is_reused(tmp_path):
    pytest.importorskip("faiss")
    from retrieval.dense import DenseRetriever
    DenseRetriever(TOY, tmp_path / "dense")
    assert (tmp_path / "dense" / "index.faiss").exists()
    assert (tmp_path / "dense" / "embeddings.npy").exists()
    mtime = (tmp_path / "dense" / "index.faiss").stat().st_mtime_ns
    r2 = DenseRetriever(TOY, tmp_path / "dense")          # must load, not rebuild
    assert (tmp_path / "dense" / "index.faiss").stat().st_mtime_ns == mtime
    assert r2.search("ATP cellular respiration", 1)[0]["doc_id"] == "D4"


# ---------------- Common interface ----------------
@pytest.fixture
def toy_interface(tmp_path, monkeypatch):
    import retrieval.retrieval_interface as ri
    monkeypatch.setattr(ri, "get_clean_corpus", lambda task: TOY)
    monkeypatch.setattr(ri, "INDEX_ROOT", tmp_path)
    monkeypatch.setattr(ri, "_RETRIEVERS", {})
    return ri


def test_interface_bm25_and_reuse(toy_interface):
    res = toy_interface.retrieve("birds winter", "toytask", method="bm25", top_k=2)
    check_format(res, "bm25")
    first = toy_interface._RETRIEVERS[("toytask", "bm25")]
    toy_interface.retrieve("plants sunlight", "toytask", method="bm25", top_k=2)
    assert toy_interface._RETRIEVERS[("toytask", "bm25")] is first   # not rebuilt


def test_interface_dense(toy_interface):
    pytest.importorskip("faiss")
    res = toy_interface.retrieve("plants and sunlight", "toytask", method="dense", top_k=2)
    check_format(res, "dense")


def test_invalid_method_needs_no_dataset():
    from retrieval.retrieval_interface import retrieve
    with pytest.raises(ValueError):
        retrieve("q", "biology", method="nope")