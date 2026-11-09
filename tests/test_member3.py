
from query_reformulation.query_generator import generate_queries
from query_reformulation.query_expansion import expand_query
from query_reformulation.document_filter import filter_documents


def test_query_generator():
    result = generate_queries(["climate change"])
    assert len(result) > 1


def test_query_expansion():
    result = expand_query("climate change and agriculture")
    assert len(result) > 1


def test_document_filter():
    docs = [
        {"doc_id": "1", "text": "Document A", "score": 0.8},
        {"doc_id": "1", "text": "Document A", "score": 0.9},
        {"doc_id": "2", "text": "Document B", "score": 0.7},
    ]

    result = filter_documents(docs)

    assert len(result) == 2
    assert result[0]["score"] == 0.9


def test_empty_input():
    assert generate_queries([]) == []
    assert expand_query("unknown topic") == ["unknown topic"]
