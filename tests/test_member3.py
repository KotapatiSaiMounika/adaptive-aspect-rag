
from query_reformulation.query_generator import (
    generate_queries,
    generate_next_query
)

from query_reformulation.query_expansion import (
    expand_query
)

from query_reformulation.document_filter import (
    filter_documents
)


def test_generate_queries():

    result = generate_queries([
        "causes of climate change"
    ])

    assert len(result) > 0


def test_query_expansion():

    result = expand_query(
        "effects of climate change"
    )

    assert len(result) > 1


def test_next_query():

    result = generate_next_query(
        "What causes climate change?",
        "effects of climate change",
        [
            {
                "doc_id": "D1",
                "text": "Global warming increases temperatures."
            }
        ]
    )

    assert "effects of climate change" in result


def test_document_filter():

    documents = [
        {
            "doc_id": "D1",
            "text": "First document",
            "score": 0.9
        },
        {
            "doc_id": "D1",
            "text": "Duplicate document",
            "score": 0.7
        },
        {
            "doc_id": "D2",
            "text": "Second document",
            "score": 0.8
        }
    ]

    result = filter_documents(documents)

    assert len(result) == 2
    assert result[0]["doc_id"] == "D1"


def test_empty_filter():

    assert filter_documents([]) == []


def test_document_filter_preserves_metadata():

    documents = [
        {
            "doc_id": "doc1",
            "text": "Climate change affects temperature.",
            "score": 5.0,
            "retrieved_for_aspects": ["A1"],
            "retrieval_queries": ["climate change"],
        },
        {
            "doc_id": "doc1",
            "text": "Climate change affects temperature.",
            "score": 8.0,
            "retrieved_for_aspects": ["A2"],
            "retrieval_queries": ["climate change effects"],
        },
    ]

    result = filter_documents(documents)

    assert len(result) == 1
    assert result[0]["score"] == 8.0
    assert set(result[0]["retrieved_for_aspects"]) == {"A1", "A2"}
    assert len(result[0]["retrieval_queries"]) == 2

