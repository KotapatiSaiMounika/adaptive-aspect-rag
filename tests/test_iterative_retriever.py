
from query_reformulation.iterative_retriever import iterative_retrieve


def test_iterative_retrieval_query_budget():
    result = iterative_retrieve(
        missing_aspects=[
            {
                "aspect_id": "A1",
                "description": "Climate change causes",
            },
            {
                "aspect_id": "A2",
                "description": "Climate change effects",
            },
        ],
        task="biology",
        method="bm25",
        top_k=2,
        max_queries=4,
        original_question="What causes climate change and what are its effects?",
        return_metadata=True,
    )

    assert result["query_count"] <= 4
    assert len(result["queries"]) == result["query_count"]
    assert "A1" in result["by_aspect"] or "A2" in result["by_aspect"]


def test_iterative_retrieval_empty_input():
    result = iterative_retrieve(
        missing_aspects=[],
        task="biology",
        max_queries=4,
        return_metadata=True,
    )

    assert result["documents"] == []
    assert result["query_count"] == 0
