
from query_reformulation.iterative_retriever import (
    retrieve_for_aspects,
    retrieve_for_missing_aspect
)


def test_initial_retrieval():

    aspects = [
        {
            "aspect_id": "A1",
            "description": "causes of climate change"
        }
    ]

    result = retrieve_for_aspects(
        aspects,
        task="biology",
        method="bm25",
        top_k=5
    )

    print(result)

    assert isinstance(result, dict)


def test_targeted_retrieval():

    result = retrieve_for_missing_aspect(
        {
            "aspect_id": "A2",
            "description": "effects of climate change"
        },
        task="biology",
        method="bm25",
        top_k=5,
        original_question="What causes climate change and what are its effects?"
    )

    print(result)

    assert isinstance(result, dict)
