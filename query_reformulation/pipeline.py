
from query_reformulation.iterative_retriever import iterative_retrieve
from query_reformulation.document_filter import filter_documents


def run_pipeline(
    missing_aspects,
    task="biology",
    method="bm25",
    top_k=10,
    max_queries=10,
    original_question="",
    previous_evidence=None,
    previous_queries=None,
    return_metadata=False,
):
    """
    Run the Member 3 query generation and retrieval pipeline.

    Member 4 supplies the missing aspects.
    Member 3 generates retrieval queries and retrieves evidence.
    """

    print("\nStarting query generation and retrieval...")

    # Run bounded iterative retrieval.
    retrieval_result = iterative_retrieve(
        missing_aspects=missing_aspects,
        task=task,
        method=method,
        top_k=top_k,
        max_queries=max_queries,
        original_question=original_question,
        previous_evidence=previous_evidence,
        previous_queries=previous_queries,
        return_metadata=True,
    )

    queries = retrieval_result["queries"]
    documents = retrieval_result["documents"]

    print("Executed queries:", len(queries))

    print("\nFiltering retrieved documents...")

    filtered_docs = filter_documents(
        documents,
        top_k=top_k,
    )

    print("Final documents:", len(filtered_docs))

    output = {
        "queries": queries,
        "documents": filtered_docs,
        "by_aspect": retrieval_result["by_aspect"],
        "query_count": retrieval_result["query_count"],
    }

    if return_metadata:
        return output

    return {
        "queries": queries,
        "documents": filtered_docs,
    }


if __name__ == "__main__":

    missing_aspects = [
        {
            "aspect_id": "A1",
            "description": "Effects of climate change on agriculture",
        }
    ]

    result = run_pipeline(
        missing_aspects=missing_aspects,
        task="biology",
        method="bm25",
        top_k=5,
        max_queries=6,
        original_question=(
            "What are the causes and effects of climate change?"
        ),
        return_metadata=True,
    )

    print("\nFINAL RESULTS")

    for doc in result["documents"]:
        print("\nDocument ID:", doc["doc_id"])
        print("Score:", doc["score"])
        print("Text:", doc["text"][:200])

    print("\nTotal queries:", result["query_count"])
