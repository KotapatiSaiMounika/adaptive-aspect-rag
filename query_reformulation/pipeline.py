
from query_reformulation.query_generator import generate_queries
from query_reformulation.query_expansion import expand_query
from query_reformulation.iterative_retriever import iterative_retrieve
from query_reformulation.document_filter import filter_documents


def run_pipeline(missing_aspects, task="biology", method="bm25", top_k=10):

    print("\nGenerating queries...")

    queries = []

    for aspect in missing_aspects:
        queries.extend(generate_queries([aspect]))
        queries.extend(expand_query(aspect))

    queries = list(dict.fromkeys(queries))

    print("Generated queries:", len(queries))

    print("\nRetrieving additional documents...")

    documents = iterative_retrieve(
        missing_aspects,
        task=task,
        method=method,
        top_k=top_k
    )

    print("Retrieved documents:", len(documents))

    print("\nFiltering documents...")

    filtered_docs = filter_documents(documents, top_k=top_k)

    print("Final documents:", len(filtered_docs))

    return {
        "queries": queries,
        "documents": filtered_docs
    }


if __name__ == "__main__":

    missing_aspects = [
        "Effects of climate change on agriculture"
    ]

    result = run_pipeline(
        missing_aspects,
        task="biology",
        method="bm25",
        top_k=10
    )

    print("\nFINAL RESULTS")

    for doc in result["documents"]:
        print("\nDocument ID:", doc["doc_id"])
        print("Score:", doc["score"])
        print("Text:", doc["text"][:200])
