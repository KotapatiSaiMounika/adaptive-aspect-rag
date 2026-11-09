
from retrieval.retrieval_interface import retrieve
from query_reformulation.query_generator import generate_queries
from query_reformulation.query_expansion import expand_query
from query_reformulation.document_filter import filter_documents


def iterative_retrieve(missing_aspects, task, method="bm25",
                       top_k=5, max_queries=10):

    all_documents = []

    for aspect in missing_aspects:
        queries = generate_queries([aspect])
        queries.extend(expand_query(aspect))
        queries = list(dict.fromkeys(queries))[:max_queries]

        for query in queries:
            documents = retrieve(
                query=query,
                task=task,
                method=method,
                top_k=top_k
            )

            all_documents.extend(documents)

    results = []

    for aspect in missing_aspects:
        filtered = filter_documents(
            all_documents,
            top_k=top_k * 2,
            query=aspect,
            min_similarity=0.05
        )
        results.extend(filtered)

    return filter_documents(results, top_k=top_k * 2)


if __name__ == "__main__":

    aspects = ["Effects of climate change on agriculture"]

    results = iterative_retrieve(
        aspects,
        task="biology",
        method="bm25",
        top_k=5
    )

    print("Relevant documents:", len(results))

    for doc in results:
        print("\n", doc["doc_id"], doc["score"])
        print(doc["text"][:200])
