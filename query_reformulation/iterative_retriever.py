
from retrieval.retrieval_interface import retrieve
from query_reformulation.query_generator import generate_queries
from query_reformulation.query_expansion import expand_query


def iterative_retrieve(missing_aspects, task, method="bm25", top_k=5):

    all_documents = []

    for aspect in missing_aspects:
        queries = generate_queries([aspect])
        expanded_queries = expand_query(aspect)

        queries.extend(expanded_queries)

        for query in set(queries):
            documents = retrieve(
                query=query,
                task=task,
                method=method,
                top_k=top_k
            )

            all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":

    missing_aspects = [
        "Effects of climate change on agriculture"
    ]

    results = iterative_retrieve(
        missing_aspects,
        task="biology",
        method="bm25",
        top_k=5
    )

    print("Retrieved documents:", len(results))

    for doc in results[:5]:
        print("\n", doc["doc_id"], doc["score"])
        print(doc["text"][:200])
