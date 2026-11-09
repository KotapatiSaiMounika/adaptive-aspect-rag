from retrieval.retrieval_interface import retrieve
from query_reformulation.query_generator import generate_queries
from query_reformulation.query_expansion import expand_query


def retrieve_for_aspects(aspects, task, method="bm25", top_k=5):
    results = {}

    for i, item in enumerate(aspects, 1):
        aspect = item.get("description", "") if isinstance(item, dict) else str(item)
        aspect_id = item.get("id", i) if isinstance(item, dict) else i

        queries = list(dict.fromkeys(
            [aspect] + generate_queries([aspect]) + expand_query(aspect)
        ))

        documents = {}
        for query in queries[:10]:
            for doc in retrieve(query, task, method, top_k):
                doc = dict(doc)
                doc["aspect_id"] = aspect_id
                doc["retrieved_for_aspect"] = aspect
                if doc["doc_id"] not in documents or doc["score"] > documents[doc["doc_id"]]["score"]:
                    documents[doc["doc_id"]] = doc

        results[str(aspect_id)] = {
            "aspect": aspect,
            "documents": sorted(
                documents.values(), key=lambda x: x["score"], reverse=True
            )[:top_k]
        }

    return results


def retrieve_for_missing_aspect(aspect, task, method="bm25", top_k=5):
    return retrieve_for_aspects([aspect], task, method, top_k)


def iterative_retrieve(missing_aspects, task, method="bm25",
                       top_k=5, max_queries=10):
    results = retrieve_for_aspects(missing_aspects, task, method, top_k)
    return [
        doc
        for result in results.values()
        for doc in result["documents"]
    ]