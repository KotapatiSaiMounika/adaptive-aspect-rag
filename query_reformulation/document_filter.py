
def filter_documents(documents, top_k=10, min_score=0.0):
    unique_docs = {}

    for doc in documents:
        doc_id = doc["doc_id"]
        score = float(doc["score"])

        if score < min_score:
            continue

        if doc_id not in unique_docs or score > unique_docs[doc_id]["score"]:
            unique_docs[doc_id] = doc

    filtered = sorted(
        unique_docs.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return filtered[:top_k]