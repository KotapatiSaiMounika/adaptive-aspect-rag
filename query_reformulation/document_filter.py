
def filter_documents(
    documents,
    top_k=10,
    min_score=None
):
    """
    Validate documents, remove duplicates, merge retrieval metadata,
    and return the highest-ranked documents.
    """

    if top_k <= 0:
        return []

    unique_documents = {}

    for doc in documents or []:

        if not isinstance(doc, dict):
            continue

        doc_id = doc.get("doc_id")
        text = doc.get("text", "")

        if not doc_id or not isinstance(text, str):
            continue

        if not text.strip():
            continue

        try:
            score = float(doc.get("score", 0.0))
        except (ValueError, TypeError):
            continue

        if min_score is not None and score < min_score:
            continue

        doc = dict(doc)
        doc["doc_id"] = str(doc_id)
        doc["text"] = text.strip()
        doc["score"] = score

        key = doc["doc_id"]

        if key not in unique_documents:
            unique_documents[key] = doc

        else:
            existing = unique_documents[key]

            # Preserve aspect associations.
            aspects = list(
                existing.get("retrieved_for_aspects", [])
            )

            for aspect_id in doc.get("retrieved_for_aspects", []):
                if aspect_id not in aspects:
                    aspects.append(aspect_id)

            # Preserve retrieval query history.
            queries = list(
                existing.get("retrieval_queries", [])
            )

            for query in doc.get("retrieval_queries", []):
                if query not in queries:
                    queries.append(query)

            # Keep the highest-scoring document version.
            if score > existing["score"]:
                unique_documents[key] = doc

            # Restore merged metadata.
            unique_documents[key]["retrieved_for_aspects"] = aspects
            unique_documents[key]["retrieval_queries"] = queries

    results = list(unique_documents.values())

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]
