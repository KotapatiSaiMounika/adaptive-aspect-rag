
def filter_documents(documents, top_k=10):
    unique_docs = {}

    for doc in documents:
        doc_id = doc["doc_id"]

        if doc_id not in unique_docs:
            unique_docs[doc_id] = doc
        elif doc["score"] > unique_docs[doc_id]["score"]:
            unique_docs[doc_id] = doc

    filtered = sorted(
        unique_docs.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return filtered[:top_k]


if __name__ == "__main__":

    sample_docs = [
        {"doc_id": "1", "text": "Climate change affects crops", "score": 0.8},
        {"doc_id": "2", "text": "Global warming affects farming", "score": 0.9},
        {"doc_id": "1", "text": "Climate change affects crops", "score": 0.95},
    ]

    results = filter_documents(sample_docs)

    print("Filtered documents:")
    for doc in results:
        print(doc["doc_id"], doc["score"])
