
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def filter_documents(documents, top_k=10, min_score=0.0,
                     query=None, min_similarity=0.05):

    unique_docs = {}

    for doc in documents:
        doc_id = doc["doc_id"]
        score = float(doc["score"])

        if score < min_score:
            continue

        if doc_id not in unique_docs or score > unique_docs[doc_id]["score"]:
            unique_docs[doc_id] = doc

    filtered = list(unique_docs.values())

    if query and filtered:
        texts = [query] + [doc["text"] for doc in filtered]

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)

        similarities = cosine_similarity(
            matrix[0:1], matrix[1:]
        )[0]

        filtered = [
            doc for doc, similarity in zip(filtered, similarities)
            if similarity >= min_similarity
        ]

    filtered.sort(key=lambda x: x["score"], reverse=True)

    return filtered[:top_k]
