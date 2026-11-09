
from retrieval.retrieval_interface import retrieve
from query_reformulation.query_generator import generate_next_query
from query_reformulation.document_filter import filter_documents


def retrieve_for_controller(
    aspect_id,
    aspect_text,
    task,
    original_question="",
    previous_evidence=None,
    previous_queries=None,
    hints=None,
    method="bm25",
):
    """
    Adapter between Member 4's adaptive controller
    and Member 3's query reformulation/retrieval pipeline.

    Returns:
        {
            "queries": [...],
            "documents": [...]
        }
    """

    hints = hints or {}
    previous_evidence = previous_evidence or []
    previous_queries = previous_queries or []

    # Use the controller's suggested top_k.
    top_k = hints.get("top_k", 5)

    # Generate a follow-up query for the weakest aspect.
    query = generate_next_query(
        original_question,
        aspect_text,
        previous_evidence,
        previous_queries,
    )

    if not query:
        return {"queries": [], "documents": []}

    # Add missing terms supplied by the controller.
    missing_terms = hints.get("missing_terms", [])

    if missing_terms:
        query = query + " " + " ".join(missing_terms)

    query = " ".join(query.split())

    # Prevent duplicate queries.
    previous = {
        str(q).casefold().strip()
        for q in previous_queries
    }

    if query.casefold().strip() in previous:
        return {"queries": [], "documents": []}

    # Retrieve using BM25 or dense retrieval.
    documents = retrieve(
        query,
        task,
        method=method,
        top_k=top_k,
    ) or []

    # Exclude documents already present in the evidence.
    excluded_ids = {
        str(doc_id)
        for doc_id in hints.get("exclude_doc_ids", [])
    }

    documents = [
        doc for doc in documents
        if isinstance(doc, dict)
        and str(doc.get("doc_id", "")) not in excluded_ids
    ]

    # Remove duplicates and retain the best results.
    documents = filter_documents(
        documents,
        top_k=top_k,
    )

    # Attach aspect and query information.
    for doc in documents:
        doc.setdefault("retrieved_for_aspects", [])

        if aspect_id not in doc["retrieved_for_aspects"]:
            doc["retrieved_for_aspects"].append(aspect_id)

        doc.setdefault("retrieval_queries", [])

        if query not in doc["retrieval_queries"]:
            doc["retrieval_queries"].append(query)

    return {
        "queries": [query],
        "documents": documents,
    }
