
from retrieval.retrieval_interface import retrieve

from query_reformulation.query_generator import (
    generate_queries,
    generate_next_query,
)

from query_reformulation.query_expansion import expand_query

from query_reformulation.document_filter import filter_documents


def get_aspect_info(aspect, index):
    """Extract aspect ID and description from supported formats."""

    if isinstance(aspect, dict):
        aspect_id = (
            aspect.get("aspect_id")
            or aspect.get("id")
            or index
        )

        description = (
            aspect.get("description")
            or aspect.get("aspect")
            or aspect.get("text")
            or aspect.get("name")
            or ""
        )

        return str(aspect_id), str(description).strip()

    return str(index), str(aspect).strip()


def _document_key(doc):
    """Create a stable key for document deduplication."""

    if doc.get("doc_id") is not None:
        return str(doc["doc_id"])

    return str(doc.get("text", "")).strip()


def _merge_documents(existing_documents, new_documents, aspect_id, query):
    """
    Merge documents without losing the aspects and queries
    associated with each document.
    """

    merged = {
        _document_key(doc): dict(doc)
        for doc in existing_documents
    }

    for doc in new_documents or []:
        if not isinstance(doc, dict):
            continue

        key = _document_key(doc)

        if not key:
            continue

        if key not in merged:
            merged[key] = dict(doc)

        else:
            # Keep the document with the higher retrieval score.
            old_score = merged[key].get("score", 0) or 0
            new_score = doc.get("score", 0) or 0

            if new_score > old_score:
                merged[key].update(doc)

        current = merged[key]

        aspects = current.setdefault("retrieved_for_aspects", [])
        if aspect_id not in aspects:
            aspects.append(aspect_id)

        queries = current.setdefault("retrieval_queries", [])
        if query and query not in queries:
            queries.append(query)

    return list(merged.values())


def retrieve_for_aspects(
    aspects,
    task,
    method="bm25",
    top_k=5,
):
    """Perform initial retrieval separately for each aspect."""

    results = {}

    for index, aspect in enumerate(aspects or [], start=1):
        aspect_id, description = get_aspect_info(aspect, index)

        if not description:
            continue

        candidates = (
            generate_queries([description])
            + expand_query(description)
        )

        queries = list(dict.fromkeys(candidates))[:3]

        documents = []
        executed_queries = []

        for query in queries:
            retrieved = retrieve(
                query,
                task,
                method=method,
                top_k=top_k,
            )

            documents.extend(retrieved or [])
            executed_queries.append(query)

        filtered = filter_documents(
            documents,
            top_k=top_k,
        )

        results[aspect_id] = {
            "aspect": description,
            "queries": executed_queries,
            "documents": filtered,
        }

    return results


def retrieve_for_missing_aspect(
    aspect,
    task,
    method="bm25",
    top_k=5,
    original_question="",
    previous_evidence=None,
    previous_queries=None,
):
    """
    Retrieve additional evidence for an aspect identified
    as insufficient by Member 4.
    """

    aspect_id, description = get_aspect_info(aspect, 1)

    if not description:
        return {}

    query = generate_next_query(
        original_question,
        description,
        previous_evidence or [],
        previous_queries or [],
    )

    if not query:
        return {}

    # Avoid repeating an existing query.
    previous = {
        str(q).casefold().strip()
        for q in (previous_queries or [])
    }

    if query.casefold().strip() in previous:
        return {}

    documents = retrieve(
        query,
        task,
        method=method,
        top_k=top_k,
    )

    filtered = filter_documents(
        documents or [],
        top_k=top_k,
    )

    return {
        aspect_id: {
            "aspect": description,
            "queries": [query],
            "documents": filtered,
        }
    }


def iterative_retrieve(
    missing_aspects,
    task,
    method="bm25",
    top_k=5,
    max_queries=10,
    original_question="",
    previous_evidence=None,
    previous_queries=None,
    return_metadata=False,
):
    """
    Bounded two-phase retrieval.

    Phase 1: Retrieve initial evidence for missing aspects.
    Phase 2: Generate evidence-guided follow-up queries.

    Member 4 decides which aspects are missing.
    Member 3 generates queries and performs retrieval.
    """

    aspects = list(missing_aspects or [])

    if max_queries <= 0 or not aspects:
        empty_output = {
            "documents": [],
            "by_aspect": {},
            "queries": list(previous_queries or []),
            "query_count": 0,
        }

        return empty_output if return_metadata else []

    query_history = list(previous_queries or [])
    executed_queries = []
    by_aspect = {}
    all_documents = []

    # Normalize previous query history for duplicate checking.
    seen_queries = {
        str(q).casefold().strip()
        for q in query_history
    }

    def run_retrieval(query, aspect_id):
        """Execute one retrieval query and update shared results."""

        documents = retrieve(
            query,
            task,
            method=method,
            top_k=top_k,
        ) or []

        executed_queries.append(query)
        query_history.append(query)
        seen_queries.add(query.casefold().strip())

        filtered = filter_documents(
            documents,
            top_k=top_k,
        )

        nonlocal all_documents

        all_documents = _merge_documents(
            all_documents,
            filtered,
            aspect_id,
            query,
        )

        return filtered

    # Reserve some of the query budget for follow-up retrieval.
    followup_slots = min(
        len(aspects),
        max_queries // 2,
    )

    initial_budget = max_queries - followup_slots

    # Phase 1: Initial aspect-wise retrieval.
    for index, aspect in enumerate(aspects, start=1):
        if len(executed_queries) >= initial_budget:
            break

        aspect_id, description = get_aspect_info(aspect, index)

        if not description:
            continue

        candidates = (
            generate_queries([description])
            + expand_query(description)
        )

        candidates = list(dict.fromkeys(candidates))

        aspect_result = by_aspect.setdefault(
            aspect_id,
            {
                "aspect": description,
                "queries": [],
                "documents": [],
            },
        )

        for query in candidates:
            if len(executed_queries) >= initial_budget:
                break

            normalized_query = query.casefold().strip()

            if not query or normalized_query in seen_queries:
                continue

            documents = run_retrieval(query, aspect_id)

            aspect_result["queries"].append(query)

            aspect_result["documents"] = filter_documents(
                aspect_result["documents"] + documents,
                top_k=top_k,
            )

    # Phase 2: Evidence-guided follow-up retrieval.
    for index, aspect in enumerate(aspects, start=1):
        if len(executed_queries) >= max_queries:
            break

        aspect_id, description = get_aspect_info(aspect, index)

        if not description:
            continue

        aspect_result = by_aspect.setdefault(
            aspect_id,
            {
                "aspect": description,
                "queries": [],
                "documents": [],
            },
        )

        evidence = list(previous_evidence or [])
        evidence.extend(aspect_result["documents"])

        query = generate_next_query(
            original_question,
            description,
            evidence,
            query_history,
        )

        if not query:
            continue

        normalized_query = query.casefold().strip()

        if normalized_query in seen_queries:
            continue

        documents = run_retrieval(query, aspect_id)

        aspect_result["queries"].append(query)

        aspect_result["documents"] = filter_documents(
            aspect_result["documents"] + documents,
            top_k=top_k,
        )

    # Final document collection.
    final_documents = filter_documents(
        all_documents,
        top_k=top_k * max(1, len(aspects)),
    )

    output = {
        "documents": final_documents,
        "by_aspect": by_aspect,
        "queries": executed_queries,
        "query_count": len(executed_queries),
    }

    if return_metadata:
        return output

    return final_documents
