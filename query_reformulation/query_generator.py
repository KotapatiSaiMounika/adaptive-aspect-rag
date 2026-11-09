
import re


def get_aspect_text(item):
    """Accept string or dictionary aspect formats."""

    if isinstance(item, dict):
        text = (
            item.get("description")
            or item.get("aspect")
            or item.get("text")
            or item.get("name")
            or ""
        )
        return str(text).strip()

    return str(item).strip()


def clean_query(query, max_length=250):
    """Normalize whitespace and limit query length."""

    query = re.sub(r"\s+", " ", str(query)).strip()
    return query[:max_length].strip()


def generate_queries(missing_aspects):
    """
    Generate initial search queries for each missing aspect.
    """

    queries = []

    for item in missing_aspects or []:
        aspect = get_aspect_text(item)

        if not aspect:
            continue

        candidates = [
            aspect,
            f"{aspect} supporting evidence",
            f"{aspect} causes and consequences",
            f"{aspect} mechanisms and factors",
            f"{aspect} research findings and examples",
        ]

        queries.extend(clean_query(q) for q in candidates)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(q for q in queries if q))


def generate_next_query(
    original_question,
    missing_aspect,
    previous_evidence,
    previous_queries=None,
):
    """
    Generate a targeted follow-up query using the original question,
    missing aspect, and previously retrieved evidence.
    """

    aspect = get_aspect_text(missing_aspect)

    if not aspect:
        return ""

    original_question = str(original_question or "").strip()

    # Collect a small amount of context from previous evidence
    evidence_context = []

    for doc in (previous_evidence or [])[:2]:
        if not isinstance(doc, dict):
            continue

        doc_text = str(doc.get("text", "")).strip()

        if doc_text:
            evidence_context.append(doc_text[:150])

    evidence_text = " ".join(evidence_context)

    # Construct a focused follow-up query
    query_parts = [aspect]

    if original_question:
        query_parts.append(original_question)

    if evidence_text:
        query_parts.append(
            f"additional evidence beyond {evidence_text}"
        )
    else:
        query_parts.append("detailed supporting evidence")

    query = clean_query(" ".join(query_parts))

    # Avoid repeating an existing query
    previous = {
        str(q).casefold().strip()
        for q in (previous_queries or [])
    }

    if query.casefold() in previous:
        fallback = clean_query(
            f"{aspect} further research findings"
        )

        if fallback.casefold() in previous:
            return ""

        return fallback

    return query
