import re

def generate_queries(missing_aspects):
    queries = []
    for item in missing_aspects:
        aspect = item.get("description", "") if isinstance(item, dict) else str(item)
        aspect = aspect.strip()
        if not aspect:
            continue
        clean = re.sub(r"^(impact of|effects of|role of)\s+", "", aspect, flags=re.I)
        queries.extend([
            aspect,
            f"{clean} causes and consequences",
            f"{clean} research findings",
            f"{clean} evidence and examples",
            f"How does {clean} affect the outcome?"
        ])
    return list(dict.fromkeys(queries))


def generate_next_query(original_question, missing_aspect, previous_evidence):
    evidence = " ".join(
        doc.get("text", "")[:300] for doc in previous_evidence[:3]
    )
    return (
        f"{original_question} {missing_aspect} "
        f"Further evidence needed beyond: {evidence}"
    ).strip()