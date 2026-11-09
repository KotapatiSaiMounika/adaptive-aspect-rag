
import re


def expand_query(aspect, max_expansions=6):
    """
    Generate alternative search queries for a reasoning aspect.

    Supports both string and dictionary aspect formats.
    """

    # Handle dictionary input
    if isinstance(aspect, dict):
        aspect = (
            aspect.get("description")
            or aspect.get("aspect")
            or aspect.get("text")
            or aspect.get("name")
            or ""
        )

    # Normalize the aspect text
    aspect = re.sub(r"\s+", " ", str(aspect)).strip()

    if not aspect:
        return []

    if max_expansions <= 0:
        return []

    # Generate alternative search formulations
    expanded_queries = [
        aspect,
        f"{aspect} supporting evidence",
        f"{aspect} causes and contributing factors",
        f"{aspect} underlying mechanisms",
        f"{aspect} effects and implications",
        f"{aspect} research findings and examples",
        f"{aspect} relevant studies and observations",
        f"{aspect} explanations and real-world examples",
    ]

    # Remove duplicates while preserving order
    expanded_queries = list(dict.fromkeys(expanded_queries))

    # Limit the number of generated queries
    return expanded_queries[:max_expansions]
