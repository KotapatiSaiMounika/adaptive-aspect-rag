
def generate_queries(missing_aspects):
    """
    Generate alternative search queries from missing aspects.

    Args:
        missing_aspects: List of aspects that need more information.

    Returns:
        List of improved search queries.
    """

    queries = []

    for aspect in missing_aspects:

        # Original aspect itself is a search query
        queries.append(aspect)

        # Generate alternative search queries
        queries.append(f"What are the {aspect}?")
        queries.append(f"Impact and effects of {aspect}")
        queries.append(f"Research and evidence about {aspect}")

    # Remove duplicate queries
    queries = list(dict.fromkeys(queries))

    return queries


if __name__ == "__main__":

    missing_aspects = [
        "Economic effects of climate change",
        "Impact of climate change on agriculture"
    ]

    queries = generate_queries(missing_aspects)

    print("Generated Search Queries:\n")

    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")
