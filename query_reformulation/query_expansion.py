
def expand_query(aspect):
    """
    Expand a missing aspect into related search queries.
    """

    expansions = {
        "agriculture": [
            "climate change and crop production",
            "global warming and farming",
            "agricultural productivity under changing climate",
            "drought and food security",
            "climate variability and crop yields"
        ],

        "economy": [
            "economic consequences",
            "financial losses",
            "economic growth and development",
            "climate change and employment",
            "cost of environmental damage"
        ],

        "health": [
            "public health consequences",
            "human health risks",
            "climate change and disease",
            "heat-related illnesses",
            "environmental health impacts"
        ]
    }

    aspect_lower = aspect.lower()

    expanded_queries = [aspect]

    for category, terms in expansions.items():
        if category in aspect_lower:
            expanded_queries.extend(terms)

    # Remove duplicates
    return list(dict.fromkeys(expanded_queries))


if __name__ == "__main__":

    aspect = "Effects of climate change on agriculture"

    queries = expand_query(aspect)

    print("Original Aspect:", aspect)
    print("\nExpanded Queries:")

    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")
