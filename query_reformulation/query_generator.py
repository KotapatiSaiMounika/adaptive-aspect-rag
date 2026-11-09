
import re


def generate_queries(missing_aspects):
    queries = []

    for aspect in missing_aspects:
        aspect = aspect.strip()

        if not aspect:
            continue

        queries.append(aspect)

        clean = re.sub(r"^(impact of|effects of|role of)\s+", "", aspect, flags=re.I)

        queries.extend([
            f"{clean} causes and consequences",
            f"{clean} research findings",
            f"{clean} real-world examples",
            f"Factors influencing {clean}",
            f"Recent studies about {clean}"
        ])

    return list(dict.fromkeys(queries))


if __name__ == "__main__":
    aspects = ["Effects of climate change on agriculture"]

    for i, query in enumerate(generate_queries(aspects), 1):
        print(f"{i}. {query}")
