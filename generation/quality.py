# generation/quality.py
from generation.claim_checker import check_support, judge_answer, SCORE
from generation.answer_generator import norm_aspects

def verify(question, ans, evidence, aspects=None, support_thr=0.8, utility_thr=4):
    claims = ans["claims"]
    labels = check_support(claims, evidence)
    scores = [SCORE[l] for l in labels]
    support_rate = sum(scores) / len(scores) if scores else 0.0
    j = judge_answer(question, ans["text"], aspects)
    uncovered = [a for a, ok in j["covered"].items() if not ok]
    confidence = round(0.6 * support_rate + 0.4 * (j["utility"] - 1) / 4, 3)
    return {
        "support_rate": round(support_rate, 3),
        "utility": j["utility"],
        "unsupported_claims": [c["text"] for c, l in zip(claims, labels) if l == "none"],
        "uncovered_aspects": uncovered,           # aspect ids, for M4
        "confidence": confidence,                  # heuristic: 0.6*support + 0.4*utility
        "needs_retrieval": support_rate < support_thr or j["utility"] < utility_thr or bool(uncovered),
        "claim_labels": labels,
    }