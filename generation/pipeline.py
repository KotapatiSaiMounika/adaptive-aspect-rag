# generation/pipeline.py
from generation.answer_generator import answer, revise
from generation.quality import verify

def generate_and_verify(question, evidence, aspects=None, max_revisions=1):
    """The single function M4's controller calls."""
    ans = answer(question, evidence, aspects)
    rep = verify(question, ans, evidence, aspects)
    for _ in range(max_revisions):
        if not rep["unsupported_claims"]:
            break
        ans = revise(question, ans, rep, evidence)
        rep = verify(question, ans, evidence, aspects)
    return {"answer": ans, "report": rep}