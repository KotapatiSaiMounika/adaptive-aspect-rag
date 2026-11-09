import pytest
from controller import AdaptiveController, ControllerConfig, LexicalScorer, normalize_aspects

A = ["birds migrate winter", "magnetic field navigation"]
D_MIG = {"doc_id": "m", "text": "Birds migrate in winter to warmer places."}
D_MAG = {"doc_id": "g", "text": "Magnetic field helps navigation in birds."}
D_BAD = {"doc_id": "x", "text": "The stock market fell yesterday."}


def ctrl(**kw):
    return AdaptiveController(A, ControllerConfig(**kw), LexicalScorer())


def test_normalize_aspects_formats():
    assert normalize_aspects(["a b"])[0] == {"id": "a0", "text": "a b"}
    assert normalize_aspects([{"aspect_id": "q-a1", "description": "d"}])[0] == {"id": "q-a1", "text": "d"}
    assert normalize_aspects({"x": "t"})[0] == {"id": "x", "text": "t"}
    assert normalize_aspects(None) == []


def test_empty_aspects_raises():
    with pytest.raises(ValueError):
        AdaptiveController([])


def test_weakest_aspect_is_targeted():
    d = ctrl().decide([D_MIG, D_BAD])
    assert d.action == "retrieve" and d.weakest_aspect == "a1"
    assert "magnetic" in d.hints["missing_terms"]


def test_stop_when_all_covered():
    d = ctrl().decide([D_MIG, D_MAG])
    assert d.action == "stop" and d.stop_reason == "sufficient"


def test_max_rounds():
    c = ctrl(max_rounds=1, min_gain=-1)  # disable stagnation
    assert c.decide([D_BAD]).action == "retrieve"
    d = c.decide([D_BAD])
    assert d.action == "stop" and d.stop_reason == "max_rounds"


def test_doc_and_token_budget():
    assert ctrl(max_docs=1).decide([D_MIG, D_BAD]).stop_reason == "doc_budget"
    assert ctrl(max_tokens=3).decide([D_MIG]).stop_reason == "token_budget"


def test_stagnation():
    c = ctrl(max_rounds=5, patience=1)
    c.decide([D_BAD])
    d = c.decide([D_BAD])  # new round gave nothing
    assert d.stop_reason == "stagnation"


def test_verification_forces_retrieval():
    c = ctrl()
    d = c.decide([D_MIG, D_MAG], verification={"supported": False, "unsupported_aspects": ["a0"]})
    assert d.action == "retrieve" and d.weakest_aspect == "a0" and d.hints["verification_triggered"]


def test_reflection_lowers_irrelevant_doc():
    c = ctrl(reflection_weight=1.0)
    d = c.decide([D_MIG, D_MAG], reflections={"a0": {"m": 0.0}})
    assert d.weakest_aspect == "a0"


def test_scores_bounded_and_summary():
    c = ctrl()
    c.decide([D_MIG, D_BAD])
    s = c.summary()
    assert all(0 <= v <= 1 for v in s["final_coverage"].values())
    assert s["rounds_used"] == 1 and len(s["coverage_trajectory"]) == 1
