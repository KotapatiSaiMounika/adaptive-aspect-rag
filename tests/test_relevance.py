import pytest

from controller import AdaptiveController, ControllerConfig, LexicalScorer
from controller.aspect_extractor import to_dicts, heuristic_extract
from controller.relevance import (AspectRelevanceScorer, ReflectionEngine, reflect,
                                  summarize_support)

ASPECTS = [{"id": "q-a1", "description": "why birds migrate in winter"},
           {"id": "q-a2", "description": "magnetic field navigation"}]
D_MIG = {"doc_id": "m", "text": "Birds migrate in winter to warmer places. They follow food."}
D_MAG = {"doc_id": "g", "text": "Magnetic field helps navigation in birds."}
D_BAD = {"doc_id": "x", "text": "The stock market fell yesterday after rates rose."}
EV = [D_MIG, D_MAG, D_BAD]


class CountingScorer:
    def __init__(self):
        self.calls = 0

    def score(self, a, d):
        self.calls += 1
        return 0.5


def test_reflection_format_and_bounds():
    r = reflect(ASPECTS, EV)
    assert set(r) == {"q-a1", "q-a2"}
    for row in r.values():
        assert set(row) == {"m", "g", "x"}
        assert all(isinstance(s, float) and 0.0 <= s <= 1.0 for s in row.values())


def test_relevant_docs_outscore_irrelevant():
    r = reflect(ASPECTS, EV)
    assert r["q-a1"]["m"] > r["q-a1"]["x"] and r["q-a2"]["g"] > r["q-a2"]["x"]
    assert r["q-a1"]["x"] == 0.0


def test_accepts_aspect_objects_from_extractor():
    asp = heuristic_extract("Why do birds migrate in winter? How do birds use magnetic fields to navigate?",
                            qid="1", task="bio")
    r = reflect(to_dicts(asp), EV)
    assert set(r) == {a.id for a in asp}


def test_passage_level_beats_scattered_mentions():
    filler = " ".join(f"Filler sentence number {i} about nothing." for i in range(30))
    focused = f"{filler} Magnetic field navigation guides birds. {filler}"
    scattered = (f"Magnetic fields exist. {filler} Navigation is hard. {filler} "
                 f"Something else entirely. {filler} Field notes.")
    s = AspectRelevanceScorer(LexicalScorer(), passage_weight=1.0)
    assert s.score("magnetic field navigation", focused) > s.score("magnetic field navigation", scattered)


def test_passage_weight_zero_equals_base_scorer():
    base = LexicalScorer()
    s = AspectRelevanceScorer(base, passage_weight=0.0)
    assert s.score(ASPECTS[0]["description"], D_MIG["text"]) == pytest.approx(
        base.score(ASPECTS[0]["description"], D_MIG["text"]), abs=1e-4)


def test_explain_returns_snippet_and_handles_empty():
    e = AspectRelevanceScorer().explain("magnetic field navigation", D_MAG["text"])
    assert set(e) == {"score", "doc_score", "passage_score", "snippet"} and "Magnetic" in e["snippet"]
    assert AspectRelevanceScorer().score("", D_MAG["text"]) == 0.0
    assert AspectRelevanceScorer().score("magnetic", "") == 0.0


def test_judge_fn_is_blended_and_clamped():
    s = AspectRelevanceScorer(CountingScorer(), passage_weight=0.0, judge_fn=lambda a, p: 7.0, judge_weight=0.5)
    assert s.score("a b", "some text") == pytest.approx(0.75)         # 0.5*0.5 + 0.5*clamp(7)=1.0


def test_cache_scores_only_new_docs():
    cs = CountingScorer()
    eng = ReflectionEngine(ASPECTS, AspectRelevanceScorer(cs, passage_weight=0.0))
    eng.reflect([D_MIG])
    n1 = cs.calls
    eng.reflect([D_MIG, D_MAG])
    assert cs.calls - n1 == len(ASPECTS)                              # only the new doc, per aspect


def test_long_documents_are_capped():
    s = AspectRelevanceScorer(max_chunks=5)
    text = " ".join(f"Sentence {i} here." for i in range(500))
    assert len(s.split_passages(text)) == 5


def test_summarize_support():
    sm = summarize_support(reflect(ASPECTS, EV), threshold=0.38)
    assert sm["q-a1"]["best_doc"] == "m" and sm["q-a1"]["supported"]
    assert "x" not in sm["q-a1"]["supporting_docs"]


def test_empty_aspects_raises():
    with pytest.raises(ValueError):
        ReflectionEngine([])


def test_integration_with_controller_reflections():
    eng = ReflectionEngine(ASPECTS)
    ctrl = AdaptiveController(ASPECTS, ControllerConfig(reflection_weight=0.4), LexicalScorer())
    d = ctrl.decide([D_MIG, D_BAD], reflections=eng.reflect([D_MIG, D_BAD]))
    assert d.action == "retrieve" and d.weakest_aspect == "q-a2"
    d = ctrl.decide([D_MIG, D_MAG], reflections=eng.reflect([D_MIG, D_MAG]))
    assert d.action == "stop" and d.stop_reason == "sufficient"


def test_relevance_scorer_works_as_controller_scorer():
    ctrl = AdaptiveController(ASPECTS, ControllerConfig(), AspectRelevanceScorer())
    assert ctrl.decide([D_MIG, D_MAG]).action == "stop"


def test_accepts_aspect_dataclass_objects():
    asp = heuristic_extract("Why do birds migrate in winter? How do birds use magnetic fields to navigate?", qid="1")
    r = reflect(asp, EV)                       # Aspect objects, not dicts
    assert set(r) == {a.id for a in asp}
    