import json

from controller import AdaptiveController, ControllerConfig, LexicalScorer, normalize_aspects
from controller.aspect_extractor import (Aspect, extract_aspects, from_dataset, get_aspects,
                                         heuristic_extract, make_aspect_id, to_dicts)

POST = ("I keep a small garden and I'm confused about composting. Why do my compost piles smell "
        "like ammonia? How do I balance green and brown materials? Also, is it safe to add citrus "
        "peels and cooked food, and how long does the process usually take?")


def test_make_aspect_id_matches_brightpro_convention():
    assert make_aspect_id(1, "0", "biology") == "biology-0-a1"
    assert make_aspect_id(3) == "a3"


def test_heuristic_splits_multi_part_question():
    asp = heuristic_extract(POST, qid="7", task="sustainable_living")
    assert 2 <= len(asp) <= 5
    ids = [a.id for a in asp]
    assert ids == sorted(set(ids), key=ids.index)               # unique
    assert ids[0] == "sustainable_living-7-a1"
    assert all(a.description and a.keywords for a in asp)
    assert any("ammonia" in a.description for a in asp)
    assert any("citrus" in a.description for a in asp)


def test_heuristic_respects_max_and_dedupes():
    q = ("Why is the sky blue during daytime? Why is the sky blue during daytime? "
         "Why is the sky blue during daytime hours? How do rainbows form after heavy rain?")
    asp = heuristic_extract(q, max_aspects=5)
    assert len(asp) == 2
    assert len(heuristic_extract(POST, max_aspects=2)) <= 2


def test_heuristic_edge_cases():
    assert heuristic_extract("") == []
    assert len(heuristic_extract("birds")) == 1                 # too short -> whole-question fallback
    code = "```python\nprint(1)\n```\nHow can I reverse a list in place efficiently?"
    asp = heuristic_extract(code)
    assert len(asp) == 1 and "print" not in asp[0].description


def test_from_dataset_reuses_existing_aspects_and_unique_ids():
    raw = [{"id": "biology-0-a1", "description": "why birds migrate"},
           {"id": "biology-0-a1", "description": "how birds navigate"},   # duplicate id
           {"id": "biology-0-a3", "description": "  "}]                   # empty -> dropped
    asp = from_dataset(raw)
    assert [a.id for a in asp] == ["biology-0-a1", "biology-0-a1_2"]
    assert all(a.source == "dataset" for a in asp)


def test_get_aspects_priority_dataset_then_llm_then_heuristic():
    q = {"qid": "0", "query": POST, "aspects": [{"id": "biology-0-a1", "description": "x y z"}]}
    assert get_aspects(q, llm_fn=lambda p: "boom")[0].source == "dataset"      # dataset wins
    q2 = {"qid": "0", "query": POST, "aspects": None}
    assert get_aspects(q2, task="t")[0].source == "heuristic"
    assert get_aspects(q2, use_dataset=False)[0].source == "heuristic"


def test_llm_extraction_parses_fenced_json():
    reply = '```json\n{"aspects": [{"description": "causes of ammonia smell", "keywords": ["Ammonia"]},' \
            ' "ideal green brown ratio"]}\n```'
    asp = extract_aspects(POST, llm_fn=lambda p: reply)
    assert [a.source for a in asp] == ["llm", "llm"]
    assert asp[0].keywords == ["ammonia"] and asp[1].description == "ideal green brown ratio"


def test_llm_failure_falls_back_to_heuristic():
    for bad in (lambda p: "not json", lambda p: json.dumps({"aspects": []}), lambda p: 1 / 0):
        asp = extract_aspects(POST, llm_fn=bad)
        assert asp and all(a.source == "heuristic" for a in asp)


def test_output_is_controller_compatible():
    asp = heuristic_extract(POST, qid="1", task="x")
    norm = normalize_aspects(to_dicts(asp))
    assert [n["id"] for n in norm] == [a.id for a in asp]
    assert [n["text"] for n in norm] == [a.description for a in asp]
    ctrl = AdaptiveController(to_dicts(asp), ControllerConfig(), LexicalScorer())
    d = ctrl.decide([{"doc_id": "d", "text": "Compost smells of ammonia when too much nitrogen."}])
    assert d.action in ("retrieve", "stop") and set(d.coverage) == {a.id for a in asp}


def test_search_query_adds_only_new_terms():
    a = Aspect("a1", "why compost smells of ammonia")
    assert a.search_query(["ammonia", "nitrogen"]) == "why compost smells of ammonia nitrogen"