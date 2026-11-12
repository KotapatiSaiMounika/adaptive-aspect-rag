from generation.answer_generator import extract_claims
from generation import llm
from generation.quality import verify

def test_extract_claims_parses_citations():
    claims = extract_claims("Birds use magnetic fields to navigate [d1]. They also use stars at night [d2, d3].",
                            {"d1", "d2", "d3"})
    assert claims[0]["cited"] == ["d1"]
    assert claims[1]["cited"] == ["d2", "d3"]

def test_invalid_citation_is_dropped():
    claims = extract_claims("Some long enough factual sentence here [fake].", {"d1"})
    assert claims[0]["cited"] == []

def test_verify_uses_labels(monkeypatch):
    def fake(system, user, **kw):
        if "CLAIM" in user:
            return {"labels": [{"id": 0, "label": "fully"}]}
        return {"utility": 5, "aspects": [{"id": 0, "covered": True}]}
    monkeypatch.setattr(llm, "chat_json", fake)
    ans = {"text": "x", "claims": [{"text": "Birds navigate by magnetism.", "cited": ["d1"]}]}
    rep = verify("q", ans, [{"doc_id": "d1", "text": "Birds navigate by magnetism."}], aspects=["navigation"])
    assert rep["support_rate"] == 1.0 and rep["needs_retrieval"] is False

def test_unicode_hyphen_in_citation_is_matched():
    text = "Studies show LED lights attract fewer insects than mercury lamps. [biology\u20110/extraction_5.txt]"
    claims = extract_claims(text, {"biology-0/extraction_5.txt"})
    assert len(claims) == 1 and claims[0]["cited"] == ["biology-0/extraction_5.txt"]

def test_citation_after_period_attaches_to_claim():
    text = "- LED lights attract fewer insects than mercury lamps. [d1]\n**A heading line**\n"
    claims = extract_claims(text, {"d1"})
    assert len(claims) == 1 and claims[0]["cited"] == ["d1"]

def test_missing_file_extension_is_matched():
    text = "LED lamps attract fewer insects than mercury-vapor lamps. [biology\u20110/extraction_5]"
    claims = extract_claims(text, {"biology-0/extraction_5.txt"})
    assert claims[0]["cited"] == ["biology-0/extraction_5.txt"]

def test_bullet_end_citation_covers_whole_bullet():
    text = "* The nasal cycle alternates congestion between sides every few hours. It is a normal process. [d1]"
    claims = extract_claims(text, {"d1"})
    assert len(claims) == 2 and all(c["cited"] == ["d1"] for c in claims)