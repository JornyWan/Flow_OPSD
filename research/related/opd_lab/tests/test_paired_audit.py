import json
from itertools import product

import pytest

from opdlab.paired_audit import (agreed_endpoints, change, choose_pairs, endpoints,
                                 guarded_endpoints, literal_label, parse_observation,
                                 parse_pair, requirements)


def field(a, b):
    return {"A": a, "B": b, "change": change(a, b), "evidence": "A: seen; B: seen"}


def encoded(fields):
    return json.dumps({"clauses": [{"id": k, **v} for k, v in fields.items()]})


def test_nullable_fields_preserve_other_valid_clauses():
    fields = {"text": field(False, True), "count": field(None, None), "full_visibility": field(True, True)}
    fields["count"]["evidence"] = None
    result, errors = parse_pair(encoded(fields), fields)
    assert not errors and result == fields
    fields["count"]["B"] = "true"
    result, errors = parse_pair(encoded(fields), fields)
    assert errors and result["text"] == fields["text"] and result["full_visibility"] == fields["full_visibility"]
    assert result["count"]["B"] is None


def test_invalid_and_contradictory_claims_cannot_become_repairs():
    for raw in ('null', '{"clauses": [], "clauses": []}', '{"clauses": NaN}', 'arbitrary prose'):
        parsed, errors = parse_pair(raw, ("text", "full_visibility"))
        assert errors and all(c["A"] is None and c["B"] is None for c in parsed.values())
    fields = {"text": field(False, True), "full_visibility": field(True, True)}
    fields["text"]["change"] = "same"
    parsed, errors = parse_pair(encoded(fields), fields)
    assert errors and parsed["text"]["A"] is None and parsed["full_visibility"]["A"] is True
    fields["text"] = {**field(False, True), "evidence": None}
    parsed, errors = parse_pair(encoded(fields), fields)
    assert errors and parsed["text"]["B"] is None


def test_reverse_alignment_and_disagreements_exhaustively():
    for a, b, c, d in product((False, True, None), repeat=4):
        forward = {"text": field(a, b), "full_visibility": field(True, True)}
        reverse = {"text": field(d, c), "full_visibility": field(True, True)}
        base, candidate = agreed_endpoints(forward, reverse)
        assert base["clauses"]["text"] is (a if a is c else None)
        assert candidate["clauses"]["text"] is (b if b is d else None)
        assert endpoints(reverse, True)[0]["clauses"]["text"] is c


def test_target_free_guard_keeps_extra_letters_and_small_labels():
    for text in ("SUNNNY", "SUNNY.", "SUNNY extra", "S UNNY"):
        obs, errors = parse_observation(json.dumps({"text_lines": [text], "text_uncertain": False, "scene_description": None}))
        assert not errors and literal_label(obs, "SUNNY") is False
    obs = {"text_lines": ["SUNNY", "label"], "text_uncertain": False}
    assert literal_label(obs, "SUNNY") is False
    obs["text_lines"] = [" sunny "]
    assert literal_label(obs, "SUNNY") is True
    obs["text_uncertain"] = True
    assert literal_label(obs, "SUNNY") is None
    pair = endpoints({"text": field(False, True), "full_visibility": field(True, True)})
    assert guarded_endpoints(pair, [True, True])[0]["clauses"]["text"] is None


def test_explicit_relation_subject_and_reference():
    record = {"constraints": [{"id": "text", "kind": "ocr", "text": "PINE"},
              {"id": "relation", "kind": "geneval", "metadata": {"tag": "position", "include":
               [{"class": "bottle", "count": 1}, {"class": "cup", "count": 1, "position": ["left of", 0]}]}}]}
    req = requirements(record)
    assert "The cup must be to the LEFT OF the bottle" in req[1]["requirement"]
    assert "Exactly 1 object(s) of class bottle" in req[1]["requirement"]
    record["constraints"][1]["metadata"]["include"][1]["position"][1] = 1
    with pytest.raises(ValueError):
        requirements(record)


def test_safe_selection_cannot_credit_unknown_or_same_image():
    safe = endpoints({"text": field(False, True), "count": field(True, True), "full_visibility": field(True, True)})
    harm = endpoints({"text": field(False, True), "count": field(True, False), "full_visibility": field(True, True)})
    assert choose_pairs({"ocr@1": safe, "geneval@1": harm}, {"ocr@1": 2, "geneval@1": 1}) == "ocr@1"
    unknown = endpoints({"text": field(None, True), "count": field(True, True), "full_visibility": field(True, True)})
    assert choose_pairs({"ocr@1": unknown, "geneval@1": harm}, {"ocr@1": 2, "geneval@1": 1}) == "noop"
    for labels in product((False, True, None), repeat=3):
        self_pair = endpoints({k: field(v, v) for k, v in zip(("text", "count", "full_visibility"), labels)})
        assert choose_pairs({"ocr@1": self_pair, "geneval@1": self_pair}, {"ocr@1": 1, "geneval@1": 1}) == "noop"
    assert choose_pairs(json.loads(json.dumps({"geneval@1": safe, "ocr@1": safe}, sort_keys=True)), {"ocr@1": 1, "geneval@1": 1}) == "ocr@1"
