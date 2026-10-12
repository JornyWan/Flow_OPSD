import json

from opdlab.paired_audit import parse_pair
from opdlab.paired_array_diagnostic import parse_array_compatible


def test_container_mapping_preserves_all_fields_and_original_parser():
    fields = [{"id": "text", "A": False, "B": True, "change": "repair", "evidence": "A: SUNNNY; B: SUNNY"},
              {"id": "full_visibility", "A": None, "B": None, "change": "unknown", "evidence": None}]
    raw = json.dumps(fields)
    expected = parse_pair(json.dumps({"clauses": fields}), ("text", "full_visibility"))
    assert parse_array_compatible(raw, ("text", "full_visibility")) == expected
    assert parse_array_compatible('```json\n' + raw + '\n```', ("text", "full_visibility")) == expected
    assert parse_pair(raw, ("text", "full_visibility"))[1] == ["Expected object"]
    assert expected[0]["text"]["evidence"] == "A: SUNNNY; B: SUNNY"


def test_no_value_recovery_or_duplicate_key_salvage():
    for raw in ('null', '{"clauses": [], "clauses": []}', '[{"id":"text","id":"count"}]',
                '[{"id":"text","A":NaN}]', 'prefix [{"id":"text"}]'):
        assert parse_array_compatible(raw, ("text",)) == parse_pair(raw, ("text",))
    bad = json.dumps([{"id": "text", "A": True, "B": True, "change": "repair", "evidence": "claimed repair"}])
    result, errors = parse_array_compatible(bad, ("text",))
    assert errors and result["text"]["A"] is None and result["text"]["B"] is None
