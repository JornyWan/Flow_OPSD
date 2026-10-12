import pytest
from opdlab.relation_audit import relation_request, parse_relation


def record(items):
    return {'prompt': 'SECRETWORD', 'constraints': [
        {'id': 'text', 'kind': 'ocr', 'text': 'SECRETWORD'},
        {'id': 'relation', 'kind': 'geneval', 'metadata': {'include': items}}]}


def test_relation_compiler_uses_subject_reference_indices_not_list_order():
    r = record([{'class': 'bowl', 'count': 1},
                {'class': 'apple', 'count': 1, 'position': ['right of', 0]}])
    s = relation_request(r)
    assert 'The apple must be to the RIGHT OF the bowl' in s
    assert 'SECRETWORD' not in s
    assert 'The apple must be to the LEFT OF the bowl' in relation_request(r, True)
    reordered = record([{'class': 'apple', 'count': 1, 'position': ['right of', 1]},
                        {'class': 'bowl', 'count': 1}])
    assert relation_request(reordered) == s
    r['constraints'][1]['metadata']['include'][1]['position'][1] = 1
    with pytest.raises(ValueError): relation_request(r)


def test_nullable_evidence_does_not_erase_a_valid_relation_abstention():
    assert parse_relation('{"satisfied":null,"evidence":null}') == {'satisfied': None, 'evidence': None}
    for s in ('{"satisfied":1,"evidence":null}', '{"satisfied":true,"evidence":null,"extra":1}'):
        with pytest.raises(ValueError): parse_relation(s)
