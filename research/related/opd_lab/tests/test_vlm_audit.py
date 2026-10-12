import json
import pytest
from opdlab.vlm_audit import OBSERVATION_PROMPT, parse_response, scene_request


def test_requests_do_not_leak_target_word_or_prior_scores():
    r={'id':'secret_identity','prompt':'secret_prompt','constraints':[
        {'id':'text','kind':'ocr','text':'SECRETWORD'},
        {'id':'count','kind':'geneval','metadata':{'tag':'counting','include':[{'class':'cup','count':2}]}}]}
    request=scene_request(r)
    for marker in ('SECRETWORD','secret_prompt','secret_identity'):
        assert marker not in request and marker not in OBSERVATION_PROMPT
    assert 'cup' in request


def test_parser_preserves_unknown_and_rejects_coercion_or_missing_clauses():
    value={'clauses':[{'id':'count','satisfied':None,'evidence':'occluded'}],
           'full_visibility':False,'visibility_evidence':'cropped'}
    assert parse_response('```json\n'+json.dumps(value)+'\n```','scene',['count'])==value
    for bad in ({**value,'full_visibility':'false'},{**value,'full_visibility':0},
                {**value,'clauses':[]},{**value,'clauses':value['clauses']*2}):
        with pytest.raises(ValueError):parse_response(json.dumps(bad),'scene',['count'])
    with pytest.raises(ValueError,match='Duplicate'):
        parse_response('{"text_lines":[],"text_lines":[],"text_uncertain":false,"scene_description":"none"}','observation')
    with pytest.raises(ValueError):parse_response('not JSON','scene',['count'])
