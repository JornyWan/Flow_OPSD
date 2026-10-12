"""Strict, audit-only VLM requests and parsing; independent of training imports."""
import json
import re


OBSERVATION_PROMPT = '''Inspect the image itself. Transcribe ALL visible letters, words, digits and symbols anywhere in the image, including small object labels. Copy literally: do not correct spelling, complete missing letters, or infer intended words. If some apparent text cannot be read reliably, set text_uncertain=true and keep any readable fragments. No target text is supplied. Also briefly describe the visible objects and any cropping or occlusion, without guessing hidden objects. Return only JSON with exactly these fields: {"text_lines": ["literal visible line"], "text_uncertain": false, "scene_description": "brief literal description"}. Use an empty text_lines list if no text is visible; distinguish that from unreadable apparent text.'''

SCENE_INSTRUCTION = '''Judge the following explicit scene constraints against the image. For each clause, enforce required object identity and EXACT count, then color or relative position if specified. Count distinct objects, not a group as one; do not invent hidden objects. Return null if identity/count/color/position cannot be reliably determined. full_visibility is separate: all required objects must be present, complete inside the image, and without substantial occlusion; missing objects or clear cropping fail it. Do not judge text and do not infer a scene from the requested constraints. Return only JSON with exactly {"clauses": [{"id": "requested id", "satisfied": true, "evidence": "visible evidence"}], "full_visibility": false, "visibility_evidence": "visible evidence"}. Every requested clause id must occur exactly once. satisfied and full_visibility must be true, false, or null. Constraints: '''


def scene_request(record):
    clauses=[{'id':c['id'],'metadata':c['metadata']} for c in record['constraints'] if c['kind']=='geneval']
    if not clauses:raise ValueError('No scene constraints')
    return SCENE_INSTRUCTION+json.dumps(clauses,sort_keys=True)


def _object(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('Duplicate JSON key')
        out[k]=v
    return out


def parse_response(raw,stage,clause_ids=()):
    value=raw.strip()
    if value.startswith('```'):
        match=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',value,re.S)
        if not match:raise ValueError('Malformed JSON fence')
        value=match.group(1)
    obj=json.loads(value,object_pairs_hook=_object,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(obj,dict):raise ValueError('Expected JSON object')
    if stage=='observation':
        if (set(obj)!={'text_lines','text_uncertain','scene_description'} or
            not isinstance(obj['text_lines'],list) or any(not isinstance(x,str) for x in obj['text_lines']) or
            type(obj['text_uncertain']) is not bool or not isinstance(obj['scene_description'],str)):
            raise ValueError('Invalid observation schema')
    elif stage=='scene':
        if (set(obj)!={'clauses','full_visibility','visibility_evidence'} or not isinstance(obj['clauses'],list) or
            not (type(obj['full_visibility']) is bool or obj['full_visibility'] is None) or not isinstance(obj['visibility_evidence'],str)):
            raise ValueError('Invalid scene schema')
        seen=[]
        for c in obj['clauses']:
            if (not isinstance(c,dict) or set(c)!={'id','satisfied','evidence'} or not isinstance(c['id'],str) or
                not (type(c['satisfied']) is bool or c['satisfied'] is None) or not isinstance(c['evidence'],str)):
                raise ValueError('Invalid clause schema')
            seen.append(c['id'])
        if len(seen)!=len(set(seen)) or set(seen)!=set(clause_ids):raise ValueError('Missing/duplicate/unexpected clause')
    else:raise ValueError('Unknown stage')
    return obj
