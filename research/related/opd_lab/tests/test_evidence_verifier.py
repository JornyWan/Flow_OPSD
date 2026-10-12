from opdlab.evidence_verifier import choose,consensus,text_label,text_rows,visibility_support
import json


def line(text,x=0,y=0,confidence=.99):
    return {"text":text,"confidence":confidence,"box":[[x,y],[x+10,y],[x+10,y+10],[x,y+10]]}


def test_spatial_assembly_keeps_extra_text_and_duplicate_characters():
    rows = text_rows([line('N',40),line('L',0),line('U',20),line('A',60)])
    assert rows[0]['text']=='luna'
    assert text_label([rows]*4,'LUNA') is True
    extras = text_rows([line('LUNA'),line('LABEL',0,30)])
    assert text_label([extras]*4,'LUNA') is False
    assert text_label([[{'text':'LUNAA','confidence':.99}]]*4,'LUNA') is False


def test_conflicting_low_confidence_and_missing_readings_abstain():
    good=[{'text':'LUNA','confidence':.99}]
    assert text_label([good,[{'text':'Lina','confidence':.99}]],'LUNA') is None
    assert text_label([good,[]],'LUNA') is None
    assert text_label([[{'text':'LUNA','confidence':.3}]]*4,'LUNA') is None
    assert consensus([True,False]) is None


def test_selection_blocks_unknown_baseline_and_partial_progress_harm():
    def endpoint(text,relation=False,visible=True):
        return {'clauses':{'text':text,'relation':relation},'full_visibility':visible}
    assert choose({'noop':endpoint(None),'ocr@1':endpoint(True,True)},{'ocr@1':1})['selected']=='noop'
    assert choose({'noop':endpoint(True),'ocr@1':endpoint(False,True)},{'ocr@1':1})['selected']=='noop'
    assert choose({'noop':endpoint(False),'ocr@1':endpoint(False)},{'ocr@1':1})['selected']=='noop'
    assert choose({'noop':endpoint(False),'ocr@1':endpoint(True)},{'ocr@1':1})['selected']=='ocr@1'
    assert choose({'noop':endpoint(False),'ocr@1':endpoint(True,visible=False)},{'ocr@1':1})['selected']=='noop'


def test_bbox_visibility_uses_original_frame_after_padding_and_missing_objects():
    constraint=[{'metadata':{'include':[{'class':'cup','count':1}]}}]
    assert visibility_support([{'boxes':{'cup':[[30,30,90,90,.99]]}}],constraint,(100,100))
    assert visibility_support([{'boxes':{'cup':[[46,46,106,106,.99]]}}],constraint,(100,100),16)
    assert not visibility_support([{'boxes':{'cup':[[16,30,60,60,.99]]}}],constraint,(100,100),16)
    assert not visibility_support([{'boxes':{}}],constraint,(100,100))


def test_saved_evidence_roundtrip_preserves_canonical_tie_break():
    base={'clauses':{'text':False},'full_visibility':True}
    good={'clauses':{'text':True},'full_visibility':True}
    evidence={'noop':base,'ocr@1':good,'geneval@1':good}
    distances={'ocr@1':.1,'geneval@1':.1}
    saved=json.loads(json.dumps(evidence,sort_keys=True))
    assert choose(evidence,distances)==choose(saved,distances)
    assert choose(saved,distances)['selected']=='ocr@1'
