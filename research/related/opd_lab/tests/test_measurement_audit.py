import numpy as np
import pytest

from opdlab.measurement_audit import ocr_checks, trace_boxes


def test_compact_reading_preserves_extra_content_and_unknown_target_scope():
    r = ocr_checks('S U N N Y', 'Sunny', 'SUNNY')
    assert r['v1'][0] is False and r['both_compact'] == (True, 1.)
    for text in ('SUNNY SUNNY', 'S SUNNY', 'SUNNY 123', 'SUNNY®', 'SUNNY!'):
        assert not ocr_checks(text, 'SUNNY', 'SUNNY')['both_compact'][0]
    assert not ocr_checks('SUNNY', '', 'SUNNY')['both_compact'][0]
    with pytest.raises(ValueError, match='alphabetic'):
        ocr_checks('HELLO WORLD', 'HELLO WORLD', 'HELLO WORLD')


def test_detector_trace_keeps_upstream_strict_threshold_and_full_raw_boxes():
    constants = dict(THRESHOLD=.3, COUNTING_THRESHOLD=.9, MAX_OBJECTS=16, NMS_THRESHOLD=1.)
    record = {'constraints':[{'id':'count','kind':'geneval','metadata':{'tag':'counting','include':[{'class':'cup','count':2}]}},
                              {'id':'color','kind':'geneval','metadata':{'tag':'colors','include':[{'class':'cup','count':2,'color':'red'}]}}]}
    boxes = [np.array([[0,10,30,40,.95],[50,10,80,40,.9],[10,20,30,40,.3]])]
    result = trace_boxes(boxes,['cup'],record,(100,100),constants)
    assert len(result['raw_requested_class_boxes']['cup']) == 3
    assert result['clauses'][0]['objects']['cup']['count'] == 1
    assert result['clauses'][1]['objects']['cup']['count'] == 2
    assert not result['clauses'][0]['count_component_satisfied']
    assert result['frame_flag']
