import pytest
from opdlab.audit_metrics import confusion, conjunction


def test_abstentions_cannot_inflate_usable_recall_or_turn_into_negatives():
    s = confusion([(True, True), (None, True), (False, False), (None, False), (True, None)])
    assert (s['TP'], s['FP'], s['TN'], s['FN']) == (1, 0, 1, 0)
    assert s['recall_both_known'] == 1
    assert s['recall_including_abstentions'] == .5
    assert s['coverage_on_reference_known'] == .5
    assert s['reference_unknown'] == 1 and s['prediction_unknown'] == 2
    assert s['predicted_positive_reference_unknown'] == 1
    assert confusion([(None, None)])['recall_including_abstentions'] is None
    with pytest.raises(ValueError): confusion([(0, False)])


def test_joint_keeps_uncertainty_unless_an_observed_failure_decides_it():
    assert conjunction([True, None]) is None
    assert conjunction([False, None]) is False
    assert conjunction([True, True]) is True
    with pytest.raises(ValueError): conjunction([])
