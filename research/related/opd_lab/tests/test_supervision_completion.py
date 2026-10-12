import pytest

from opdlab.supervision_completion import tasks


def test_completion_selects_only_missing_slots_and_replays_each_observed_path():
    plan={"arms":["single"],"evaluation":[{"pair_id":"a","id":"p","seed":1,"single_action":True},
                                               {"pair_id":"b","id":"q","seed":2,"single_action":False}]}
    _,all_rows=tasks(plan,[])
    assert len(all_rows)==5
    existing=all_rows[:3]
    replay,missing=tasks(plan,existing)
    assert missing==all_rows[3:]
    assert replay==[existing[0],existing[2]]
    _,none=tasks(plan,all_rows)
    assert none==[]
    with pytest.raises(AssertionError):
        tasks(plan,existing+[existing[0]])
    with pytest.raises(AssertionError):
        tasks(plan,[{**existing[0],"seed":99}])
