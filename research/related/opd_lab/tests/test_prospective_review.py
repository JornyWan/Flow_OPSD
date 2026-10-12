import runpy

from opdlab.io import ROOT


def functions():
    return runpy.run_path(str(ROOT / "scripts/review_prospective.py"))


def test_unknown_baseline_cannot_certify_safe_repair_and_harm_still_counts():
    compare = functions()["compare"]
    base = {"clauses":{"text":None,"relation":False},"full_visibility":True}
    target = {"clauses":{"text":True,"relation":True},"full_visibility":True}
    assert compare(base,target)["status"] == "unknown"
    cropped = {**target,"full_visibility":False}
    result = compare(base,cropped)
    assert result["status"] == "unsafe" and result["known_harm"] and result["has_unknown"]
    base["clauses"]["text"] = True
    assert compare(base,target)["safe_repair"]
    assert compare(base,base)["status"] == "non_repair"


def test_ratio_preserves_zero_denominators_and_joint_prompt_resampling():
    ratio = functions()["ratio"]
    empty = ratio([False]*4,[False]*4)
    assert empty["mean"] is None and empty["ci95"] is None
    perfect = ratio([True,False,True,False],[True,False,True,False])
    assert perfect["numerator"] == perfect["denominator"] == 2
    assert perfect["mean"] == 1 and perfect["ci95"] == [1,1]
    assert 0 < perfect["bootstrap_nonzero_denominator"] < 2000


def test_complete_scene_does_not_treat_unknown_as_success():
    visual = functions()["visual"]
    obs = {"clauses":{"text":True,"count":None},"full_visibility":True}
    assert not visual(obs)["complete_scene"] and visual(obs)["has_unknown"]
