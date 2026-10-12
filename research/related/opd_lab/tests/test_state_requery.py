import torch

from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.state_requery import intervene
from opdlab.trajectory_probe import rollout


def test_requery_keeps_current_prefix_and_uses_current_student_continuation():
    b = ToyBackend(Experiment(backend="toy", precision="no", steps=4))
    b.load_student({"weight": torch.tensor([.4, -.6])}, refresh_behavior=False)
    q = {"seed": 42, "index": 1, "context": "fixture"}
    states, actions = rollout(b, q, list(range(4)), 4)
    ss, aa = intervene(b, states, actions, q, actions[1], 4)
    assert all(torch.equal(x, y) for x, y in zip(ss, states))
    assert all(torch.equal(x, y) for x, y in zip(aa, actions))
    correction = actions[1] + torch.ones_like(actions[1])
    ss, aa = intervene(b, states, actions, q, correction, 4)
    assert all(torch.equal(ss[i], states[i]) for i in (0, 1))
    assert not torch.equal(ss[2], states[2])
    assert torch.equal(aa[1], correction)
    assert torch.equal(aa[2], b.predict(ss[2], 2, q["context"], "student"))
    assert not torch.equal(aa[2], b.predict(ss[2], 2, q["context"], "behavior"))


def test_visual_repair_cannot_trade_away_visibility_or_hide_unknowns():
    import runpy
    from opdlab.io import ROOT
    compare = runpy.run_path(str(ROOT / "scripts/review_state_requery.py"))["compare_observations"]
    base = {"clauses": {"relation": False, "text": True}, "full_visibility": True}
    good = {"clauses": {"relation": True, "text": True}, "full_visibility": True}
    assert compare(base, good)["accepted_encoded_repair"]
    cropped = {**good, "full_visibility": False}
    assert not compare(base, cropped)["accepted_encoded_repair"]
    assert compare(base, cropped)["known_harm"]
    unclear = {"clauses": {"relation": True, "text": None}, "full_visibility": True}
    assert not compare(base, unclear)["accepted_encoded_repair"]
    assert compare(base, unclear)["has_unknown"] and not compare(base, unclear)["known_harm"]
