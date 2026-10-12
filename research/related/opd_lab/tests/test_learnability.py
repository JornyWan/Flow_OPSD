import copy
import json

import pytest
import torch

from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT
from opdlab.learnability import action_metrics, run_fixed, select_positives
from opdlab.verifiers import ToyVerifier


def test_positive_selection_rejects_unknowns_and_collateral_damage():
    case = {"case": 1, "id": "x", "mapping": {"A": {"action": "noop", "row": {}},
                                               "B": {"action": "target", "row": {}}}}
    observation = {"case": 1, "better": "B", "observations": {
        "A": {"clauses": {"text": True, "count": False}, "full_visibility": False},
        "B": {"clauses": {"text": True, "count": True}, "full_visibility": True}}}
    assert select_positives([case], [observation])[0]["repaired"] == ["count"]
    for key, value in (("text", False), ("count", None)):
        changed = copy.deepcopy(observation)
        changed["observations"]["B"]["clauses"][key] = value
        assert not select_positives([case], [changed])
    changed = copy.deepcopy(observation)
    changed["observations"]["A"]["full_visibility"] = None
    assert not select_positives([case], [changed])


def test_relative_error_distinguishes_underfit_and_off_direction():
    noop, target = torch.zeros(2), torch.tensor([2., 0.])
    halfway = action_metrics(torch.tensor([1., 0.]), target, noop)
    assert halfway["relative_target_mse"] == .25 and halfway["gamma"] == .5
    off_ray = action_metrics(torch.tensor([2., 2.]), target, noop)
    assert off_ray["gamma"] == 1 and off_ray["orthogonal_mse_over_initial"] == 1
    with pytest.raises(ValueError):
        action_metrics(noop, noop, noop)


def test_conflicting_single_fits_start_identically_and_joint_cancels(tmp_path):
    c = Experiment(backend="toy", precision="no", steps=4, learning_rate=.1, weight_decay=0)
    b, verifier = ToyBackend(c), ToyVerifier(c)
    initial_hash = tensor_digest(b.behavior_state())
    records = load_records(ROOT / "data/demo/toy.jsonl", "train")
    queries = [{"record": r, "seed": 42 + i, "index": 1, "state": torch.zeros(1, 2),
                "context": "", "dt": -.25, "accepted": True,
                "target": torch.full((1, 2), 1. if i == 0 else -1.)} for i, r in enumerate(records)]
    plan = {"selected": [{"case": i, "id": r["id"], "old_endpoints": {
                          name: {"image_sha256": "fixture"} for name in ("noop", "target")}}
                         for i, r in enumerate(records)], "max_updates": 2, "milestones": [0, 1, 2]}
    run_fixed(b, verifier, queries, c, plan, tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    starts = [r for r in rows if r["optimizer_step"] == 0]
    assert {r["student_sha256"] for r in starts} == {initial_hash}
    ends = [r for r in rows if r["optimizer_step"] == 2]
    assert all(r["relative_target_mse"] < .7 for r in ends if r["arm"] != "joint")
    assert all(r["relative_target_mse"] == 1 for r in ends if r["arm"] == "joint")
    assert b.counters["backward_queries"] == 8
    assert tensor_digest(b.behavior_state()) == initial_hash
    assert len((tmp_path / "endpoints.jsonl").read_text().splitlines()) == 16
    for arm in ("single_case00", "single_case01", "joint"):
        state = torch.load(tmp_path / "checkpoints" / arm / "training.pt", weights_only=False)
        assert state["optimizer_step"] == 2
        assert (tmp_path / "checkpoints" / arm / "raw_lora/toy.pt").is_file()
