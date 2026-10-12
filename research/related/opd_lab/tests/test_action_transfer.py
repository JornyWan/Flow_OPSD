import copy
import pytest
import torch

from opdlab.action_transfer import branch_actions, require_exact, select_pairs, trace_baseline
from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.fitting_probe import tensor_digest
from opdlab.learnability import single_endpoint


def test_shared_prefix_suffix_and_no_update():
    config = Experiment(backend="toy", steps=10)
    backend = ToyBackend(config)
    record = {"prompt": "x"}
    query, base = trace_baseline(backend, record, 42, 10, 3)
    assert torch.equal(base, single_endpoint(backend, query, query["actions"]["noop"], 10))
    frozen = tensor_digest(backend.behavior_state())
    backend.load_student({"weight": torch.tensor([0.2, -0.1])}, refresh_behavior=False)
    learned = tensor_digest(backend.student_state())
    actions = branch_actions(backend, query, True)
    assert set(actions) == {"student_single", "ocr@1", "geneval@1"}
    counts = copy.deepcopy(backend.counters)
    endpoint = single_endpoint(backend, query, actions["student_single"], 10)
    assert backend.counters["forward_behavior"] - counts["forward_behavior"] == 6
    assert backend.counters["forward_student"] == counts["forward_student"]
    assert not torch.equal(base, endpoint)
    assert tensor_digest(backend.behavior_state()) == frozen
    assert tensor_digest(backend.student_state()) == learned
    assert not actions["student_single"].requires_grad
    assert set(branch_actions(backend, query, False)) == {"student_single"}
    assert backend.counters["forward_ocr"] == backend.counters["forward_geneval"] == 1
    assert backend.counters["backward_queries"] == 0


def test_all_pairs_coverage_and_fail_closed_replay():
    points = [{"id": str(i), "seed": i, "label_target": "ocr@1"} for i in range(6)]
    pairs = [{"id": str(i), "seed": i + j * 10, "pair_id": f"{i}-{j}",
              "cohort": "fitting_seed_diagnostic" if j == 0 else "positive_fresh_seed"}
             for i in range(6) for j in range(3)]
    assert select_pairs({"points": points, "evaluation": pairs}) == (points, pairs)
    with pytest.raises(AssertionError):
        select_pairs({"points": points, "evaluation": pairs[:-1]})
    with pytest.raises(RuntimeError, match="exact replay"):
        require_exact({"image_sha256": "changed"}, "original")
