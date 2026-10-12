from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_coverage_probe import seed_values
from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.coverage_probe import INDICES, candidate_specs, make_pairs, propose, trace
from opdlab.fitting_probe import tensor_digest
from opdlab.learnability import single_endpoint


def test_seed_inventory_and_disjoint_balanced_pairs():
    assert seed_values({"query": {"seed": 15}, "eval_seeds": [16, 17], "step": 18, "label": True}) == {15, 16, 17}
    points = [{"id": str(i)} for i in range(6)]
    pairs = make_pairs(points, [42])
    assert len({p["seed"] for p in pairs}) == 12
    assert all(sum(p["id"] == str(i) for p in pairs) == 2 for i in range(6))
    with pytest.raises(ValueError, match="seed collision"):
        make_pairs(points, [pairs[0]["seed"]])


def test_common_prefix_alphas_suffix_and_query_budget():
    backend = ToyBackend(Experiment(backend="toy", steps=10))
    initial = tensor_digest(backend.student_state())
    trajectory, base = trace(backend, {"prompt": "test"}, 17, 10)
    modes = []
    for index in INDICES:
        q = propose(backend, trajectory, index)
        assert torch.equal(single_endpoint(backend, q, q["actions"]["noop"], 10), base)
        for spec in candidate_specs():
            if spec["index"] != index:
                continue
            a = q["actions"][spec["mode"]]
            assert torch.equal(a, q["actions"]["noop"] + spec["alpha"] * (q["experts"][spec["teacher"]] - q["actions"]["noop"]))
            before = backend.counters["forward_behavior"]
            single_endpoint(backend, q, a, 10)
            assert backend.counters["forward_behavior"] - before == 9 - index
            assert not a.requires_grad
            modes.append(spec["mode"])
    assert len(set(modes)) == 12
    assert backend.counters["forward_ocr"] == backend.counters["forward_geneval"] == 3
    assert backend.counters["backward_queries"] == 0
    assert tensor_digest(backend.student_state()) == tensor_digest(backend.behavior_state()) == initial
