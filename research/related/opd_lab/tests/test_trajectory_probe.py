import pytest
import torch

from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.fitting_probe import tensor_digest
from opdlab.trajectory_probe import MODES, rollout, student_indices


def test_query_is_included_and_prefix_changes_only_the_approach_state():
    c = Experiment(backend="toy", precision="no", steps=6)
    b = ToyBackend(c)
    b.load_student({"weight": torch.tensor([.7, -.4])}, refresh_behavior=False)
    old_hash, new_hash = tensor_digest(b.behavior_state()), tensor_digest(b.student_state())
    q = {"seed": 42, "context": "fixture", "index": 2}
    outputs = {mode: rollout(b, q, student_indices(mode, 2, 6), 6)[0]
               for mode in ("base", *MODES)}
    assert student_indices("prefix", 2, 6) == [0, 1, 2]
    assert student_indices("suffix", 2, 6) == [2, 3, 4, 5]
    assert student_indices("window4", 4, 6) == [4, 5]
    for mode in ("local", "suffix", "window2", "window4"):
        assert all(torch.equal(outputs[mode][i], outputs["base"][i]) for i in range(3))
        assert not torch.equal(outputs[mode][3], outputs["base"][3])
    assert not torch.equal(outputs["prefix"][2], outputs["base"][2])
    assert not torch.equal(outputs["suffix"][-1], outputs["local"][-1])
    assert torch.equal(outputs["full"][-1], b.generate({"prompt": "fixture"}, 42, 6))
    assert tensor_digest(b.behavior_state()) == old_hash
    assert tensor_digest(b.student_state()) == new_hash


def test_zero_index_duplicate_rules_are_identical_and_invalid_indices_fail():
    b = ToyBackend(Experiment(backend="toy", precision="no", steps=4))
    b.load_student({"weight": torch.tensor([.7, -.4])}, refresh_behavior=False)
    q = {"seed": 42, "context": "fixture", "index": 0}
    for first, second in (("local", "prefix"), ("suffix", "full")):
        a = rollout(b, q, student_indices(first, 0, 4), 4)
        z = rollout(b, q, student_indices(second, 0, 4), 4)
        assert all(torch.equal(x, y) for xs, ys in zip(a, z) for x, y in zip(xs, ys))
    with pytest.raises(ValueError):
        student_indices("local", 4, 4)
    with pytest.raises(ValueError):
        rollout(b, q, [0, 0], 4)
