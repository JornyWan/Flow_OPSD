import pytest
import torch

from opdlab.action_probe import finish_action, project_action, ray_action, same_verdict
from opdlab.backends import ToyBackend
from opdlab.config import Experiment


def test_projection_removes_only_orthogonal_component_without_clipping():
    base, target = torch.tensor([5., 2.]), torch.tensor([7., 2.])
    for gamma in (-.5, .25, 1.5):
        actual = base + torch.tensor([2 * gamma, 3.])
        projected, stats = project_action(base, target, actual)
        torch.testing.assert_close(projected, base + torch.tensor([2 * gamma, 0.]))
        assert stats["gamma"] == gamma
        assert stats["gamma_outside_unit_interval"] == (gamma < 0 or gamma > 1)
        assert stats["orthogonal_squared_norm"] == 9
        assert stats["orthogonality_error"] == 0
    with pytest.raises(ValueError, match="nonzero"):
        project_action(base, base, target)


def test_ray_endpoints_preserve_exact_original_tensors():
    base, target = torch.tensor([1e8, 1.]), torch.tensor([1., 1e-8])
    assert ray_action(base, target, 0).data_ptr() == base.data_ptr()
    assert ray_action(base, target, 1).data_ptr() == target.data_ptr()
    torch.testing.assert_close(ray_action(base, target, .25), (base.double() * .75 + target.double() * .25).float())


def test_suffix_uses_frozen_behavior_after_student_changes():
    b = ToyBackend(Experiment(backend="toy", steps=4))
    b.load_student({"weight": torch.tensor([20., 20.])}, refresh_behavior=False)
    q = {"state": torch.tensor([[1., 2.]]), "index": 1, "context": ""}
    action = torch.tensor([[.5, -.5]])
    expected = q["state"] - .25 * action
    for _ in range(2):
        expected = expected - .25 * .15 * expected
    torch.testing.assert_close(finish_action(b, q, action, 4), expected)
    assert b.counters["forward_behavior"] == 2 and b.counters["forward_student"] == 0


def test_verdict_reproduction_survives_json_without_hiding_value_changes():
    import json
    from opdlab.method import Verdict
    original = Verdict((True, False), (1., .25)).as_dict(.5)
    decoded = json.loads(json.dumps(original))
    assert same_verdict(original, decoded)
    for field, value in (("valid", False), ("utility", .7), ("joint_success", True),
                         ("satisfied", [True, True]), ("progress", [1., .25000001])):
        assert not same_verdict(original, {**decoded, field: value})
