from copy import deepcopy

import pytest
import torch

from opdlab.io import sha256, write_json
from opdlab.training_replay import MODE, qualification, training_execution, validate_training_replay


def qualified_plan(tmp_path):
    path = tmp_path / "evidence.json"
    modes = {mode: {"initial_actions_exact": True, "arms": {a: {"optimizer_updates": 2} for a in "abc"},
                    "repeats": {a: {"exact": mode == "deterministic"} for a in "bc"}}
             for mode in ("default", "deterministic")}
    write_json(path, {"status": "complete", "counts": {"backward_queries": 168}, "new_images": 0,
                     "new_teacher_queries": 0, "gpu": "test_gpu", "torch": torch.__version__, **modes,
                     "current": {"single": {"historical_replay_exact": True, "optimizer_updates": 2}}})
    return {"training_execution": {"mode": MODE, "qualification_result": str(path), "qualification_sha256": sha256(path)},
            "historical_reference_model_hashes": {"single": "old"}}


def test_original_gate_cannot_be_silently_relaxed():
    with pytest.raises(AssertionError):
        validate_training_replay({}, {"arms": {"single": {"historical_replay_exact": False}}})
    assert validate_training_replay({}, {"arms": {"single": {"historical_replay_exact": True}}})["historical_model_exact"]


def test_qualification_is_required_and_historical_difference_is_explicit(tmp_path, monkeypatch):
    plan = qualified_plan(tmp_path)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test_gpu")
    before = torch.are_deterministic_algorithms_enabled()
    with training_execution(plan) as execution:
        assert torch.are_deterministic_algorithms_enabled()
    assert torch.are_deterministic_algorithms_enabled() == before
    result = {"training_execution": execution, "model_hashes": {"single": "new"}, "historical_model_comparison": {"single": False}}
    assert not validate_training_replay(plan, result)["historical_model_exact"]
    bad = deepcopy(result); bad["historical_model_comparison"]["single"] = True
    with pytest.raises(AssertionError):
        validate_training_replay(plan, bad)
    with pytest.raises(RuntimeError):
        with training_execution(plan):
            raise RuntimeError("fitting failed")
    assert torch.are_deterministic_algorithms_enabled() == before
    write_json(plan["training_execution"]["qualification_result"], {})
    with pytest.raises(AssertionError):
        qualification(plan)
