"""Explicit, evidence-backed numerical protocol for saved-query training."""
from contextlib import contextmanager
import os

import torch

from .io import read_json, sha256

MODE = "deterministic_training_qualified_v1"


def qualification(plan):
    policy = plan["training_execution"]
    assert policy["mode"] == MODE
    assert sha256(policy["qualification_result"]) == policy["qualification_sha256"]
    result = read_json(policy["qualification_result"])
    assert result["status"] == "complete" and result["counts"]["backward_queries"] == 168
    assert result["new_images"] == result["new_teacher_queries"] == 0
    for mode in ("default", "deterministic"):
        assert result[mode]["initial_actions_exact"]
        assert len(result[mode]["arms"]) == 3 and all(r["optimizer_updates"] == 2 for r in result[mode]["arms"].values())
        assert len(result[mode]["repeats"]) == 2
        assert all(c["exact"] == (mode == "deterministic") for c in result[mode]["repeats"].values())
    assert result["current"]["single"]["historical_replay_exact"]
    assert result["current"]["single"]["optimizer_updates"] == 2
    assert not plan.get("reference_model_hashes")
    assert set(plan["historical_reference_model_hashes"]) == {"single"}
    return result


@contextmanager
def training_execution(plan):
    if "training_execution" not in plan:
        yield None
        return
    evidence = qualification(plan)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert torch.__version__ == evidence["torch"] and torch.cuda.get_device_name() == evidence["gpu"]
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    assert not previous and not torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield {**plan["training_execution"], "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
               "deterministic_algorithms": True, "warn_only": False, "cublas_workspace_config": ":4096:8",
               "scope": "All fitting arms only; reference generation and evaluation retain the original execution mode."}
        assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def validate_training_replay(plan, result):
    if "training_execution" not in plan:
        assert result["arms"]["single"]["historical_replay_exact"]
        return {"historical_model_exact": True, "policy": "historical_final_weight_exact"}
    evidence = qualification(plan)
    actual = result["training_execution"]
    assert all(actual[k] == v for k, v in plan["training_execution"].items())
    assert actual["deterministic_algorithms"] is True and actual["warn_only"] is False
    assert actual["torch"] == evidence["torch"] and actual["gpu"] == evidence["gpu"]
    assert actual["cublas_workspace_config"] == ":4096:8"
    exact = result["model_hashes"]["single"] == plan["historical_reference_model_hashes"]["single"]
    assert result["historical_model_comparison"]["single"] == exact
    return {"historical_model_exact": exact, "policy": MODE, "qualified_old_new_fitter_updates": 2,
            "limit": "GPU equivalence measured at two updates; no claim of reproducing the historical nondeterministic 128-update model."}
