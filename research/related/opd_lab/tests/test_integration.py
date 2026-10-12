from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.engine import train_loop
from opdlab.evaluate import compare, evaluate
from opdlab.io import ROOT, read_json, write_json
from opdlab.runtime import prepare, verify_run


def config():
    return Experiment(backend="toy", dataset=str(ROOT / "data/demo/toy.jsonl"), precision="no",
                      method="fixed", steps=4, outer_epochs=2, queries_per_rank=4,
                      accumulation=2, learning_rate=0.03, ema_interval=1, save_every=1)


def test_resume_recovers_student_optimizer_ema_and_rng(tmp_path):
    full_config, resumed_config = config(), config()
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    prepare(full_config, full, None)
    prepare(resumed_config, resumed, None)
    train_loop(full_config, None, None, full)
    train_loop(resumed_config, None, None, resumed, stop_after=1)
    train_loop(resumed_config, None, None, resumed, resume=resumed / "checkpoints/epoch-000001")
    left = torch.load(full / "checkpoints/epoch-000002/training.pt", weights_only=False)
    right = torch.load(resumed / "checkpoints/epoch-000002/training.pt", weights_only=False)
    assert left["optimizer_step"] == right["optimizer_step"] == 4
    torch.testing.assert_close(left["student"]["weight"], right["student"]["weight"], rtol=0, atol=0)
    torch.testing.assert_close(left["ema"]["weight"], right["ema"]["weight"], rtol=0, atol=0)
    for key in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(left["optimizer"]["state"][0][key], right["optimizer"]["state"][0][key], rtol=0, atol=0)
    rng_a = torch.load(full / "checkpoints/epoch-000002/rng-rank-0.pt", weights_only=False)
    rng_b = torch.load(resumed / "checkpoints/epoch-000002/rng-rank-0.pt", weights_only=False)
    torch.testing.assert_close(rng_a["torch"], rng_b["torch"], rtol=0, atol=0)


def test_all_rejected_queries_do_not_apply_adam_decay_or_momentum(tmp_path):
    c = replace(config(), method="random", accept_probability=0)
    run = prepare(c, tmp_path / "reject", None)
    result = train_loop(c, None, None, run)
    torch.testing.assert_close(result["weight"], torch.zeros(2), rtol=0, atol=0)
    assert read_json(run / "result.json")["optimizer_step"] == 0


def test_unified_eval_is_paired_and_rejects_step_mismatch(tmp_path):
    c = config()
    left, right, mismatched = tmp_path / "base", tmp_path / "copy", tmp_path / "different"
    evaluate(c, None, None, left, steps=4, seeds=(42, 43))
    evaluate(c, None, None, right, steps=4, seeds=(42, 43))
    result = compare(left, right)
    assert result["utility"]["paired_delta"] == 0
    assert result["utility"]["prompt_count"] == 2
    assert set(result["clause_accuracy_delta"]) == {"x", "y"}
    assert all(v["mean"] == 0 for v in result["clause_accuracy_delta"].values())
    evaluate(c, None, None, mismatched, steps=8, seeds=(42, 43))
    with pytest.raises(ValueError, match="steps"):
        compare(left, mismatched)
    # Partial/corrupted eval cannot quietly change the denominator.
    rows = (right / "per_image.jsonl").read_text().splitlines()
    (right / "per_image.jsonl").write_text("\n".join(rows[:-1]))
    with pytest.raises(ValueError, match="changed"):
        compare(left, right)


def test_protocol_rejects_raw_ema_mismatch(tmp_path):
    c = config()
    # Use labelled mock toy adapters to exercise loading and averaging metadata.
    parent = tmp_path / "checkpoint"
    parent.mkdir()
    write_json(parent / "metadata.json", {"raw_lora": "raw", "ema_lora": "EMA"})
    for kind in ("raw", "ema"):
        adapter = parent / f"{kind}_lora"
        adapter.mkdir()
        torch.save({"weight": torch.zeros(2)}, adapter / "toy.pt")
        evaluate(c, None, None, tmp_path / kind, adapter, kind, steps=4)
    with pytest.raises(ValueError, match="adapter_kind"):
        compare(tmp_path / "raw", tmp_path / "ema")


def test_split_leakage_and_modified_run_data_are_rejected(tmp_path):
    rows = [json.loads(s) for s in (ROOT / "data/demo/toy.jsonl").read_text().splitlines()]
    rows[-1]["group"] = rows[0]["group"]
    invalid = tmp_path / "leaky.jsonl"
    invalid.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="leakage"):
        load_records(invalid, "train")
    c = config()
    run = prepare(c, tmp_path / "run", None)
    verify_run(run)
    with (run / "data.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="modified"):
        verify_run(run)


def test_resume_rejects_hyperparameter_change(tmp_path):
    c = config()
    run = prepare(c, tmp_path / "run", None)
    train_loop(c, None, None, run, stop_after=1)
    c.learning_rate *= 2
    with pytest.raises(ValueError, match="mismatch"):
        train_loop(c, None, None, run, resume=run / "checkpoints/epoch-000001")


def test_one_invalid_candidate_stops_before_optimizer_update(tmp_path, monkeypatch):
    from opdlab.method import Verdict
    from opdlab.verifiers import ToyVerifier
    class PartialFailure(ToyVerifier):
        def __call__(self, image, record):
            result = super().__call__(image, record)
            if self.counters["verifier_images"] == 2:
                return Verdict((False, False), (0, 0), valid=False, error="injected failure")
            return result
    monkeypatch.setattr("opdlab.engine.build_verifier", lambda *args: PartialFailure(None))
    c = config()
    run = prepare(c, tmp_path / "invalid", None)
    with pytest.raises(RuntimeError, match="incomplete candidate set"):
        train_loop(c, None, None, run)
    assert not (run / "checkpoints").exists()


def test_scheduler_metadata_order_is_not_a_sampling_difference(tmp_path):
    from opdlab.io import digest
    c = config()
    for name, fields in (("left", ["shift", "invert_sigmas"]), ("right", ["invert_sigmas", "shift"])):
        output = tmp_path / name
        evaluate(c, None, None, output, steps=4)
        protocol = read_json(output / "protocol.json")
        protocol["scheduler"] = {"shift": 3.0, "_use_default_values": fields}
        write_json(output / "protocol.json", protocol)
        summary = read_json(output / "summary.json")
        summary["protocol_digest"] = digest(protocol)
        write_json(output / "summary.json", summary)
    assert compare(tmp_path / "left", tmp_path / "right")["utility"]["paired_delta"] == 0
    protocol["scheduler"]["shift"] = 1.0
    write_json(tmp_path / "right/protocol.json", protocol)
    summary["protocol_digest"] = digest(protocol)
    write_json(tmp_path / "right/summary.json", summary)
    with pytest.raises(ValueError, match="scheduler"):
        compare(tmp_path / "left", tmp_path / "right")
