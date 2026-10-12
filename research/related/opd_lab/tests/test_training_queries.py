from collections import Counter
from dataclasses import replace

import pytest
import torch

from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.engine import train_loop
from opdlab.io import ROOT, read_json, write_json
from opdlab.runtime import prepare, verify_run
from opdlab.training_queries import make_training_manifest, validate_training_manifest


def test_every_batch_balances_tasks_and_every_pass_covers_all_prompts():
    c = Experiment(outer_epochs=12, queries_per_rank=16, accumulation=16)
    records = load_records(ROOT / "data/joint_pilot_v1/records.jsonl", "train")
    groups = {r["id"]: r["group"] for r in records}
    plan = make_training_manifest(c, records)
    rows = plan["queries"]
    assert len(rows) == 192
    for start in range(0, len(rows), 16):
        assert set(Counter(groups[r["id"]] for r in rows[start:start + 16]).values()) == {4}
    for start in range(0, len(rows), 64):
        assert {r["id"] for r in rows[start:start + 64]} == set(groups)
    assert make_training_manifest(replace(c, method="verified_fixed", learning_rate=1e-5), records) == plan
    plan["queries"][0]["query_index"] = 100
    with pytest.raises(ValueError, match="manifest"):
        validate_training_manifest(plan, c, records, 1)


def test_manifest_training_resume_and_tamper_rejection(tmp_path):
    c = Experiment(backend="toy", dataset=str(ROOT / "data/demo/toy.jsonl"), precision="no",
                   method="fixed", steps=4, outer_epochs=2, queries_per_rank=2, accumulation=2,
                   save_every=1, ema_interval=1, learning_rate=.03)
    path = tmp_path / "queries.json"
    plan = make_training_manifest(c, load_records(c.dataset, "train"))
    write_json(path, plan)
    c.training_manifest = str(path)
    full_config, resume_config = replace(c), replace(c)
    full, resumed = prepare(full_config, tmp_path / "full", None), prepare(resume_config, tmp_path / "resume", None)
    train_loop(full_config, None, None, full)
    train_loop(resume_config, None, None, resumed, stop_after=1)
    train_loop(resume_config, None, None, resumed, resume=resumed / "checkpoints/epoch-000001")
    a = torch.load(full / "checkpoints/epoch-000002/training.pt", weights_only=False)
    b = torch.load(resumed / "checkpoints/epoch-000002/training.pt", weights_only=False)
    torch.testing.assert_close(a["student"]["weight"], b["student"]["weight"], atol=0, rtol=0)
    assert a["optimizer_step"] == b["optimizer_step"] == 2
    verify_run(resumed)
    changed = read_json(resumed / "training_queries.json")
    changed["queries"][0]["seed"] += 1
    write_json(resumed / "training_queries.json", changed)
    with pytest.raises(ValueError, match="manifest was modified"):
        verify_run(resumed)
