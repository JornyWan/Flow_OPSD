import copy
import hashlib
import json

import pytest
import torch

from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT
from opdlab.transfer_eval import generate_models, make_pairs, validate_pairs
from opdlab.verifiers import ToyVerifier


def test_transfer_pairs_are_complete_deterministic_and_exclude_fit_seeds():
    train = load_records(ROOT / "data/joint_pilot_v1/records.jsonl", "train")
    validation = load_records(ROOT / "data/joint_pilot_v1/records.jsonl", "validation")
    selected = [{"id": r["id"], "seed": 100 + i} for i, r in enumerate(train[:5])]
    pairs = make_pairs(selected, train, validation)
    assert pairs == make_pairs(list(reversed(selected)), train, list(reversed(validation)))
    assert len(pairs) == 42 and sum(p["cohort"] == "train_new_seed" for p in pairs) == 10
    with pytest.raises(ValueError, match="Missing"):
        validate_pairs(pairs[:-1], selected, validation)
    bad = copy.deepcopy(pairs)
    bad[0]["seed"] = selected[0]["seed"]
    with pytest.raises(ValueError, match="fitting seed"):
        validate_pairs(bad, selected, validation)


def test_multiple_raw_models_use_paired_noise_and_ordinary_frozen_generation(tmp_path):
    config = Experiment(backend="toy", precision="no", steps=3)
    backend, verifier = ToyBackend(config), ToyVerifier(config)
    records = load_records(ROOT / "data/demo/toy.jsonl", "train")
    record = records[0]
    models = {}
    for name, value in (("a", .1), ("b", .3)):
        state = {"weight": torch.full((2,), value)}
        path = tmp_path / f"{name}.pt"
        torch.save(state, path)
        backend.load_student(state, refresh_behavior=False)
        expected = backend.generate(record, 90, 3)
        models[name] = {"weights": str(path), "anchor": {"id": record["id"], "seed": 90,
                          "image_sha256": hashlib.sha256(expected.numpy().tobytes()).hexdigest()}}
    initial_behavior = tensor_digest(backend.behavior_state())
    backend.counters.clear()
    plan = {"models": models, "pairs": [{"id": record["id"], "seed": seed, "pair_id": str(seed),
                                         "cohort": "fixture", "replicate": i} for i, seed in enumerate((91, 92))]}
    checks = generate_models(backend, verifier, records, config, plan, tmp_path)
    rows = [json.loads(s) for s in (tmp_path / "endpoints.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and len(checks["noise_hashes"]) == 2
    for pair in ("91", "92"):
        group = [r for r in rows if r["pair_id"] == pair]
        assert len({r["noise_sha256"] for r in group}) == 1
        assert len({r["image_sha256"] for r in group}) == 2
    assert tensor_digest(backend.behavior_state()) == initial_behavior
    assert backend.counters["forward_student"] == 18 and backend.counters["backward_queries"] == 0
