import hashlib
import json

import pytest
import torch

from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT
from opdlab.method import Candidate, Verdict, select
from opdlab.prospective import collect, make_queries
from opdlab.verifiers import ToyVerifier


def test_prospective_complete_train_only_balanced_and_fresh():
    records = load_records(ROOT / "data/joint_pilot_v1/records.jsonl", "train")
    queries = make_queries(records, [])
    assert queries == make_queries(list(reversed(records)), [])
    assert {q["id"] for q in queries} == {r["id"] for r in records}
    assert all(len({q["group"] for q in queries[i:i+4]}) == 4 for i in range(0,64,4))
    with pytest.raises(ValueError, match="seed collision"):
        make_queries(records, [queries[0]["seed"]])
    with pytest.raises(ValueError, match="all64"):
        make_queries(records[:-1], [])
    with pytest.raises(ValueError, match="train-only"):
        make_queries([{**records[0], "split": "test"}, *records[1:]], [])


def test_prospective_retains_all_actions_and_matches_ordinary_noop(tmp_path):
    config = Experiment(backend="toy", precision="no", alphas=[1.0], steps=10)
    backend, verifier = ToyBackend(config), ToyVerifier(config)
    backend.load_student({"weight": torch.tensor([.2, -.1])})
    records = load_records(ROOT / "data/demo/toy.jsonl", "train")[:1]
    spec = {"id": records[0]["id"], "seed": 71, "index": 3, "ordinal": 0, "query_id": "fixture"}
    expected = backend.generate(records[0], 71, 10)
    expected_hash = hashlib.sha256(expected.numpy().tobytes()).hexdigest()
    initial = tensor_digest(backend.student_state())
    backend.counters.clear()
    collect(backend, verifier, records, config, [spec], tmp_path)
    row = json.loads((tmp_path / "candidates.jsonl").read_text())
    saved = torch.load(tmp_path / row["saved_query"], weights_only=True)
    assert row["candidates"][0]["image_sha256"] == expected_hash
    assert set(saved["actions"]) == {"noop", "ocr@1", "geneval@1"}
    assert torch.equal(saved["target"], saved["actions"][row["selected"]])
    candidates = [Candidate(c["name"],c["teacher"],c["alpha"],saved["actions"][c["name"]],c["distance"],
                  Verdict(tuple(c["satisfied"]),tuple(c["progress"]),c["valid"],c["quality"])) for c in row["candidates"]]
    assert select(candidates,config,71,0) == saved["winner"]
    assert backend.counters["forward_behavior"] == 22
    assert backend.counters["forward_ocr"] == backend.counters["forward_geneval"] == 1
    assert backend.counters["decode_images"] == 3 and backend.counters["backward_queries"] == 0
    assert tensor_digest(backend.student_state()) == tensor_digest(backend.behavior_state()) == initial
