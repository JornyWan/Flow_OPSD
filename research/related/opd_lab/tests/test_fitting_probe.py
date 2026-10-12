import copy
import random

import torch

from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.fitting_probe import fit_audit, restore, snapshot, tensor_digest, update_once
from opdlab.engine import seed_all


def test_common_adam_prefix_frozen_behavior_and_second_step_penalty():
    seed_all(42)
    c = Experiment(backend="toy", precision="no", steps=4, learning_rate=0.1, weight_decay=0)
    b = ToyBackend(c)
    opt = torch.optim.AdamW(b.parameters(), lr=c.learning_rate, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    from types import SimpleNamespace
    queries = [{"state": torch.zeros(1, 2), "index": 1, "context": "", "dt": -0.25,
                "target": torch.ones(1, 2) if accepted else torch.zeros(1, 2),
                "accepted": accepted, "record": {"id": str(accepted)},
                "candidates": [SimpleNamespace(action=torch.zeros(1, 2))]} for accepted in (True, False)]
    frozen = tensor_digest(b.behavior_state())
    assert fit_audit(b, queries, c)[1]["noop_drift"] == 0
    update_once(b, queries, opt, scaler, c, 0)
    shared = snapshot(b, opt, scaler, 1)
    expected_adam = copy.deepcopy(shared["optimizer"])
    assert fit_audit(b, queries, c)[1]["noop_drift"] > 0
    restore(b, opt, scaler, shared)
    random_number = random.random()
    torch_number = torch.rand(1)
    update_once(b, queries, opt, scaler, c, 0)
    no_keep = b.student_state()["weight"]
    restore(b, opt, scaler, shared)
    assert random.random() == random_number
    assert torch.equal(torch.rand(1), torch_number)
    for k, v in opt.state_dict()["state"][0].items():
        assert torch.equal(v, expected_adam["state"][0][k])
    update_once(b, queries, opt, scaler, c, 1)
    keep = b.student_state()["weight"]
    assert keep.abs().sum() < no_keep.abs().sum()
    assert tensor_digest(b.behavior_state()) == frozen
    b.load_student(b.student_state())
    assert tensor_digest(b.behavior_state()) != frozen


def test_probe_completes_all_four_prespecified_checkpoints(tmp_path):
    import json
    from opdlab.data import load_records
    from opdlab.fitting_probe import ARMS, run_probe
    from opdlab.io import ROOT, read_json, sha256, write_json
    from opdlab.runtime import prepare
    records = load_records(ROOT / "data/demo/toy.jsonl", "train")
    c = Experiment(backend="toy", precision="no", dataset=str(ROOT / "data/demo/toy.jsonl"),
                   steps=4, accumulation=len(records), queries_per_rank=len(records), learning_rate=0.03)
    run = prepare(c, tmp_path / "probe", None)
    write_json(run / "probe_plan.json", {"arms": {k: list(v) for k, v in ARMS.items()},
               "queries": [{"id": r["id"], "seed": i + 42, "query_index": 1} for i, r in enumerate(records)]})
    (run / "PROTOCOL.md").write_text("CPU contract fixture")
    write_json(run / "probe_attestation.json", {"plan_sha256": sha256(run / "probe_plan.json"),
               "protocol_sha256": sha256(run / "PROTOCOL.md")})
    run_probe(run)
    result = read_json(run / "result.json")
    assert result["status"] == "complete" and result["unique_optimizer_updates"] == 7
    assert result["counts"]["backward_queries"] == 7 * len(records)
    metrics = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
    prefix = [m for m in metrics if m["optimizer_step"] == 1]
    assert len(prefix) == 3 and len({m["student_sha256"] for m in prefix}) == 1
    assert len({m["behavior_sha256"] for m in metrics}) == 1
    assert len((run / "postfit.jsonl").read_text().splitlines()) == 4 * len(records)
    for arm, (step, keep) in ARMS.items():
        meta = read_json(run / "checkpoints" / arm / "metadata.json")
        assert meta["optimizer_step"] == step and meta["rejected_noop_coefficient"] == keep
        exported = torch.load(run / "checkpoints" / arm / "raw_lora/toy.pt", weights_only=True)
        assert tensor_digest(exported) == meta["student_sha256"]
