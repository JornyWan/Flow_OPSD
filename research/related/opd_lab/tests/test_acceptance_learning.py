from collections import Counter

import torch

from opdlab.acceptance_learning import ARMS, fit_saved, schedules, update_saved_actions
from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.engine import seed_all
from opdlab.fitting_probe import tensor_digest, update_once
from opdlab.supervision_study import fit


def test_acceptance_schedule_matches_counts_without_moving_protection():
    points = [{"id": str(i), "key": f"{i}-{j}", "replicate": j, "role": "positive" if i < 6 else "preservation",
               "target": "geneval@1" if i < 6 and j < (i % 3) else "noop"}
              for i in range(12) for j in range(8 if i < 6 else 1)]
    result, perm = schedules(points, 128)
    assert (result, perm) == schedules(list(reversed(points)), 128)
    bykey = {p["key"]: p for p in points}
    for arm in ARMS:
        assert len(result[arm]) == 128 and all(len(r) == 12 for r in result[arm])
        counts = Counter(e["key"] for row in result[arm] for e in row)
        assert all(counts[p["key"]] == (16 if p["role"] == "positive" else 128) for p in points)
    for start in range(0, 128, 8):
        totals = []
        for arm in ("certified", "random_matched"):
            totals.append(Counter((bykey[e["key"]]["id"], e["target"]) for row in result[arm][start:start+8] for e in row))
        assert totals[0] == totals[1]
    for triplet in zip(*(result[a] for a in ARMS)):
        assert len({tuple(e["key"] for e in row) for row in triplet}) == 1
        assert all(e["target"] == "noop" for row in triplet for e in row if bykey[e["key"]]["role"] == "preservation")
    assert any(all(e["target"] == "noop" for e in row) for row in result["certified"])


def test_saved_update_matches_original_and_accepts_all_noop_batches():
    config = Experiment(backend="toy", precision="no", steps=4, learning_rate=.03)
    states = []
    for original in (True, False):
        seed_all(42)
        backend = ToyBackend(config)
        optimizer = torch.optim.AdamW(backend.parameters(), lr=.03)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        q = {"state": torch.zeros(1, 2), "index": 1, "context": "", "dt": -.25, "target": torch.ones(1, 2)*.1, "accepted": True}
        if original:
            update_once(backend, [q, {**q, "target": torch.zeros(1,2), "accepted":False}], optimizer, scaler, config, keep=1)
        else:
            update_saved_actions(backend, [q, {**q, "target":torch.zeros(1,2)}], optimizer, scaler, config)
        states.append(tensor_digest(backend.student_state()))
    assert states[0] == states[1]
    before = tensor_digest(backend.student_state())
    update_saved_actions(backend, [{**q, "target":torch.zeros(1,2), "accepted":False}], optimizer, scaler, config)
    assert tensor_digest(backend.student_state()) != before and backend.counters["backward_queries"] == 3


def test_saved_fitter_matches_previous_core_weights_optimizer_and_resets(tmp_path):
    config = Experiment(backend="toy", precision="no", steps=4, learning_rate=.03)
    zero = torch.zeros(1,2)
    queries = {str(i): {"state":zero, "index":1, "context":"", "dt":-.25, "seed":i, "record":{"id":str(i)},
                        "label_target":"geneval@1" if i == 0 else "noop", "actions":{"noop":zero, "geneval@1":torch.ones_like(zero)*.1}} for i in range(2)}
    row = [{"key":k,"target":q["label_target"]} for k,q in queries.items()]
    plan = {"arms":ARMS,"schedules":{a:[row]*2 for a in ARMS}, "audits":[0,2]}
    saved = []
    for name, runner in (("old",fit),("new",fit_saved)):
        seed_all(42)
        backend = ToyBackend(config)
        run=tmp_path/name;run.mkdir()
        reports=runner(backend,queries,config,plan,run)
        assert len({r["final_sha256"] for r in reports.values()}) == 1
        assert all(r["optimizer_updates"] == 2 and r["counts"]["backward_queries"] == 4 for r in reports.values())
        saved.append(torch.load(run/"checkpoints/certified/training.pt",weights_only=False))
    assert tensor_digest(saved[0]["student"]) == tensor_digest(saved[1]["student"])
    for k, state in saved[0]["optimizer"]["state"].items():
        assert all(torch.equal(v,saved[1]["optimizer"]["state"][k][n]) for n,v in state.items())
