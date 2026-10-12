import copy
import json

import pytest
import torch

from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.fitting_probe import cpu_tree, tensor_digest
from opdlab.io import ROOT
from opdlab.state_requery import intervene
from opdlab.trajectory_fit import build_case, fit_arms
from opdlab.trajectory_probe import rollout
from opdlab.verifiers import ToyVerifier


def fixture():
    config = Experiment(backend="toy", precision="no", steps=3, learning_rate=.03, weight_decay=0)
    backend = ToyBackend(config)
    backend.load_student({"weight": torch.tensor([.2, -.1])}, refresh_behavior=True)
    cases = []
    for i, record in enumerate(load_records(ROOT / "data/demo/toy.jsonl", "train")):
        query = {"record": record, "seed": 42 + i, "index": 1, "context": record["prompt"]}
        states, actions = rollout(backend, query, [], 3)
        ss, aa = intervene(backend, states, actions, query, actions[1] + .3, 3)
        saved = cpu_tree({"record": record, "seed": query["seed"], "query_index": 1,
                         "branches": {"base": {"states": states, "actions": actions},
                                      "geneval@1": {"states": ss, "actions": aa}}})
        spec = {"id": record["id"], "case": i, "candidate": "geneval@1" if i == 0 else "base"}
        cases.append(build_case(backend, saved, query, spec, 3))
    return config, backend, cases


def test_corrected_suffix_and_teacher_provenance_are_preserved():
    config, backend, cases = fixture()
    positive = cases[0]
    assert sum(p["accepted"] for c in cases for p in c["points"]) == 1
    assert all(torch.equal(p["target"], p["noop"]) for c in cases for p in c["points"] if not p["accepted"])
    suffix = positive["points"][2]
    assert not torch.equal(suffix["target"], positive["saved"]["branches"]["base"]["actions"][2])
    assert positive["sparse"] is positive["points"][1]
    broken = copy.deepcopy(positive["saved"])
    broken["branches"]["geneval@1"]["actions"][2].add_(.1)
    with pytest.raises(ValueError, match="transition"):
        build_case(backend, broken, positive["query"], positive["spec"], config.steps)


def test_arms_reset_optimizer_weights_and_match_backward_budget(tmp_path):
    config, backend, cases = fixture()
    original = tensor_digest(backend.behavior_state())
    plan = {"arms": {"sparse": {"queries_per_update": 2, "updates": 6, "audits": [0, 2, 6]},
                     "dense": {"queries_per_update": 6, "updates": 2, "audits": [0, 2]}}}
    result = fit_arms(backend, ToyVerifier(config), cases, config, plan, tmp_path)
    assert all(r["initial_student_sha256"] == original and r["counts"]["backward_queries"] == 12 for r in result.values())
    assert backend.counters["backward_queries"] == 24
    assert tensor_digest(backend.behavior_state()) == original
    rows = [json.loads(s) for s in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert {r["student_sha256"] for r in rows if r["optimizer_step"] == 0} == {original}
    assert len((tmp_path / "endpoints.jsonl").read_text().splitlines()) == 12
    for arm, step in (("sparse", 6), ("dense", 2)):
        state = torch.load(tmp_path / "checkpoints" / arm / "training.pt", weights_only=False)
        assert state["optimizer_step"] == step and state["optimizer"]["state"]
