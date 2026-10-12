from collections import Counter
from pathlib import Path
import sys

import pytest
import torch

from opdlab.annotation_probe import ARMS, fit_models
from opdlab.backends import ToyBackend
from opdlab.config import Experiment
from opdlab.engine import seed_all
from opdlab.fitting_probe import tensor_digest
from opdlab.supervision_study import fit, schedules


def points():
    return [{"key": f"{i}-{s}", "id": str(i), "seed": s, "group": str(i // 4), "original": s == 0,
             "target": "noop" if i % 4 < 2 else "ocr@1" if i % 4 == 2 else "geneval@1"}
            for i in range(12) for s in range(3 if i == 2 else 2 if i == 6 else 1)]


def test_schedules_preserve_prompt_weights_and_match_random_roles():
    specs = points()
    result = schedules(specs, 128)
    assert result == schedules(list(reversed(specs)), 128)
    bykey = {p["key"]: p for p in specs}
    for arm, rows in result.items():
        assert len(rows) == 128 and all(len(row) == 12 for row in rows)
        assert Counter(bykey[e["key"]]["id"] for row in rows for e in row) == {str(i): 128 for i in range(12)}
    counts = Counter(e["key"] for row in result["multi"] for e in row)
    assert sorted(counts[f"2-{s}"] for s in range(3)) == [42,43,43]
    assert counts["6-0"] == counts["6-1"] == 64
    for multi, random in zip(result["multi"], result["random_matched"]):
        assert [e["key"] for e in multi] == [e["key"] for e in random]
        assert Counter((bykey[e["key"]]["group"],e["target"]) for e in multi) == Counter((bykey[e["key"]]["group"],e["target"]) for e in random)
    assert {e["key"] for row in result["single"] for e in row} == {p["key"] for p in specs if p["original"]}
    with pytest.raises(AssertionError):
        schedules(specs + [specs[0]], 1)


def test_saved_schedule_runner_exactly_replays_old_fit_and_resets_arms(tmp_path):
    config = Experiment(backend="toy", precision="no", steps=4, learning_rate=.03)
    seed_all(42)
    backend = ToyBackend(config)
    frozen = tensor_digest(backend.student_state())
    specs = [{"key": f"{i}-0", "id": str(i), "seed": 0, "group": "task", "original": True,
              "target": "ocr@1" if i == 0 else "noop"} for i in range(4)]
    qs = {}
    for p in specs:
        z = torch.zeros(1,2)
        qs[p["key"]] = {"record":{"id":p["id"]}, "seed":0,"index":1,"context":"","state":z,"dt":-.25,
                        "label_target":p["target"],"actions":{"noop":z,"ocr@1":torch.ones_like(z)*.1,"geneval@1":torch.ones_like(z)*-.1}}
    old = tmp_path / "old"; old.mkdir()
    targets = {arm: {p["id"]:p["target"] for p in specs} for arm in ARMS}
    fit_models(backend,list(qs.values()),config,{"targets":targets,"updates":2,"audits":[0,2]},old)
    reference = tensor_digest(torch.load(old/'checkpoints/label_selected/step002.pt',weights_only=True))
    seed_all(42)
    fresh = ToyBackend(config)
    assert tensor_digest(fresh.student_state()) == frozen
    new = tmp_path / "new"; new.mkdir()
    plan={"arms":["single","multi","random_matched"],"schedules":schedules(specs,2),"audits":[0,2],"reference_model_hashes":{"single":reference}}
    reports=fit(fresh,qs,config,plan,new)
    assert reports["single"]["historical_replay_exact"]
    assert all(r["initial_sha256"] == frozen and r["counts"]["backward_queries"] == 8 for r in reports.values())
    assert tensor_digest(fresh.behavior_state()) == frozen
    for arm in plan["arms"]:
        saved=torch.load(new/f"checkpoints/{arm}/training.pt",weights_only=False)
        assert saved["optimizer_step"]==2 and saved["optimizer"]["state"]


def test_eval_seeds_are_fresh_and_local_paths_are_predeclared():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from prepare_supervision_study import evaluation_pairs
    old={"evaluation":[{"id":str(i//2),"replicate":i%2,"cohort":"positive_fresh_seed" if i<12 else "preservation_fresh_seed" if i<24 else "unfitted_prompt_fresh_seed"} for i in range(32)]}
    extra=[{"id":str(i),"seed":100+i,"original":False} for i in range(3)]
    pairs=evaluation_pairs(old,extra,[42])
    assert len(pairs)==35 and sum(p["single_action"] for p in pairs)==15
    assert sum(p["cohort"]=="added_fitting_states" for p in pairs)==3
    with pytest.raises(AssertionError):
        evaluation_pairs(old,extra,[pairs[0]["seed"]])
