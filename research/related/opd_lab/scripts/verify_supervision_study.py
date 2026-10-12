"""Verify the saved-query study, its matched schedules and all frozen evaluation joins."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import torch

from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT, assert_environment, read_json, sha256, write_json
from opdlab.runtime import verify_run
from opdlab.supervision_study import schedules
from opdlab.training_replay import validate_training_replay
from prepare_coverage_probe import seed_values
from prepare_supervision_study import evaluation_pairs
from review_action_transfer import check_labels
from review_prospective import lines
from review_supervision import build_report, validate
from verify_state_probes import historical_index
from verify_supervision_completion import validate_completion


def verify(run, review, campaign):
    asset_audit = verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    assert not any((run / n).exists() for n in ("RUNNING", "failure.json"))
    result, plan = read_json(run / "result.json"), read_json(run / "probe_plan.json")
    completion = validate_completion(run) if "completion" in plan else None
    rows = validate(run)
    assert plan["schedules"] == schedules(plan["points"], plan["updates"])
    points = {p["key"]: p for p in plan["points"]}
    assert len(points) == 15 and sum(p["target"] != "noop" for p in points.values()) == 9
    inventory = read_json(run / "seed_inventory.json")
    excluded = set()
    for p, value in inventory["files"].items():
        assert sha256(p) == value["sha256"]
        seeds = seed_values(lines(p) if p.endswith(".jsonl") else read_json(p))
        assert sorted(seeds) == value["seeds"]
        excluded.update(seeds)
    assert sorted(excluded) == inventory["excluded_seeds"]
    assert evaluation_pairs(read_json(ROOT / "runs/le15_annotation_001/probe_plan.json"), plan["points"], excluded) == plan["evaluation"]
    for key, expected in plan["expected_counts"].items():
        assert result["counts"].get(key, 0) == expected
    updates = lines(run / "updates.jsonl")
    assert len(updates) == 384
    for arm in plan["arms"]:
        actual = [r for r in updates if r["arm"] == arm]
        assert [r["step"] for r in actual] == list(range(1,129))
        assert [r["queries"] for r in actual] == plan["schedules"][arm]
        assert result["arms"][arm]["counts"]["backward_queries"] == 1536
        assert result["arms"][arm]["initial_sha256"] == result["model_hashes"]["joint128"]
        model = torch.load(run / f"checkpoints/{arm}/step128.pt", map_location="cpu", weights_only=True)
        assert tensor_digest(model) == result["model_hashes"][arm] == result["arms"][arm]["final_sha256"]
        training = torch.load(run / f"checkpoints/{arm}/training.pt", map_location="cpu", weights_only=False)
        assert training["optimizer_step"] == 128 and tensor_digest(training["student"]) == tensor_digest(model)
        assert training["optimizer"]["state"] and all(int(v["step"]) == 128 for v in training["optimizer"]["state"].values())
    replay = validate_training_replay(plan, result)
    historical_hash = tensor_digest(torch.load(plan["reference_model"],map_location="cpu",weights_only=True))
    assert (historical_hash == result["model_hashes"]["single"]) == replay["historical_model_exact"]
    bykey = {p["key"]: p for p in plan["points"]}
    for single, multi, random in zip(*(plan["schedules"][a] for a in plan["arms"])):
        assert all(len(row)==12 for row in (single,multi,random))
        assert [e["key"] for e in multi] == [e["key"] for e in random]
        assert Counter((bykey[e["key"]]["group"],e["target"]) for e in multi) == Counter((bykey[e["key"]]["group"],e["target"]) for e in random)
    references = lines(run / "references.jsonl")
    assert len(references) == 24 and all(r["exact"] for r in references)
    for r in references:
        assert r["image_sha256"] == points[r["key"]]["images"][r["mode"]]["sha256"] == sha256(run / r["image"])
    metrics = lines(run / "metrics.jsonl")
    assert len(metrics)==135 and Counter((r["arm"],r["step"]) for r in metrics)=={(a,s):15 for a in plan["arms"] for s in (0,32,128)}
    for p in plan["points"]:
        q = torch.load(p["saved_query"],map_location="cpu",weights_only=False)
        assert sha256(p["saved_query"])==p["saved_query_sha256"]
        assert {n:tensor_digest({"action":q["actions"][source]}) for n,source in p["action_names"].items()}==p["action_hashes"]
    for p in plan["evaluation"]:
        sample = [r for r in rows if r["pair_id"]==p["pair_id"]]
        assert len(sample)==(7 if p["single_action"] else 4)
        assert {r["noise_sha256"] for r in sample}=={result["noise_hashes"][p["pair_id"]]}
        if p["single_action"]:
            q=torch.load(run / "eval_queries" / f"{p['pair_id']}.pt",map_location="cpu",weights_only=False)
            assert (q["record"]["id"],q["seed"],q["index"])==(p["id"],p["seed"],3)
            noise=torch.randn(q["state"].shape,generator=torch.Generator().manual_seed(p["seed"]),dtype=torch.float32)
            assert tensor_digest({"noise":noise})==q["noise_sha256"]==result["noise_hashes"][p["pair_id"]]
    assert check_labels(review)==read_json(review / "observations.json")
    assert json.loads(json.dumps(build_report(run,review)))==read_json(campaign / "results.json")
    items={i["case"]:i for i in read_json(review/"items.json")}
    for entry in read_json(review/"key.json"):
        for letter,r in entry["mapping"].items():
            assert sha256(review/items[entry["case"]]["images"][letter])==r["image_sha256"]
    for n in ("observations","observations_frozen"):
        assert sha256(campaign/f"{n}.json")==sha256(review/f"{n}.json")
    upstream={}
    for name,spec in read_json(ROOT/"upstream.lock.json").items():
        path=ROOT/"upstream"/name
        assert not subprocess.check_output(["git","status","--porcelain"],cwd=path).strip()
        head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=path).decode().strip()
        assert head==spec["commit"]
        upstream[name]=head
    history={str(p):historical_index(p,rev) for p,rev in ((6,"e384a09"),(7,"8c1e160"),(8,"d9c392c"),(9,"d9c392c"),(10,"9dcc18b"),(11,"3923b6a"),(12,"8e9c655"),(13,"bb804ca"),(14,"a23a429"),(15,"f1b9e79"),(16,"198dc28"),(17,"70a1213"))}
    jobs=read_json(campaign/"slurm_status.json")["jobs"]
    assert any(j["run"]==str(run) and j["state"]=="COMPLETED" and j["exit_code"]=="0:0" for j in jobs)
    write_json(campaign/"verification.json",{"runtime_asset_audit":asset_audit,"reference_images_exact":24,
               "training_replay":replay,"completion":completion,"training_states":15,"matched_updates_each_arm":128,
               "backward_queries_each_arm":1536,"evaluation_images":185,"paired_local_queries":15,
               "review_slots":200,"fresh_seeds":32,"statistics_rebuilt":True,"historical_artifacts":history,
               "upstream":upstream,"allocated_gpu_hours_including_attempts":sum(j["allocated_gpu_hours"] for j in jobs),
               "limits":"Assistant labels; shared large model assets stat-audited rather than full content rehash; fixed-query mechanism study."})
    print("Verified24 references, declared training replay policy,384 updates,185 endpoints/200 review slots, fresh seeds and unchanged phases6–17.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--review",required=True)
    parser.add_argument("--campaign",required=True)
    args=parser.parse_args()
    verify(Path(args.run).resolve(),Path(args.review).resolve(),Path(args.campaign).resolve())
