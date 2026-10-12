"""Verify phase17 frozen inputs, trajectories, labels, target pool and archived phases."""
from collections import Counter
import json
import math
from pathlib import Path
import subprocess

import torch

from opdlab.coverage_probe import INDICES, make_pairs
from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT, assert_environment, read_json, sha256, write_json
from opdlab.runtime import verify_run
from prepare_coverage_probe import seed_values
from review_coverage_probe import build_report, pools, validate
from review_action_transfer import check_labels
from review_prospective import lines
from verify_state_probes import historical_index


def main():
    campaign, run, review = ROOT / "experiments/local_expert_v17", ROOT / "runs/le17_coverage_001", ROOT / "runs/le17_review_001"
    result, plan = read_json(run / "result.json"), read_json(run / "probe_plan.json")
    audit = verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    assert not any((run / n).exists() for n in ("RUNNING", "failure.json", "updates.jsonl", "checkpoints"))
    endpoints = validate(run)
    inventory = read_json(run / "seed_inventory.json")
    excluded = set()
    for p, row in inventory["files"].items():
        assert sha256(p) == row["sha256"]
        data = lines(p) if p.endswith(".jsonl") else read_json(p)
        seeds = seed_values(data)
        assert sorted(seeds) == row["seeds"]
        excluded.update(seeds)
    assert sorted(excluded) == inventory["excluded_seeds"] == plan["excluded_seeds"]
    assert make_pairs(plan["points"], excluded) == plan["pairs"]
    assert result["teacher_queries"] == 72 and result["optimizer_updates"] == 0
    for k, v in plan["expected_counts"].items():
        assert result["counts"].get(k, 0) == v, (k, v)
    initial = torch.load(plan["student"], map_location="cpu", weights_only=True)
    assert tensor_digest(initial) == result["student_sha256"]
    refs = lines(run / "references.jsonl")
    specs = {p["id"]: p for p in plan["points"]}
    assert len(refs) == 12 and all(r["exact"] for r in refs)
    for r in refs:
        assert r["image_sha256"] == specs[r["id"]]["images"][r["mode"]]["sha256"]
    gate = read_json(run / "replay_gate.json")
    assert gate["all_exact"] and gate["counts_before_collection"].get("forward_ocr", 0) == gate["counts_before_collection"].get("forward_geneval", 0) == 0
    qs = lines(run / "queries.jsonl")
    assert len(qs) == 36 and Counter(r["index"] for r in qs) == {1: 12, 3: 12, 5: 12}
    trajectories = {}
    for r in qs:
        q = torch.load(run / r["saved_query"], map_location="cpu", weights_only=False)
        if r["pair_id"] not in trajectories:
            t = torch.load(run / r["trajectory"], map_location="cpu", weights_only=False)
            trajectories[r["pair_id"]] = t
            assert len(t["states"]) == 11 and len(t["actions"]) == 10
            noise = torch.randn(t["states"][0].shape, generator=torch.Generator().manual_seed(r["seed"]), dtype=torch.float32)
            assert torch.equal(noise, t["states"][0])
            for i in range(10):
                assert torch.equal(t["states"][i + 1], t["states"][i] + (t["sigmas"][i + 1] - t["sigmas"][i]) * t["actions"][i])
        t = trajectories[r["pair_id"]]
        assert sha256(run / r["saved_query"]) == r["saved_query_sha256"]
        assert sha256(run / r["trajectory"]) == r["trajectory_sha256"]
        assert (q["record"]["id"], q["seed"], q["index"], q["dt"]) == (r["id"], r["seed"], r["index"], r["dt"])
        assert torch.equal(q["state"], t["states"][r["index"]]) and torch.equal(q["actions"]["noop"], t["actions"][r["index"]])
        assert all(torch.equal(a, b) for a, b in zip(q["context"], t["context"]))
        assert tensor_digest({"state": q["state"]}) == r["state_sha256"]
        assert {n: tensor_digest({"action": a}) for n, a in q["actions"].items()} == r["action_hashes"]
        assert {n: tensor_digest({"action": a}) for n, a in q["experts"].items()} == r["expert_hashes"]
        assert r["prefix_sha256"] == r["suffix_sha256"] == result["student_sha256"]
        assert tensor_digest({"noise": t["states"][0]}) == r["noise_sha256"]
        for e in endpoints:
            if (e["pair_id"], e["index"]) != (r["pair_id"], r["index"]):
                continue
            a = q["actions"][e["mode"]]
            assert torch.equal(a, q["actions"]["noop"] + e["alpha"] * (q["experts"][e["teacher"]] - q["actions"]["noop"]))
            distance = float((a - q["actions"]["noop"]).double().square().mean())
            assert math.isclose(distance, e["action_distance"], rel_tol=1e-12, abs_tol=1e-12)
    costs = lines(run / "costs.jsonl")
    assert len(costs) == 12
    for row in costs:
        assert row["counts"] == {"forward_behavior": 82, "forward_examples": 173, "decode_images": 13,
                                 "forward_ocr": 3, "forward_geneval": 3, "verifier_images": 13,
                                 "ocr_images": 13, "geneval_clause_images": 13}
    assert check_labels(review) == read_json(review / "observations.json")
    rebuilt = json.loads(json.dumps(build_report(run, review)))
    assert rebuilt == read_json(campaign / "results.json")
    assert read_json(campaign / "target_manifest.json")["targets"] == rebuilt["target_manifest"]
    for target in rebuilt["target_manifest"]:
        if target["accepted"]:
            q = torch.load(target["saved_query"], map_location="cpu", weights_only=False)
            assert tensor_digest({"action": q["actions"][target["mode"]]}) == target["action_sha256"]
    items = {i["case"]: i for i in read_json(review / "items.json")}
    for item in read_json(review / "key.json"):
        for letter, row in item["mapping"].items():
            assert sha256(review / items[item["case"]]["images"][letter]) == row["image_sha256"]
    for n in ("observations", "observations_frozen"):
        assert sha256(campaign / f"{n}.json") == sha256(review / f"{n}.json")
    upstream = {}
    for name, spec in read_json(ROOT / "upstream.lock.json").items():
        path = ROOT / "upstream" / name
        assert subprocess.check_output(["git", "status", "--porcelain"], cwd=path).strip() == b""
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
        assert head == spec["commit"]
        upstream[name] = head
    history = {str(p): historical_index(p, rev) for p, rev in ((6,"e384a09"),(7,"8c1e160"),(8,"d9c392c"),(9,"d9c392c"),
               (10,"9dcc18b"),(11,"3923b6a"),(12,"8e9c655"),(13,"bb804ca"),(14,"a23a429"),(15,"f1b9e79"),(16,"198dc28"))}
    status = read_json(campaign / "slurm_status.json")
    assert len(status["jobs"]) == 1
    job = status["jobs"][0]
    assert job["state"] == "COMPLETED" and job["exit_code"] == "0:0"
    write_json(campaign / "verification.json", {"runtime_asset_audit": audit, "excluded_seed_count": len(excluded),
               "seed_source_files": len(inventory["files"]), "fresh_noise_exact": 12, "exact_reference_images": 12,
               "new_images_and_reviewed_views": 156, "trajectories_verified": 12, "saved_query_payloads_verified": 36,
               "all_action_interpolations_exact": 144, "new_teacher_queries": 72, "optimizer_updates": 0,
               "labels_statistics_and_target_selection_rebuilt": True, "historical_artifacts": history, "upstream": upstream,
               "job_id": job["job_id"], "allocated_gpu_hours": job["allocated_gpu_hours"],
               "limitations": "Selected prompt cohort, assistant labels, development collection, oracle target pool, shared large assets stat-checked rather than full content-rehashed."})
    print("Verified phase17 references, all36 saved queries,144 action blends,156 labels, seed exclusion, target ranking, unchanged phases6-16.")


if __name__ == "__main__":
    main()
