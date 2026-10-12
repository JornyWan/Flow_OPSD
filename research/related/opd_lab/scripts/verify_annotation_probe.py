"""Replay phase15 analysis and verify pairing, fitting budgets and frozen archives."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from opdlab.annotation_probe import ARMS, cohort, control_targets, evaluation_pairs
from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT, assert_environment, read_json, sha256, write_json
from opdlab.runtime import verify_run
from review_annotation_probe import build_report, lines, validate
from verify_state_probes import historical_index


def main():
    campaign = ROOT / "experiments/local_expert_v15"
    run, review = ROOT / "runs/le15_annotation_001", ROOT / "runs/le15_review_001"
    result, plan = read_json(run / "result.json"), read_json(run / "probe_plan.json")
    assert not any((run / n).exists() for n in ("RUNNING", "failure.json"))
    audit = verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    endpoints = validate(run)
    source = Path(plan["source"])
    points = cohort(lines(source / "candidates.jsonl"), read_json(ROOT / "experiments/local_expert_v12/results.json")["queries"])
    assert points == plan["points"] and control_targets(points) == plan["targets"]
    records = [r for r in lines(run / "data.jsonl") if r["split"] == "train"]
    assert evaluation_pairs(points, records, plan["excluded_seeds"]) == plan["evaluation"]
    assert len(points) == result["distinct_fitting_queries"] == 12
    for key, count in plan["expected_counts"].items():
        assert result["counts"].get(key, 0) == count, (key, result["counts"])
    assert result["new_teacher_queries"] == 0
    assert all(result["counts"].get("forward_" + t, 0) == 0 for t in ("ocr", "geneval"))
    for p, h in result["checkpoint_hashes"].items():
        assert sha256(run / p) == h
    assert len(result["checkpoint_hashes"]) == 12
    initial = tensor_digest(torch.load(plan["student"], map_location="cpu", weights_only=True))
    assert initial == result["initial_student_sha256"] == result["model_hashes"]["joint128"]
    updates, metrics = lines(run / "updates.jsonl"), lines(run / "metrics.jsonl")
    assert len(updates) == 384 and len(metrics) == 108
    for arm in ARMS:
        fitting = [u for u in updates if u["arm"] == arm]
        assert [u["step"] for u in fitting] == list(range(1, 129))
        assert all(u["backward_queries"] == 12 and np.isfinite(u["objective"]) and np.isfinite(u["gradient_norm"]) for u in fitting)
        assert result["arms"][arm]["counts"]["backward_queries"] == 1536
        assert result["arms"][arm]["counts"]["forward_student"] == 1572
        assert result["arms"][arm]["initial_student_sha256"] == initial
        assert result["arms"][arm]["optimizer_updates"] == 128
        for step in (0, 32, 128):
            assert {m["id"] for m in metrics if (m["arm"], m["step"]) == (arm, step)} == {p["id"] for p in points}
            state = torch.load(run / "checkpoints" / arm / f"step{step:03d}.pt", map_location="cpu", weights_only=True)
            h = tensor_digest(state)
            assert h == (initial if step == 0 else result["model_hashes"][arm] if step == 128 else h)
        training = torch.load(run / "checkpoints" / arm / "training.pt", map_location="cpu", weights_only=False)
        assert tensor_digest(training["student"]) == result["model_hashes"][arm] == result["arms"][arm]["final_student_sha256"]
        assert training["optimizer_step"] == 128 and training["scaler"] and training["rng"]
        assert training["optimizer"]["state"] and all(s["step"].item() == 128 for s in training["optimizer"]["state"].values())
    indexed = {p["id"]: p for p in points}
    refs = lines(run / "references.jsonl")
    assert len(refs) == 18 and Counter(r["mode"] for r in refs) == {"noop": 12, "target": 6}
    for r in refs:
        p = indexed[r["id"]]
        name = "noop" if r["mode"] == "noop" else p["label_target"]
        assert r["exact"] and r["image_sha256"] == p["images"][name]["sha256"] == sha256(run / r["image"])
    noises = defaultdict(set)
    for r in endpoints:
        noises[r["pair_id"]].add(r["noise_sha256"])
        assert r["student_sha256"] == result["model_hashes"][r["model"]]
        if r["cohort"] == "fitting_seed_diagnostic" and r["model"] == "joint128":
            assert r["image_sha256"] == indexed[r["id"]]["images"]["noop"]["sha256"]
    assert len(noises) == 38 and all(len(s) == 1 for s in noises.values())
    assert {p: next(iter(s)) for p, s in noises.items()} == result["noise_hashes"]
    rebuilt = json.loads(json.dumps(build_report(run, review)))
    assert rebuilt == read_json(campaign / "results.json")
    for n in ("observations", "observations_frozen"):
        assert sha256(campaign / f"{n}.json") == sha256(review / f"{n}.json")
    assert sha256(run / "result.json") == read_json(review / "inputs.json")["result_sha256"]
    # Check public review pixels against the private key after labels are frozen.
    items = {i["case"]: i for i in read_json(review / "items.json")}
    labels = {o["case"]: o for o in read_json(review / "observations.json")}
    seen, checked = {}, 0
    inherited = read_json(review / "inherited.json")
    for entry in read_json(review / "key.json"):
        for letter, row in entry["mapping"].items():
            h = sha256(review / items[entry["case"]]["images"][letter])
            assert h == row["image_sha256"]
            obs = labels[entry["case"]]["views"][letter]
            semantic = (obs["clauses"], obs["full_visibility"])
            if h in seen:
                assert seen[h] == semantic
            seen[h] = semantic
            old = inherited.get(str(entry["case"]), {}).get(letter)
            if old:
                assert semantic == (old["clauses"], old["full_visibility"])
            checked += 1
    assert checked == 152
    upstream = {}
    for name, spec in read_json(ROOT / "upstream.lock.json").items():
        path = ROOT / "upstream" / name
        assert subprocess.check_output(["git", "status", "--porcelain"], cwd=path).strip() == b""
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
        assert head == spec["commit"]
        upstream[name] = head
    historical = {str(p): historical_index(p, revision) for p, revision in
                  ((6, "e384a09"), (7, "8c1e160"), (8, "d9c392c"), (9, "d9c392c"), (10, "9dcc18b"),
                   (11, "3923b6a"), (12, "8e9c655"), (13, "bb804ca"), (14, "a23a429"))}
    status = read_json(campaign / "slurm_status.json")
    assert len(status["jobs"]) == 1
    job = status["jobs"][0]
    assert job["state"] == "COMPLETED" and job["exit_code"] == "0:0"
    write_json(campaign / "verification.json", {"runtime_asset_audit": audit, "input_and_checkpoint_hashes_valid": True,
               "exact_reference_images": 18, "saved_noop_replays": 12, "evaluated_images_and_reviewed_views": 152,
               "common_noise_pairs": 38, "new_noise_pairs": 32, "inherited_review_views": sum(len(v) for v in inherited.values()),
               "updates_per_arm": 128, "total_backward_queries": 4608, "checkpoint_files": 12,
               "optimizer_state_steps_checked": True, "all_statistics_recomputed": True, "historical_artifacts": historical,
               "upstream": upstream, "job_id": job["job_id"], "allocated_gpu_hours": job["allocated_gpu_hours"],
               "limitations": "Selected train cohorts; assistant annotations; fixed saved states; flow10 only; full shared base/reward content hashes not recomputed."})
    print({"verified": "phase15, 18 references, 152 images, matched updates/pairing, unchanged phases6-14"})


if __name__ == "__main__":
    main()
