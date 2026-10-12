"""Read-only evidence checks for phases8/9; write their verification summaries."""
import json
from pathlib import Path
import subprocess

import torch

from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import verify_run


def historical_index(phase, revision):
    entries = read_json(ROOT / f"experiments/local_expert_v{phase}/report_index.json")["artifacts_sha256"]
    import hashlib
    for name, expected in entries.items():
        actual = hashlib.sha256(subprocess.check_output(
            ["git", "show", f"{revision}:README.md"], cwd=ROOT)).hexdigest() if name == "README.md" else sha256(ROOT / name)
        if actual != expected:
            raise ValueError(f"Historical phase{phase} artifact changed: {name}")
    return {"artifact_count": len(entries), "all_recorded_artifacts_unchanged": True,
            "README_historical_revision": revision}


def verify_phase(phase, run_name, output_name, tensor_key):
    run, review = ROOT / "runs" / run_name, ROOT / f"runs/le{phase}_review_001"
    audit = verify_run(run)
    inputs = read_json(run / "probe_attestation.json")["inputs"]
    for path, expected in inputs.items():
        assert sha256(path) == expected, path
    result = read_json(run / "result.json")
    assert result["status"] == "complete" and result["optimizer_updates"] == 0
    assert not any((run / name).exists() for name in ("RUNNING", "failure.json"))
    assert sha256(run / f"{output_name}.jsonl") == result[f"{output_name}_sha256"]
    rows = [json.loads(s) for s in (run / f"{output_name}.jsonl").read_text().splitlines()]
    expected_count = 35 if phase == 8 else 25
    assert len(rows) == result["counts"]["decode_images"] == result["counts"]["verifier_images"] == expected_count
    for row in rows:
        assert sha256(run / row["image"]) == row["image_sha256"]
    for name, expected in result[tensor_key].items():
        assert sha256(run / name) == expected, name
    frozen, review_inputs = read_json(review / "observations_frozen.json"), read_json(review / "inputs.json")
    for name in ("key", "inherited", "observations"):
        assert sha256(review / f"{name}.json") == frozen[f"{name}_sha256"]
    rubric = ROOT / "experiments/local_expert_v8/REVIEW_RUBRIC.md" if phase == 8 else run / "REVIEW_RUBRIC.md"
    assert sha256(rubric) == frozen["rubric_sha256"]
    assert frozen["frozen_before_key_join"] is True
    assert sha256(run / "result.json") == review_inputs["result_sha256"]
    keys, items = read_json(review / "key.json"), read_json(review / "items.json")
    assert len(keys) == len(items) == review_inputs["new_unique_images"]
    for entry, item in zip(keys, items):
        assert entry["case"] == item["case"]
        assert sha256(review / item["image"]) == entry["image_sha256"]
    campaign = ROOT / f"experiments/local_expert_v{phase}"
    assert read_json(campaign / "results.json")["gpu_result"] == result
    for name in ("observations", "observations_frozen"):
        assert sha256(campaign / f"{name}.json") == sha256(review / f"{name}.json")
    slurm = read_json(campaign / "slurm_status.json")
    assert len(slurm["jobs"]) == 1
    job = slurm["jobs"][0]
    assert job["state"] == "COMPLETED" and job["exit_code"] == "0:0"
    return {"frozen_inputs_valid": True, "attested_input_count": len(inputs), "asset_audit": audit,
            "generated_image_hashes_valid": len(rows), "trajectory_file_hashes_valid": len(result[tensor_key]),
            "review_hashes_and_coverage_valid": True, "unseen_review_images": len(items),
            "result_sha256": sha256(run / "result.json"), "optimizer_updates": 0,
            "job_id": job["job_id"], "allocated_gpu_hours": job["allocated_gpu_hours"]}


def main():
    prior = {"phase6": historical_index(6, "e384a09"), "phase7": historical_index(7, "8c1e160")}
    upstream = {}
    for name, spec in read_json(ROOT / "upstream.lock.json").items():
        path = ROOT / "upstream" / name
        assert subprocess.check_output(["git", "status", "--porcelain"], cwd=path).strip() == b""
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
        assert head == spec["commit"]
        upstream[name] = head
    reports = {
        8: verify_phase(8, "le8_trajectory_001", "endpoints", "trajectory_hashes"),
        9: verify_phase(9, "le9_requery_001", "candidates", "branches_sha256"),
    }
    r8 = read_json(ROOT / "runs/le8_trajectory_001/result.json")
    r9 = read_json(ROOT / "runs/le9_requery_001/result.json")
    assert len(r8["replay_checks"]) == 15 and all(c["exact"] for c in r8["replay_checks"])
    assert len(r8["duplicate_checks"]) == 4 and all(c["exact"] for c in r8["duplicate_checks"])
    assert len(r9["baseline_replay"]) == 5 and all(c["exact"] for c in r9["baseline_replay"])
    checkpoint = torch.load(ROOT / "runs/le7_fit_001/checkpoints/joint/step128.pt", map_location="cpu", weights_only=True)
    assert tensor_digest(checkpoint) == r8["student_sha256"] == r9["student_sha256"]
    behavior = torch.load(ROOT / "runs/le3_probe_001/frozen_behavior.pt", map_location="cpu", weights_only=True)
    assert tensor_digest(behavior) == r8["behavior_sha256"]
    branch_checks = 0
    for case in (9, 11, 12, 17, 19):
        old = torch.load(ROOT / f"runs/le8_trajectory_001/trajectory-{case:02d}.pt", map_location="cpu", weights_only=True)
        new = torch.load(ROOT / f"runs/le9_requery_001/branches-{case:02d}.pt", map_location="cpu", weights_only=True)
        base, k = new["branches"]["base"], new["query_index"]
        for key, length in (("states", 11), ("actions", 10)):
            assert len(old["full"][key]) == len(base[key]) == length
            assert all(torch.equal(a, b) for a, b in zip(old["full"][key], base[key]))
        for name, branch in new["branches"].items():
            assert len(branch["states"]) == 11 and len(branch["actions"]) == 10
            assert all(torch.equal(a, b) for a, b in zip(base["states"][:k + 1], branch["states"][:k + 1]))
            assert all(torch.equal(a, b) for a, b in zip(base["actions"][:k], branch["actions"][:k]))
            if name != "base":
                teacher, alpha = name.split("@")
                expected = base["actions"][k] + float(alpha) * (new["experts"][teacher] - base["actions"][k])
                torch.testing.assert_close(branch["actions"][k], expected)
            branch_checks += 1
    reports[8].update(phase7_image_replay_checks=15, index_zero_duplicate_checks=4)
    reports[9].update(phase8_image_replay_checks=5, phase8_full_trajectory_tensor_replay_checks=5,
                      branch_prefix_shape_and_action_checks=branch_checks,
                      suffix_provenance="Current student verified by frozen worker source and CPU unit test; no second GPU rerun")
    for phase, report in reports.items():
        report.update(prior_artifacts=prior, upstream=upstream, original_checkpoint_hashes_valid=True,
                      limits="Large shared base/reward assets use recorded inventory validation, not a newly computed full content hash")
        write_json(ROOT / f"experiments/local_expert_v{phase}/verification.json", report)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
