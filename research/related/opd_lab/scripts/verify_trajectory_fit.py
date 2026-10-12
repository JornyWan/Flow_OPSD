"""Verify completed phase10 artifacts without regenerating images or training."""
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import verify_run
from review_trajectory_fit import validate
from verify_state_probes import historical_index


def main():
    campaign = ROOT / "experiments/local_expert_v10"
    run, review = ROOT / "runs/le10_fit_001", ROOT / "runs/le10_review_001"
    report = {"asset_audit": verify_run(run), "endpoint_count": len(validate(run))}
    result, plan = read_json(run / "result.json"), read_json(run / "probe_plan.json")
    assert result["counts"]["backward_queries"] == 3200
    for name, expected in {"forward_student": 3600, "forward_behavior": 223,
                           "forward_examples": 7646, "decode_images": 36, "verifier_images": 36}.items():
        assert result["counts"][name] == expected, name
    assert result["new_teacher_queries"] == 0 and result["source_expert_actions"] == 1
    assert all(result["counts"].get(f"forward_{name}", 0) == 0 for name in ("ocr", "geneval"))
    for name, expected in result["checkpoint_hashes"].items():
        assert sha256(run / name) == expected, name
    assert sha256(run / "supervision.json") == result["supervision_sha256"]
    supervision = read_json(run / "supervision.json")
    assert len(supervision) == 50 and sum(r["origin"] == "external_expert" for r in supervision) == 1
    references = [json.loads(s) for s in (run / "references.jsonl").read_text().splitlines()]
    assert len(references) == 6 and all(r["exact"] and sha256(run / r["image"]) == r["image_sha256"] for r in references)
    updates = [json.loads(s) for s in (run / "updates.jsonl").read_text().splitlines()]
    metrics = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(updates) == 352 and len(metrics) == 250
    original = torch.load(plan["student"], map_location="cpu", weights_only=True)
    initial_hash = tensor_digest(original)
    assert result["initial_student_sha256"] == initial_hash
    for arm, spec in plan["arms"].items():
        group = [r for r in updates if r["arm"] == arm]
        assert [r["optimizer_step"] for r in group] == list(range(1, spec["updates"] + 1))
        assert sum(r["backward_queries"] for r in group) == result["arms"][arm]["counts"]["backward_queries"] == 1600
        assert result["arms"][arm]["initial_student_sha256"] == initial_hash
        root = run / "checkpoints" / arm
        assert tensor_digest(torch.load(root / "step000.pt", map_location="cpu", weights_only=True)) == initial_hash
        training = torch.load(root / "training.pt", map_location="cpu", weights_only=False)
        final = torch.load(root / f"step{spec['updates']:03d}.pt", map_location="cpu", weights_only=True)
        assert tensor_digest(training["student"]) == tensor_digest(final)
        assert training["optimizer_step"] == spec["updates"]
        assert all(int(s["step"]) == spec["updates"] for s in training["optimizer"]["state"].values())
        raw = load_file(root / "raw_lora/adapter_model.safetensors")
        assert tensor_digest(raw) == tensor_digest(final)
        for step in spec["audits"]:
            group = [r for r in metrics if r["arm"] == arm and r["optimizer_step"] == step]
            assert len(group) == 50 and len({(r["id"], r["index"]) for r in group}) == 50
            checksum = tensor_digest(torch.load(root / f"step{step:03d}.pt", map_location="cpu", weights_only=True))
            assert all(r["student_sha256"] == checksum for r in group)
            if step:
                for spec_case in plan["selected"]:
                    trajectory = torch.load(root / f"trajectory-step{step:03d}-case{spec_case['case']:02d}.pt", map_location="cpu", weights_only=True)
                    assert len(trajectory["states"]) == 11 and len(trajectory["actions"]) == 10
    frozen, inputs = read_json(review / "observations_frozen.json"), read_json(review / "inputs.json")
    assert frozen["frozen_before_key_join"] is True
    for name in ("observations", "inherited", "key"):
        assert sha256(review / f"{name}.json") == frozen[f"{name}_sha256"]
    assert sha256(run / "REVIEW_RUBRIC.md") == frozen["rubric_sha256"]
    assert inputs["result_sha256"] == sha256(run / "result.json")
    for key, item in zip(read_json(review / "key.json"), read_json(review / "items.json")):
        assert key["case"] == item["case"] and sha256(review / item["image"]) == key["image_sha256"]
    summary = read_json(campaign / "results.json")
    assert summary["gpu_result"] == result and len(summary["joined_review"]) == 30
    for name in ("observations", "observations_frozen"):
        assert sha256(campaign / f"{name}.json") == sha256(review / f"{name}.json")
    slurm = read_json(campaign / "slurm_status.json")
    assert len(slurm["jobs"]) == 1
    assert slurm["jobs"][0]["state"] == "COMPLETED" and slurm["jobs"][0]["exit_code"] == "0:0"
    prior = {str(p): historical_index(p, revision) for p, revision in ((6, "e384a09"), (7, "8c1e160"), (8, "d9c392c"), (9, "d9c392c"))}
    report.update(initial_weight_identity=True, terminal_optimizer_steps_valid=True, raw_exports_match=True,
                  checkpoint_file_hashes=len(result["checkpoint_hashes"]), baseline_and_target_replays=6,
                  backward_queries_per_arm=1600, optimizer_updates=352, metric_rows=250,
                  frozen_review_hashes_valid=True, historical_artifacts=prior,
                  allocated_gpu_hours=slurm["allocated_gpu_hours_including_failed_attempts_so_far"],
                  limitations="Saved artifacts/source contracts checked; no additional GPU replay. Shared large assets use recorded inventories, not new full hashes.")
    write_json(campaign / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
