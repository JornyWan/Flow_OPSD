"""Read-only artifact checks and independent aggregate checks for phase11."""
import json
import subprocess

import torch

from opdlab.data import load_records
from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import verify_run
from opdlab.transfer_eval import make_pairs
from review_transfer import validate, visual_metrics
from verify_state_probes import historical_index


def main():
    run, review, campaign = ROOT / "runs/le11_transfer_001", ROOT / "runs/le11_review_001", ROOT / "experiments/local_expert_v11"
    report = {"asset_audit": verify_run(run)}
    rows = validate(run)
    result, plan = read_json(run / "result.json"), read_json(run / "probe_plan.json")
    assert len(rows) == result["transfer_images"] == 168 and result["optimizer_updates"] == result["new_teacher_queries"] == 0
    for name, expected in {"forward_student": 1720, "forward_examples": 3440, "decode_images": 172, "verifier_images": 172}.items():
        assert result["counts"][name] == expected, name
    assert all(result["counts"].get(k, 0) == 0 for k in ("backward_queries", "forward_behavior", "forward_ocr", "forward_geneval"))
    assert plan["pairs"] == make_pairs(plan["selected"], load_records(run / "data.jsonl", "train"), load_records(run / "data.jsonl", "validation"))
    anchors = [json.loads(s) for s in (run / "anchors.jsonl").read_text().splitlines()]
    assert len(anchors) == 4
    for model, spec in plan["models"].items():
        expected = tensor_digest(torch.load(spec["weights"], map_location="cpu", weights_only=True))
        assert expected == result["model_hashes"][model]
        assert all(r["student_sha256"] == expected for r in rows if r["model"] == model)
        anchor = next(r for r in anchors if r["model"] == model)
        assert anchor["exact"] and sha256(run / anchor["image"]) == anchor["image_sha256"] == spec["anchor"]["image_sha256"]
    for pair in plan["pairs"]:
        group = [r for r in rows if r["pair_id"] == pair["pair_id"]]
        assert len(group) == 4 and {r["noise_sha256"] for r in group} == {result["noise_hashes"][pair["pair_id"]]}
    frozen = read_json(review / "observations_frozen.json")
    assert frozen["frozen_before_key_join"] is True
    for name in ("observations", "key", "items"):
        assert sha256(review / f"{name}.json") == frozen[f"{name}_sha256"]
    assert sha256(run / "REVIEW_RUBRIC.md") == frozen["rubric_sha256"]
    for key, item in zip(read_json(review / "key.json"), read_json(review / "items.json")):
        assert key["case"] == item["case"]
        for letter in "ABCD":
            assert sha256(review / item["images"][letter]) == key["mapping"][letter]["image_sha256"]
    summary = read_json(campaign / "results.json")
    assert summary["gpu_result"] == result and len(summary["joined_review"]) == 168 and len(summary["paired_rows"]) == 126
    pixel_labels = {}
    for entry in summary["joined_review"]:
        h, obs = entry["row"]["image_sha256"], entry["observation"]
        label = (obs["clauses"], obs["full_visibility"])
        assert h not in pixel_labels or pixel_labels[h] == label, "Identical pixels have inconsistent frozen labels"
        pixel_labels[h] = label
    for cohort, n in (("train_new_seed", 10), ("validation", 32)):
        for model in plan["models"]:
            group = [e for e in summary["joined_review"] if e["row"]["cohort"] == cohort and e["row"]["model"] == model]
            assert len(group) == n
            for metric in ("complete_scene", "joint_clauses", "full_visibility", "has_unknown"):
                total = sum(visual_metrics(e["observation"])[metric] for e in group)
                assert total == summary["scores"][cohort][model][metric]["sum"]
    status = read_json(campaign / "slurm_status.json")
    assert len(status["jobs"]) == 1 and status["jobs"][0]["state"] == "COMPLETED" and status["jobs"][0]["exit_code"] == "0:0"
    upstream = {}
    for name, spec in read_json(ROOT / "upstream.lock.json").items():
        path = ROOT / "upstream" / name
        assert subprocess.check_output(["git", "status", "--porcelain"], cwd=path).strip() == b""
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
        assert head == spec["commit"]
        upstream[name] = head
    prior = {str(p): historical_index(p, revision) for p, revision in ((6,"e384a09"),(7,"8c1e160"),(8,"d9c392c"),(9,"d9c392c"),(10,"9dcc18b"))}
    report.update(transfer_image_hashes=168, exact_anchor_checks=4, matched_noise_pairs=42, unchanged_model_states=4,
                  frozen_review_views=168, unique_transfer_images=len(pixel_labels), aggregate_counts_valid=True, historical_artifacts=prior, upstream=upstream,
                  allocated_gpu_hours=status["allocated_gpu_hours_including_failed_attempts_so_far"],
                  limits="Shared large assets checked by recorded inventories, not new full content hashes; no second GPU replay")
    write_json(campaign / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
