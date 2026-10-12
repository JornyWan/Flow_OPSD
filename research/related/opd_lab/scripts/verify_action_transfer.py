"""Recompute phase16 evidence, and check all frozen sources and archived phases."""
from collections import Counter
import json
import math
from pathlib import Path
import subprocess

import torch

from opdlab.action_transfer import OLD, FRESH, select_pairs
from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT, assert_environment, read_json, sha256, write_json
from opdlab.learnability import action_metrics
from opdlab.runtime import verify_run
from review_action_transfer import build_report, check_labels, lines, validate
from verify_state_probes import historical_index


def main():
    run, review, campaign = ROOT / "runs/le16_transfer_001", ROOT / "runs/le16_review_001", ROOT / "experiments/local_expert_v16"
    result, plan = read_json(run / "result.json"), read_json(run / "probe_plan.json")
    assert not any((run / n).exists() for n in ("RUNNING", "failure.json", "updates.jsonl", "checkpoints"))
    audit = verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    validate(run)
    assert select_pairs(read_json(Path(plan["previous"]) / "probe_plan.json")) == (plan["points"], plan["pairs"])
    for k, v in plan["expected_counts"].items():
        assert result["counts"].get(k, 0) == v, (k, v)
    assert result["optimizer_updates"] == 0 and result["new_teacher_queries"] == 24
    model_hashes = {n: tensor_digest(torch.load(p, map_location="cpu", weights_only=True)) for n, p in plan["checkpoints"].items()}
    assert model_hashes == result["model_hashes"]
    refs, queries = lines(run / "references.jsonl"), lines(run / "queries.jsonl")
    assert len(refs) == 42 and len(queries) == 18
    assert Counter(r["mode"] for r in refs) == {"joint128": 18, "label_selected": 18, "selected_teacher": 6}
    specs = {p["id"]: p for p in plan["points"]}
    for r in refs:
        expected = (specs[r["id"]]["images"][specs[r["id"]]["label_target"]]["sha256"] if r["mode"] == "selected_teacher"
                    else plan["references"][r["pair_id"]][r["mode"]]["image_sha256"])
        assert r["exact"] and r["image_sha256"] == r["reference_sha256"] == expected
    max_metric_difference = 0.0
    for r in queries:
        q = torch.load(run / r["saved_query"], map_location="cpu", weights_only=False)
        assert sha256(run / r["saved_query"]) == r["saved_query_sha256"]
        assert (q["record"]["id"], q["seed"], q["index"], q["dt"]) == (r["id"], r["seed"], r["index"], r["dt"])
        assert q["index"] == 3
        assert tensor_digest({"state": q["state"]}) == r["state_sha256"]
        assert tensor_digest({str(i): c for i, c in enumerate(q["context"])}) == r["context_sha256"]
        assert {n: tensor_digest({"action": a}) for n, a in q["actions"].items()} == r["action_hashes"]
        assert r["prefix_sha256"] == r["suffix_sha256"] == model_hashes["joint128"]
        assert r["student_sha256"] == model_hashes["label_selected"]
        assert r["noise_sha256"] == q["noise_sha256"] == plan["references"][r["pair_id"]]["joint128"]["noise_sha256"]
        if r["cohort"] == OLD:
            old = torch.load(Path(plan["query_source"]) / specs[r["id"]]["saved_query"], map_location="cpu", weights_only=False)
            assert torch.equal(q["state"], old["state"])
            assert all(torch.equal(a, b) for a, b in zip(q["context"], old["context"]))
            assert all(torch.equal(a, q["actions"][n]) for n, a in old["actions"].items())
        for n in ("ocr@1", "geneval@1"):
            metrics = action_metrics(q["actions"]["student_single"], q["actions"][n], q["actions"]["noop"])
            for k, value in metrics.items():
                stored = r["action_metrics"][n][k]
                # CPU vs CUDA FP64 reductions may differ in their last bits.
                assert math.isclose(value, stored, rel_tol=1e-12, abs_tol=1e-12), (k, value, stored)
                max_metric_difference = max(max_metric_difference, abs(value - stored))
    gate = read_json(run / "replay_gate.json")
    assert gate["all_exact"] and gate["reference_images"] == 42
    assert gate["counts_before_branches"].get("forward_ocr", 0) == gate["counts_before_branches"].get("forward_geneval", 0) == 0
    assert check_labels(review) == read_json(review / "observations.json")
    rebuilt = json.loads(json.dumps(build_report(run, review)))
    assert rebuilt == read_json(campaign / "results.json")
    for n in ("observations", "observations_frozen"):
        assert sha256(campaign / f"{n}.json") == sha256(review / f"{n}.json")
    items = {i["case"]: i for i in read_json(review / "items.json")}
    for item in read_json(review / "key.json"):
        for letter, row in item["mapping"].items():
            assert sha256(review / items[item["case"]]["images"][letter]) == row["image_sha256"]
    upstream = {}
    for name, spec in read_json(ROOT / "upstream.lock.json").items():
        path = ROOT / "upstream" / name
        assert subprocess.check_output(["git", "status", "--porcelain"], cwd=path).strip() == b""
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
        assert head == spec["commit"]
        upstream[name] = head
    historical = {str(p): historical_index(p, rev) for p, rev in
                  ((6, "e384a09"), (7, "8c1e160"), (8, "d9c392c"), (9, "d9c392c"), (10, "9dcc18b"),
                   (11, "3923b6a"), (12, "8e9c655"), (13, "bb804ca"), (14, "a23a429"), (15, "f1b9e79"))}
    status = read_json(campaign / "slurm_status.json")
    assert len(status["jobs"]) == 1
    job = status["jobs"][0]
    assert job["state"] == "COMPLETED" and job["exit_code"] == "0:0"
    write_json(campaign / "verification.json", {"runtime_asset_audit": audit, "input_and_checkpoint_hashes_valid": True,
               "exact_reference_images": 42, "new_images_and_reviewed_views": 42, "saved_queries_and_actions": 18,
               "old_state_context_action_exact": 6, "common_noise_pairs": 18, "no_updates": True,
               "new_teacher_queries": 24, "all_statistics_and_action_metrics_recomputed": True,
               "max_cpu_cuda_metric_abs_difference": max_metric_difference,
               "historical_artifacts": historical, "upstream": upstream, "job_id": job["job_id"],
               "allocated_gpu_hours": job["allocated_gpu_hours"], "limitations": "Exposed seeds, six prompts, assistant review; shared large weights stat-checked, not fully content-rehashed."})
    print("Verified phase16 counts, references, saved states/actions, labels/statistics, unchanged phases6-15.")


if __name__ == "__main__":
    main()
