"""Audit reference reuse, saved actions, blinded joins and the preceding frozen experiment."""
import argparse
import json
import math
from pathlib import Path

import torch

from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT, assert_environment, read_json, sha256, write_json
from opdlab.learnability import action_metrics
from opdlab.runtime import verify_run
from opdlab.teacher_opportunity import rows
from review_teacher_opportunity import build_report, validate
from verify_state_probes import historical_index


def verify(run, review, campaign):
    audit = verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    validate(run)
    assert not any((run / n).exists() for n in ("updates.jsonl", "checkpoints"))
    plan, result = read_json(run / "probe_plan.json"), read_json(run / "result.json")
    hashes = {n: tensor_digest(torch.load(p, map_location="cpu", weights_only=True)) for n, p in plan["checkpoints"].items()}
    assert hashes == result["model_hashes"]
    queries = rows(run / "queries.jsonl")
    assert len(queries) == 12 and {q["pair_id"] for q in queries} == {p["pair_id"] for p in plan["pairs"]}
    max_difference = 0.
    for r in queries:
        q = torch.load(run / r["saved_query"], map_location="cpu", weights_only=False)
        old = torch.load(Path(plan["source"]) / "eval_queries" / f"{r['pair_id']}.pt", map_location="cpu", weights_only=False)
        assert sha256(run / r["saved_query"]) == r["saved_query_sha256"]
        assert (q["record"]["id"], q["seed"], q["index"], q["dt"]) == (r["id"], r["seed"], 3, r["dt"])
        assert all(q[k] == old[k] for k in ("record", "seed", "index", "dt", "noise_sha256"))
        assert torch.equal(q["state"], old["state"]) and torch.equal(q["actions"]["noop"], old["actions"]["noop"])
        assert all(torch.equal(a, b) for a, b in zip(q["context"], old["context"]))
        assert tensor_digest({"state": q["state"]}) == r["state_sha256"]
        assert tensor_digest({str(i): c for i, c in enumerate(q["context"])}) == r["context_sha256"]
        assert {n: tensor_digest({"action": a}) for n, a in q["actions"].items()} == r["action_hashes"]
        assert set(q["actions"]) == {"noop", "student_single", "ocr@1", "geneval@1"}
        assert r["prefix_sha256"] == r["suffix_sha256"] == hashes["initial"] and r["student_sha256"] == hashes["student"]
        assert r["noise_sha256"] == q["noise_sha256"] == plan["references"][r["pair_id"]]["noop"]["noise_sha256"]
        for mode in ("ocr@1", "geneval@1"):
            for k, v in action_metrics(q["actions"]["student_single"], q["actions"][mode], q["actions"]["noop"]).items():
                stored = r["action_metrics"][mode][k]
                assert math.isclose(v, stored, rel_tol=1e-12, abs_tol=1e-12)
                max_difference = max(max_difference, abs(v - stored))
    assert json.loads(json.dumps(build_report(run, review))) == read_json(campaign / "results.json")
    items = {i["case"]: i for i in read_json(review / "items.json")}
    known = {e["row"]["image_sha256"]: e["observation"] for e in read_json(ROOT / "experiments/local_expert_v18/results.json")["joined_review"]}
    inherited = read_json(review / "inherited.json")
    for item in read_json(review / "key.json"):
        for letter, row in item["mapping"].items():
            assert sha256(review / items[item["case"]]["images"][letter]) == row["image_sha256"]
            if row["image_sha256"] in known:
                assert inherited[str(item["case"])][letter] == known[row["image_sha256"]]
    for n in ("observations.json", "observations_frozen.json"):
        assert sha256(campaign / n) == sha256(review / n)
    jobs = read_json(campaign / "slurm_status.json")["jobs"]
    assert any(j["run"] == str(run) and j["state"] == "COMPLETED" and j["exit_code"] == "0:0" for j in jobs)
    history = historical_index(18, "bd585a7")
    write_json(campaign / "verification.json", {"runtime_asset_audit": audit, "exact_reference_images": 24,
               "new_teacher_images": 24, "queries_and_original_states_verified": 12, "review_slots": 48,
               "statistics_and_action_metrics_rebuilt": True, "max_cpu_cuda_metric_abs_difference": max_difference,
               "phase18_archive": history, "optimizer_updates": 0,
               "allocated_gpu_hours": sum(j["allocated_gpu_hours"] for j in jobs),
               "limitations": "Provisional assistant labels, exposed seeds; large shared assets stat-audited."})
    print("Verified24 exact references,24 teacher images,12 original states,48 labels, statistics and unchanged phase18 archive.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", default="runs/le19_opportunity_001")
    parser.add_argument("--review", default="runs/le19_review_001")
    parser.add_argument("--campaign", default="experiments/local_expert_v19")
    args = parser.parse_args()
    verify(Path(args.run).resolve(), Path(args.review).resolve(), Path(args.campaign).resolve())
