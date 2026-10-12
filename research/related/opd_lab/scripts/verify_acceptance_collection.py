"""Verify collection provenance, seed isolation, all labels and the binding coverage decision."""
import argparse
import json
from pathlib import Path

import torch

from opdlab.acceptance_collection import GATE, new_pairs, rows
from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT, assert_environment, read_json, sha256, write_json
from opdlab.runtime import verify_run
from prepare_coverage_probe import seed_values
from review_acceptance_collection import build_report, validate
from verify_state_probes import historical_index


def verify(run, review, campaign):
    asset_audit = verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    validate(run)
    plan, result = read_json(run / "probe_plan.json"), read_json(run / "result.json")
    assert not any((run / n).exists() for n in ("updates.jsonl", "checkpoints", "eval_queries"))
    inventory = read_json(run / "seed_inventory.json")
    excluded = set()
    for p, row in inventory["files"].items():
        assert sha256(p) == row["sha256"]
        values = seed_values(rows(p) if p.endswith(".jsonl") else read_json(p))
        assert sorted(values) == row["seeds"]
        excluded.update(values)
    assert sorted(excluded) == inventory["excluded_seeds"]
    assert (plan["collection"], plan["evaluation"]) == new_pairs(read_json(plan["source_plan"]), excluded)
    assert plan["coverage_gate"] == GATE
    model_hash = tensor_digest(torch.load(plan["student"], map_location="cpu", weights_only=True))
    assert model_hash == result["student_sha256"]
    records = {r["id"]: r for r in rows(run / "data.jsonl")}
    queries = rows(run / "queries.jsonl")
    assert len(queries) == 48 and {q["pair_id"] for q in queries} == {p["pair_id"] for p in plan["collection"]}
    endpoints = rows(run / "endpoints.jsonl")
    for row in queries:
        assert sha256(run / row["saved_query"]) == row["saved_query_sha256"]
        q = torch.load(run / row["saved_query"], map_location="cpu", weights_only=False)
        assert q["record"] == records[row["id"]] and (q["seed"], q["index"], q["dt"]) == (row["seed"], 3, row["dt"])
        assert q["dt"] < 0 and torch.isfinite(q["state"]).all()
        assert tensor_digest({"state": q["state"]}) == row["state_sha256"]
        assert tensor_digest({str(i): c for i, c in enumerate(q["context"])}) == row["context_sha256"]
        assert set(q["actions"]) == {"noop", "geneval@1"}
        assert {n: tensor_digest({"action": a}) for n, a in q["actions"].items()} == row["action_hashes"]
        assert all(torch.isfinite(a).all() for a in q["actions"].values())
        noise = torch.randn(q["state"].shape, generator=torch.Generator().manual_seed(q["seed"]), dtype=torch.float32)
        assert tensor_digest({"noise": noise}) == q["noise_sha256"] == row["noise_sha256"]
        assert row["prefix_sha256"] == row["suffix_sha256"] == model_hash
        matches = [e for e in endpoints if e["pair_id"] == row["pair_id"]]
        assert len(matches) == 2 and all(e["noise_sha256"] == row["noise_sha256"] and e["student_sha256"] == model_hash and not e["exact"] for e in matches)
    expected_refs = {(s["pair"]["pair_id"], m): r for s in plan["references"] for m, r in s["images"].items()}
    references = rows(run / "references.jsonl")
    assert len(references) == 12 and {(r["pair_id"], r["mode"]) for r in references} == set(expected_refs)
    for r in references:
        old = expected_refs[r["pair_id"], r["mode"]]
        assert r["exact"] and r["image_sha256"] == old["image_sha256"] and r["noise_sha256"] == old["noise_sha256"]
    gate = read_json(run / "replay_gate.json")
    assert gate["all_exact"] and gate["counts_before_collection"]["forward_geneval"] == 6
    assert gate["counts_before_collection"]["decode_images"] == 12
    rebuilt = json.loads(json.dumps(build_report(run, review)))
    assert rebuilt == read_json(campaign / "results.json")
    manifest = read_json(campaign / "target_manifest.json")
    assert manifest == {"training_allowed_by_coverage_gate": rebuilt["coverage_gate"]["passed"],
                        "source_results_sha256": sha256(campaign / "results.json"), "targets": rebuilt["targets"]}
    items = {i["case"]: i for i in read_json(review / "items.json")}
    for item in read_json(review / "key.json"):
        for letter, r in item["mapping"].items():
            assert sha256(review / items[item["case"]]["images"][letter]) == r["image_sha256"]
    for n in ("observations.json", "observations_frozen.json"):
        assert sha256(review / n) == sha256(campaign / n)
    jobs = read_json(campaign / "slurm_status.json")["jobs"]
    assert any(j["run"] == str(run) and j["state"] == "COMPLETED" and j["exit_code"] == "0:0" for j in jobs)
    history = historical_index(19, "9cbbf01")
    write_json(campaign / "verification.json", {"runtime_asset_audit": asset_audit, "phase19_archive": history,
               "reference_images_exact": 12, "new_states": 48, "new_images": 96, "review_slots": 96,
               "new_seeds_disjoint_from_prior_inventory": 80, "reserved_evaluation_seeds_unrendered": 32,
               "observations_and_target_selection_rebuilt": True, "coverage_gate": rebuilt["coverage_gate"],
               "optimizer_updates": 0, "allocated_gpu_hours": sum(j["allocated_gpu_hours"] for j in jobs),
               "limits": "Provisional assistant supervision; six familiar prompts; large shared assets stat-audited."})
    print("Verified48 states/96 images,12 references,80 isolated seeds, frozen labels and unchanged phase19 archive.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", default="runs/le20_collection_001")
    parser.add_argument("--review", default="runs/le20_review_001")
    parser.add_argument("--campaign", default="experiments/local_expert_v20")
    args = parser.parse_args()
    verify(Path(args.run).resolve(), Path(args.review).resolve(), Path(args.campaign).resolve())
