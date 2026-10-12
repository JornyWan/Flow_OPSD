"""Collect a historical seed inventory and freeze the phase17 candidate grid."""
import argparse
import json
from pathlib import Path
import shutil

from opdlab.config import Experiment
from opdlab.coverage_probe import VERSION, candidate_specs, make_pairs
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import prepare


def seed_values(value, seed_field=False):
    if isinstance(value, dict):
        return set().union(*(seed_values(v, seed_field or "seed" in k.lower()) for k, v in value.items()))
    if isinstance(value, list):
        return set().union(*(seed_values(v, seed_field) for v in value))
    return {value} if seed_field and type(value) is int and 0 <= value < 2**31 else set()


def seed_inventory():
    paths = set()
    patterns = ("*plan*.json", "*protocol*.json", "*manifest*.json", "config.json", "endpoints.jsonl", "candidates.jsonl", "results.jsonl", "references.jsonl", "queries.jsonl", "results.json")
    for pattern in patterns:
        paths.update((ROOT / "runs").glob(f"*/{pattern}"))
        paths.update((ROOT / "experiments").glob(f"local_expert_v*/{pattern}"))
    inventory, excluded = {}, set()
    for path in sorted(paths):
        if "le17_" in str(path) or "local_expert_v17" in str(path):
            continue
        values = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.suffix == ".jsonl" else read_json(path)
        seeds = seed_values(values)
        inventory[str(path)] = {"sha256": sha256(path), "seeds": sorted(seeds)}
        excluded.update(seeds)
    return {"scope": "Prior lab campaign and run top-level plans/manifests/configs/protocols/query/endpoint/results records, including failed attempts; not unrelated legacy projects", "files": inventory, "excluded_seeds": sorted(excluded)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--attempt", default="001")
    parser.add_argument("--site", default="site.local.json")
    args = parser.parse_args()
    campaign, previous = ROOT / "experiments/local_expert_v17", ROOT / "runs/le16_transfer_001"
    old = read_json(previous / "probe_plan.json")
    inventory = seed_inventory()
    pairs = make_pairs(old["points"], inventory["excluded_seeds"])
    config = Experiment.load(previous / "config.json")
    config.alphas = [.5, 1.]
    run = prepare(config, ROOT / f"runs/le17_coverage_{args.attempt}", read_json(args.site))
    plan = {"version": VERSION, "student": old["checkpoints"]["joint128"], "points": old["points"], "pairs": pairs,
            "query_source": old["query_source"], "candidates": candidate_specs(), "excluded_seeds": inventory["excluded_seeds"],
            "expected_counts": {"forward_student": 60, "forward_behavior": 1026, "forward_ocr": 36, "forward_geneval": 36,
                                "forward_examples": 2280, "decode_images": 168, "verifier_images": 168, "text_encode_prompts": 12, "backward_queries": 0},
            "scope": "Prospective development/target collection on six previously selected prompts, not held-out confirmation or deployable selection"}
    write_json(run / "probe_plan.json", plan)
    write_json(run / "seed_inventory.json", inventory)
    inputs = [Path(p) for p in inventory["files"]]
    inputs += [previous / "probe_plan.json", Path(plan["student"]), Path(__file__).resolve(), ROOT / "scripts/review_coverage_probe.py"]
    inputs += [ROOT / "experiments/local_expert_v16" / n for n in ("NEXT.md", "report_index.json")]
    for point in plan["points"]:
        inputs.append(Path(plan["query_source"]) / point["saved_query"])
        inputs.extend(Path(plan["query_source"]) / v["image"] for v in point["images"].values())
    for n in ("PROTOCOL.md", "REVIEW_RUBRIC.md"):
        shutil.copyfile(campaign / n, run / n)
        inputs.append(run / n)
    inputs += [run / "probe_plan.json", run / "seed_inventory.json"]
    write_json(run / "probe_attestation.json", {"inputs": {str(p): sha256(p) for p in inputs}})
    write_json(campaign / f"attempt_{args.attempt}.json", {"run": str(run), "plan_sha256": sha256(run / "probe_plan.json"),
               "protocol_sha256": sha256(run / "PROTOCOL.md"), "excluded_seed_count": len(inventory["excluded_seeds"]),
               "seed_source_files": len(inventory["files"]), "new_images": 156, "teacher_queries": 72, "updates": 0})
    print({"run": str(run), "pairs": len(pairs), "excluded_seed_count": len(inventory["excluded_seeds"]), "counts": plan["expected_counts"]})


if __name__ == "__main__":
    main()
