"""Freeze collection, untouched evaluation seeds and minimum coverage before labeling."""
import argparse
from pathlib import Path
import shutil

from opdlab.acceptance_collection import GATE, VERSION, new_pairs, rows
from opdlab.config import Experiment
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import prepare
from prepare_coverage_probe import seed_values


def inventory():
    paths = set()
    patterns = ("*plan*.json", "*protocol*.json", "*manifest*.json", "config.json", "endpoints.jsonl",
                "candidates.jsonl", "results.jsonl", "references.jsonl", "queries.jsonl", "results.json", "supervision_inventory.json")
    for pattern in patterns:
        paths.update((ROOT / "runs").glob(f"*/{pattern}"))
        paths.update((ROOT / "experiments").glob(f"local_expert_v*/{pattern}"))
    files, excluded = {}, set()
    for p in sorted(paths):
        if any(x.startswith("le20_") or x == "local_expert_v20" for x in p.parts):
            continue
        seeds = seed_values(rows(p) if p.suffix == ".jsonl" else read_json(p))
        files[str(p)] = {"sha256": sha256(p), "seeds": sorted(seeds)}
        excluded.update(seeds)
    return {"files": files, "excluded_seeds": sorted(excluded),
            "scope": "Prior lab top-level plans/configs/manifests/query/endpoint/results seed fields including phases18/19 and failed runs."}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--attempt", default="001")
    args = parser.parse_args()
    campaign, previous = ROOT / "experiments/local_expert_v20", ROOT / "runs/le19_opportunity_001"
    source = ROOT / "runs/le18_completion_001/probe_plan.json"
    old, prior = read_json(source), read_json(previous / "probe_plan.json")
    seeds = inventory()
    collection, evaluation = new_pairs(old, seeds["excluded_seeds"])
    refs, endpoints = [], rows(previous / "references.jsonl") + rows(previous / "endpoints.jsonl")
    queries = {q["pair_id"]: q for q in rows(previous / "queries.jsonl")}
    for pair in prior["pairs"]:
        if pair["replicate"] == 0:
            q = queries[pair["pair_id"]]
            refs.append({"pair": pair, "saved_query": str(previous / q["saved_query"]),
                         "images": {m: next(r for r in endpoints if r["pair_id"] == pair["pair_id"] and r["mode"] == m)
                                    for m in ("noop", "geneval@1")}})
    assert len(refs) == 6
    run = prepare(Experiment.load(previous / "config.json"), ROOT / f"runs/le20_collection_{args.attempt}", read_json(previous / "site.json"))
    plan = {"version": VERSION, "source_plan": str(source), "student": prior["checkpoints"]["initial"],
            "collection": collection, "evaluation": evaluation, "references": refs, "coverage_gate": GATE,
            "required_gpu": prior["required_gpu"], "required_torch": prior["required_torch"],
            "expected_counts": {"forward_student": 540, "forward_behavior": 330, "forward_geneval": 54,
                "forward_ocr": 0, "forward_examples": 1794, "decode_images": 108, "verifier_images": 108,
                "backward_queries": 0, "text_encode_prompts": 12},
            "scope": "48 fresh development collection states;32 independent evaluation seeds reserved and not generated. Single GenEval, fixed query index3 and direct alpha1 action; provisional assistant certification."}
    write_json(run / "probe_plan.json", plan)
    write_json(run / "seed_inventory.json", seeds)
    paths = {Path(p) for p in seeds["files"]}
    paths.update([source, Path(plan["student"]), Path(__file__).resolve(), ROOT / "scripts/review_acceptance_collection.py"])
    paths.update(ROOT / "experiments/local_expert_v19" / n for n in ("NEXT.md", "report_index.json", "verification.json"))
    for ref in refs:
        paths.add(Path(ref["saved_query"]))
        paths.update(previous / r["image"] for r in ref["images"].values())
    for n in ("PROTOCOL.md", "REVIEW_RUBRIC.md"):
        shutil.copyfile(campaign / n, run / n)
        paths.add(run / n)
    paths.update([run / "probe_plan.json", run / "seed_inventory.json"])
    write_json(run / "probe_attestation.json", {"inputs": {str(p): sha256(p) for p in sorted(paths)}})
    write_json(campaign / f"attempt_{args.attempt}.json", {"run": str(run), "plan_sha256": sha256(run / "probe_plan.json"),
               "collection_states": 48, "reserved_evaluation_states": 32, "new_images": 96, "reference_images": 12,
               "excluded_seed_count": len(seeds["excluded_seeds"]), "coverage_gate": GATE})
    print({"run": str(run), "collection": 48, "reserved_evaluation": 32, "counts": plan["expected_counts"]})


if __name__ == "__main__":
    main()
