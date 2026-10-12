"""Freeze a small label-assisted learning pilot using unchanged phase12 targets."""
import argparse
from pathlib import Path
import shutil

from opdlab.annotation_probe import VERSION, cohort, control_targets, evaluation_pairs
from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import prepare
from review_prospective import compare, lines


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--site", default="site.local.json")
    parser.add_argument("--attempt", default="001")
    args = parser.parse_args()
    campaign, source = ROOT / "experiments/local_expert_v15", ROOT / "runs/le12_collect_002"
    rows = lines(source / "candidates.jsonl")
    review = read_json(ROOT / "experiments/local_expert_v12/results.json")["queries"]
    assert sha256(source / "candidates.jsonl") == read_json(source / "result.json")["candidates_sha256"]
    for q in review:
        for name in ("ocr@1", "geneval@1"):
            assert compare(q["observations"]["noop"], q["observations"][name]) == q["comparisons"][name]
    points = cohort(rows, review)
    config = Experiment.load(source / "config.json")
    records = load_records(config.dataset, "train")
    previous = read_json(source / "probe_plan.json")
    excluded = sorted({r["seed"] for r in rows} | set(previous["excluded_seeds"]))
    pairs = evaluation_pairs(points, records, excluded)
    config.queries_per_rank = config.accumulation = len(points)
    run = prepare(config, ROOT / f"runs/le15_annotation_{args.attempt}", read_json(args.site))
    plan = {"version": VERSION, "source": str(source), "student": previous["student"], "points": points,
            "targets": control_targets(points), "updates": 128, "audits": [0, 32, 128],
            "evaluation": pairs, "excluded_seeds": excluded,
            "scope": "selected-cohort label-assisted fixed-target pilot; no automatic-verifier or native OPD claim",
            "expected_counts": {"backward_queries": 4608, "forward_behavior": 48, "forward_student": 6356,
                                "forward_examples": 12808, "decode_images": 170, "verifier_images": 170}}
    write_json(run / "probe_plan.json", plan)
    inputs = [source / n for n in ("result.json", "candidates.jsonl", "probe_plan.json", "probe_attestation.json")]
    inputs += [ROOT / "experiments/local_expert_v12" / n for n in ("results.json", "observations.json", "observations_frozen.json", "report_index.json")]
    inputs += [ROOT / "experiments/local_expert_v14" / n for n in ("report_index.json", "results.json", "array_diagnostic.json", "NEXT.md")]
    inputs.append(Path(plan["student"]))
    for p in points:
        path = source / p["saved_query"]
        assert sha256(path) == p["saved_query_sha256"]
        inputs.append(path)
        for image in p["images"].values():
            path = source / image["image"]
            assert sha256(path) == image["sha256"]
            inputs.append(path)
    for name in ("PROTOCOL.md", "REVIEW_RUBRIC.md"):
        shutil.copyfile(campaign / name, run / name)
        inputs.append(run / name)
    inputs.append(run / "probe_plan.json")
    write_json(run / "probe_attestation.json", {"inputs": {str(p): sha256(p) for p in inputs}})
    write_json(campaign / f"attempt_{args.attempt}.json", {"run": str(run), "plan_sha256": sha256(run / "probe_plan.json"),
               "protocol_sha256": sha256(run / "PROTOCOL.md"), "backward_queries": 4608, "evaluation_images": 152})
    print({"run": str(run), "points": len(points), "evaluation_states": len(pairs), "targets": plan["targets"]})


if __name__ == "__main__":
    main()
