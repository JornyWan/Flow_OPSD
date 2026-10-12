"""Freeze64 train queries,192 candidate endpoints and the existing selector."""
import argparse
from pathlib import Path
import shutil

from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.prospective import make_queries
from opdlab.runtime import prepare


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--site", required=True)
    p.add_argument("--attempt", default="001")
    args = p.parse_args()
    old, campaign = ROOT / "runs/le11_transfer_001", ROOT / "experiments/local_expert_v12"
    previous = read_json(old / "probe_plan.json")
    model = previous["models"]["joint128"]
    excluded = sorted({s["seed"] for s in previous["selected"]} | {s["seed"] for s in previous["pairs"]})
    config = Experiment.load(old / "config.json")
    config.alphas, config.queries_per_rank, config.accumulation = [1.0], 64, 1
    config.save_candidates = True
    config.validate()
    queries = make_queries(load_records(config.dataset, "train"), excluded)
    inputs = [old / n for n in ("probe_plan.json", "result.json", "probe_attestation.json")]
    inputs += [ROOT / "experiments/local_expert_v11" / n for n in ("NEXT.md", "STATUS.md", "results.json", "observations.json", "observations_frozen.json", "report_index.json")]
    inputs += [Path(model["weights"]), ROOT / "python/opdlab/method.py"]
    run = prepare(config, ROOT / f"runs/le12_collect_{args.attempt}", read_json(args.site))
    write_json(run / "probe_plan.json", {"version": 1, "student": model["weights"], "anchor": model["anchor"],
               "excluded_seeds": excluded, "queries": queries, "candidate_order": ["noop", "ocr@1", "geneval@1"],
               "selector_source_sha256": sha256(ROOT / "python/opdlab/method.py"),
               "expected_counts": {"forward_behavior": 1408, "forward_ocr": 64, "forward_geneval": 64,
                                   "forward_student": 10, "forward_examples": 3028, "decode_images": 193, "verifier_images": 193}})
    for name in ("PROTOCOL.md", "REVIEW_RUBRIC.md"):
        shutil.copyfile(campaign / name, run / name)
        inputs.append(run / name)
    inputs.append(run / "probe_plan.json")
    write_json(run / "probe_attestation.json", {"inputs": {str(p): sha256(p) for p in inputs}})
    write_json(campaign / f"attempt_{args.attempt}.json", {"run": str(run), "plan_sha256": sha256(run / "probe_plan.json"),
               "protocol_sha256": sha256(run / "PROTOCOL.md"), "candidate_images": 192, "additional_anchor_images": 1})
    print(run)


if __name__ == "__main__":
    main()
