"""Freeze fresh teacher re-querying at phase8's current-student states."""
import argparse
import json
from pathlib import Path
import shutil

from opdlab.config import Experiment
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import prepare


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--site", required=True)
    p.add_argument("--attempt", default="001")
    args = p.parse_args()
    old, campaign = ROOT / "runs/le8_trajectory_001", ROOT / "experiments/local_expert_v9"
    previous = read_json(old / "probe_plan.json")
    baselines = {r["id"]: r for r in [json.loads(x) for x in (old / "endpoints.jsonl").read_text().splitlines()]
                 if r["mode"] == "full"}
    queries = [{**{k: q[k] for k in ("id", "case", "seed", "index")},
                "baseline_image_sha256": baselines[q["id"]]["image_sha256"]} for q in previous["queries"]]
    inputs = [old / "result.json", old / "probe_plan.json", old / "endpoints.jsonl",
              Path(previous["student"]), Path(previous["frozen_queries"]),
              ROOT / "experiments/local_expert_v8/STATUS.md", ROOT / "experiments/local_expert_v8/results.json"]
    for name in ("le7_review_001", "le8_review_001"):
        inputs += [ROOT / "runs" / name / f"{key}.json" for key in ("key", "observations", "observations_frozen")]
    inputs.append(ROOT / "runs/le8_review_001/inherited.json")
    for b in baselines.values():
        image = old / b["image"]
        if sha256(image) != b["image_sha256"]:
            raise ValueError("Changed baseline image")
        inputs.append(image)
    config = Experiment.load(old / "config.json")
    run = prepare(config, ROOT / f"runs/le9_requery_{args.attempt}", read_json(args.site))
    write_json(run / "probe_plan.json", {"version": 1, "student": previous["student"],
               "frozen_queries": previous["frozen_queries"], "queries": queries})
    shutil.copyfile(campaign / "PROTOCOL.md", run / "PROTOCOL.md")
    shutil.copyfile(ROOT / "experiments/local_expert_v8/REVIEW_RUBRIC.md", run / "REVIEW_RUBRIC.md")
    inputs += [run / "probe_plan.json", run / "PROTOCOL.md", run / "REVIEW_RUBRIC.md"]
    write_json(run / "probe_attestation.json", {"inputs": {str(p): sha256(p) for p in inputs}})
    write_json(campaign / f"attempt_{args.attempt}.json", {"run": str(run), "source_phase8": str(old),
               "plan_sha256": sha256(run / "probe_plan.json"), "protocol_sha256": sha256(run / "PROTOCOL.md")})
    print(run)


if __name__ == "__main__":
    main()
