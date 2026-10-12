"""Freeze the phase7 selected positives and both local diagnostic workers."""
import argparse
import json
import shutil

from opdlab.config import Experiment
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.learnability import select_positives
from opdlab.runtime import prepare


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--site", required=True)
    p.add_argument("--attempt", default="001")
    args = p.parse_args()
    campaign = ROOT / "experiments/local_expert_v7"
    old, review = ROOT / "runs/le3_probe_001", ROOT / "runs/le4_target_review_001"
    selected = select_positives(read_json(review / "key.json"), read_json(review / "observations.json"))
    if [s["case"] for s in selected] != [9, 11, 12, 17, 19]:
        raise ValueError("Selection differs from the preregistered list")
    prior = [old / "frozen_queries.pt", old / "frozen_behavior.pt", old / "queries.jsonl",
             review / "key.json", review / "observations.json"]
    for spec in selected:
        for endpoint in spec["old_endpoints"].values():
            path = old / endpoint["image"]
            if sha256(path) != endpoint["image_sha256"]:
                raise ValueError("Original reviewed image changed")
            prior.append(path)
    common = {"version": 1, "selected": selected, "max_updates": 128,
              "milestones": [0, 1, 4, 16, 64, 128], "frozen_queries": str(prior[0]),
              "frozen_behavior": str(prior[1]), "learning_rate": 3e-5,
              "capability_queries": [{k: row[k] for k in ("id", "seed")} for row in
                                     [json.loads(x) for x in (old / "queries.jsonl").read_text().splitlines()]]}
    descriptor = {"native_run": str(ROOT / f"runs/le7_native_ocr_{args.attempt}")}
    for stage in ("fit", "capability"):
        config = Experiment(dataset=str(ROOT / "data/joint_pilot_v1/records.jsonl"),
                            queries_per_rank=5, accumulation=5, outer_epochs=1, learning_rate=3e-5)
        run = prepare(config, ROOT / f"runs/le7_{stage}_{args.attempt}", read_json(args.site))
        write_json(run / "phase7_plan.json", {**common, "stage": stage})
        shutil.copyfile(campaign / "PROTOCOL.md", run / "PROTOCOL.md")
        inputs = prior + [run / "phase7_plan.json", run / "PROTOCOL.md"]
        write_json(run / "phase7_attestation.json", {"inputs": {str(path): sha256(path) for path in inputs}})
        descriptor[f"{stage}_run"] = str(run)
    write_json(campaign / f"attempt_{args.attempt}.json", descriptor)
    write_json(campaign / "selection.json", selected)
    print(descriptor)


if __name__ == "__main__":
    main()
