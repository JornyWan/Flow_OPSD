"""Freeze phase8 execution masks and all phase7 references before GPU use."""
import argparse
import json
from pathlib import Path
import shutil

from opdlab.config import Experiment
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import prepare
from opdlab.trajectory_probe import MODES


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--site", required=True)
    p.add_argument("--attempt", default="001")
    args = p.parse_args()
    campaign, old = ROOT / "experiments/local_expert_v8", ROOT / "runs/le7_fit_001"
    prior = read_json(old / "phase7_plan.json")
    endpoints = [json.loads(x) for x in (old / "endpoints.jsonl").read_text().splitlines()]
    endpoints = [r for r in endpoints if r["arm"] == "joint" and r["optimizer_step"] == 128]
    references = [json.loads(x) for x in (old / "references.jsonl").read_text().splitlines()]
    inputs = [old / "phase7_plan.json", old / "result.json", old / "endpoints.jsonl", old / "references.jsonl",
              old / "checkpoints/joint/step128.pt", ROOT / "runs/le7_review_001/key.json",
              ROOT / "runs/le7_review_001/observations.json", ROOT / "runs/le7_review_001/observations_frozen.json",
              *map(Path, [prior["frozen_queries"], prior["frozen_behavior"]])]
    queries = []
    for spec in prior["selected"]:
        rows = [r for r in endpoints if r["id"] == spec["id"]]
        noop = next(r for r in references if r["id"] == spec["id"] and r["kind"] == "noop")
        hashes = {"base": noop["image_sha256"], **{("local" if r["mode"] == "single" else "full"): r["image_sha256"] for r in rows}}
        for row in [noop, *rows]:
            image = old / row["image"]
            if sha256(image) != row["image_sha256"]:
                raise ValueError("Old reference changed")
            inputs.append(image)
        queries.append({"id": spec["id"], "case": spec["case"], "seed": rows[0]["seed"],
                        "index": rows[0]["query_index"], "reference_images": hashes})
    config = Experiment.load(old / "config.json")
    run = prepare(config, ROOT / f"runs/le8_trajectory_{args.attempt}", read_json(args.site))
    write_json(run / "probe_plan.json", {"version": 1, "modes": list(MODES), "queries": queries,
               "student": str(old / "checkpoints/joint/step128.pt"),
               "frozen_queries": prior["frozen_queries"], "frozen_behavior": prior["frozen_behavior"]})
    shutil.copyfile(campaign / "PROTOCOL.md", run / "PROTOCOL.md")
    inputs += [run / "probe_plan.json", run / "PROTOCOL.md", campaign / "REVIEW_RUBRIC.md"]
    write_json(run / "probe_attestation.json", {"inputs": {str(p): sha256(p) for p in inputs}})
    write_json(campaign / f"attempt_{args.attempt}.json", {"run": str(run), "source_phase7": str(old),
               "plan_sha256": sha256(run / "probe_plan.json"), "protocol_sha256": sha256(run / "PROTOCOL.md")})
    print(run)


if __name__ == "__main__":
    main()
