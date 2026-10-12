"""Freeze phase-4 action interventions and all referenced phase-3 artifacts."""
import argparse
import json
from pathlib import Path
import shutil

from opdlab.action_probe import BETAS
from opdlab.config import Experiment
from opdlab.fitting_probe import ARMS
from opdlab.io import ROOT, digest, read_json, sha256, write_json
from opdlab.runtime import prepare, verify_run


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--parent", required=True)
    p.add_argument("--site", required=True)
    p.add_argument("--attempt", default="001")
    args = p.parse_args()
    parent = Path(args.parent).resolve()
    verify_run(parent)
    result = read_json(parent / "result.json")
    if result["status"] != "complete" or result["accepted"] != 22:
        raise ValueError("Expected the completed phase-3 source with 22 accepted queries")
    rows = [r for r in map(json.loads, (parent / "queries.jsonl").read_text().splitlines()) if r["accepted"]]
    names = ["frozen_queries.pt", "frozen_behavior.pt", "config.json", "site.json", "data.jsonl", "manifest.json",
             "queries.jsonl", "postfit.jsonl", "metrics.jsonl", "result.json"]
    for arm in ARMS:
        names.extend([f"checkpoints/{arm}/metadata.json", f"checkpoints/{arm}/raw_lora/adapter_model.safetensors",
                      f"checkpoints/{arm}/raw_lora/adapter_config.json"])
    for row in rows:
        names.extend(c["image"] for c in row["candidates"] if c["name"] in ("noop", row["selected"]))
    post = list(map(json.loads, (parent / "postfit.jsonl").read_text().splitlines()))
    names.extend(r["single"]["image"] for r in post if r["accepted"])
    plan = {"version": 1, "parent_run": str(parent), "parent_files": {n: sha256(parent / n) for n in names},
            "query_ids": [r["id"] for r in rows], "arms": list(ARMS), "betas": list(BETAS),
            "review_ids": sorted([r["id"] for r in rows], key=lambda i: digest(["phase4-review-case", i])),
            "expected_images": 286, "optimizer_updates": 0, "split": "train"}
    campaign = ROOT / "experiments/local_expert_v4"
    plan_path = campaign / "action_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Frozen campaign plan changed")
    if not plan_path.exists():
        write_json(plan_path, plan)
    config = Experiment.load(parent / "config.json")
    run = prepare(config, ROOT / f"runs/le4_action_{args.attempt}", read_json(args.site))
    shutil.copyfile(plan_path, run / "action_plan.json")
    shutil.copyfile(campaign / "PROTOCOL.md", run / "PROTOCOL.md")
    attestation = {"plan_sha256": sha256(run / "action_plan.json"), "protocol_sha256": sha256(run / "PROTOCOL.md")}
    write_json(run / "action_attestation.json", attestation)
    write_json(campaign / f"attempt_{args.attempt}.json", {"run": str(run), **attestation})
    print({"run": str(run), **attestation})


if __name__ == "__main__":
    main()
