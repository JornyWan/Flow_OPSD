"""Freeze the phase-3 protocol, balanced queries, source and evaluation worker."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil

from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.fitting_probe import ARMS
from opdlab.io import ROOT, digest, read_json, sha256, stable_seed, write_json
from opdlab.runtime import prepare


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--site", required=True)
    p.add_argument("--attempt", default="001")
    args = p.parse_args()
    campaign = ROOT / "experiments/local_expert_v3"
    c = Experiment(dataset=str(ROOT / "data/joint_pilot_v1/records.jsonl"), fixed_teacher="geneval",
                   outer_epochs=1, queries_per_rank=64, accumulation=64, fitting_epochs=4,
                   learning_rate=3e-5, save_every=1, save_candidates=True)
    records = load_records(c.dataset, "train")
    families = defaultdict(list)
    for r in records:
        families[r["group"]].append(r)
    groups = sorted(families)
    for group in groups:
        families[group].sort(key=lambda r: digest(["phase3_order", c.seed, r["id"]]))
    if len(groups) != 4 or any(len(v) != 16 for v in families.values()):
        raise ValueError("Expected four balanced training families")
    ordered = [families[group][i] for i in range(16) for group in groups]
    plan = {"version": 1, "arms": {k: list(v) for k, v in ARMS.items()},
            "noise_key": "phase3_query", "time_key": "phase3_time",
            "queries": [{"id": r["id"], "seed": stable_seed(c.seed, "phase3_query", r["id"]),
                         "query_index": stable_seed(c.seed, "phase3_time", r["id"]) % 9} for r in ordered],
            "review_ids": [r["id"] for g in groups for r in sorted(families[g],
                           key=lambda r: digest(["phase3_review", r["id"]]))[:2]],
            "evaluation": {"split": "validation", "seeds": [42], "cells": [["flow", 10], ["flow", 40]]}}
    plan_path = campaign / "probe_plan.json"
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("Existing frozen plan differs")
    else:
        write_json(plan_path, plan)
    run = prepare(c, ROOT / f"runs/le3_probe_{args.attempt}", read_json(args.site))
    shutil.copyfile(plan_path, run / "probe_plan.json")
    shutil.copyfile(campaign / "PROTOCOL.md", run / "PROTOCOL.md")
    attestation = {"plan_sha256": sha256(run / "probe_plan.json"),
                   "protocol_sha256": sha256(run / "PROTOCOL.md")}
    write_json(run / "probe_attestation.json", attestation)
    evaluation = ROOT / f"runs/le3_eval_{args.attempt}"
    evaluation.mkdir(exist_ok=False)
    shutil.copyfile(ROOT / "scripts/run_fitting_probe_eval.py", evaluation / "worker.py")
    descriptor = {"training_run": str(run), "evaluation_run": str(evaluation), **attestation,
                  "evaluation_worker_sha256": sha256(evaluation / "worker.py")}
    write_json(evaluation / "inputs.json", descriptor)
    write_json(campaign / f"attempt_{args.attempt}.json", descriptor)
    print(descriptor)


if __name__ == "__main__":
    main()
