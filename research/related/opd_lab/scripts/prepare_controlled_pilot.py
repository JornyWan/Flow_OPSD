"""Freeze phase-2's predeclared four-arm on-policy screening experiment."""
import argparse
from pathlib import Path

from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.runtime import prepare
from opdlab.training_queries import make_training_manifest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--site", required=True)
    args = p.parse_args()
    campaign = ROOT / "experiments/local_expert_v2"
    plan = campaign / "training_queries.json"
    if plan.exists():
        raise FileExistsError("Phase-2 training design is already frozen")
    c = Experiment(dataset=str(ROOT / "data/joint_pilot_v1/records.jsonl"),
                   fixed_teacher="geneval", outer_epochs=12, queries_per_rank=16, accumulation=16,
                   learning_rate=3e-5, ema_interval=1, save_every=4, save_candidates=True)
    write_json(plan, make_training_manifest(c, load_records(c.dataset, "train")))
    c.training_manifest = str(plan)
    runs = {}
    for name, method, protect in [("fixed_geneval", "fixed", True),
                                  ("verified_fixed", "verified_fixed", True),
                                  ("verified_multi", "verified", True),
                                  ("utility_only", "verified", False)]:
        c.method, c.protect_satisfied = method, protect
        config_path = campaign / f"{name}.json"
        write_json(config_path, c.as_dict())
        config = Experiment.load(config_path)
        run = prepare(config, ROOT / f"runs/le2_train_{name}_001", read_json(args.site))
        runs[name] = str(run)
    write_json(campaign / "runs.json", {"protocol_sha256": sha256(campaign / "PROTOCOL.md"),
               "query_manifest_sha256": sha256(plan), "training_runs": runs,
               "evaluation": {"split": "validation", "seeds": [42, 43], "checkpoint_epoch": 12,
                              "adapter_kind": "raw", "cells": [["flow", 10], ["flow", 40], ["dpm2", 40]]}})
    print(runs)


if __name__ == "__main__":
    main()
