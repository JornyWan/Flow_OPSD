"""Run all predefined phase-3 ordinary validation cells from a frozen context."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--inputs", required=True)
    args = p.parse_args()
    inputs = json.loads(Path(args.inputs).read_text())
    context, output = Path(inputs["training_run"]), Path(inputs["evaluation_run"])
    sys.path.insert(0, str(context / "framework/python"))
    from opdlab.evaluate import compare, read_evaluation
    from opdlab.fitting_probe import ARMS
    from opdlab.io import read_json, sha256, write_json
    from opdlab.runtime import launch_eval
    if sha256(__file__) != inputs["evaluation_worker_sha256"]:
        raise ValueError("Frozen evaluation worker changed")
    if sha256(context / "probe_plan.json") != inputs["plan_sha256"]:
        raise ValueError("Probe plan changed")
    if read_json(context / "result.json")["status"] != "complete":
        raise ValueError("Probe training incomplete")
    if (output / "result.json").exists():
        raise FileExistsError("Evaluation already complete")
    plan = read_json(context / "probe_plan.json")["evaluation"]
    results = {"status": "running", "plan_sha256": inputs["plan_sha256"], "cells": {}}
    for sampler, steps in plan["cells"]:
        cell = output / f"{sampler}{steps}"
        cell.mkdir(exist_ok=False)
        for name in ("base", *ARMS):
            adapter = None if name == "base" else context / "checkpoints" / name / "raw_lora"
            if adapter and not read_json(adapter.parent / "metadata.json")["complete"]:
                raise ValueError("Incomplete checkpoint")
            print({"event": "evaluate", "arm": name, "sampler": sampler, "steps": steps}, flush=True)
            launch_eval(context, cell / name, adapter, "raw" if adapter else "base", sampler,
                        steps, plan["seeds"], plan["split"], save_images=True)
            read_evaluation(cell / name)
        contrasts = [(name, "base") for name in ARMS] + [("fit2_keep", "fit2"),
                    ("fit4_keep", "fit4"), ("fit4", "fit2"), ("fit4_keep", "fit2_keep")]
        results["cells"][f"{sampler}{steps}"] = {f"{a}_minus_{b}": compare(cell / b, cell / a)
                                                    for a, b in contrasts}
        write_json(output / "progress.json", results)
    results["status"] = "complete"
    write_json(output / "result.json", results)


if __name__ == "__main__":
    main()
