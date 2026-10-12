"""Two-epoch engineering checks, with real resume and raw/EMA evaluation. No efficacy claim."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from opdlab.evaluate import compare
from opdlab.io import read_json, write_json


def invoke(run, *args):
    run = Path(run).resolve()
    site = read_json(run / "site.json")
    env = dict(os.environ, PYTHONPATH=str(run / "framework/python"))
    command = [site["python"], "-m", "opdlab", *args]
    print(json.dumps({"event": "smoke_command", "command": command}), flush=True)
    subprocess.run(command, env=env, check=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--control-run", required=True)
    p.add_argument("--verified-run", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    runs = {"round_robin": Path(args.control_run).resolve(), "verified": Path(args.verified_run).resolve()}
    evidence = {"scope": "engineering_smoke_only; two epochs; validation subset; no statistical efficacy claim", "training": {}, "comparisons": {}}
    for name, run in runs.items():
        invoke(run, "run", "--run", str(run), "--stop-after", "1")
        first = run / "checkpoints/epoch-000001"
        invoke(run, "run", "--run", str(run), "--resume", str(first))
        checkpoint = run / "checkpoints/epoch-000002"
        state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
        before = torch.load(first / "training.pt", map_location="cpu", weights_only=False)
        maximum = max(float((state["student"][k] - before["student"][k]).abs().max()) for k in state["student"])
        if name == "round_robin" and (state["optimizer_step"] != 2 or maximum == 0):
            raise RuntimeError("Control did not execute both real optimizer updates")
        evidence["training"][name] = {"result": read_json(run / "result.json"),
                                      "epoch1_to_epoch2_max_parameter_change": maximum,
                                      "optimizer_state_entries": len(state["optimizer"]["state"]),
                                      "checkpoint": str(checkpoint)}
        del state, before
    # Common same-hardware protocol; both raw and EMA exports must load, including DPM2.
    for sampler in ("flow", "dpm2"):
        base = out / f"base_{sampler}10"
        reference = runs["round_robin"]
        invoke(reference, "eval", "--run", str(reference), "--out", str(base), "--sampler", sampler,
               "--steps", "10", "--seeds", "42", "--split", "validation", "--save-images")
        for name, run in runs.items():
            for kind in ("raw", "ema"):
                target = out / f"{name}_{kind}_{sampler}10"
                adapter = run / f"checkpoints/epoch-000002/{kind}_lora"
                invoke(run, "eval", "--run", str(run), "--out", str(target), "--adapter", str(adapter),
                       "--adapter-kind", kind, "--sampler", sampler, "--steps", "10", "--seeds", "42",
                       "--split", "validation", "--save-images")
                evidence["comparisons"][target.name] = compare(base, target)
                write_json(out / "progress.json", evidence)
    evidence["status"] = "engineering_smoke_passed"
    write_json(out / "result.json", evidence)
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
