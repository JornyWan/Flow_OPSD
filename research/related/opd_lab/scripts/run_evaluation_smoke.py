"""Evaluate completed smoke checkpoints in one frozen context; no retraining."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from opdlab.evaluate import compare
from opdlab.io import read_json, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--context-run", required=True)
    p.add_argument("--round-robin-checkpoint", required=True)
    p.add_argument("--verified-checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", nargs="+", type=int, default=[10])
    args = p.parse_args()
    context, out = Path(args.context_run).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    site = read_json(context / "site.json")
    env = dict(os.environ, PYTHONPATH=str(context / "framework/python"))
    checkpoints = {"round_robin": Path(args.round_robin_checkpoint).resolve(),
                   "verified": Path(args.verified_checkpoint).resolve()}
    results = {"scope": "Engineering validation subset; no efficacy claim", "context_run": str(context), "comparisons": {}}
    for sampler in ("flow", "dpm2"):
        for steps in args.steps:
            baseline = out / f"base_{sampler}{steps}"
            variants = [("base", None, "base")] + [(name, path / f"{kind}_lora", kind)
                        for name, path in checkpoints.items() for kind in ("raw", "ema")]
            for name, adapter, kind in variants:
                target = baseline if adapter is None else out / f"{name}_{kind}_{sampler}{steps}"
                command = [site["python"], "-m", "opdlab", "eval", "--run", str(context), "--out", str(target),
                           "--sampler", sampler, "--steps", str(steps), "--seeds", "42", "--split", "validation", "--save-images"]
                if adapter:
                    command += ["--adapter", str(adapter), "--adapter-kind", kind]
                print(json.dumps({"event": "evaluate", "command": command}), flush=True)
                subprocess.run(command, env=env, check=True)
                if adapter:
                    results["comparisons"][target.name] = compare(baseline, target)
                write_json(out / "progress.json", results)
    results["status"] = "evaluation_smoke_passed"
    write_json(out / "result.json", results)
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
