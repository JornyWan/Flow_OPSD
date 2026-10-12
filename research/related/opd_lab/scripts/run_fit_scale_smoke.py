"""One-update learning-rate check with identical pre-update candidate evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from opdlab.io import read_json, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = {"scope": "One-update fitting check on four training prompts; not validation or efficacy tuning.",
                "arms": [], "identical_candidates": False}
    reference = None
    for item in args.runs:
        run = Path(item).resolve()
        site = read_json(run / "site.json")
        env = dict(os.environ, PYTHONPATH=str(run / "framework/python"))
        subprocess.run([site["python"], "-m", "opdlab", "run", "--run", str(run)], env=env, check=True)
        subprocess.run([site["python"], str(Path(__file__).with_name("report_training_smoke.py")),
                        "--run", str(run), "--out", str(run / "postfit_audit.json")], env=env, check=True)
        result = read_json(run / "result.json")
        if result != {"status": "complete", "outer_epoch": 1, "optimizer_step": 1}:
            raise ValueError("Expected one completed real update")
        rows = [json.loads(s) for s in (run / "queries-rank-0.jsonl").read_text().splitlines()]
        # Paths change between attempts, but all initial actions, images, scores,
        # query positions and selections must match before comparing fitting.
        signature = []
        for row in rows:
            clean = dict(row)
            clean["candidates"] = [{k: v for k, v in c.items() if k != "image"} for c in row["candidates"]]
            signature.append(clean)
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("Initial candidate/query evidence differs between learning rates")
        audit = read_json(run / "postfit_audit.json")
        evidence["arms"].append({"run": str(run), "learning_rate": read_json(run / "config.json")["learning_rate"],
                                 "fitting": audit["epochs"]["1"], "rows": audit["rows"]})
        write_json(out / "progress.json", evidence)
    evidence["identical_candidates"] = True
    evidence["status"] = "fit_scale_smoke_passed"
    write_json(out / "result.json", evidence)
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
