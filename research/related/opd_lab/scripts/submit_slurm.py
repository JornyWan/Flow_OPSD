"""Submit one frozen experiment attempt and append its exact command to the ledger."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def record(path, value):
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), **value}) + "\n")
        f.flush()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--account", default="minilab")
    p.add_argument("--partition", default="minilab-gpu")
    p.add_argument("--gres", default="gpu:1")
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--memory", default="96G")
    p.add_argument("--time", default="01:00:00")
    p.add_argument("--dependency", help="Explicit Slurm dependency, e.g. afterok:12345")
    p.add_argument("--site", default=str(ROOT / "site.local.json"))
    p.add_argument("--ledger", default=str(ROOT / "experiments/local_expert_v1/ledger.jsonl"))
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        p.error("Supply the executable and arguments after --")
    if args.nodes < 1:
        p.error("Positive node count required")
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    native_plan = run / "plan.json"
    if native_plan.exists():
        plan = json.loads(native_plan.read_text())
        if plan.get("name") in {"diffusionopd", "diffusionopsd"}:
            if args.nodes != plan.get("nodes", 1) or int(args.gres.split(":")[-1]) * args.nodes != plan["world_size"]:
                raise ValueError("Slurm node/GPU request does not match the frozen native world size")
    elif args.nodes > 1:
        raise ValueError("Multi-node submission is currently implemented for native plans only")
    site = json.loads(Path(args.site).read_text())
    # The CLI loads the run's own source snapshot before starting its worker.
    pythonpath = run / "framework/python"
    if not pythonpath.exists():
        pythonpath = ROOT / "python"
    lines = ["#!/bin/bash", "set -euo pipefail", f"cd {shlex.quote(str(ROOT))}",
             'export TMPDIR=$(mktemp -d "/tmp/opdlab.${SLURM_JOB_ID}.XXXXXX")',
             'export TRITON_CACHE_DIR="$TMPDIR/triton"',
             "trap 'rm -rf -- \"$TMPDIR\"' EXIT",
             "export PYTHONNOUSERSITE=1", "export PYTHONUNBUFFERED=1",
             f"export OMP_NUM_THREADS={min(args.cpus, 4)}", f"export OPENBLAS_NUM_THREADS={min(args.cpus, 4)}",
             f"export PYTHONPATH={shlex.quote(str(pythonpath))}"]
    for key, value in site.get("environment", {}).items():
        if key not in {"HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "HF_HUB_OFFLINE", "CUDA_HOME"}:
            raise ValueError("Unsupported environment binding")
        lines.append(f"export {key}={shlex.quote(value)}")
    dispatched = ["srun", "--kill-on-bad-exit=1", *command] if args.nodes > 1 else command
    lines.append(shlex.join(dispatched))
    script = run / "job.sh"
    with script.open("x") as f:
        f.write("\n".join(lines) + "\n")
    submission = ["sbatch", "--parsable", f"--account={args.account}",
                  f"--partition={args.partition}", f"--gres={args.gres}", f"--nodes={args.nodes}",
                  f"--ntasks={args.nodes}", "--ntasks-per-node=1",
                  f"--cpus-per-task={args.cpus}", f"--mem={args.memory}", f"--time={args.time}",
                  f"--job-name=opdlab_{args.stage}", f"--output={run}/slurm-%j.log", str(script)]
    entry = {"stage": args.stage, "run": str(run), "command": submission,
             "worker_command": command, "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    if args.dependency:
        submission.insert(2, f"--dependency={args.dependency}")
    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    record(ledger, {"event": "submission_intent", **entry})
    result = subprocess.run(submission, text=True, capture_output=True)
    if result.returncode:
        record(ledger, {"event": "submission_failed", **entry, "error": result.stderr})
        raise RuntimeError(result.stderr)
    entry.update(job_id=result.stdout.strip().split(";")[0])
    (run / "submission.json").write_text(json.dumps(entry, indent=2) + "\n")
    record(ledger, {"event": "submitted", **entry})
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
