"""Reconcile only this campaign's submitted jobs; never submit/cancel jobs or promote results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from submit_slurm import ROOT, record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--campaign", default=str(ROOT / "experiments/local_expert_v1"))
    args = parser.parse_args()
    campaign = Path(args.campaign)
    entries = [json.loads(s) for s in (campaign / "ledger.jsonl").read_text().splitlines() if s]
    submissions = {e["job_id"]: e for e in entries if e["event"] == "submitted"}
    accounted = {e["job_id"] for e in entries if e["event"] == "slurm_final"}
    command = ["sacct", "-X", "-n", "-P", "-j", ",".join(submissions),
               "--format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode,NodeList"]
    rows = []
    for line in subprocess.check_output(command, text=True).splitlines():
        job, state, elapsed, tres, exitcode, nodes = line.split("|")[:6]
        if job not in submissions:
            continue
        gpu_count = max([0] + [int(x) for x in re.findall(r"gres/gpu(?::[^,=]+)?=(\d+)", tres)])
        row = {"job_id": job, "stage": submissions[job]["stage"], "run": submissions[job]["run"],
               "state": state, "elapsed_seconds": int(elapsed), "allocated_tres": tres,
               "exit_code": exitcode, "nodes": nodes, "allocated_gpu_hours": int(elapsed) * gpu_count / 3600}
        rows.append(row)
        terminal = state.split()[0] in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL"}
        if terminal and job not in accounted:
            record(campaign / "ledger.jsonl", {"event": "slurm_final", **row,
                   "meaning": "scheduler accounting only; experiment validity requires artifact inspection"})
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "jobs": rows,
              "allocated_gpu_hours_including_failed_attempts_so_far": sum(r["allocated_gpu_hours"] for r in rows)}
    temporary = campaign / "slurm_status.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(campaign / "slurm_status.json")
    for row in rows:
        print(f"{row['job_id']:>8} {row['state']:<23} {row['stage']:<20} {row['allocated_gpu_hours']:.4f} GPU-h")


if __name__ == "__main__":
    main()
