"""Prepare and submit a fresh attempt of the frozen three-cell validation matrix."""
import argparse
from pathlib import Path
import shutil
import subprocess

from opdlab.io import ROOT, read_json, sha256, write_json
from submit_slurm import record


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--attempt", required=True, type=int)
    p.add_argument("--defer-ocr", action="store_true")
    args = p.parse_args()
    campaign, context = Path(args.campaign).resolve(), Path(args.context).resolve()
    site = read_json(context / "site.json")
    plan = read_json(campaign / "runs.json")
    if sha256(campaign / "PROTOCOL.md") != plan["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    active = campaign / "evaluation_jobs.json"
    if active.exists():
        archived = campaign / f"evaluation_jobs_before_attempt_{args.attempt:03d}.json"
        with archived.open("x") as f:
            f.write(active.read_text())
    jobs = []
    for sampler, steps in plan["evaluation"]["cells"]:
        job = ROOT / f"runs/le2_eval_{sampler}{steps}_job_{args.attempt:03d}"
        job.mkdir(exist_ok=False)
        shutil.copytree(ROOT / "python/opdlab", job / "framework/python/opdlab",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(ROOT / "scripts/run_controlled_evaluation.py", job / "run_controlled_evaluation.py")
        inputs = job / "input_copies"
        inputs.mkdir()
        attestation = {}
        for filename in ("runs.json", "ocr_audit_config.json", "PROTOCOL.md"):
            shutil.copyfile(campaign / filename, inputs / filename)
            attestation[filename] = sha256(inputs / filename)
        attestation["worker_script_sha256"] = sha256(job / "run_controlled_evaluation.py")
        write_json(job / "input_attestation.json", attestation)
        worker = [site["python"], str(job / "run_controlled_evaluation.py"), "--context", str(context),
                  "--runs", str(inputs / "runs.json"), "--ocr-config", str(inputs / "ocr_audit_config.json"),
                  "--sampler", sampler, "--steps", str(steps), "--out", str(job / "evaluations")]
        if args.defer_ocr:
            worker.append("--defer-ocr")
            write_json(job / "cpu_finalization_command.json", [*worker[:-1], "--score-existing"])
        command = [site["python"], str(ROOT / "scripts/submit_slurm.py"), "--run", str(job),
                   "--stage", f"eval_{sampler}{steps}_a{args.attempt}", "--gres", "gpu:l40:1",
                   "--time", "00:40:00", "--site", str(context / "site.json"),
                   "--ledger", str(campaign / "ledger.jsonl"), "--", *worker]
        subprocess.run(command, check=True, capture_output=True, text=True)
        submission = read_json(job / "submission.json")
        jobs.append({"sampler": sampler, "steps": steps, "job": str(job), "job_id": submission["job_id"]})
        print(submission["job_id"], job.name, flush=True)
    write_json(active, {"operational_note": "Fresh attempt after pre-generation audit-reader startup failures. "
                       "Same frozen samples/models/metrics; optional CPU OCR finalization runs on the calibrated host. "
                       "Failed attempts remain in the ledger and archived job registry.", "jobs": jobs})
    record(campaign / "ledger.jsonl", {"event": "evaluation_attempt_prepared", "attempt": args.attempt,
           "defer_ocr": args.defer_ocr, "jobs": jobs, "scientific_protocol_unchanged": True})


if __name__ == "__main__":
    main()
