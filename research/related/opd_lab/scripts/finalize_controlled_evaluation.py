"""Finish ready GPU evaluations using the exact calibrated OCR reader on this CPU host."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import time

from opdlab.io import read_json, sha256, write_json
from submit_slurm import record


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--refresh-directory-cache", action="store_true",
                   help="Refresh output directory mtimes for NFS clients with long negative-entry caches; file payloads unchanged")
    args = p.parse_args()
    campaign = Path(args.campaign).resolve()
    for item in read_json(campaign / "evaluation_jobs.json")["jobs"]:
        job = Path(item["job"])
        if args.refresh_directory_cache:
            os.utime(job, None)
            evaluation = job / "evaluations"
            if evaluation.exists():
                os.utime(evaluation, None)
                for arm in ("base", "fixed_geneval", "utility_only", "verified_fixed", "verified_multi"):
                    path = evaluation / arm
                    if path.exists():
                        os.utime(path, None)
        target = job / "evaluations/result.json"
        if not target.exists():
            continue
        status = read_json(target)
        if status["status"] == "complete":
            continue
        if status["status"] != "awaiting_secondary_ocr":
            raise ValueError("Unexpected evaluation stage")
        attestation = read_json(job / "input_attestation.json")
        for filename in ("runs.json", "ocr_audit_config.json", "PROTOCOL.md"):
            if sha256(job / "input_copies" / filename) != attestation[filename]:
                raise ValueError("Frozen evaluation input changed")
        if sha256(job / "run_controlled_evaluation.py") != attestation["worker_script_sha256"]:
            raise ValueError("Frozen scoring implementation changed")
        primary_files = [p for name in status["arms"] for p in (job / "evaluations" / name).glob("*.json*")]
        hashes = {str(p): sha256(p) for p in primary_files if p.name != "secondary_ocr.json"}
        saved = job / "evaluations/primary_generation_result.json"
        if not saved.exists():
            write_json(saved, status)
        command = read_json(job / "cpu_finalization_command.json")
        env = dict(os.environ, PYTHONPATH=str(job / "framework/python"), PYTHONNOUSERSITE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        started = time.perf_counter()
        record(campaign / "ledger.jsonl", {"event": "cpu_ocr_started", "job": str(job),
               "command": command, "host": socket.gethostname()})
        with (job / "cpu_ocr.log").open("a") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        metadata = {"job": str(job), "host": socket.gethostname(), "command": command,
                    "elapsed_seconds": time.perf_counter() - started, "exit_code": completed.returncode,
                    "allocated_gpu_hours": 0, "primary_artifacts_unchanged": all(sha256(p) == h for p, h in hashes.items()),
                    "primary_artifacts_sha256": hashes, "worker_script_sha256": attestation["worker_script_sha256"]}
        write_json(job / "cpu_finalization.json", metadata)
        record(campaign / "ledger.jsonl", {"event": "cpu_ocr_finished", **metadata})
        if completed.returncode or not metadata["primary_artifacts_unchanged"]:
            raise RuntimeError(f"CPU OCR finalization failed: {job}")
        print(job.name, "complete", flush=True)


if __name__ == "__main__":
    main()
