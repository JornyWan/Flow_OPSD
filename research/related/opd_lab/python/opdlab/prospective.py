"""Phase12: frozen prospective teacher candidates and unchanged automatic selection."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import time

import torch

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .diagnostic import image_writer
from .engine import atomic_torch_save, seed_all
from .fitting_probe import cpu_tree, tensor_digest
from .io import append_jsonl, assert_environment, digest, read_json, sha256, stable_seed, write_json
from .learnability import save_verdict
from .method import collect_query, query_log
from .runtime import verify_run
from .verifiers import build_verifier


def make_queries(records, excluded_seeds):
    groups = defaultdict(list)
    for record in sorted(records, key=lambda r: r["id"]):
        if record["split"] != "train":
            raise ValueError("Prospective collection is train-only")
        groups[record["group"]].append(record)
    if len(records) != 64 or len({r["id"] for r in records}) != 64 or sorted(map(len, groups.values())) != [16] * 4:
        raise ValueError("Expected all64 distinct train prompts in four balanced groups")
    queries = []
    for i in range(16):
        for group in sorted(groups):
            record = groups[group][i]
            row = {"id": record["id"], "group": group, "ordinal": len(queries), "index": 3,
                   "seed": stable_seed(12012, "phase12_prospective_v1", record["id"])}
            queries.append({**row, "query_id": digest(row)[:24]})
    seeds = {q["seed"] for q in queries}
    if len(seeds) != 64 or seeds & set(excluded_seeds):
        raise ValueError("Fresh seed collision with this cohort or prior fitting/transfer")
    return queries


@torch.no_grad()
def collect(backend, verifier, records, config, specs, run):
    records = {r["id"]: r for r in records}
    (run / "queries").mkdir(exist_ok=False)
    initial = tensor_digest(backend.student_state())
    if tensor_digest(backend.behavior_state()) != initial:
        raise ValueError("Behavior must start at the same frozen student")
    for spec in specs:
        start, before = time.perf_counter(), backend.counters + verifier.counters
        query = collect_query(backend, verifier, records[spec["id"]], spec["seed"], spec["index"], config,
                              query_ordinal=spec["ordinal"], save_image=image_writer(run, spec["query_id"]))
        if any(not c.verdict.valid for c in query["candidates"]):
            raise RuntimeError("Invalid verifier result; no selection audit from invalid candidates")
        saved = cpu_tree({k: query[k] for k in ("record", "seed", "index", "state", "context", "dt", "target", "accepted", "winner")})
        saved["actions"] = {c.name: cpu_tree(c.action) for c in query["candidates"]}
        path = run / "queries" / f"{spec['query_id']}.pt"
        atomic_torch_save(saved, path)
        row = {"query": spec, "sigma": float(backend.sigmas[spec["index"]]),
               "constraint_ids": [c["id"] for c in records[spec["id"]]["constraints"]],
               "noise_sha256": tensor_digest({"noise": backend.initial(spec["seed"])}),
               "state_sha256": tensor_digest({"state": saved["state"]}),
               "target_sha256": tensor_digest({"action": saved["target"]}),
               "action_hashes": {k: tensor_digest({"action": a}) for k, a in saved["actions"].items()},
               "saved_query": str(path.relative_to(run)), "saved_query_sha256": sha256(path),
               "elapsed_seconds": time.perf_counter() - start,
               "counts": dict(backend.counters + verifier.counters - before), **query_log(query, config)}
        if row["target_sha256"] != row["action_hashes"][row["selected"]]:
            raise RuntimeError("Saved target differs from selected action")
        append_jsonl(run / "candidates.jsonl", row)
        # Do not expose the selected arm or score before the subsequent blind review.
        print({"event": "query_complete", "ordinal": spec["ordinal"], "total": len(specs)}, flush=True)
    if tensor_digest(backend.student_state()) != initial or tensor_digest(backend.behavior_state()) != initial:
        raise RuntimeError("Frozen student or behavior changed")


def run_collect(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for path, expected in read_json(run / "probe_attestation.json")["inputs"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    if any((run / p).exists() for p in ("RUNNING", "result.json", "failure.json", "candidates.jsonl", "queries")):
        raise FileExistsError("Use a fresh numbered attempt")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    started = time.perf_counter()
    try:
        config, site, plan = Experiment.load(run / "config.json"), read_json(run / "site.json"), read_json(run / "probe_plan.json")
        records = load_records(config.dataset, "train")
        if plan["queries"] != make_queries(records, plan["excluded_seeds"]):
            raise ValueError("Prospective query plan changed")
        if config.steps != 10 or config.alphas != [1.0] or config.teachers != ["ocr", "geneval"] or config.method != "verified":
            raise ValueError("Unexpected prospective grid/selector")
        seed_all(config.seed)
        # This backend flag also loads external teacher adapters; all predictions
        # below still use no-grad and there is no optimizer in this worker.
        backend = build_backend(config, site, run / "upstream", "cuda", training=True)
        state = torch.load(plan["student"], weights_only=True, map_location="cpu")
        backend.load_student(state, refresh_behavior=True)
        verifier = build_verifier(config, site, backend.device, records)
        anchor = plan["anchor"]
        record = next(r for r in records if r["id"] == anchor["id"])
        verdict = save_verdict(run, backend, verifier, record,
                              backend.generate(record, anchor["seed"], config.steps), ["phase12_anchor"], config.utility_joint_weight)
        if verdict["image_sha256"] != anchor["image_sha256"]:
            raise RuntimeError("Initial student failed exact anchor replay")
        write_json(run / "anchor.json", {**anchor, **verdict, "exact": True})
        collect(backend, verifier, records, config, plan["queries"], run)
        elapsed = time.perf_counter() - started
        write_json(run / "result.json", {"status": "complete", "queries": 64, "candidate_images": 192,
                   "anchor_images": 1, "optimizer_updates": 0, "teacher_queries": 128,
                   "elapsed_seconds": elapsed, "worker_gpu_hours": elapsed / 3600,
                   "counts": dict(backend.counters + verifier.counters), "student_sha256": tensor_digest(state),
                   "candidates_sha256": sha256(run / "candidates.jsonl"), "anchor_sha256": sha256(run / "anchor.json")})
    except Exception as e:
        write_json(run / "failure.json", {"error": repr(e), "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    run_collect(parser.parse_args().run)
