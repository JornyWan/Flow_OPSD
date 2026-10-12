"""Phase11: immutable raw checkpoints, fresh paired seeds, ordinary generation."""
import argparse
import os
from pathlib import Path
import time

import torch

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import seed_all
from .fitting_probe import tensor_digest
from .io import append_jsonl, assert_environment, digest, read_json, sha256, stable_seed, write_json
from .learnability import save_verdict
from .runtime import verify_run
from .verifiers import build_verifier


def make_pairs(selected, train, validation):
    train_ids = {r["id"] for r in train}
    if len(selected) != 5 or len({s["id"] for s in selected}) != 5 or not all(s["id"] in train_ids for s in selected):
        raise ValueError("Expected the five distinct fitted training prompts")
    if len(validation) != 32 or any(r["split"] != "validation" for r in validation):
        raise ValueError("Expected the complete32-prompt validation split")
    pairs = []
    for cohort, ids, repetitions in (("train_new_seed", [s["id"] for s in selected], 2),
                                      ("validation", [r["id"] for r in validation], 1)):
        for ident in sorted(ids):
            for rep in range(repetitions):
                pairs.append({"id": ident, "cohort": cohort, "replicate": rep,
                              "seed": stable_seed(11011, "phase11_transfer_v1", cohort, ident, rep),
                              "pair_id": digest(["phase11", cohort, ident, rep])[:24]})
    validate_pairs(pairs, selected, validation)
    return pairs


def validate_pairs(pairs, selected, validation):
    expected = {(s["id"], "train_new_seed", r) for s in selected for r in (0, 1)}
    expected |= {(r["id"], "validation", 0) for r in validation}
    actual = {(p["id"], p["cohort"], p["replicate"]) for p in pairs}
    if len(pairs) != 42 or len(actual) != len(pairs) or actual != expected:
        raise ValueError("Missing, duplicate or unexpected transfer pair")
    if len({p["pair_id"] for p in pairs}) != len(pairs) or len({p["seed"] for p in pairs}) != len(pairs):
        raise ValueError("Pair or seed collision")
    if {p["seed"] for p in pairs} & {s["seed"] for s in selected}:
        raise ValueError("A transfer seed repeats a fitting seed")


@torch.no_grad()
def generate_models(backend, verifier, records, config, plan, run):
    records = {r["id"]: r for r in records}
    noise, model_hashes = {}, {}
    initial_behavior = tensor_digest(backend.behavior_state())
    for model, spec in plan["models"].items():
        state = torch.load(spec["weights"], map_location="cpu", weights_only=True)
        backend.load_student(state, refresh_behavior=False)
        expected_hash = tensor_digest(state)
        anchor = spec["anchor"]
        image = backend.generate(records[anchor["id"]], anchor["seed"], config.steps, sampler="flow")
        verdict = save_verdict(run, backend, verifier, records[anchor["id"]], image,
                              ["anchor", model], config.utility_joint_weight)
        if verdict["image_sha256"] != anchor["image_sha256"]:
            raise RuntimeError("Checkpoint anchor did not replay exactly")
        append_jsonl(run / "anchors.jsonl", {"model": model, "id": anchor["id"], "seed": anchor["seed"], "exact": True, **verdict})
        for i, pair in enumerate(plan["pairs"]):
            checksum = tensor_digest({"noise": backend.initial(pair["seed"])})
            if pair["pair_id"] in noise and noise[pair["pair_id"]] != checksum:
                raise RuntimeError("Paired initial noise differs between models")
            noise[pair["pair_id"]] = checksum
            record = records[pair["id"]]
            image = backend.generate(record, pair["seed"], config.steps, sampler="flow")
            verdict = save_verdict(run, backend, verifier, record, image,
                                  ["transfer", model, pair["pair_id"]], config.utility_joint_weight)
            append_jsonl(run / "endpoints.jsonl", {**pair, "model": model, "student_sha256": expected_hash,
                         "noise_sha256": checksum, "constraint_ids": [c["id"] for c in record["constraints"]], **verdict})
            if (i + 1) % 10 == 0:
                print({"event": "model_progress", "model": model, "pairs": i + 1}, flush=True)
        if tensor_digest(backend.student_state()) != expected_hash or tensor_digest(backend.behavior_state()) != initial_behavior:
            raise RuntimeError("Frozen weights changed")
        model_hashes[model] = expected_hash
        print({"event": "model_complete", "model": model, "pairs": len(plan["pairs"])}, flush=True)
    return {"model_hashes": model_hashes, "noise_hashes": noise, "frozen_behavior_sha256": initial_behavior}


def run_eval(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for path, expected in read_json(run / "probe_attestation.json")["inputs"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    if any((run / name).exists() for name in ("RUNNING", "result.json", "failure.json", "endpoints.jsonl")):
        raise FileExistsError("Use a fresh numbered attempt")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    started = time.perf_counter()
    try:
        config, site, plan = Experiment.load(run / "config.json"), read_json(run / "site.json"), read_json(run / "probe_plan.json")
        train, validation = load_records(config.dataset, "train"), load_records(config.dataset, "validation")
        validate_pairs(plan["pairs"], plan["selected"], validation)
        if set(plan["models"]) != {"joint128", "sparse32", "sparse320", "dense32"}:
            raise ValueError("Unexpected checkpoint matrix")
        seed_all(config.seed)
        backend = build_backend(config, site, run / "upstream", "cuda", training=False)
        initial = torch.load(plan["models"]["joint128"]["weights"], map_location="cpu", weights_only=True)
        backend.load_student(initial, refresh_behavior=True)
        verifier = build_verifier(config, site, backend.device, train + validation)
        checks = generate_models(backend, verifier, train + validation, config, plan, run)
        elapsed = time.perf_counter() - started
        write_json(run / "result.json", {"status": "complete", "optimizer_updates": 0, "new_teacher_queries": 0,
                   "pairs": 42, "transfer_images": 168, "anchor_images": 4, "elapsed_seconds": elapsed,
                   "worker_gpu_hours": elapsed / 3600, "counts": dict(backend.counters + verifier.counters),
                   "output_hashes": {p.name: sha256(p) for p in sorted(run.glob("*.jsonl"))}, **checks})
    except Exception as e:
        write_json(run / "failure.json", {"error": repr(e), "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    run_eval(parser.parse_args().run)
