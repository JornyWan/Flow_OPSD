"""Frozen counterfactual queries with immutable identities and complete image evidence."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .backends import build_backend
from .data import load_records
from .engine import barrier, distributed_device, seed_all
from .io import digest, read_json, sha256, stable_seed, write_json
from .method import collect_query, query_log
from .verifiers import build_verifier


def make_manifest(dataset, split, seeds, indices, steps, output, ids=None):
    records = load_records(dataset, split)
    if ids:
        if len(set(ids)) != len(ids) or set(ids) - {r["id"] for r in records}:
            raise ValueError("Unknown or duplicated requested record ids")
        records = [r for r in records if r["id"] in ids]
    if not seeds or len(set(seeds)) != len(seeds) or not indices or len(set(indices)) != len(indices):
        raise ValueError("Use nonempty distinct seeds and time indices")
    if steps < 1 or any(type(i) is not int or not 0 <= i < steps for i in indices):
        raise ValueError("Query indices must be on the requested grid")
    rows = []
    for record in records:
        for replicate in seeds:
            noise_seed = stable_seed(replicate, "diagnostic", record["id"])
            for index in indices:
                query = {"id": record["id"], "replicate": replicate, "seed": noise_seed,
                         "query_index": index, "ordinal": len(rows)}
                rows.append({**query, "query_id": digest(query)[:24]})
    result = {"format": 1, "dataset_sha256": sha256(dataset), "split": split, "steps": steps,
              "seed_rule": "sha256(replicate,diagnostic,record_id); independent of intervention time",
              "record_ids": [r["id"] for r in records], "seeds": list(seeds), "indices": list(indices),
              "queries": rows}
    if Path(output).exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result


def validate_manifest(manifest, config, records):
    if manifest["format"] != 1 or manifest["steps"] != config.steps or manifest["split"] != config.split:
        raise ValueError("Diagnostic grid/split mismatch")
    if manifest["dataset_sha256"] != sha256(config.dataset):
        raise ValueError("Diagnostic manifest was made for different data bytes")
    record_ids = {r["id"] for r in records}
    keys, ids = set(), set()
    for row in manifest["queries"]:
        key = (row["id"], row["replicate"], row["query_index"])
        if key in keys or row["query_id"] in ids or row["id"] not in record_ids:
            raise ValueError("Duplicate or unknown diagnostic query")
        if not 0 <= row["query_index"] < config.steps:
            raise ValueError("Diagnostic query time outside schedule")
        if row["seed"] != stable_seed(row["replicate"], "diagnostic", row["id"]):
            raise ValueError("Diagnostic noise seed was changed")
        if row["ordinal"] != len(keys):
            raise ValueError("Query ordinals must be contiguous and ordered")
        unsigned = {k: row[k] for k in ("id", "replicate", "seed", "query_index", "ordinal")}
        if row["query_id"] != digest(unsigned)[:24]:
            raise ValueError("Query identity hash mismatch")
        keys.add(key)
        ids.add(row["query_id"])
    expected = {(r, s, i) for r in manifest["record_ids"] for s in manifest["seeds"] for i in manifest["indices"]}
    if not keys or keys != expected:
        raise ValueError("Diagnostic query grid is incomplete")


def image_writer(run, query_id):
    directory = Path(run) / "query_results" / query_id
    directory.mkdir(parents=True, exist_ok=True)
    def save(name, image):
        stem = digest([query_id, name])[:20]
        extension = ".pt" if isinstance(image, torch.Tensor) else ".png"
        path = directory / (stem + extension)
        if isinstance(image, torch.Tensor):
            torch.save(image, path)
            # torch.save archives include a filename; hash pixels separately for exact noop checks.
            content_hash = __import__("hashlib").sha256(image.numpy().tobytes()).hexdigest()
        else:
            image.save(path)
            content_hash = sha256(path)
        return str(path.relative_to(run)), content_hash
    return save


def validate_saved(row, run, signature, expected):
    if row["signature"] != signature or row["query"] != expected:
        raise ValueError("Existing diagnostic query belongs to different experiment inputs")
    for candidate in row["candidates"]:
        path = Path(run) / candidate["image"]
        if path.suffix == ".png":
            actual = sha256(path)
        else:
            image = torch.load(path, weights_only=True)
            actual = __import__("hashlib").sha256(image.numpy().tobytes()).hexdigest()
        if actual != candidate["image_sha256"]:
            raise ValueError(f"Candidate evidence changed: {path}")


def run_diagnostic(config, site, runtime, run):
    run = Path(run)
    records = load_records(config.dataset, config.split)
    manifest = read_json(config.diagnostic_manifest)
    validate_manifest(manifest, config, records)
    record_map = {r["id"]: r for r in records}
    rank, world, device = distributed_device(config.backend)
    seed_all(config.seed)
    signature = digest({"manifest": read_json(run / "manifest.json"), "queries": manifest})
    started = time.perf_counter()
    backend = build_backend(config, site, runtime, device)
    verifier = build_verifier(config, site, device, records)
    backend.refresh_behavior()
    for spec in manifest["queries"][rank::world]:
        path = run / "query_results" / spec["query_id"] / "result.json"
        if path.exists():
            validate_saved(read_json(path), run, signature, spec)
            print(json.dumps({"event": "resume_skip_complete", "query_id": spec["query_id"]}), flush=True)
            continue
        start = time.perf_counter()
        before = backend.counters + verifier.counters
        query = collect_query(backend, verifier, record_map[spec["id"]], spec["seed"], spec["query_index"],
                              config, query_ordinal=spec["ordinal"], save_image=image_writer(run, spec["query_id"]))
        if device.type == "cuda":
            torch.cuda.synchronize()
        row = {"outer_epoch": 0, "query": spec, "signature": signature, "rank": rank,
               "sigma": float(backend.sigmas[spec["query_index"]]),
               "sigma_next": float(backend.sigmas[spec["query_index"] + 1]),
               "elapsed_seconds": time.perf_counter() - start,
               "counts": dict(backend.counters + verifier.counters - before), **query_log(query, config)}
        write_json(path, row)  # One atomic completion marker per query, after every candidate is saved.
        print(json.dumps({"event": "query_complete", "ordinal": spec["ordinal"], "total": len(manifest["queries"]),
                          "seconds": row["elapsed_seconds"], "accepted": row["accepted"]}), flush=True)
    barrier()
    if rank == 0:
        rows, noops, counts = [], {}, Counter()
        for spec in manifest["queries"]:
            row = read_json(run / "query_results" / spec["query_id"] / "result.json")
            validate_saved(row, run, signature, spec)
            key = (spec["id"], spec["replicate"])
            noop = row["candidates"][0]
            if key in noops and noop["image_sha256"] != noops[key]:
                raise ValueError(f"Noop endpoint changed across query positions: {key}")
            noops[key] = noop["image_sha256"]
            counts.update(row["counts"])
            rows.append(row)
        # Compatible with analyze_queries; rebuilt only from complete, verified per-query records.
        (run / "queries-rank-0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        invalid = sum(not c["valid"] for r in rows for c in r["candidates"])
        report = {"status": "diagnosed" if invalid == 0 else "diagnostic_invalid_verifier",
                  "queries": len(rows), "candidates": sum(len(r["candidates"]) for r in rows),
                  "unique_noise_trajectories": len(noops), "noop_identical_across_positions": True,
                  "invalid_candidates": invalid, "counts": dict(counts),
                  "worker_elapsed_seconds_this_attempt": time.perf_counter() - started,
                  "optimizer_updates": 0, "signature": signature, "query_rows_digest": digest(rows)}
        if device.type == "cuda":
            report["hardware"] = {"gpu": torch.cuda.get_device_name(device),
                                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        write_json(run / "result.json", report)
        print(json.dumps(report), flush=True)
    barrier()
    if rank == 0 and invalid:
        raise RuntimeError(f"{invalid} invalid verifier results; diagnostic evidence is incomplete")
