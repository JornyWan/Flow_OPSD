"""One held-out sampling protocol for all adapters, with paired prompt statistics."""
from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .backends import build_backend
from .data import load_records, records_digest
from .engine import seed_all
from .io import append_jsonl, digest, environment, read_json, sha256, stable_seed, tree_fingerprint, write_json
from .verifiers import build_verifier


def canonical_scheduler(config):
    config = copy.deepcopy(dict(config))
    # Diffusers serializes this set in hash-randomized order across processes.
    # Keep every effective scheduler parameter and normalize only this set's order.
    if "_use_default_values" in config:
        config["_use_default_values"] = sorted(config["_use_default_values"])
    return config


def adapter_identity(adapter, kind):
    if not adapter:
        if kind != "base":
            raise ValueError("No adapter requires adapter_kind=base")
        return {"kind": "base", "adapter": None}
    if kind not in {"raw", "ema", "upstream_unknown"}:
        raise ValueError("Explicitly label the adapter raw, ema, or upstream_unknown")
    adapter = Path(adapter).resolve()
    metadata = adapter.parent / "metadata.json"
    if not metadata.exists():
        metadata = adapter.parent / "opdlab_checkpoint.json"
    if metadata.exists():
        declared = read_json(metadata).get(adapter.name)
        if not declared and kind != "upstream_unknown":
            raise ValueError("Metadata does not label this adapter; use upstream_unknown")
        if declared and str(declared).lower() != kind:
            raise ValueError(f"Checkpoint metadata says {declared}, requested {kind}")
    elif kind != "upstream_unknown":
        raise ValueError("Unlabelled external adapter: use upstream_unknown; do not guess raw/EMA")
    return {"kind": kind, "adapter": tree_fingerprint(adapter, full=True)}


def evaluate(config, site, runtime, output, adapter=None, kind="base", sampler="flow", steps=40,
             seeds=(42,), split="test", save_images=False, allow_training_split=False):
    if config.selection_protocol != "legacy":
        raise ValueError("Semantic runs require the frozen A/B task and five-quality evaluation pipeline; "
                         "legacy JointVerifier evaluation would change label semantics")
    if split == "train" and not allow_training_split:
        raise ValueError("Held-out evaluation refuses train split; explicit diagnostic override required")
    if steps < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive steps and distinct replicate seeds required")
    identity = adapter_identity(adapter, kind)
    config = copy.deepcopy(config)
    config.initial_adapter = str(Path(adapter).resolve()) if adapter and config.backend == "sd3" else None
    records = load_records(config.dataset, split)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0" if config.backend == "sd3" else "cpu")
    seed_all(config.seed)
    start = time.perf_counter()
    backend = build_backend(config, site, runtime, device, training=False)
    if adapter and config.backend == "toy":
        backend.load_student(torch.load(Path(adapter) / "toy.pt", weights_only=True))
    verifier = build_verifier(config, site, device, records)
    protocol = {"version": 1, "backend": config.backend, "split": split,
                "dataset": records_digest(records), "sampler": sampler, "steps": steps,
                "resolution": config.resolution, "guidance_scale": config.guidance_scale,
                "max_sequence_length": config.max_sequence_length, "precision": config.precision,
                "noise": "cpu_fp32_sha256_sample_id_v1", "replicate_seeds": list(seeds),
                "batch_size": 1, "negative_prompt": "", "verifier": "toy_v1" if config.backend == "toy" else "joint_constraints_v1",
                "ocr_normalization": "NFKC_casefold_collapse_whitespace_exact_match",
                "geneval": "per_clause_upstream_strict_count_and_clipped_progress",
                "utility_joint_weight": config.utility_joint_weight,
                "quality_floor": config.quality_floor, "quality_drop_tolerance": config.quality_drop_tolerance,
                "adapter_kind": kind, "environment": environment()["packages"]}
    if config.backend == "sd3":
        protocol.update(model=tree_fingerprint(site["model"]), reward_weights=tree_fingerprint(site["reward_ckpts"]),
                        auxiliary_models={k: tree_fingerprint(v) for k, v in site.get("auxiliary_models", {}).items()},
                        scheduler=canonical_scheduler(backend.pipeline.scheduler.config),
                        upstream_content=tree_fingerprint(Path(runtime) / "flow_grpo", full=True)["inventory_digest"],
                        gpu=torch.cuda.get_device_name(device))
    expected = [{"id": r["id"], "replicate": seed, "noise_seed": stable_seed(seed, "eval", r["id"])}
                for r in records for seed in seeds]
    write_json(output / "protocol.json", protocol)
    write_json(output / "checkpoint.json", identity)
    write_json(output / "environment.json", environment())
    write_json(output / "expected.json", expected)
    rows = []
    with torch.no_grad():
        for record in records:
            for seed in seeds:
                noise_seed = stable_seed(seed, "eval", record["id"])
                image = backend.generate(record, noise_seed, steps, sampler)
                verdict = verifier(image, record)
                row = {"id": record["id"], "replicate": seed, "noise_seed": noise_seed,
                       "constraint_ids": [c["id"] for c in record["constraints"]],
                       **verdict.as_dict(config.utility_joint_weight)}
                if save_images and config.backend == "sd3":
                    path = output / "images" / f"{digest([record['id'], seed])[:24]}.png"
                    path.parent.mkdir(exist_ok=True)
                    image.save(path)
                    row["image"] = str(path.relative_to(output))
                    row["image_sha256"] = sha256(path)
                append_jsonl(output / "per_image.jsonl", row)
                rows.append(row)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    valid = [r for r in rows if r["valid"]]
    summary = {"status": "complete" if len(valid) == len(rows) else "invalid_verifier_results",
               "images": len(rows), "valid_images": len(valid), "elapsed_seconds": elapsed,
               "allocated_gpu_hours": elapsed / 3600 if device.type == "cuda" else 0.0,
               "counts": dict(backend.counters + verifier.counters),
               "protocol_digest": digest(protocol), "per_image_sha256": sha256(output / "per_image.jsonl"),
               "utility": float(np.mean([r["utility"] for r in valid])) if valid else None,
               "joint_success": float(np.mean([r["joint_success"] for r in valid])) if valid else None}
    write_json(output / "summary.json", summary)
    return summary


def read_evaluation(path):
    path = Path(path)
    summary, protocol = read_json(path / "summary.json"), read_json(path / "protocol.json")
    if summary["status"] != "complete" or summary["protocol_digest"] != digest(protocol):
        raise ValueError("Evaluation is incomplete, invalid, or protocol was changed")
    if sha256(path / "per_image.jsonl") != summary["per_image_sha256"]:
        raise ValueError("Per-image results changed after evaluation")
    rows = [json.loads(s) for s in (path / "per_image.jsonl").read_text().splitlines() if s]
    expected = read_json(path / "expected.json")
    keys = [(r["id"], r["replicate"], r["noise_seed"]) for r in rows]
    expected_keys = [(r["id"], r["replicate"], r["noise_seed"]) for r in expected]
    if len(keys) != len(set(keys)) or set(keys) != set(expected_keys) or len(keys) != summary["images"]:
        raise ValueError("Duplicate/missing/unexpected evaluation samples")
    if any(not r["valid"] for r in rows):
        raise ValueError("Invalid verifier rows cannot enter a paired comparison")
    for row in rows:
        if "image_sha256" in row and sha256(path / row["image"]) != row["image_sha256"]:
            raise ValueError("Evaluation image changed after scoring")
    return protocol, {k: r for k, r in zip(keys, rows)}


def compare(base_path, method_path, allow_base=True):
    base_protocol, base = read_evaluation(base_path)
    method_protocol, method = read_evaluation(method_path)
    left, right = dict(base_protocol), dict(method_protocol)
    for protocol in (left, right):
        if "scheduler" in protocol:
            protocol["scheduler"] = canonical_scheduler(protocol["scheduler"])
    # Base has no averaging convention. Raw-vs-EMA trained-model comparisons
    # remain forbidden; both are allowed to compare against the same base.
    if allow_base and "base" in {left["adapter_kind"], right["adapter_kind"]}:
        left.pop("adapter_kind")
        right.pop("adapter_kind")
    differences = [k for k in sorted(set(left) | set(right)) if left.get(k) != right.get(k)]
    if differences or base.keys() != method.keys():
        raise ValueError(f"Paired evaluation protocol mismatch: {differences or 'sample identities'}")
    result = {"baseline": str(base_path), "method": str(method_path), "images": len(base),
              "statistical_unit": "prompt; average replicate seeds within prompt before paired bootstrap",
              "interval": "paired prompt bootstrap percentile 95%, 2000 resamples, seed 0"}
    for metric in ("utility", "joint_success"):
        by_prompt = defaultdict(list)
        for key in base:
            by_prompt[key[0]].append(float(method[key][metric]) - float(base[key][metric]))
        deltas = np.array([np.mean(v) for v in by_prompt.values()])
        interval = None
        if len(deltas) > 1:
            rng = np.random.default_rng(0)
            samples = [float(rng.choice(deltas, len(deltas), replace=True).mean()) for _ in range(2000)]
            interval = np.quantile(samples, [0.025, 0.975]).tolist()
        result[metric] = {"paired_delta": float(deltas.mean()), "prompt_count": len(deltas), "ci95": interval}
    # Component accuracy and preservation can move oppositely to aggregate utility.
    from .report import prompt_summary
    clauses, damaged = defaultdict(list), defaultdict(list)
    for key in base:
        before, after = base[key], method[key]
        if before["constraint_ids"] != after["constraint_ids"]:
            raise ValueError("Constraint identities differ in paired evaluation")
        for name, old, new in zip(before["constraint_ids"], before["satisfied"], after["satisfied"]):
            clauses[name].append((key[0], int(new) - int(old)))
            if old:
                damaged[name].append((key[0], not new))
    result["clause_accuracy_delta"] = {name: prompt_summary(values) for name, values in clauses.items()}
    result["regression_given_baseline_satisfied"] = {name: prompt_summary(values) for name, values in damaged.items()}
    return result


def analyze_queries(run):
    rows = []
    for path in sorted(Path(run).glob("queries-rank-*.jsonl")):
        rows.extend(json.loads(s) for s in path.read_text().splitlines() if s)
    if not rows:
        raise ValueError("No diagnostic query records")
    arms = defaultdict(list)
    headroom, selected = [], []
    errors = 0
    for row in rows:
        base = row["candidates"][0]
        if not base["valid"]:
            errors += 1
            continue
        eligible = [c for c in row["candidates"] if c["valid"] and c["admissible"]]
        headroom.append(max([base["utility"]] + [c["utility"] for c in eligible]) - base["utility"])
        for c in row["candidates"][1:]:
            if c["valid"]:
                arms[c["name"]].append({"gain": c["utility"] - base["utility"], "damage": c["collateral_damage"]})
        chosen = next(c for c in row["candidates"] if c["name"] == row["selected"])
        if chosen["valid"]:
            selected.append(chosen["utility"] - base["utility"])
    by_arm = {name: {"n": len(values), "mean_local_gain": float(np.mean([v["gain"] for v in values])),
                    "harm_rate": float(np.mean([v["gain"] < 0 for v in values])),
                    "collateral_damage_rate": float(np.mean([v["damage"] for v in values]))}
              for name, values in arms.items()}
    return {"queries": len(rows), "invalid_baseline_queries": errors, "arms": by_arm,
            "accepted_fraction": sum(r["accepted"] for r in rows) / len(rows),
            "oracle_admissible_headroom": float(np.mean(headroom)) if headroom else None,
            "selected_local_gain": float(np.mean(selected)) if selected else None,
            "interpretation": "Within frozen-policy interventions on collected states; NOT held-out trained-policy improvement"}
