"""Collect one frozen GenEval action per fresh state before choosing supervision."""
import argparse
import os
from pathlib import Path
import time

import torch

from .action_transfer import require_exact, trace_baseline
from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import atomic_torch_save, seed_all
from .fitting_probe import cpu_tree, tensor_digest
from .io import append_jsonl, assert_environment, digest, read_json, sha256, stable_seed, write_json
from .learnability import save_verdict, single_endpoint
from .runtime import verify_run
from .supervision_completion import rows
from .verifiers import build_verifier

VERSION = "geneval_acceptance_v1"
GATE = {"safe_states": 8, "prompts_with_two_safe": 3, "safe_complete_states": 4}


def new_pairs(source, excluded):
    ids = sorted({p["id"] for p in source["evaluation"] if p["cohort"] == "positive_fresh_seed"})
    assert len(ids) == 6
    collection = [{"id": i, "replicate": r, "cohort": "collection"} for i in ids for r in range(8)]
    evaluation = [{k: p[k] for k in ("id", "replicate", "cohort")} for p in source["evaluation"] if "all_fresh" in p["cohorts"]]
    assert len(evaluation) == 32
    for stage, pairs in (("collection", collection), ("evaluation", evaluation)):
        for p in pairs:
            p["seed"] = stable_seed(20020, VERSION, stage, p["id"], p["replicate"])
            p["pair_id"] = digest([VERSION, stage, p["id"], p["seed"]])[:24]
    seeds = [p["seed"] for p in collection + evaluation]
    assert len(set(seeds)) == 80 and not set(seeds) & set(excluded)
    return collection, evaluation


def coverage_gate(cases):
    by_prompt = {}
    for c in cases:
        by_prompt.setdefault(c["id"], 0)
        by_prompt[c["id"]] += int(c["safe_repair"])
    measured = {"safe_states": sum(by_prompt.values()), "prompts_with_two_safe": sum(n >= 2 for n in by_prompt.values()),
                "safe_complete_states": sum(c["safe_repair"] and c["complete_scene"] for c in cases)}
    checks = {k: measured[k] >= v for k, v in GATE.items()}
    return {"thresholds": GATE, "measured": measured, "checks": checks, "passed": all(checks.values()), "safe_by_prompt": by_prompt}


@torch.no_grad()
def execute(backend, verifier, config, plan, run):
    initial = torch.load(plan["student"], map_location="cpu", weights_only=True)
    backend.load_student(initial, refresh_behavior=True)
    model_hash = tensor_digest(initial)
    records = {r["id"]: r for r in load_records(config.dataset, "train")}

    def save(pair, q, mode, image, reference=None):
        v = save_verdict(run, backend, verifier, records[pair["id"]], image,
                         [VERSION, "reference" if reference else "collection", pair["pair_id"], mode], config.utility_joint_weight)
        if reference:
            require_exact(v, reference["image_sha256"])
        append_jsonl(run / ("references.jsonl" if reference else "endpoints.jsonl"),
                     {**pair, "mode": mode, "index": 3, "noise_sha256": q["noise_sha256"],
                      "exact": bool(reference), "student_sha256": model_hash, **v})

    # Reference selection is replicate0 for each prompt, regardless of outcomes.
    for spec in plan["references"]:
        pair, old = spec["pair"], torch.load(spec["saved_query"], map_location="cpu", weights_only=False)
        q, image = trace_baseline(backend, records[pair["id"]], pair["seed"], config.steps, 3)
        assert all(q[k] == old[k] for k in ("record", "seed", "index", "dt", "noise_sha256"))
        assert torch.equal(q["state"].cpu(), old["state"])
        assert all(torch.equal(a, b) for a, b in zip(q["context"], old["context"]))
        noop = backend.predict(q["state"], 3, q["context"], "behavior")
        teacher = backend.predict(q["state"], 3, q["context"], "geneval")
        assert torch.equal(noop.cpu(), old["actions"]["noop"]) and torch.equal(noop, q["actions"]["noop"])
        assert torch.equal(teacher.cpu(), old["actions"]["geneval@1"])
        save(pair, q, "noop", image, spec["images"]["noop"])
        save(pair, q, "geneval@1", single_endpoint(backend, q, teacher, config.steps), spec["images"]["geneval@1"])
    write_json(run / "replay_gate.json", {"all_exact": True, "reference_images": 12,
               "original_states_and_geneval_actions": 6, "counts_before_collection": dict(backend.counters + verifier.counters)})
    print({"event": "references_exact", "images": 12}, flush=True)
    (run / "queries").mkdir(exist_ok=False)
    for pair in plan["collection"]:
        q, image = trace_baseline(backend, records[pair["id"]], pair["seed"], config.steps, 3)
        teacher = backend.predict(q["state"], 3, q["context"], "geneval")
        q["actions"]["geneval@1"] = teacher
        save(pair, q, "noop", image)
        save(pair, q, "geneval@1", single_endpoint(backend, q, teacher, config.steps))
        path = Path("queries") / f"{pair['pair_id']}.pt"
        atomic_torch_save(cpu_tree(q), run / path)
        append_jsonl(run / "queries.jsonl", {**pair, "saved_query": str(path), "saved_query_sha256": sha256(run / path),
                     "state_sha256": tensor_digest({"state": q["state"]}), "noise_sha256": q["noise_sha256"],
                     "context_sha256": tensor_digest({str(i): c for i, c in enumerate(q["context"])}),
                     "action_hashes": {n: tensor_digest({"action": a}) for n, a in q["actions"].items()},
                     "prefix_sha256": model_hash, "suffix_sha256": model_hash, "index": 3, "dt": q["dt"]})
        print({"event": "pair_complete", "pair_id": pair["pair_id"]}, flush=True)
    assert tensor_digest(backend.student_state()) == tensor_digest(backend.behavior_state()) == model_hash
    counts = dict(backend.counters + verifier.counters)
    for k, v in plan["expected_counts"].items():
        assert counts.get(k, 0) == v, (k, counts.get(k), v)
    return {"status": "complete", "counts": counts, "student_sha256": model_hash, "optimizer_updates": 0,
            "reference_images": 12, "new_images": 96, "collection_teacher_queries": 48, "reference_teacher_queries": 6,
            "evaluation_images_generated": 0,
            "output_hashes": {str(p.relative_to(run)): sha256(p) for p in run.rglob("*") if p.is_file() and
                              (p.parent.name == "queries" or p.parent == run and p.suffix == ".jsonl" or p.name == "replay_gate.json")}}


def run_collection(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for p, h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(p) == h, p
    if any((run / n).exists() for n in ("RUNNING", "result.json", "failure.json", "references.jsonl")):
        raise FileExistsError("Use a new attempt")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    started = time.perf_counter()
    try:
        plan, config = read_json(run / "probe_plan.json"), Experiment.load(run / "config.json")
        assert plan["version"] == VERSION and plan["coverage_gate"] == GATE
        assert (plan["collection"], plan["evaluation"]) == new_pairs(read_json(plan["source_plan"]), read_json(run / "seed_inventory.json")["excluded_seeds"])
        assert torch.cuda.get_device_name() == plan["required_gpu"] and torch.__version__ == plan["required_torch"]
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8" and not torch.are_deterministic_algorithms_enabled()
        seed_all(config.seed)
        site = read_json(run / "site.json")
        backend = build_backend(config, site, run / "upstream", "cuda", training=True)
        verifier = build_verifier(config, site, backend.device, load_records(config.dataset, "train"))
        result = execute(backend, verifier, config, plan, run)
        write_json(run / "result.json", {**result, "elapsed_seconds": time.perf_counter() - started})
    except Exception as exc:
        write_json(run / "failure.json", {"error": repr(exc), "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    run_collection(parser.parse_args().run)
