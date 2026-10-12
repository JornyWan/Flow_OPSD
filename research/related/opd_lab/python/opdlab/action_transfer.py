"""Phase16: frozen checkpoints, shared-prefix single-action diagnostics, no fitting."""
import argparse
import os
from pathlib import Path
import time

import torch

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import atomic_torch_save, seed_all
from .fitting_probe import cpu_tree, tensor_digest
from .io import append_jsonl, assert_environment, read_json, sha256, write_json
from .learnability import action_metrics, save_verdict, single_endpoint
from .runtime import verify_run
from .verifiers import build_verifier

VERSION = "action_transfer_v1"
OLD = "fitting_seed_diagnostic"
FRESH = "positive_fresh_seed"


def select_pairs(previous):
    points = [p for p in previous["points"] if p["label_target"] != "noop"]
    pairs = [p for p in previous["evaluation"] if p["cohort"] in (OLD, FRESH)]
    assert len(points) == 6 and len(pairs) == 18
    assert len({p["pair_id"] for p in pairs}) == 18
    for point in points:
        group = [p for p in pairs if p["id"] == point["id"]]
        assert len(group) == 3 and sum(p["cohort"] == FRESH for p in group) == 2
        assert [p["seed"] for p in group if p["cohort"] == OLD] == [point["seed"]]
    assert {p["id"] for p in pairs} == {p["id"] for p in points}
    return points, pairs


@torch.no_grad()
def trace_baseline(backend, record, seed, steps, index):
    """One full baseline replay also captures the exact state before action index."""
    backend.schedule(steps)
    z, context = backend.initial(seed), backend.encode(record["prompt"])
    query = {"record": record, "seed": seed, "index": index, "context": context,
             "noise_sha256": tensor_digest({"noise": z})}
    for j in range(steps):
        action = backend.predict(z, j, context, "student")
        if j == index:
            query.update(state=z.clone(), actions={"noop": action.clone()},
                         dt=float(backend.sigmas[j + 1] - backend.sigmas[j]))
        z = backend.step(z, action, j)
    return query, backend.decode(z)


def require_exact(verdict, expected):
    if verdict["image_sha256"] != expected:
        raise RuntimeError("Reference image failed exact replay; stop before new branches")


@torch.no_grad()
def branch_actions(backend, query, fresh):
    actions = {"student_single": backend.predict(query["state"], query["index"], query["context"], "student")}
    if fresh:
        for teacher in ("ocr", "geneval"):
            actions[teacher + "@1"] = backend.predict(query["state"], query["index"], query["context"], teacher)
    return actions


@torch.no_grad()
def execute(backend, verifier, config, plan, run):
    states = {n: torch.load(p, map_location="cpu", weights_only=True) for n, p in plan["checkpoints"].items()}
    hashes = {n: tensor_digest(s) for n, s in states.items()}
    backend.load_student(states["joint128"], refresh_behavior=True)
    records = {r["id"]: r for r in load_records(config.dataset, "train")}
    specs = {p["id"]: p for p in plan["points"]}
    queries = {}

    def save(image, pair, mode, reference=None):
        verdict = save_verdict(run, backend, verifier, records[pair["id"]], image,
                              [pair["pair_id"], mode], config.utility_joint_weight)
        row = {**pair, "mode": mode, **verdict}
        if reference is not None:
            require_exact(verdict, reference)
            row.update(exact=True, reference_sha256=reference)
        append_jsonl(run / ("references.jsonl" if reference else "endpoints.jsonl"), row)

    # All 42 exact references must pass before any new teacher prediction/branch.
    for pair in plan["pairs"]:
        q, image = trace_baseline(backend, records[pair["id"]], pair["seed"], config.steps, plan["index"])
        pid = pair["pair_id"]
        assert q["noise_sha256"] == plan["references"][pid]["joint128"]["noise_sha256"]
        save(image, pair, "joint128", plan["references"][pid]["joint128"]["image_sha256"])
        noop = backend.predict(q["state"], q["index"], q["context"], "behavior")
        assert torch.equal(noop, q["actions"]["noop"])
        if pair["cohort"] == OLD:
            spec = specs[pair["id"]]
            old = torch.load(Path(plan["query_source"]) / spec["saved_query"], map_location="cpu", weights_only=False)
            assert (old["record"], old["seed"], old["index"], old["dt"]) == (q["record"], q["seed"], q["index"], q["dt"])
            assert torch.equal(old["state"], q["state"].cpu())
            assert all(torch.equal(a, b) for a, b in zip(old["context"], q["context"]))
            for n, a in old["actions"].items():
                assert tensor_digest({"action": a}) == spec["action_hashes"][n]
            assert torch.equal(old["actions"]["noop"], noop.cpu())
            q["actions"].update({n: a.to(backend.device) for n, a in old["actions"].items()})
            target = q["actions"][spec["label_target"]]
            save(single_endpoint(backend, q, target, config.steps), pair, "selected_teacher",
                 spec["images"][spec["label_target"]]["sha256"])
        queries[pid] = q
    backend.load_student(states["label_selected"], refresh_behavior=False)
    for pair in plan["pairs"]:
        image = backend.generate(records[pair["id"]], pair["seed"], config.steps)
        save(image, pair, "label_selected", plan["references"][pair["pair_id"]]["label_selected"]["image_sha256"])
    write_json(run / "replay_gate.json", {"all_exact": True, "reference_images": 42, "old_state_action_context_replays": 6,
               "noop_replays": 18, "counts_before_branches": dict(backend.counters + verifier.counters)})
    print({"event": "all_references_exact", "images": 42}, flush=True)
    (run / "queries").mkdir(exist_ok=False)
    for pair in plan["pairs"]:
        q = queries[pair["pair_id"]]
        actions = branch_actions(backend, q, pair["cohort"] == FRESH)
        q["actions"].update(actions)
        saved = Path("queries") / (pair["pair_id"] + ".pt")
        atomic_torch_save(cpu_tree(q), run / saved)
        action_hashes = {n: tensor_digest({"action": a}) for n, a in q["actions"].items()}
        metrics = {n: action_metrics(actions["student_single"], a, q["actions"]["noop"])
                   for n, a in q["actions"].items() if n in ("ocr@1", "geneval@1")}
        append_jsonl(run / "queries.jsonl", {**pair, "saved_query": str(saved), "saved_query_sha256": sha256(run / saved),
                     "noise_sha256": q["noise_sha256"], "state_sha256": tensor_digest({"state": q["state"]}),
                     "context_sha256": tensor_digest({str(i): c for i, c in enumerate(q["context"])}),
                     "action_hashes": action_hashes, "action_metrics": metrics, "selected_teacher": specs[pair["id"]]["label_target"],
                     "prefix_sha256": hashes["joint128"], "suffix_sha256": hashes["joint128"],
                     "student_sha256": hashes["label_selected"], "index": plan["index"], "dt": q["dt"]})
        for name, action in actions.items():
            save(single_endpoint(backend, q, action, config.steps), pair, name)
        print({"event": "branches", "pair_id": pair["pair_id"], "images": len(actions)}, flush=True)
    assert tensor_digest(backend.student_state()) == hashes["label_selected"]
    assert tensor_digest(backend.behavior_state()) == hashes["joint128"]
    counts = dict(backend.counters + verifier.counters)
    for key, expected in plan["expected_counts"].items():
        assert counts.get(key, 0) == expected, (key, expected, counts)
    return {"status": "complete", "counts": counts, "model_hashes": hashes, "optimizer_updates": 0,
            "reference_images": 42, "new_images": 42, "new_teacher_queries": 24,
            "output_hashes": {str(p.relative_to(run)): sha256(p) for p in run.rglob("*")
                              if p.is_file() and (p.parent.name == "queries" or p.name in
                                 ("references.jsonl", "endpoints.jsonl", "queries.jsonl", "replay_gate.json"))}}


def run_probe(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for p, h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(p) == h, p
    if any((run / n).exists() for n in ("RUNNING", "result.json", "failure.json", "references.jsonl")):
        raise FileExistsError("Use a new attempt")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    start = time.perf_counter()
    try:
        config, plan, site = Experiment.load(run / "config.json"), read_json(run / "probe_plan.json"), read_json(run / "site.json")
        assert plan["version"] == VERSION
        seed_all(config.seed)
        # training=True only loads the expert adapters; execute uses no_grad, no optimizer.
        backend = build_backend(config, site, run / "upstream", "cuda", training=True)
        verifier = build_verifier(config, site, backend.device, load_records(config.dataset, "train"))
        result = execute(backend, verifier, config, plan, run)
        write_json(run / "result.json", {**result, "elapsed_seconds": time.perf_counter() - start})
    except Exception as exc:
        write_json(run / "failure.json", {"error": repr(exc), "elapsed_seconds": time.perf_counter() - start})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    run_probe(parser.parse_args().run)
