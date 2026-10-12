"""Measure teacher opportunity on every saved state of a completed evaluation cohort."""
import argparse
import os
from pathlib import Path
import shutil
import time

import torch

from .action_transfer import require_exact, trace_baseline
from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import atomic_torch_save, seed_all
from .fitting_probe import cpu_tree, tensor_digest
from .io import ROOT, append_jsonl, assert_environment, read_json, sha256, write_json
from .learnability import action_metrics, save_verdict, single_endpoint
from .runtime import prepare, verify_run
from .supervision_completion import rows
from .verifiers import build_verifier

VERSION = "teacher_opportunity_v1"
TEACHERS = ("ocr", "geneval")


def pairs_from_plan(plan):
    """Select by the already-frozen cohort only, never by success or uncertainty."""
    pairs = [p for p in plan["evaluation"] if "positive_fresh_seed" in p["cohorts"]]
    assert len(pairs) == len({p["pair_id"] for p in pairs}) == 12
    assert all(p["single_action"] for p in pairs)
    assert len({p["id"] for p in pairs}) == 6
    assert all(sum(q["id"] == p["id"] for q in pairs) == 2 for p in pairs)
    return pairs


def prepare_probe(source, run, campaign):
    source, run, campaign = map(lambda p: Path(p).resolve(), (source, run, campaign))
    old, result = read_json(source / "probe_plan.json"), read_json(source / "result.json")
    assert result["status"] == "complete"
    pairs = pairs_from_plan(old)
    endpoints = rows(source / "endpoints.jsonl")
    refs = {p["pair_id"]: {mode: next(r for r in endpoints if
            (r["pair_id"], r["model"], r["path"]) == (p["pair_id"], model, path))
            for mode, model, path in (("noop", "joint128", "full"), ("student_single", "multi", "single"))}
            for p in pairs}
    inputs = [source / n for n in ("probe_plan.json", "result.json", "endpoints.jsonl", "manifest.json")]
    inputs += [ROOT / "experiments/local_expert_v18" / n for n in
               ("results.json", "observations_frozen.json", "verification.json", "report_index.json")]
    checkpoints = {"initial": old["student"], "student": str(source / "checkpoints/multi/step128.pt")}
    inputs += list(map(Path, checkpoints.values()))
    for p in pairs:
        inputs.append(source / "eval_queries" / f"{p['pair_id']}.pt")
        for r in refs[p["pair_id"]].values():
            assert sha256(source / r["image"]) == r["image_sha256"]
            inputs.append(source / r["image"])
    prepare(Experiment.load(source / "config.json"), run, read_json(source / "site.json"))
    plan = {"version": VERSION, "source": str(source), "pairs": pairs, "references": refs,
            "checkpoints": checkpoints, "index": 3, "alpha": 1, "teachers": list(TEACHERS),
            "required_gpu": "NVIDIA L40S", "required_torch": result["training_execution"]["torch"],
            "expected_counts": {"forward_student": 132, "forward_behavior": 228,
                "forward_ocr": 12, "forward_geneval": 12, "forward_examples": 756,
                "decode_images": 48, "verifier_images": 48, "backward_queries": 0,
                "text_encode_prompts": 12},
            "scope": "Post-Phase18 development diagnosis on all12 primary states; no fitting or outcome-based state selection. Teacher availability is an oracle diagnostic, not a deployable selector."}
    write_json(run / "probe_plan.json", plan)
    for name in ("PROTOCOL.md", "REVIEW_RUBRIC.md"):
        shutil.copyfile(campaign / name, run / name)
        inputs.append(run / name)
    inputs.append(run / "probe_plan.json")
    write_json(run / "probe_attestation.json", {"inputs": {str(p): sha256(p) for p in inputs}})
    write_json(campaign / "attempt_001.json", {"run": str(run), "plan_sha256": sha256(run / "probe_plan.json"),
               "new_images": 24, "reference_images": 24, "teacher_queries": 24, "updates": 0})
    print({"run": str(run), "states": 12, "new_images": 24, "updates": 0})


@torch.no_grad()
def execute(backend, verifier, config, plan, run):
    initial, student = (torch.load(plan["checkpoints"][n], map_location="cpu", weights_only=True)
                        for n in ("initial", "student"))
    hashes = {"initial": tensor_digest(initial), "student": tensor_digest(student)}
    backend.load_student(initial, refresh_behavior=True)
    records = {r["id"]: r for r in load_records(config.dataset, "train")}
    queries = {}

    def save(pair, mode, image, reference=None):
        v = save_verdict(run, backend, verifier, records[pair["id"]], image,
                         [VERSION, pair["pair_id"], mode], config.utility_joint_weight)
        if reference is not None:
            require_exact(v, reference["image_sha256"])
        row = {**pair, "mode": mode, "noise_sha256": queries[pair["pair_id"]]["noise_sha256"], **v}
        if reference is not None:
            row.update(exact=True, reference_sha256=reference["image_sha256"])
        append_jsonl(run / ("references.jsonl" if reference is not None else "endpoints.jsonl"), row)

    # Finish every baseline/state replay, then every learned-action replay, before any teacher query.
    for pair in plan["pairs"]:
        q, image = trace_baseline(backend, records[pair["id"]], pair["seed"], config.steps, plan["index"])
        pid = pair["pair_id"]
        old = torch.load(Path(plan["source"]) / "eval_queries" / f"{pid}.pt", map_location="cpu", weights_only=False)
        assert all(q[k] == old[k] for k in ("record", "seed", "index", "dt", "noise_sha256"))
        assert torch.equal(q["state"].cpu(), old["state"])
        assert all(torch.equal(a, b) for a, b in zip(q["context"], old["context"]))
        noop = backend.predict(q["state"], q["index"], q["context"], "behavior")
        assert torch.equal(noop.cpu(), old["actions"]["noop"]) and torch.equal(noop, q["actions"]["noop"])
        assert q["noise_sha256"] == plan["references"][pid]["noop"]["noise_sha256"]
        queries[pid] = q
        save(pair, "noop", image, plan["references"][pid]["noop"])
    backend.load_student(student, refresh_behavior=False)
    for pair in plan["pairs"]:
        q = queries[pair["pair_id"]]
        q["actions"]["student_single"] = backend.predict(q["state"], q["index"], q["context"], "student")
        save(pair, "student_single", single_endpoint(backend, q, q["actions"]["student_single"], config.steps),
             plan["references"][pair["pair_id"]]["student_single"])
    write_json(run / "replay_gate.json", {"all_exact": True, "reference_images": 24, "states": 12,
               "counts_before_teachers": dict(backend.counters + verifier.counters)})
    print({"event": "all_references_exact", "images": 24}, flush=True)
    (run / "queries").mkdir(exist_ok=False)
    for pair in plan["pairs"]:
        q = queries[pair["pair_id"]]
        metrics = {}
        for teacher in TEACHERS:
            mode = teacher + "@1"
            action = backend.predict(q["state"], q["index"], q["context"], teacher)
            q["actions"][mode] = action
            metrics[mode] = action_metrics(q["actions"]["student_single"], action, q["actions"]["noop"])
            save(pair, mode, single_endpoint(backend, q, action, config.steps))
        saved = Path("queries") / f"{pair['pair_id']}.pt"
        atomic_torch_save(cpu_tree(q), run / saved)
        append_jsonl(run / "queries.jsonl", {**pair, "saved_query": str(saved), "saved_query_sha256": sha256(run / saved),
                     "state_sha256": tensor_digest({"state": q["state"]}), "noise_sha256": q["noise_sha256"],
                     "context_sha256": tensor_digest({str(i): x for i, x in enumerate(q["context"])}),
                     "action_hashes": {n: tensor_digest({"action": a}) for n, a in q["actions"].items()},
                     "action_metrics": metrics, "prefix_sha256": hashes["initial"], "suffix_sha256": hashes["initial"],
                     "student_sha256": hashes["student"], "index": q["index"], "dt": q["dt"]})
        print({"event": "teacher_pair_complete", "pair_id": pair["pair_id"]}, flush=True)
    assert tensor_digest(backend.student_state()) == hashes["student"]
    assert tensor_digest(backend.behavior_state()) == hashes["initial"]
    counts = dict(backend.counters + verifier.counters)
    for k, v in plan["expected_counts"].items():
        assert counts.get(k, 0) == v, (k, counts.get(k), v)
    return {"status": "complete", "counts": counts, "model_hashes": hashes, "optimizer_updates": 0,
            "reference_images": 24, "new_images": 24, "teacher_queries": 24,
            "output_hashes": {str(p.relative_to(run)): sha256(p) for p in run.rglob("*") if p.is_file() and
                              (p.parent.name == "queries" or p.parent == run and p.suffix == ".jsonl" or p.name == "replay_gate.json")}}


def run_probe(run):
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
        assert plan["version"] == VERSION and plan["pairs"] == pairs_from_plan(read_json(Path(plan["source"]) / "probe_plan.json"))
        assert plan["index"] == 3 and plan["alpha"] == 1 and plan["teachers"] == list(TEACHERS) and config.steps == 10
        assert torch.__version__ == plan["required_torch"] and torch.cuda.get_device_name() == plan["required_gpu"]
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
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--source", default="runs/le18_completion_001")
    parser.add_argument("--campaign", default="experiments/local_expert_v19")
    args = parser.parse_args()
    prepare_probe(args.source, args.run, args.campaign) if args.stage == "prepare" else run_probe(args.run)
