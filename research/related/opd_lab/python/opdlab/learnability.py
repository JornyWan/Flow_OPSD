"""Phase7 fixed-positive controls; no on-policy method or reward changes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import torch

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .diagnostic import image_writer
from .engine import atomic_torch_save, preserve_rng, seed_all
from .fitting_probe import restore, snapshot, tensor_digest, update_once
from .io import append_jsonl, assert_environment, digest, read_json, sha256, write_json
from .runtime import verify_run
from .verifiers import build_verifier


def select_positives(key, observations):
    """Apply the preregistered rule to OLD observations, never new scores."""
    labels = {r["case"]: r for r in observations}
    selected = []
    for case in key:
        obs = labels[case["case"]]
        sides = {v["action"]: side for side, v in case["mapping"].items()}
        before, after = (obs["observations"][sides[k]] for k in ("noop", "target"))
        a, b = before["clauses"], after["clauses"]
        if set(a) != set(b):
            raise ValueError("Inconsistent clause labels")
        values = [*a.values(), *b.values(), before["full_visibility"], after["full_visibility"]]
        repaired = [k for k in a if a[k] is False and b[k] is True]
        harmed = [k for k in a if a[k] is True and b[k] is False]
        if (obs["better"] == sides["target"] and repaired and not harmed
                and all(type(v) is bool for v in values)
                and not (before["full_visibility"] and not after["full_visibility"])):
            selected.append({"case": case["case"], "id": case["id"], "repaired": repaired,
                             "preserved": [k for k in a if a[k]],
                             "complete_visibility_subset": after["full_visibility"] and all(b.values()),
                             "old_observations": {"noop": before, "target": after},
                             "old_endpoints": {k: case["mapping"][sides[k]]["row"] for k in sides}})
    return selected


def action_metrics(prediction, target, noop):
    direction, displacement = (target - noop).double(), (prediction - noop).double()
    denominator = direction.square().mean()
    if denominator <= 0:
        raise ValueError("Positive target must differ from noop")
    gamma = (displacement * direction).mean() / denominator
    return {"relative_target_mse": float((prediction.double() - target.double()).square().mean() / denominator),
            "target_mse": float((prediction.double() - target.double()).square().mean()),
            "gamma": float(gamma),
            "orthogonal_mse_over_initial": float((displacement - gamma * direction).square().mean() / denominator)}


@torch.no_grad()
def single_endpoint(backend, query, action, steps):
    backend.schedule(steps)
    z = backend.step(query["state"].clone(), action, query["index"])
    for index in range(query["index"] + 1, steps):
        z = backend.step(z, backend.predict(z, index, query["context"], "behavior"), index)
    return backend.decode(z)


def save_verdict(run, backend, verifier, record, image, identity, weight):
    path, checksum = image_writer(run, digest(identity)[:24])("endpoint", image)
    verdict = verifier(image, record)
    if not verdict.valid:
        raise RuntimeError("Invalid verifier result; image retained for diagnosis")
    return {**verdict.as_dict(weight), "image": path, "image_sha256": checksum}


def run_fixed(backend, verifier, queries, config, plan, run):
    optimizer = torch.optim.AdamW(backend.parameters(), lr=config.learning_rate,
                betas=tuple(config.adam_betas), eps=config.adam_epsilon, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
    initial = snapshot(backend, optimizer, scaler, 0)
    initial_hash = tensor_digest(initial["student"])
    specs = {s["id"]: s for s in plan["selected"]}
    for q in queries:
        backend.schedule(config.steps)
        q["noop"] = backend.predict(q["state"], q["index"], q["context"], "behavior").detach()
        for label, action in (("noop", q["noop"]), ("target", q["target"])):
            verdict = save_verdict(run, backend, verifier, q["record"],
                        single_endpoint(backend, q, action, config.steps),
                        ["reference", q["record"]["id"], label], config.utility_joint_weight)
            old = specs[q["record"]["id"]]["old_endpoints"][label]
            append_jsonl(run / "references.jsonl", {"id": q["record"]["id"], "kind": label,
                         **verdict, "old_image_sha256": old["image_sha256"],
                         "old_image_exact_match": verdict["image_sha256"] == old["image_sha256"]})
    arms = [(f"single_case{s['case']:02d}", [q]) for s, q in zip(plan["selected"], queries)] + [("joint", queries)]
    for arm, active in arms:
        (run / "checkpoints" / arm).mkdir(parents=True, exist_ok=False)
        restore(backend, optimizer, scaler, initial)
        if tensor_digest(backend.student_state()) != initial_hash or optimizer.state:
            raise RuntimeError("Arm did not start from the identical fresh optimizer and student")
        for step in range(plan["max_updates"] + 1):
            if step:
                metrics = update_once(backend, active, optimizer, scaler, config, keep=0)
                append_jsonl(run / "updates.jsonl", {"arm": arm, "optimizer_step": step, **metrics})
            if step not in plan["milestones"]:
                continue
            with preserve_rng(), torch.no_grad():
                if tensor_digest(backend.behavior_state()) != initial_hash:
                    raise RuntimeError("Frozen behavior changed")
                state = backend.student_state()
                state_hash = tensor_digest(state)
                atomic_torch_save(state, run / "checkpoints" / arm / f"step{step:03d}.pt")
                for q in active:
                    backend.schedule(config.steps)
                    p = backend.predict(q["state"], q["index"], q["context"], "student")
                    row = {"arm": arm, "optimizer_step": step, "id": q["record"]["id"],
                           "student_sha256": state_hash, "seed": q["seed"], "query_index": q["index"],
                           **action_metrics(p, q["target"], q["noop"])}
                    append_jsonl(run / "metrics.jsonl", row)
                    if step:
                        for mode, image in (("single", single_endpoint(backend, q, p, config.steps)),
                                            ("full", backend.generate(q["record"], q["seed"], config.steps))):
                            verdict = save_verdict(run, backend, verifier, q["record"], image,
                                        [arm, step, q["record"]["id"], mode], config.utility_joint_weight)
                            append_jsonl(run / "endpoints.jsonl", {**row, "mode": mode, **verdict})
                print({"event": "fit_milestone", "arm": arm, "step": step,
                       "counts": dict(backend.counters)}, flush=True)
        target = run / "checkpoints" / arm
        atomic_torch_save(snapshot(backend, optimizer, scaler, plan["max_updates"]), target / "training.pt")
        backend.export(target / "raw_lora")
        write_json(target / "metadata.json", {"complete": True, "optimizer_step": plan["max_updates"],
                   "raw_lora": "raw", "training_sha256": sha256(target / "training.pt"),
                   "initial_student_sha256": initial_hash, "ids": [q["record"]["id"] for q in active]})


@torch.no_grad()
def run_capability(backend, verifier, queries, records, config, plan, run):
    from peft import get_peft_model_state_dict
    initial, original_cfg = backend.student_state(), config.guidance_scale
    for teacher in config.teachers:
        teacher_state = get_peft_model_state_dict(backend.model, adapter_name=backend.adapters[teacher])
        backend.load_student(teacher_state, refresh_behavior=False)
        config.guidance_scale = config.teacher_guidance[teacher]
        for q in queries:
            backend.schedule(config.steps)
            p = backend.predict(q["state"], q["index"], q["context"], "student")
            t = backend.predict(q["state"], q["index"], q["context"], teacher)
            error = float((p - t).abs().max())
            append_jsonl(run / "teacher_copy.jsonl", {"teacher": teacher, "guidance": config.guidance_scale,
                         "id": q["record"]["id"], "max_abs_error": error, "exact": torch.equal(p, t)})
            if not torch.equal(p, t):
                raise RuntimeError("Matched-CFG copied teacher does not exactly reproduce teacher prediction")
    backend.load_student(initial, refresh_behavior=False)
    config.guidance_scale = original_cfg
    record_map = {r["id"]: r for r in records}
    conditions = [("base_cfg4.5", "behavior", 4.5), ("base_cfg1", "behavior", 1.0),
                  ("ocr_cfg4.5", "ocr", 4.5), ("geneval_cfg1", "geneval", 1.0),
                  ("geneval_cfg4.5", "geneval", 4.5)]
    for name, adapter, guidance in conditions:
        if adapter in config.teachers:
            config.teacher_guidance[adapter] = guidance
        else:
            config.guidance_scale = guidance
        for spec in plan["capability_queries"]:
            r = record_map[spec["id"]]
            image = backend.generate(r, spec["seed"], config.steps, adapter=adapter)
            verdict = save_verdict(run, backend, verifier, r, image,
                        ["capability", name, r["id"]], config.utility_joint_weight)
            append_jsonl(run / "capability.jsonl", {"condition": name, "id": r["id"],
                         "seed": spec["seed"], "guidance": guidance, **verdict})
        print({"event": "capability_condition_complete", "condition": name}, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--stage", required=True, choices=["fit", "capability"])
    args = parser.parse_args()
    run = Path(args.run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    attest = read_json(run / "phase7_attestation.json")
    for path, expected in attest["inputs"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    plan = read_json(run / "phase7_plan.json")
    if plan["stage"] != args.stage:
        raise ValueError("Wrong stage for frozen run")
    config, site = Experiment.load(run / "config.json"), read_json(run / "site.json")
    if any((run / p).exists() for p in ("RUNNING", "result.json", "updates.jsonl", "capability.jsonl", "failure.json")):
        raise FileExistsError("Use a fresh numbered attempt")
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    started = time.perf_counter()
    try:
        seed_all(config.seed)
        backend = build_backend(config, site, run / "upstream", "cuda")
        initial = torch.load(plan["frozen_behavior"], map_location="cpu", weights_only=True)
        backend.load_student(initial, refresh_behavior=True)
        records = load_records(config.dataset, "train")
        verifier = build_verifier(config, site, backend.device, records)
        all_queries = torch.load(plan["frozen_queries"], map_location="cpu", weights_only=False)
        indexed = {q["record"]["id"]: q for q in all_queries}
        queries = [indexed[s["id"]] for s in plan["selected"]]
        for q in queries:
            if not q["accepted"]:
                raise ValueError("Selected positive is not the saved accepted target")
            q["state"], q["target"] = q["state"].to(backend.device), q["target"].to(backend.device)
            backend.schedule(config.steps)
            if abs(float(backend.sigmas[q["index"] + 1] - backend.sigmas[q["index"]]) - q["dt"]) > 1e-7:
                raise ValueError("Saved query belongs to a different schedule")
        if args.stage == "fit":
            run_fixed(backend, verifier, queries, config, plan, run)
        else:
            run_capability(backend, verifier, queries, records, config, plan, run)
        if tensor_digest(backend.behavior_state()) != tensor_digest(initial):
            raise RuntimeError("Frozen behavior changed")
        elapsed = time.perf_counter() - started
        write_json(run / "result.json", {"status": "complete", "stage": args.stage,
                   "elapsed_seconds": elapsed, "worker_gpu_hours": elapsed / 3600,
                   "counts": dict(backend.counters + verifier.counters),
                   "output_hashes": {p.name: sha256(p) for p in sorted(run.glob("*.jsonl"))},
                   "plan_sha256": sha256(run / "phase7_plan.json")})
    except Exception as e:
        write_json(run / "failure.json", {"error": repr(e), "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    main()
