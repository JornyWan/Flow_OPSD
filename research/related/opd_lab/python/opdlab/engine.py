"""Collect frozen-policy interventions, then fit detached targets.

All control arms use this same engine. Native upstream baselines use native.py.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from .backends import build_backend
from .data import load_records
from .io import append_jsonl, digest, read_json, stable_seed, write_json
from .method import collect_query, query_log, transition_mean_loss
from .verifiers import build_verifier


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def distributed_device(backend):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "sd3":
        if not torch.cuda.is_available():
            raise RuntimeError("No allocated GPU")
        torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if backend == "sd3" else "gloo")
    return rank, world, torch.device(f"cuda:{local_rank}" if backend == "sd3" else "cpu")


def barrier():
    if dist.is_initialized():
        dist.barrier()


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def checkpoint(run, epoch, step, backend, optimizer, scaler, ema, signature, rank, world, elapsed):
    target = Path(run) / "checkpoints" / f"epoch-{epoch:06d}"
    if rank == 0:
        target.mkdir(parents=True, exist_ok=False)
        atomic_torch_save({"format": 1, "epoch": epoch, "optimizer_step": step,
                           "student": backend.student_state(), "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict(), "ema": ema, "signature": signature,
                           "world_size": world, "elapsed_seconds": elapsed}, target / "training.pt")
    barrier()
    atomic_torch_save(rng_state(), target / f"rng-rank-{rank}.pt")
    barrier()
    if rank == 0:
        # Export can switch adapters; restore all RNG so saving has no training effect.
        with preserve_rng():
            backend.export(target / "raw_lora")
            backend.export(target / "ema_lora", ema)
        write_json(target / "metadata.json", {"format": 1, "outer_epoch": epoch, "optimizer_step": step,
                   "raw_lora": "raw", "ema_lora": "EMA", "complete": True, "signature": signature})
        write_json(Path(run) / "latest.json", {"checkpoint": str(target)})
    barrier()
    return target


@torch.no_grad()
def postfit(backend, verifier, queries, config):
    rows = []
    backend.schedule(config.steps)
    for q in queries:
        # Two distinct questions: can we fit the local action, and does the
        # UPDATED complete policy improve a fresh full rollout?
        prediction = backend.predict(q["state"], q["index"], q["context"], "student")
        error = float(transition_mean_loss(prediction, q["target"], q["dt"]))
        state = backend.step(q["state"].clone(), prediction, q["index"])
        for j in range(q["index"] + 1, config.steps):
            state = backend.step(state, backend.predict(state, j, q["context"], "behavior"), j)
        local = verifier(backend.decode(state), q["record"])
        full = verifier(backend.generate(q["record"], q["seed"], config.steps), q["record"])
        rows.append({"id": q["record"]["id"], "seed": q["seed"], "query_index": q["index"],
                     "accepted": q["accepted"], "local_target_loss": error,
                     "postfit_single_intervention": local.as_dict(config.utility_joint_weight),
                     "postfit_full_policy_train_prompt": full.as_dict(config.utility_joint_weight)})
    return rows


def train_loop(config, site, runtime, run, resume=None, stop_after=None, diagnostic=False):
    config.validate()
    run = Path(run)
    if config.selection_protocol in {"semantic_batch_v1", "semantic_veto_v1"} and int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Semantic batch v1 supports one process only")
    rank, world, device = distributed_device(config.backend)
    records = load_records(config.dataset, config.split)
    semantic = config.selection_protocol in {"semantic_batch_v1", "semantic_veto_v1"}
    veto = config.selection_protocol == "semantic_veto_v1"
    selection = None
    if semantic:
        if veto:
            from . import veto_selection as selection_module
        else:
            from . import online_selection as selection_module
        selection = selection_module.BatchSelection(config, records, read_json(run / "manifest.json"), world)
    if config.split != "train" and not diagnostic:
        raise ValueError("Training on validation/test split is forbidden")
    training_queries = None
    record_map = {r["id"]: r for r in records}
    if config.training_manifest:
        from .training_queries import validate_training_manifest
        manifest = read_json(config.training_manifest)
        validate_training_manifest(manifest, config, records, world)
        training_queries = manifest["queries"]
    # Identical adapter initialization on every rank; per-query rollout seeds
    # are keyed by global query id, separately from training RNG.
    seed_all(config.seed)
    started = time.perf_counter()
    backend = build_backend(config, site, runtime, device)
    if semantic:
        from .online_selection import SemanticVerifier
        verifier = SemanticVerifier(config)
    else:
        verifier = build_verifier(config, site, device, records)
    optimizer = torch.optim.AdamW(backend.parameters(), lr=config.learning_rate,
                 betas=tuple(config.adam_betas), eps=config.adam_epsilon, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.backend == "sd3" and config.precision == "fp16")
    signature = digest({"config": config.as_dict(), "world_size": world,
                        "records": records, "manifest": read_json(run / "manifest.json")})
    epoch0, step, previous_elapsed = 0, 0, 0.0
    ema = copy.deepcopy(backend.student_state())
    if resume:
        resume = Path(resume)
        if not read_json(resume / "metadata.json").get("complete"):
            raise ValueError("Cannot resume an incomplete checkpoint")
        state = torch.load(resume / "training.pt", map_location="cpu", weights_only=False)
        if state["signature"] != signature or state["world_size"] != world:
            raise ValueError("Resume config, data, source, assets, or world size mismatch")
        backend.load_student(state["student"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        ema, epoch0, step = state["ema"], state["epoch"], state["optimizer_step"]
        previous_elapsed = state["elapsed_seconds"]
        restore_rng(torch.load(resume / f"rng-rank-{rank}.pt", weights_only=False))
        if epoch0 >= config.outer_epochs:
            raise ValueError("This training budget has already completed")
    log = run / f"queries-rank-{rank}.jsonl"
    post_log = run / f"postfit-rank-{rank}.jsonl"
    # Resuming an older checkpoint with later logs would mix two branches.
    extra_logs = [run / "selection-batches.jsonl"] if semantic else []
    if config.record_selected_states or config.optimizer_step_rule != "constant" or veto:
        extra_logs += [run / "optimizer_updates.jsonl", run / "selected_states.jsonl"]
    if config.record_fit_blocks:
        extra_logs += [run / "fit_blocks.jsonl"]
    for path in [log, post_log, *extra_logs] + ([run / "metrics.jsonl"] if rank == 0 else []):
        if path.exists():
            rows = [json.loads(s) for s in path.read_text().splitlines() if s]
            if any(row["outer_epoch"] > epoch0 for row in rows):
                raise ValueError("Logs extend beyond resume checkpoint; use a fresh explicit branch directory")
    end = min(config.outer_epochs, stop_after) if stop_after is not None else config.outer_epochs
    if end <= epoch0:
        raise ValueError("stop_after must be later than the resume checkpoint")
    if diagnostic:
        end = epoch0 + 1
    for epoch in range(epoch0, end):
        backend.refresh_behavior()
        backend.schedule(config.steps)
        queries = []
        counters_before = backend.counters.copy() + verifier.counters.copy()
        for local in range(config.queries_per_rank):
            global_id = epoch * world * config.queries_per_rank + rank * config.queries_per_rank + local
            record = records[global_id % len(records)]
            seed = stable_seed(config.seed, "train_query", global_id)
            count = int(config.steps * config.timestep_fraction)
            index = stable_seed(config.seed, "query_time", global_id) % count
            if training_queries is not None:
                spec = training_queries[global_id]
                record, seed, index = record_map[spec["id"]], spec["seed"], spec["query_index"]
            save_image = None
            if config.save_candidates:
                from .diagnostic import image_writer
                save_image = image_writer(run, digest(["training", global_id])[:24])
            q = collect_query(backend, verifier, record, seed, index, config,
                              query_ordinal=global_id, save_image=save_image, defer_selection=semantic)
            q["global_query_id"] = global_id
            queries.append(q)
            if not semantic:
                append_jsonl(log, {"outer_epoch": epoch + 1, "global_query_id": global_id, **query_log(q, config)})
        failed = torch.tensor(int(any(not c.verdict.valid for q in queries for c in q["candidates"])), device=device)
        if world > 1:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError("Invalid verifier candidate; refusing to train on an incomplete candidate set")
        if semantic:
            from .online_selection import semantic_query_log
            for row in selection.select_epoch(queries, epoch + 1):
                append_jsonl(run / "selection-batches.jsonl", row)
            for q in queries:
                append_jsonl(log, {"outer_epoch": epoch + 1, "global_query_id": q["global_query_id"],
                                   **semantic_query_log(q, config)})
        local_loss_sum, updates = 0.0, 0
        if not diagnostic:
            fit_block = None
            if config.record_fit_blocks:
                from .repeated_fitting import FitBlock
                fit_block = FitBlock(backend, queries, epoch + 1, step, config.fitting_epochs)
            for fitting in range(config.fitting_epochs):
                if fit_block is not None:
                    fit_block.check_inputs(queries, fitting)
                # Stable shared query order: no hidden resampling after selection.
                for offset in range(0, len(queries), config.accumulation):
                    group = queries[offset:offset + config.accumulation]
                    accepted = torch.tensor(sum(q["accepted"] for q in group), device=device)
                    if world > 1:
                        dist.all_reduce(accepted)
                    if accepted.item() == 0:
                        # Adam momentum/decay can move parameters even on zero loss.
                        # No accepted supervision => no optimizer or EMA update.
                        continue
                    from .sparse_updates import set_step_learning_rate, selected_state_snapshot, parameter_step_l2
                    step_lr = set_step_learning_rate(optimizer, config, int(accepted.item()), config.accumulation * world)
                    if config.record_selected_states and fitting == 0:
                        with preserve_rng():
                            selected_state_snapshot(run, epoch + 1, offset // config.accumulation, backend, group)
                    before_parameters = backend.student_state() if config.record_selected_states else None
                    optimizer.zero_grad(set_to_none=True)
                    for q in group:
                        prediction = backend.predict(q["state"], q["index"], q["context"], "student", grad=True)
                        loss = transition_mean_loss(prediction, q["target"], q["dt"])
                        # Keep the original all-query loss denominator. Adam's actual
                        # step size need not scale proportionally with this loss.
                        loss = loss * float(q["accepted"]) / config.accumulation
                        local_loss_sum += float(loss.detach())
                        scaler.scale(loss).backward()
                        backend.counters["backward_queries"] += 1
                    # Reduce scaled gradients BEFORE unscale, so all ranks observe
                    # the same overflow and GradScaler makes the same skip decision.
                    if world > 1:
                        for p in backend.parameters():
                            if p.grad is None:
                                raise RuntimeError("Missing student gradient")
                            dist.all_reduce(p.grad)
                            p.grad.div_(world)
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(backend.parameters(), config.max_grad_norm,
                                                   error_if_nonfinite=not scaler.is_enabled())
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    did_update = scaler.get_scale() >= old_scale
                    optimizer.zero_grad(set_to_none=True)
                    if semantic and not did_update:
                        raise RuntimeError("Semantic count-matched run had a skipped scaler update; stop")
                    if did_update:
                        step += 1
                        updates += 1
                        if (step + 1) % config.ema_interval == 0:
                            decay = min((1 + step) / (10 + step), config.ema_decay)
                            for key, value in backend.student_state().items():
                                ema[key].lerp_(value, 1 - decay)
                    if config.record_selected_states or config.optimizer_step_rule != "constant" or veto:
                        after_parameters = backend.student_state() if before_parameters is not None else None
                        identity = {}
                        if fit_block is not None:
                            identity = fit_block.observe_update(before_parameters, after_parameters, fitting)
                        append_jsonl(run / "optimizer_updates.jsonl", {
                            "outer_epoch": epoch + 1, "group": offset // config.accumulation,
                            "fitting": fitting, "optimizer_step": step, "accepted": int(accepted.item()),
                            "total_queries": config.accumulation * world, "learning_rate": step_lr,
                            "step_rule": config.optimizer_step_rule, "gradient_norm_before_clip": float(gradient_norm),
                            "did_update": did_update, "parameter_step_l2": parameter_step_l2(
                                before_parameters, after_parameters) if before_parameters is not None else None,
                            **identity})
                        del after_parameters
                    del before_parameters
            if fit_block is not None:
                fit_block.finish(run, backend, queries, step, updates)
            if config.audit_postfit:
                with preserve_rng():
                    for row in postfit(backend, verifier, queries, config):
                        append_jsonl(post_log, {"outer_epoch": epoch + 1, **row})
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = previous_elapsed + time.perf_counter() - started
        counters = backend.counters + verifier.counters - counters_before
        summary = {"accepted": sum(q["accepted"] for q in queries), "queries": len(queries),
                   "local_loss_sum": local_loss_sum, "updates": updates, "counts": dict(counters)}
        summaries = [summary]
        if world > 1:
            summaries = [None] * world
            dist.all_gather_object(summaries, summary)
        if rank == 0:
            counts = Counter()
            for item in summaries:
                counts.update(item["counts"])
            metrics = {"outer_epoch": epoch + 1, "optimizer_step": step, "diagnostic_only": diagnostic,
                       "accepted": sum(v["accepted"] for v in summaries),
                       "queries": sum(v["queries"] for v in summaries),
                       "loss_sum_rank_mean": sum(v["local_loss_sum"] for v in summaries) / world,
                       "counts": dict(counts), "elapsed_seconds": elapsed,
                       "allocated_gpu_hours": elapsed * world / 3600 if device.type == "cuda" else 0.0}
            append_jsonl(run / "metrics.jsonl", metrics)
            print(json.dumps(metrics), flush=True)
        if not diagnostic and ((epoch + 1) % config.save_every == 0 or epoch + 1 == end):
            checkpoint(run, epoch + 1, step, backend, optimizer, scaler, ema, signature, rank, world, elapsed)
    barrier()
    if rank == 0:
        if semantic and not diagnostic and end == config.outer_epochs:
            selection_module.export_count_reference(run, config, records, read_json(run / "manifest.json"))
        write_json(run / "result.json", {"status": "diagnosed" if diagnostic else (
                   "complete" if end == config.outer_epochs else "paused_at_checkpoint"),
                   "outer_epoch": end, "optimizer_step": step})
    return backend.student_state()
