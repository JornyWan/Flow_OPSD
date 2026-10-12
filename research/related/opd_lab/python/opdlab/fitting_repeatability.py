"""Bounded GPU replay diagnostic using the unchanged historical and current fitters."""
import argparse
import os
from pathlib import Path
import time

import torch

from .annotation_probe import ARMS, fit_models
from .backends import build_backend
from .config import Experiment
from .engine import rng_state, restore_rng, seed_all
from .fitting_probe import tensor_digest
from .io import assert_environment, read_json, sha256, write_json
from .runtime import verify_run
from .supervision_study import fit


def compare_states(a, b):
    assert a.keys() == b.keys()
    differences = [float((a[k].double() - b[k].double()).abs().max()) for k in a]
    return {"exact": tensor_digest(a) == tensor_digest(b),
            "max_abs_difference": max(differences), "different_tensors": sum(d != 0 for d in differences)}


def run_diagnostic(run):
    run = Path(run).resolve()
    verify_run(run)
    assert_environment(read_json(run / "manifest.json")["environment"])
    for p, h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(p) == h, p
    if any((run / n).exists() for n in ("RUNNING", "result.json", "failure.json")):
        raise FileExistsError("Use a fresh diagnostic attempt")
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    (run / "RUNNING").write_text(f"pid={os.getpid()}\n")
    started = time.perf_counter()
    try:
        plan = read_json(run / "probe_plan.json")
        config = Experiment.load(run / "config.json")
        seed_all(config.seed)
        backend = build_backend(config, read_json(run / "site.json"), run / "upstream", "cuda", training=True)
        initial = torch.load(plan["student"], map_location="cpu", weights_only=True)
        backend.load_student(initial, refresh_behavior=True)
        queries = {}
        for p in plan["points"]:
            q = torch.load(p["saved_query"], map_location="cpu", weights_only=False)
            actions = {n: q["actions"][source] for n, source in p["action_names"].items()}
            assert {n: tensor_digest({"action": a}) for n, a in actions.items()} == p["action_hashes"]
            q.update(state=q["state"].to(backend.device), label_target=p["target"],
                     actions={n: a.to(backend.device) for n, a in actions.items()})
            queries[p["key"]] = q
        old_points = [queries[e["key"]] for e in plan["schedule"]]
        target_map = {q["record"]["id"]: q["label_target"] for q in old_points}
        old_plan = {"updates": plan["updates"], "audits": [0, plan["updates"]],
                    "targets": {a: target_map for a in ARMS}}
        initial_rng = rng_state()
        reports = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                   "attention_processors": sorted({type(p).__name__ for p in backend.model.attn_processors.values()}),
                   "scope": "All historical arm names use identical targets here: repeated updates, not a scientific comparison."}
        for mode, deterministic in (("default", False), ("deterministic", True)):
            torch.use_deterministic_algorithms(deterministic)
            backend.load_student(initial, refresh_behavior=False)
            restore_rng(initial_rng)
            backend.schedule(config.steps)
            with torch.no_grad():
                exact_actions = all(torch.equal(backend.predict(q["state"], q["index"], q["context"], "student"),
                                               q["actions"]["noop"]) for q in queries.values())
            assert exact_actions, f"{mode} changed the saved initial action"
            out = run / mode
            out.mkdir()
            result = fit_models(backend, old_points, config, old_plan, out)
            states = {a: torch.load(out / f"checkpoints/{a}/step{plan['updates']:03d}.pt",
                                   map_location="cpu", weights_only=True) for a in ARMS}
            comparisons = {a: compare_states(states[ARMS[0]], states[a]) for a in ARMS[1:]}
            reports[mode] = {"initial_actions_exact": exact_actions, "arms": result, "repeats": comparisons}
            write_json(run / "progress.json", reports)
            print({"event": "repeated_historical_fitter", "mode": mode, "comparisons": comparisons}, flush=True)
        assert all(c["exact"] for c in reports["deterministic"]["repeats"].values())
        backend.load_student(initial, refresh_behavior=False)
        restore_rng(initial_rng)
        out = run / "current"
        out.mkdir()
        current_plan = {"arms": ["single"], "schedules": {"single": [plan["schedule"]] * plan["updates"]},
                        "audits": old_plan["audits"], "reference_model_hashes": {"single": tensor_digest(states[ARMS[0]])}}
        reports["current"] = fit(backend, queries, config, current_plan, out)
        assert reports["current"]["single"]["historical_replay_exact"]
        reports.update(status="complete", counts=dict(backend.counters),
                       elapsed_seconds=time.perf_counter() - started, new_images=0, new_teacher_queries=0)
        write_json(run / "result.json", reports)
    except Exception as exc:
        write_json(run / "failure.json", {"error": repr(exc), "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        (run / "RUNNING").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True)
    run_diagnostic(parser.parse_args().run)
