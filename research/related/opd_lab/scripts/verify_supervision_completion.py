"""Verify interrupted-job provenance separately from scientific result comparisons."""
from pathlib import Path

from opdlab.io import read_json, sha256
from opdlab.supervision_completion import inspect_training, key, rows, tasks


def validate_completion(run):
    plan, result = read_json(run / "probe_plan.json"), read_json(run / "result.json")
    recovery = plan["completion"]
    source = Path(recovery["source"])
    assert {k:v for k,v in plan.items() if k != "completion"} == read_json(source / "probe_plan.json")
    reports = inspect_training(source)
    assert result["arms"] == recovery["training_reports"] == reports
    for p in (source / "checkpoints").rglob("*.pt"):
        assert sha256(p) == sha256(run / p.relative_to(source))
    for p in (source / "eval_queries").glob("*.pt"):
        assert sha256(p) == sha256(run / p.relative_to(source))
    for name in ("updates.jsonl","metrics.jsonl"):
        assert sha256(source / name) == sha256(run / name)
    old, new = rows(source / "endpoints.jsonl"), rows(run / "endpoints.jsonl")
    replays, missing = tasks(plan, old)
    assert recovery["replay"] == replays and recovery["missing"] == missing
    bykey = {key(r):r for r in new}
    assert len(bykey) == len(new) == 185 and len(old) == 158 and len(missing) == 27
    assert all(bykey[key(r)] == r for r in old)
    actual = rows(run / "replays.jsonl")
    assert len(actual) == len(replays) == 7
    for a, b in zip(actual, replays):
        assert key(a) == key(b)
        for field in ("image_sha256","student_sha256","noise_sha256"):
            assert a[field] == b[field]
        assert sha256(run / a["image"]) == b["image_sha256"]
    completion = result["completion"]
    assert completion["source"] == str(source) and completion["new_optimizer_updates"] == 0
    assert (completion["source_endpoints"],completion["new_endpoints"],completion["exact_replays"]) == (158,27,31)
    for k,v in recovery["expected_runtime_counts"].items():
        assert completion["runtime_counts"].get(k,0) == v
    assert result["counts"] == plan["expected_counts"] and "reconstructed" in result["counts_basis"]
    assert all(r["elapsed_seconds"] is None for r in reports.values())
    return {"source":str(source),"training_reused_updates":384,"new_updates":0,"source_endpoints":158,
            "new_endpoints":27,"replay_images":31,"runtime_counts":completion["runtime_counts"],
            "counts_basis":result["counts_basis"],"source_per_arm_timers_available":False}
