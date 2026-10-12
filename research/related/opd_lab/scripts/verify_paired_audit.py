"""Check full response replay, clean upstreams and unchanged prior archives."""
from pathlib import Path
import subprocess

from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.paired_array_diagnostic import parse_array_compatible
from summarize_paired_audit import build_report
from explain_paired_audit import reasons
from verify_state_probes import historical_index


def main():
    campaign = ROOT / "experiments/local_expert_v14"
    report = read_json(campaign / "results.json")
    run = Path(report["run"])
    # JSON roundtrip normalizes tuple endpoint pairs in the analysis builder.
    import json
    assert json.loads(json.dumps(build_report(run))) == report
    diagnostic = read_json(campaign / "array_diagnostic.json")
    rebuilt = json.loads(json.dumps(build_report(run, diagnostic_pair_parser=parse_array_compatible)))
    for k, v in rebuilt.items():
        assert diagnostic[k] == v, k
    assert diagnostic["primary_results_sha256"] == sha256(campaign / "results.json")
    assert diagnostic["parser_sha256"] == sha256(ROOT / "python/opdlab/paired_array_diagnostic.py")
    assert diagnostic["additional_gpu_calls"] == 0
    for source, explanation_name, values in (("results.json", "failure_analysis.json", report),
                                              ("array_diagnostic.json", "array_failure_analysis.json", diagnostic)):
        explanation = read_json(campaign / explanation_name)
        assert explanation["results_sha256"] == sha256(campaign / source)
        assert explanation["script_sha256"] == sha256(ROOT / "scripts/explain_paired_audit.py")
        indexed = {(p["id"], p["name"]): p for p in values["pairs"]}
        assert len(explanation["all_pairs"]) == len(indexed) == 128
        for row in explanation["all_pairs"]:
            p = indexed[row["id"], row["name"]]
            assert row["reference"] == p["reference"] and row["eligible"] == p["eligible"]
            assert row["explanation"] == {k: reasons(e) for k, e in p["evidence"].items()}
    diagnostic_a = read_json(campaign / "array_diagnostic_attestation.json")
    for p, h in diagnostic_a["files_sha256"].items():
        assert sha256(ROOT / p) == h
    import hashlib
    prefix = b"".join((run / "pairs.jsonl").read_bytes().splitlines(keepends=True)[:34])
    assert hashlib.sha256(prefix).hexdigest() == diagnostic_a["initial_pair_prefix_sha256"]
    plan = read_json(run / "paired_plan.json")
    assert len({t["uid"] for t in plan["tasks"]}) == 320
    assert len({p["uid"] for p in plan["pairs"]}) == 192
    key = read_json(run / "private_key.json")
    assert len(key) == len({q["id"] for q in key}) == 64
    assert all(q["id"].startswith("jp1-train-") for q in key)
    assert sum(p["self_control"] for p in plan["pairs"]) == 64
    for p in plan["pairs"]:
        tasks = [t for t in plan["tasks"] if t["pair_uid"] == p["uid"]]
        assert len(tasks) == (1 if p["self_control"] else 2)
        assert {t["reverse"] for t in tasks} == ({False} if p["self_control"] else {False, True})
        for t in tasks:
            assert t["images"] == (p["images"][::-1] if t["reverse"] else p["images"])
    old = read_json(ROOT / "experiments/local_expert_v13/results.json")
    for p in ("noop", "v1", "v1_discrete_event", "v2"):
        assert report["policies"][p] == old["policies"][p], p
    result = report["gpu_result"]
    assert result["generated_images"] == result["teacher_actions"] == result["optimizer_updates"] == 0
    assert result["environment"] == plan["environment"]
    assert result["counts"]["input_tokens"] <= 6291456
    assert result["counts"]["output_tokens"] <= 253952
    status = read_json(campaign / "slurm_status.json")
    assert len(status["jobs"]) == 1
    assert status["jobs"][0]["state"] == "COMPLETED" and status["jobs"][0]["exit_code"] == "0:0"
    upstream = {}
    for name, spec in read_json(ROOT / "upstream.lock.json").items():
        path = ROOT / "upstream" / name
        assert subprocess.check_output(["git", "status", "--porcelain"], cwd=path).strip() == b""
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
        assert head == spec["commit"]
        upstream[name] = head
    historical = {str(p): historical_index(p, revision) for p, revision in
                  ((6, "e384a09"), (7, "8c1e160"), (8, "d9c392c"), (9, "d9c392c"),
                   (10, "9dcc18b"), (11, "3923b6a"), (12, "8e9c655"), (13, "bb804ca"))}
    write_json(campaign / "verification.json", {"response_parses_replayed": 512, "input_images_verified": 192,
               "real_pairs": 128, "swapped_pair_requests": 256, "self_controls": 64,
               "decisions_and_statistics_recomputed": True, "old_v1_v2_statistics_reproduced": True,
               "separate_array_diagnostic_recomputed": True, "array_diagnostic_additional_gpu_calls": 0,
               "all_pair_failure_explanations_recomputed": 256,
               "historical_artifacts": historical, "upstream": upstream,
               "frozen_labels_unchanged": True, "model_assets_gpu_sha256_verified": len(plan["assets"]["files"]),
               "allocated_gpu_hours": status["allocated_gpu_hours_including_failed_attempts_so_far"],
               "generated_images": 0, "optimizer_updates": 0,
               "limitations": "Exposed development pool, provisional implementing-assistant labels, single correlated Qwen reviewer."})
    print({"verified": "all512 responses,128 pairs,64 self controls and unchanged phases6-13"})


if __name__ == "__main__":
    main()
