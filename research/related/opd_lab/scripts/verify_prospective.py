"""Verify phase12 candidates, selected targets, blind labels and historical integrity."""
from collections import Counter
import json
from pathlib import Path
import subprocess

import torch

from opdlab.data import load_records
from opdlab.fitting_probe import tensor_digest
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.prospective import make_queries
from opdlab.runtime import verify_run
from review_prospective import compare, validate, visual
from verify_state_probes import historical_index


def main():
    campaign = ROOT / "experiments/local_expert_v12"
    result = read_json(campaign / "results.json")
    run = Path(result["review_inputs"]["run"])
    review = ROOT / "runs/le12_review_001"
    report = {"asset_audit":verify_run(run)}
    rows = validate(run)
    plan, gpu = read_json(run / "probe_plan.json"),read_json(run / "result.json")
    assert gpu == result["gpu_result"] and gpu["optimizer_updates"] == 0 and gpu["teacher_queries"] == 128
    assert plan["queries"] == make_queries(load_records(run / "data.jsonl","train"),plan["excluded_seeds"])
    failed = ROOT / "runs/le12_collect_001"
    assert verify_run(failed) == []
    assert read_json(failed / "probe_plan.json") == plan
    assert read_json(failed / "failure.json")["error"] == "KeyError('ocr')"
    assert not (failed / "candidates.jsonl").exists()
    assert sha256(ROOT / "python/opdlab/method.py") == plan["selector_source_sha256"]
    state = torch.load(plan["student"],map_location="cpu",weights_only=True)
    assert tensor_digest(state) == gpu["student_sha256"]
    anchor = read_json(run / "anchor.json")
    assert sha256(run / "anchor.json") == gpu["anchor_sha256"] and anchor["exact"]
    assert sha256(run / anchor["image"]) == anchor["image_sha256"] == plan["anchor"]["image_sha256"]
    for k,v in plan["expected_counts"].items():
        assert gpu["counts"][k] == v,(k,gpu["counts"].get(k))
    assert gpu["counts"].get("backward_queries",0) == 0
    counts = Counter()
    for row in rows:
        counts.update(row["counts"])
        saved = torch.load(run / row["saved_query"],map_location="cpu",weights_only=True)
        assert saved["record"]["id"] == row["id"] and saved["seed"] == row["seed"] and saved["index"] == 3
        assert tensor_digest({"state":saved["state"]}) == row["state_sha256"]
        assert torch.equal(saved["target"],saved["actions"][row["selected"]])
        for name,action in saved["actions"].items():
            assert tensor_digest({"action":action}) == row["action_hashes"][name]
            assert torch.isfinite(action).all() and not action.requires_grad
    for k,v in {"forward_behavior":1408,"forward_ocr":64,"forward_geneval":64,"decode_images":192,"verifier_images":192,"forward_examples":3008}.items():
        assert counts[k] == v
    frozen = read_json(review / "observations_frozen.json")
    assert frozen["frozen_before_key_join"]
    for n in ("observations","key","items"):
        assert sha256(review / f"{n}.json") == frozen[f"{n}_sha256"]
    assert sha256(review / "observations.json") == sha256(campaign / "observations.json")
    assert sha256(review / "observations_frozen.json") == sha256(campaign / "observations_frozen.json")
    assert sha256(run / "REVIEW_RUBRIC.md") == frozen["rubric_sha256"]
    seen = {}
    for e in result["joined_review"]:
        candidate,obs = e["candidate"],e["observation"]
        path = review / f"image-{e['case']:02d}-{e['view']}.png"
        assert sha256(path) == candidate["image_sha256"]
        label,h = (obs["clauses"],obs["full_visibility"]),candidate["image_sha256"]
        assert h not in seen or seen[h] == label
        seen[h] = label
    assert len(result["queries"]) == 64 and len(result["joined_review"]) == 192
    for q in result["queries"]:
        for name,c in q["comparisons"].items():
            assert c == compare(q["observations"]["noop"],q["observations"][name])
    for policy,metrics in result["policy_scores"].items():
        for metric in ("complete_scene","joint_clauses","full_visibility","has_unknown"):
            total = sum(visual(q["observations"][q["selected"] if policy == "selected" else policy])[metric] for q in result["queries"])
            assert total == metrics[metric]["sum"]
    diagnosis = read_json(campaign / "measurement_diagnosis.json")
    assert diagnosis["results_sha256"] == sha256(campaign / "results.json")
    assert diagnosis["script_sha256"] == sha256(ROOT / "scripts/diagnose_prospective.py")
    assert len(diagnosis["accepted_queries"]) == sum(q["accepted"] for q in result["queries"])
    harmed = sum(len(q["selected_comparison"]["harmed_clauses"]) for q in result["queries"] if q["accepted"])
    assert harmed == len(diagnosis["accepted_harmed_clause_witnesses"])
    for witness in diagnosis["accepted_harmed_clause_witnesses"]:
        q = next(q for q in result["queries"] if q["case"] == witness["case"])
        assert witness["clause"] in q["selected_comparison"]["harmed_clauses"]
        for name,prefix in (("noop","baseline"),(q["selected"],"selected")):
            candidate = next(e["candidate"] for e in result["joined_review"]
                             if e["case"] == q["case"] and e["candidate"]["name"] == name)
            clause_ids = [c["id"] for c in candidate["details"]["clauses"]]
            assert witness[f"{prefix}_proxy_satisfied"] == candidate["satisfied"][clause_ids.index(witness["clause"])]
    status = read_json(campaign / "slurm_status.json")
    assert sorted(j["state"] for j in status["jobs"]) == ["COMPLETED","FAILED"]
    assert next(j for j in status["jobs"] if j["state"] == "COMPLETED")["exit_code"] == "0:0"
    upstream = {}
    for name,spec in read_json(ROOT / "upstream.lock.json").items():
        path = ROOT / "upstream" / name
        assert subprocess.check_output(["git","status","--porcelain"],cwd=path).strip() == b""
        head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=path).decode().strip()
        assert head == spec["commit"]
        upstream[name] = head
    prior = {str(p):historical_index(p,revision) for p,revision in ((6,"e384a09"),(7,"8c1e160"),(8,"d9c392c"),(9,"d9c392c"),(10,"9dcc18b"),(11,"3923b6a"))}
    report.update(candidate_images=192,unique_candidate_images=len(seen),frozen_review_views=192,saved_states_and_all_actions=64,
                  automatic_selections_recomputed=64,exact_anchor_checks=1,unchanged_selector=True,aggregate_counts_valid=True,
                  retry_plan_identical=True,failed_attempt_retained=True,upstream=upstream,historical_artifacts=prior,
                  measurement_diagnosis_harm_witnesses_verified=harmed,
                  allocated_gpu_hours_including_failed_attempt=status["allocated_gpu_hours_including_failed_attempts_so_far"],
                  limits="Shared large pretrained assets use recorded inventories, not newly computed full content hashes; implementing-assistant review; single grid and seed per train prompt")
    write_json(campaign / "verification.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
